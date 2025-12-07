import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from prompts import prompts
from train_datasets import get_batch_iterator
from utils import (
    formatted_dict,
    slice_and_move_batch_for_device,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    move_batch_to_device,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple
import math

def lipo_lambda_loss(
    policy_logps: List[Dict[str, torch.Tensor]],
    reference_logps: List[Dict[str, torch.Tensor]],
    relevance_rank: List[List[Tuple[int, int]]],
    beta: float = 0.1,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    
    device = next(iter(policy_logps[0].values())).device
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    sample_losses = []

    for sample_idx in range(len(policy_logps)):
        policy_sample = policy_logps[sample_idx]
        reference_sample = reference_logps[sample_idx]
        ranking_sample = relevance_rank[sample_idx]
        items = list(policy_sample.keys())
        
        r = torch.stack([policy_sample[i] - reference_sample[i] for i in items], dim=0)
        s = beta * r

        rels, ranks = zip(*ranking_sample)
        rels = torch.tensor(rels, dtype=torch.float, device=device)
        max_rel = rels.max()
        psi = rels / max_rel if max_rel > 0 else rels

        tau = torch.tensor(ranks, dtype=torch.float, device=device)

        # Gains and discounts
        G = 2 * psi - 1                          
        D = torch.log1p(tau)

        K = s.shape[0]
        loss_b = torch.tensor(0.0, device=device, requires_grad=True)
        for i in range(K):
            for j in range(K):
                if psi[i] > psi[j]:
                    weight_ij = torch.abs(G[i] - G[j]) * torch.abs(1.0 / D[i] - 1.0 / D[j])
                    loss_b = loss_b + weight_ij * ( - F.logsigmoid(-(s[i] - s[j])))

        loss_b = -loss_b
        total_loss = total_loss + loss_b
        sample_losses.append(loss_b)

    return total_loss, sample_losses

def rdpo_loss(
    policy_logps: List[Dict[str, torch.FloatTensor]],
    reference_logps: List[Dict[str, torch.FloatTensor]],
    relevance_rank: List[List[Tuple[int, int]]],
    beta: float = 0.1, ablation: int = 0
) -> Tuple[torch.Tensor, List[torch.Tensor], List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:  
    """
    Computes R-DPO loss for a batch with multiple samples, each with multiple candidates.

    Args:
        policy_logps: List of dictionaries mapping movie names to log probabilities (policy model).
        reference_logps: List of dictionaries mapping movie names to log probabilities (reference model).
        relevance_rank: List (per sample) of lists (per movie) of (relevance, rank) tuples.
                      The order must match the movie order in the batch.
        beta: Temperature parameter for the DPO loss.

    Returns:
        A tuple of:
            - total_loss: Scalar R-DPO loss aggregated over the batch.
            - sample_losses: List of per-sample losses.
            - rewards: List of per-sample rewards dictionaries.
    """
    device = next(iter(policy_logps[0].values())).device
    total_loss = torch.tensor(0.0, device=device)
    sample_losses = []
    avg_m = []

    for sample_idx in range(len(policy_logps)):
        policy_sample = policy_logps[sample_idx]
        reference_sample = reference_logps[sample_idx]
        ranking_sample = relevance_rank[sample_idx]  # List of (relevance, rank) tuples
        movies = list(policy_sample.keys())

        r = torch.stack([policy_sample[movie] - reference_sample[movie] for movie in movies])

        # Extract relevance and rank from ranking_sample.
        relevances = torch.tensor([item[0] for item in ranking_sample][:len(r)]).float().to(device)
        ranks = torch.tensor([item[1] for item in ranking_sample][:len(r)]).float().to(device)
        # 
        # 

        # original
        # Compute weights: (2**relevance - 1) / log2(1 + rank)
        if ablation == 0:
            weights = (2 ** relevances - 1) / torch.log2(1 + ranks)

        # Ablation(1) --> w(i) = 1/log(1+i)
        elif ablation == 1:
            weights = 1.0 / torch.log2(1 + ranks)

        # Ablation(2) --> w(i) = (2^y - 1) / i
        elif ablation == 2:
            weights = (2 ** relevances - 1) / ranks
        else:
            weights = (2 ** relevances - 1) / torch.log2(1 + ranks)
        

        # Compute pairwise differences: for each i, j: diff = r_j - r_i.
        # r.unsqueeze(0) has shape (1, N) and r.unsqueeze(1) has shape (N, 1) so that
        # their difference yields a (N, N) tensor where element (i,j) is r[j] - r[i].
        diffs = (r.unsqueeze(0) - r.unsqueeze(1)) - torch.eye(r.shape[0]).to(device) * 100

        # Compute sum_exp for each i: sum_{j} exp(beta*(r_j - r_i))
        log_sum_exp = torch.logsumexp(beta * diffs, dim=1) - np.log(diffs.shape[0] - 1)

        # Compute the log sigmoid loss for each movie.
        losses = weights * F.logsigmoid(-log_sum_exp)

        # Aggregate the sample loss.
        sample_loss = -losses.mean()

        total_loss = total_loss + sample_loss
        sample_losses.append(sample_loss)
        # avg_m.append((r.unsqueeze(0) - r.unsqueeze(1)).detach().clone().mean(-1)[:10])
    # avg_m = torch.stack(avg_m).mean(0).tolist()
    avg_m = None
    return total_loss, sample_losses, None, None, avg_m

def sft_loss(
    policy_logps: List[Dict[str, torch.FloatTensor]],
    relevance_rank: List[List[Tuple[int, int]]],
):
    device = next(iter(policy_logps[0].values())).device
    total_loss = torch.tensor(0.0, device=device)

    for sample_idx in range(len(policy_logps)):
        policy_sample = policy_logps[sample_idx]
        ranking_sample = relevance_rank[sample_idx]  # List of (relevance, rank) tuples
        movies = list(policy_sample.keys())

        logits = torch.stack([policy_sample[movie] for movie in movies])

        # Extract relevance and rank from ranking_sample.
        relevances = torch.tensor([item[0] for item in ranking_sample][:len(logits)]).float().to(device)

        # Compute the log sigmoid loss for each movie.
        losses = torch.log_softmax(logits, dim=-1)

        # Aggregate the sample loss.
        sample_loss = -losses.mean()

        total_loss = total_loss + sample_loss
    return total_loss

def listwise_concatenated_inputs(batch: Dict[str, Union[str, List, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Create a concatenated version of the batch inputs.
    """
    concatenated_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if k.startswith('response_'):
                concatenated_key = f'concatenated_{k[9:]}'
            else:
                concatenated_key = f'concatenated_{k}'
            concatenated_batch[concatenated_key] = v
        else:
            concatenated_batch[k] = v
    concatenated_batch['relevance_rank'] = batch['relevance_rank']
    return concatenated_batch

class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting either SFT or DPO training.
           
           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        
        print(config)

        self.device = torch.cuda.current_device()
        rank0_print(f'Using GPU device {self.device}')
        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=config.shuffle,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            # sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy
        self.reference_model = reference_model

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size,
                                                 silent=rank != 0, cache_dir=get_local_dir(config.local_dirs), embed_dir=config.embed_dirs, prompt_template=prompts[config.model.name_or_path])
        rank0_print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0,
                                                cache_dir=get_local_dir(config.local_dirs), embed_dir=config.embed_dirs, prompt_template=prompts[config.model.name_or_path])
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in {'dpo', 'ipo', 'rdpo', 'lipo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)
        # if self.example_counter % 1500 == 0:  # Print every 500 examples
                # print(f"[Debug GET BATCH SAMPLES] Example {self.example_counter}")
                # print("Prompt:", batch.get("prompt", "[No prompt found]"))
                # print("Expected:", batch.get("response", "[No expected response found]"))
                # print("Model Output:", policy_output_decoded)
        if self.config.loss.name in {'dpo', 'ipo', 'rdpo', 'lipo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    
    def calculate_movie_logps(self, all_logits: torch.FloatTensor, concatenated_batch: Dict[str, torch.LongTensor], movies_list: List[str], average_log_prob=False) -> List[Dict[str, torch.Tensor]]:
        """
        Calculate the log probabilities for each movie in the response, per sample.
        Args:
            all_logits: Logits from the model (batch_size, sequence_length, vocab_size).
            concatenated_batch: Dictionary containing input IDs.
            movies_list: List of movies for each sample.
        Returns:
            A list of dictionaries mapping movie names to their log probabilities.
        """
        log_probs = all_logits.log_softmax(dim=-1)
        input_ids = concatenated_batch['concatenated_input_ids']
        movie_ids = concatenated_batch['response_movie_ids']
        batch_size, sequence_length, vocab_size = log_probs.shape
        batch_movie_logps = []
        if isinstance(movies_list[0], str):
            movies_list = [movies_list]
        for batch_idx in range(batch_size):
            sample_movies = movies_list[batch_idx]
            sample_movie_logps = {}
            sample_log_probs = log_probs[batch_idx]  
            sample_input_ids = input_ids[batch_idx]
            sample_movie_ids = movie_ids[batch_idx]
            if not isinstance(sample_movie_ids, torch.Tensor):
                    sample_movie_ids = torch.tensor(sample_movie_ids, device=log_probs.device)
                    
            for movie_index, movie in enumerate(sample_movies):
                mask = (sample_movie_ids == movie_index)
                if mask.dim() == 0:
                    mask = mask.unsqueeze(0)                
                token_positions = mask.nonzero(as_tuple=False).squeeze(1)                    
                if token_positions.numel() == 0:
                    print(f"Warning: No tokens found for movie '{movie}' in sample {batch_idx}.")
                    sample_movie_logps[movie] = torch.tensor(-100.0, device=log_probs.device)
                    continue      
                movie_token_ids = sample_input_ids[token_positions]         # (num_tokens,)
                token_logits = sample_log_probs[token_positions]    # (num_tokens, vocab_size)
                token_correct_logps = token_logits.gather(1, movie_token_ids.unsqueeze(1)).squeeze(1)   
                movie_logp = token_correct_logps.mean() if average_log_prob else token_correct_logps.sum()
                sample_movie_logps[movie] = movie_logp
            batch_movie_logps.append(sample_movie_logps)
        return batch_movie_logps

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass for listwise preference learning.
        Args:
            model: The language model.
            batch: Batch containing movie sequences and metadata.
        Returns:
            List of dictionaries mapping each movie to its log probability.
        """
        concatenated_batch = listwise_concatenated_inputs(batch)
        input_ids = concatenated_batch.get('concatenated_input_ids')
        movie_ids = concatenated_batch.get('response_movie_ids')
        attention_mask = concatenated_batch.get('concatenated_attention_mask')
        movies_list = concatenated_batch.get('movies', [])
        if input_ids is None or attention_mask is None or movie_ids is None:
            raise ValueError("Missing concatenated_input_ids or concatenated_attention_mask in batch!")
        if not isinstance(movies_list, list):
            raise ValueError("Expected movies to be a list, but got:", type(movies_list))
        all_logits = model(input_ids, attention_mask=attention_mask).logits.to(torch.float32)
        movie_logps = self.calculate_movie_logps(all_logits=all_logits, concatenated_batch=concatenated_batch, movies_list=movies_list, average_log_prob=True)
        return movie_logps

    def get_batch_metrics(self, batch: Dict[str, Union[str, List]], loss_config, train: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the R-DPO loss and aggregated metrics for the given batch.
        This version computes the average top-K mean relevance and NDCG over samples.
        """
        metrics = {}
        mode = 'train' if train else 'eval'
        
        # Forward pass on policy and reference models
        policy_logps = self.concatenated_forward(self.policy, batch)
        policy_logps = [
            {movie: value.to(self.device) for movie, value in sample.items()}
            for sample in policy_logps
        ]

        if loss_config.name in {'dpo', 'ipo', 'rdpo', 'lipo'}:
            with torch.no_grad():
                reference_logps = self.concatenated_forward(self.reference_model, batch)
                reference_logps = [
                {movie: value.to(self.device) for movie, value in sample.items()}
                for sample in reference_logps
            ]
        else:
            reference_logps = None
        
        # Compute R-DPO loss
        if loss_config.name == "rdpo":
            loss, _, final_rewards,discounted_rewards, ave_margin = rdpo_loss(
                policy_logps=policy_logps,
                reference_logps=reference_logps,
                relevance_rank=batch["relevance_rank"],
                beta=loss_config.beta,
                ablation=loss_config.ablation
            )
        elif loss_config.name == "sft":
            loss = sft_loss(
                policy_logps=policy_logps,
                relevance_rank=batch["relevance_rank"],
            )
        elif loss_config.name == "lipo":
            loss, sample_losses = lipo_lambda_loss(
                policy_logps=policy_logps,
                reference_logps=reference_logps,
                relevance_rank=batch["relevance_rank"],
                beta=loss_config.beta
            )
            
        metrics[f'loss/{mode}'] = loss.item()
        
        # Define the k values for our metrics
        k_values = [1, 2, 3, 4, 5]
        agg_topk_mean_rel = {k: [] for k in k_values}
        agg_ndcg = {k: [] for k in k_values}
        
        # Process each sample using the ordering provided by the batch
        for sample_idx, policy_logps_sample in enumerate(policy_logps):
            movies_list = batch["movies"][sample_idx]
            # Get policy scores in the order of movies from the batch
            policy_scores = torch.stack([policy_logps_sample[movie] for movie in movies_list])
            predicted_ranks = torch.argsort(policy_scores, descending=True)
            
            # Map each movie to its true relevance (from ranking info)
            relevance_dict = {}
            for movie, movie_rankings in zip(movies_list, batch["relevance_rank"][sample_idx]):
                relevance_dict[movie] = movie_rankings[0]
            
            true_relevance = torch.tensor(
                [relevance_dict[movie] for movie in movies_list],
                device=policy_scores.device,
                dtype=torch.float
            )
            # Compute metrics for each k for this sample and add to aggregator
            for k in k_values:
                if k <= len(predicted_ranks):
                    top_k_indices = predicted_ranks[:k]
                    top_k_relevance = true_relevance[top_k_indices]
                    # Top-K mean relevance
                    top_k_mean_rel = top_k_relevance.mean().item()
                    agg_topk_mean_rel[k].append(top_k_mean_rel)
                    
                    # Compute NDCG
                    gains = (2 ** top_k_relevance - 1)
                    discounts = torch.log2(torch.arange(2, k + 2, device=policy_scores.device).float())
                    dcg = (gains / discounts).sum()
                    
                    ideal_true_relevance, _ = torch.sort(true_relevance, descending=True)
                    ideal_top_k = ideal_true_relevance[:k]
                    ideal_gains = (2 ** ideal_top_k - 1)
                    idcg = (ideal_gains / discounts).sum()
                    
                    ndcg = (dcg / idcg).item() if idcg > 0 else 0.0
                    agg_ndcg[k].append(ndcg)
        
        # Compute aggregated (mean) metrics across samples for each k value
        for k in k_values:
            if agg_topk_mean_rel[k]:
                avg_topk_mean_rel = sum(agg_topk_mean_rel[k]) / len(agg_topk_mean_rel[k])
                metrics[f'top{k}_mean_relevance/{mode}_avg'] = avg_topk_mean_rel
            if agg_ndcg[k]:
                avg_ndcg = sum(agg_ndcg[k]) / len(agg_ndcg[k])
                metrics[f'ndcg@{k}/{mode}_avg'] = avg_ndcg
            # if ave_margin[k]:
            #     metrics[f'margin@{k}/{mode}_avg'] = ave_margin[k]
        return loss, metrics

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo', 'rdpo', 'lipo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:               
            ############################################################################################## BEGIN EVALUATION ##############################################################################################
                # print(f"\n[Debug] Example {self.example_counter}")
                # print("Prompt:", batch.get("prompt", "[No prompt found]"))
                # print("Expected Output:", batch.get("response", "[No expected response found]"))          
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                # if self.config.sample_during_eval:
                #     all_policy_samples, all_reference_samples = [], []
                #     policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                #     if self.config.loss.name in {'dpo', 'ipo', 'rdpo'}:
                #         reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    local_microbatch = move_batch_to_device(eval_batch, device=self.device)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=False)
                        print(f"Eval metrics: {eval_metrics}")

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].append(v)

                # if self.config.sample_during_eval:
                #     if self.config.n_eval_model_samples < self.config.eval_batch_size:
                #         rank0_print(
                #             f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                #         sample_batches = self.eval_batches[:1]
                #     else:
                #         n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                #         sample_batches = self.eval_batches[:n_sample_batches]
                #     for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                #         local_eval_batch = move_batch_to_device(eval_batch, self.rank)
                #         policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                #         all_policy_samples.extend(policy_samples)
                #         all_reference_samples.extend(reference_samples)

                #         for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                #             policy_text_table.add_data(self.example_counter, prompt, sample)
                #         if self.config.loss.name in {'dpo', 'ipo', 'rdpo'}:
                #             for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                #                 reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {
                    k: sum(v) / len(v) for k, v in all_eval_metrics.items() if v and isinstance(v[0], (int, float))
                }

                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                # if self.config.sample_during_eval:
                #     rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                #     if self.config.loss.name in {'dpo', 'ipo', 'rdpo'}:
                #         rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    # if self.config.sample_during_eval:
                    #     wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                    #     if self.config.loss.name in {'dpo', 'ipo', 'rdpo'}:
                    #         wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

                if self.example_counter > 0 and (self.example_counter % (self.config.eval_every * 30) == 0):
                    if self.config.debug:
                        rank0_print('skipping save in debug mode')
                    else:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, mean_eval_metrics)
            ############################################################################################## END EVALUATION ##############################################################################################

            ############################################################################################## BEGIN TRAINING ##############################################################################################
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                loss, metrics = self.get_batch_metrics(global_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].append(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            ############################################################################################## END TRAINING ##############################################################################################

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        self.write_state_dict(self.example_counter, self.policy.state_dict(), metrics, 'policy.pt', output_dir)

        # self.write_state_dict(self.example_counter, self.optimizer.state_dict(), metrics, 'optimizer.pt', output_dir)

        # self.write_state_dict(self.example_counter, self.scheduler.state_dict(), metrics, 'scheduler.pt', output_dir)



















class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.
        
           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class}, )

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in {'dpo', 'ipo', 'rdpo', 'lipo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        dist.barrier()


class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)

        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {'dpo', 'ipo', 'rdpo', 'lipo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()

        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
