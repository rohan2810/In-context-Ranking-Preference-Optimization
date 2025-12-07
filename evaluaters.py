import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
import contextlib

from prompts import prompts
from evaluate_datasets import get_batch_iterator
from utils import (
    all_gather_if_needed,
    pad_to_length,
    rank0_print,
    move_batch_to_device,
    get_local_dir,
)
import numpy as np
import tqdm

import random
import os
from collections import defaultdict
from typing import Optional, Dict, List, Union, Tuple
import math

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
        self.saved_rankings: List[Dict[str, Union[str, List]]] = []
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
            shuffle=False,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            # sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy
        self.reference_model = reference_model

        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='eval', n_epochs=config.n_epochs, batch_size=config.eval_batch_size, silent=rank != 0,
                                                cache_dir=get_local_dir(config.local_dirs), embed_dir=config.embed_dirs, prompt_template=prompts[config.model.name_or_path], num_negtive=config.num_negtive) 
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""
        print("Getting batch samples")
        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        device = next(self.policy.parameters()).device

        # 1) Extract prompt IDs/masks and response IDs/masks from the batch
        # prompt_ids   = batch["prompt_input_ids"].to(device)       # (B, L_p)
        # prompt_mask  = batch["prompt_attention_mask"].to(device)  # (B, L_p)
        response_ids   = batch["response_input_ids"].to(device)   # (B, L_r)
        response_mask  = batch["response_attention_mask"].to(device)  # (B, L_r)        
        # concatenated_ids  = torch.cat([prompt_ids, response_ids], dim=1)    # (B, L_p + L_r)
        # concatenated_mask = torch.cat([prompt_mask, response_mask], dim=1)  # (B, L_p + L_r)
        total_length = response_ids.shape[1]
        max_new      = 256 
        max_len      = total_length + max_new
        self.tokenizer.padding_side = "left"
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                response_ids,
                attention_mask = response_mask,
                max_length     = max_len,
                pad_token_id   = self.tokenizer.pad_token_id,
                do_sample      = False,                  # greedy is fine for reranking
                temperature = None,
                top_p = None,
                top_k = None,
                )
                # batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, 
                # pad_token_id=self.tokenizer.pad_token_id)

        # if self.config.loss.name in {'dpo', 'ipo', 'rdpo'}:
        #     ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        #     with ctx():
        #         reference_output = self.reference_model.generate(
        #             batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        # policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        # if self.config.loss.name in {'dpo', 'ipo', 'rdpo'}:
        #     reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
        #     reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
        #     reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        # else:
        #     reference_output_decoded = []

        return policy_output_decoded, None
    
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
    
    def listwise_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass for listwise preference learning.
        Args:
            model: The language model.
            batch: Batch containing movie sequences and metadata.
        Returns:
            List of dictionaries mapping each movie to its log probability.
        """
        concatenated_batch = listwise_concatenated_inputs(batch)
        prompts = concatenated_batch.get('prompt')
        relevance_ranks = concatenated_batch.get('relevance_rank')
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

        best_movie_logps = dict()
        for movie_logp, prompt, relevance_rank in zip(movie_logps, prompts, relevance_ranks):
            avg_logp = sum([v for k, v in movie_logp.items()]) / len(movie_logp)
            if prompt not in best_movie_logps or avg_logp > best_movie_logps[prompt]["logit"]:
                best_movie_logps[prompt] = {"logit": avg_logp, "movie_logp": {k: torch.tensor(len(movie_logp) - idx).float() for idx, (k, v) in enumerate(movie_logp.items())}, "relevance_rank": relevance_rank}
        relevance_ranks = [v["relevance_rank"] for k, v in best_movie_logps.items()]
        best_movie_logps = [v["movie_logp"] for k, v in best_movie_logps.items()]
        return relevance_ranks, best_movie_logps

    def get_batch_metrics(self, batch: Dict[str, Union[str, List]], loss_config, train: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the R-DPO loss and aggregated metrics for the given batch.
        This version computes the average top-K mean relevance, NDCG, Recall, and MRR over samples.
        """
        metrics = {}
        mode = 'train' if train else 'eval'
        
        # Define the k values for our metrics
        k_values = [1, 3, 5, 10, 20]
        agg_topk_mean_rel = {k: [] for k in k_values}
        agg_ndcg = {k: [] for k in k_values}
        agg_recall = {k: [] for k in k_values}
        agg_mrr = []  # MRR is computed per sample


        # Forward
        if loss_config.name in  {"rdpo"}:
            policy_logps = self.concatenated_forward(self.policy, batch)
            policy_logps = [
                {movie: value.to(self.device) for movie, value in sample.items()}
                for sample in policy_logps
            ]
            relevance_rank = batch["relevance_rank"]
        elif loss_config.name in  {"sft", "lipo"}:
            relevance_rank, policy_logps = self.listwise_forward(self.policy, batch)
            policy_logps = [
                {movie: value.to(self.device) for movie, value in sample.items()}
                for sample in policy_logps
            ]
        else:
            raise NotImplementedError

        # Process each sample using the ordering provided by the batch
        for sample_idx, policy_logps_sample in enumerate(policy_logps):
            candidate_texts = batch["movies"][sample_idx]            # List[str], length=K
            rel_pairs       = relevance_rank[sample_idx]             # List[(ψ, τ)] in same order
            relevance_dict  = {
                movie_text: rel_pairs[i][0]
                for i, movie_text in enumerate(candidate_texts)
            }

            # 2) Sort the (movie_text, log-prob) items descending by log-prob
            sorted_items = sorted(
                policy_logps_sample.items(),
                key=lambda kv: kv[1],
                reverse=True
            )
            ranked_candidates = [kv[0] for kv in sorted_items]      # List[str]

            # 3) Save for reporting
            prompt_text = batch["prompt"][sample_idx]
            self.saved_rankings.append({
                "prompt": prompt_text,
                "ranked_candidates": ranked_candidates,
            })

            # 4) Build true_relevance tensor in the same order as ranked_candidates
            true_relevance = torch.tensor(
                [relevance_dict[movie_text] for movie_text in ranked_candidates],
                device=self.device,
                dtype=torch.float
            )
            total_relevant = (true_relevance > 0).sum().item()
            
            
            mrr_value = 0.0
            for rank_idx, rel in enumerate(true_relevance):
                if rel > 0:
                    mrr_value = 1.0 / (rank_idx + 1)
                    break
            agg_mrr.append(mrr_value)

            # 6) Compute Top-K metrics
            for k in k_values:
                if k <= len(ranked_candidates):
                    top_k_rel = true_relevance[:k]  # tensor of length k
                    
                    # Top-K mean relevance
                    top_k_mean_rel = top_k_rel.mean().item()
                    agg_topk_mean_rel[k].append(top_k_mean_rel)
                    
                    # NDCG@k
                    gains     = (2 ** top_k_rel - 1)
                    discounts = torch.log2(torch.arange(2, k + 2, device=self.device).float())
                    dcg       = (gains / discounts).sum()

                    # Ideal DCG: sort true_relevance descending, take first k
                    ideal_rel, _ = torch.sort(true_relevance, descending=True)
                    ideal_top_k  = ideal_rel[:k]
                    ideal_gains  = (2 ** ideal_top_k - 1)
                    idcg         = (ideal_gains / discounts).sum()

                    ndcg_val = (dcg / idcg).item() if idcg > 0 else 0.0
                    agg_ndcg[k].append(ndcg_val)
                    
                    # Recall@k
                    if total_relevant > 0:
                        relevant_in_top_k = (top_k_rel > 0).sum().item()
                        recall_k = relevant_in_top_k / total_relevant
                    else:
                        recall_k = 0.0
                    agg_recall[k].append(recall_k)
        

        # if self.rank == 0 and saved_rankings:
        #     import os, json
        #     output_path = os.path.join(self.run_dir, "reranked_examples.json")
        #     print(f"Saving {len(saved_rankings)} reranked examples to {output_path}")
        #     with open(output_path, "w") as fp:
        #         json.dump(saved_rankings, fp, indent=2)

        # Compute aggregated (mean) metrics across samples for each k value
        for k in k_values:
            if agg_topk_mean_rel[k]:
                avg_topk_mean_rel = sum(agg_topk_mean_rel[k]) / len(agg_topk_mean_rel[k])
                metrics[f'mr@{k}'] = avg_topk_mean_rel
            if agg_ndcg[k]:
                avg_ndcg = sum(agg_ndcg[k]) / len(agg_ndcg[k])
                metrics[f'ndcg@{k}'] = avg_ndcg
            if agg_recall[k]:
                avg_recall = sum(agg_recall[k]) / len(agg_recall[k])
                metrics[f'recall@{k}'] = avg_recall
        
        # Compute aggregated MRR across samples
        if agg_mrr:
            avg_mrr = sum(agg_mrr) / len(agg_mrr)
            metrics[f'mrr'] = avg_mrr
        
        return metrics

    def test(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.example_counter = 0
        self.batch_counter = 0
        
        
        ############################################################################################## BEGIN SAVE EXAMPLES ##############################################################################################
        saved_examples: List[Dict[str,str]] = []
        saved_count = 0
        max_save = getattr(self.config, "num_save_examples", 5)

        ############################################################################################## END SAVE EXAMPLES ##############################################################################################

        ############################################################################################## BEGIN EVALUATION ##############################################################################################      
        rank0_print(f'Running evaluation after {self.example_counter} train examples')
        self.policy.eval()


        all_eval_metrics = defaultdict(list)

        for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
            local_microbatch = move_batch_to_device(eval_batch, device=self.device)


            # if self.rank == 0 and saved_count < max_save:
            #     print("Saving examples")
            #     # get_batch_samples returns: (policy_output_decoded, reference_output_decoded)
            #     policy_gen, _ = self.get_batch_samples(local_microbatch)
            #     # eval_batch['prompt'] is a list of prompt‐strings
            #     prompts = eval_batch["prompt"]  
            #     for prompt_str, gen_str in zip(prompts, policy_gen):
            #         if saved_count >= max_save:
            #             break
            #         saved_examples.append({
            #             "prompt": prompt_str,
            #             "generated": gen_str
            #         })
            #         saved_count += 1



            with torch.no_grad():
                eval_metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=False)
                print(f"Eval metrics: {eval_metrics}")

            for k, v in eval_metrics.items():
                all_eval_metrics[k].append(v)

        mean_eval_metrics = {
            k: sum(v) / len(v) for k, v in all_eval_metrics.items() if v and isinstance(v[0], (int, float))
        }
        # if self.rank == 0 and saved_examples:
        #     import os, json
        #     save_path = os.path.join(self.run_dir, "generated_examples.json")
        #     rank0_print(f"Saving {len(saved_examples)} generated example(s) to {save_path}")
        #     with open(save_path, "w") as fp:
        #         json.dump(saved_examples, fp, indent=2)
        if self.rank == 0 and self.saved_rankings:
            import os, json
            save_path = os.path.join(self.run_dir, "reranked_examples.json")
            rank0_print(f"Saving {len(self.saved_rankings)} reranked example(s) to {save_path}")
            with open(save_path, "w") as fp:
                json.dump(self.saved_rankings, fp, indent=2)        
        return mean_eval_metrics

        ############################################################################################## END EVALUATION ##############################################################################################

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
