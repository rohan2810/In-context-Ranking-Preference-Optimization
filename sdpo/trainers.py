import torch
from prompts import prompts
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

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
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


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    # print(f"chosen:{chosen_logratios}")
    rejected_logratios = {}
    for key in policy_rejected_logps:
        rejected_logratios[key] = policy_rejected_logps[key] - reference_rejected_logps[key]
        # print(f"{key}_logratios:{rejected_logratios[key].shape}")
    # if reference_free:
    #     ref_logratios = 0

    # logits = pi_logratios - ref_logratios
    temp = sum(torch.exp(beta * (rejected_logratios[key] - chosen_logratios)) for key in rejected_logratios)
    temp1 = -torch.log(temp)
    losses = -F.logsigmoid(temp1)
    # losses = -F.logsigmoid(self.beta * logits)
    rejected_rewards = {}
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    for key in policy_rejected_logps:
        rejected_rewards[key] = beta * (policy_rejected_logps[key] - reference_rejected_logps[key]).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    rejected_max_len = max([batch[key].shape[1] for key in batch if key.startswith("rejected") and key.endswith("_input_ids")])
    max_length = max(batch["chosen_input_ids"].shape[1], rejected_max_len)
    concatenated_batch = {}
    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            # concatenated_key = k.replace("rejected", "concatenated")
            prefix = k.split("_")[0]
            concatenated_key = "concatenated" + k[len(prefix):] 
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0)
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

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy
        self.reference_model = reference_model

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs),embed_dir=config.embed_dirs,prompt_template=prompts[config.model.name_or_path])
        rank0_print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs),embed_dir=config.embed_dirs,prompt_template=prompts[config.model.name_or_path])
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in {'dpo', 'ipo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    
    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor], torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        all_logps = _get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        step = batch["chosen_input_ids"].shape[0]
        rejected_logps = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logps[f"rejected{cnt}"] = all_logps[step*cnt : step*(cnt+1)]

        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        rejected_logits = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logits[f"rejected{cnt}"] = all_logits[step*cnt : step*(cnt+1)]
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)


    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'

        policy_chosen_logps, policy_rejected_logps,policy_chosen_logits,policy_rejected_logits = self.concatenated_forward(self.policy, batch)
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps,_,_ = self.concatenated_forward(self.reference_model, batch)

        if loss_config.name == 'dpo':
            loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}
        else:
            raise ValueError(f'unknown loss {loss_config.name}')

        losses, chosen_rewards, rejected_rewards = preference_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)
        reward_accuracies = None
        for key in rejected_rewards:
            if reward_accuracies is None:
                reward_accuracies = (chosen_rewards > rejected_rewards[key]).float()
            else:
                reward_accuracies *= (chosen_rewards > rejected_rewards[key]).float()

        prefix = "eval_" if train_test == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/{key}"] = rejected_rewards[key].cpu().numpy().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/margins-{key}"] = (chosen_rewards - rejected_rewards[key]).cpu().numpy().mean()
        for key in policy_rejected_logps:    
            metrics[f"{prefix}logps/rejected-{key}"] = policy_rejected_logps[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        for key in policy_rejected_logits:    
            metrics[f"{prefix}logits/rejected-{key}"] = policy_rejected_logits[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()

        return losses.mean(), metrics

        # chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
        # rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
        # reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

        # metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
        # metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
        # metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
        # metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

        # policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
        # metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        # policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        # metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        # all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        # metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()


    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
    
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name in {'dpo', 'ipo'}:
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].append(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.config.loss.name in {'dpo', 'ipo'}:
                            for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                if self.config.sample_during_eval:                    
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name in {'dpo', 'ipo'}:
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.config.loss.name in {'dpo', 'ipo'}:
                            wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

                if self.example_counter > 0 and (self.example_counter % (self.config.eval_every * 30) == 0):
                    if self.config.debug:
                        rank0_print('skipping save in debug mode')
                    else:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, mean_eval_metrics)
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
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
            #### END TRAINING ####


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

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        # optimizer_state_dict = self.optimizer.state_dict()
        # self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        # del optimizer_state_dict

        # scheduler_state_dict = self.scheduler.state_dict()
        # self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)