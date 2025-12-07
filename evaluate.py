import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout
import os
import hydra
from omegaconf import OmegaConf, DictConfig
import evaluaters
import json
import socket
from typing import Optional, Set

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""

    TrainerClass = getattr(evaluaters, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)

    return trainer.test()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    print(OmegaConf.to_yaml(config))
    print('=' * 80)

    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {'device_map': 'auto'} if config.trainer == 'BasicTrainer' else {} # , "attn_implementation": 'eager'
    policy_dtype = getattr(torch, config.model.policy_dtype)

    if config.model.archive is not None and os.path.isdir(config.model.archive):
        policy = transformers.AutoModelForCausalLM.from_pretrained(config.model.archive, low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
        print("Load from", config.model.archive)
        model_name = config.model.archive.split('/')[-1]
    else:
        policy = transformers.AutoModelForCausalLM.from_pretrained(config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
        if config.model.archive is not None:    
            state_dict = torch.load(config.model.archive, map_location='cpu')
            step, metrics = state_dict['step_idx'], state_dict['metrics']
            print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
            policy.load_state_dict(state_dict['state'])
            print("Load the states", config.model.archive)
            model_name = config.model.archive.split('/')[-2]
        else:
            print("Load the pretrained model", config.model.name_or_path)
            model_name = config.model.name_or_path.split('/')[-1]

    disable_dropout(policy)
    print('starting single-process worker')
    metrics = worker_main(0, 1, config, policy, None)

    dataset = config.embed_dirs.split('/')[-1].split('_')[0]
    model = config.model.name_or_path.split('/')[-1]
    save_dir = f"./metrics_examples_new/{dataset}/{model}"
    os.makedirs(save_dir, exist_ok=True)
    json.dump(metrics, open(f"{save_dir}/{model_name}.json", 'w'))


if __name__ == '__main__':
    main()
