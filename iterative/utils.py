import os
import getpass
from datetime import datetime
import torch
import random
import numpy as np
import torch.distributed as dist
import inspect
import importlib.util
import socket
import os
from typing import Dict, Union, Type, List
import math
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple

def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) # bind to all interfaces and use an OS provided port
        return s.getsockname()[1] # return only the port number


def get_remote_file(remote_path, local_path=None):
    hostname, path = remote_path.split(':')
    local_hostname = socket.gethostname()
    if hostname == local_hostname or hostname == local_hostname[:local_hostname.find('.')]:
        return path
    
    if local_path is None:
        local_path = path
    # local_path = local_path.replace('/scr-ssd', '/scr')    
    if os.path.exists(local_path):
        return local_path
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    print(f'Copying {hostname}:{path} to {local_path}')
    os.system(f'scp {remote_path} {local_path}')
    return local_path


def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"
    

def get_local_run_dir(exp_name: str, local_dirs: List[str]) -> str:
    """Create a local directory to store outputs for this run, and return its path."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = f"{get_local_dir(local_dirs)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def slice_and_move_batch_for_device(batch: dict, idx: int, total: int, device: str) -> dict:
    total_samples = len(list(batch.values())[0])
    chunk_size = math.ceil(total_samples / total)
    start = idx * chunk_size
    end = min(start + chunk_size, total_samples)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device

def move_batch_to_device(batch: Dict, device: str) -> Dict:
    """Move a batch to the specified device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}
    

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_gpu_memory(rank: int = None, message: str = ''):
    """Print the amount of GPU memory currently allocated for each GPU."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.device(f'cuda:{i}')
            allocated_bytes = torch.cuda.memory_allocated(device)
            if allocated_bytes == 0:
                continue
            print('*' * 40)
            print(f'[{message} rank {rank} ] GPU {i}: {allocated_bytes / 1024**2:.2f} MB')
        print('*' * 40)


def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")


def get_block_class_from_model_class_and_block_name(model_class: Type, block_class_name: str) -> Type:
    filepath = inspect.getfile(model_class)
    assert filepath.endswith('.py'), f"Expected a .py file, got {filepath}"
    assert os.path.exists(filepath), f"File {filepath} does not exist"
    assert "transformers" in filepath, f"Expected a transformers model, got {filepath}"

    module_name = filepath[filepath.find('transformers'):].replace('/', '.')[:-3]
    print(f"Searching in file {filepath}, module {module_name} for class {block_class_name}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class dynamically
    class_ = getattr(module, block_class_name)
    print(f"Found class {class_} in module {module_name}")
    return class_


def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str = 'nccl'):
    print(rank, 'initializing distributed')
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class TemporarilySeededRandom:
    def __init__(self, seed):
        """Temporarily set the random seed, and then restore it when exiting the context."""
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        # Store the current random state
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # Set the random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the random state
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)


def tokenize_batch_element(prompt: str, chosen: str, truncation_mode: str, tokenizer,
                           max_length: int, max_prompt_length: int) -> Dict:
    """
    Tokenize a single batch element.
    
    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    in case the prompt + chosen responses is too long. We first tokenize and possibly
    truncate the prompt, then tokenize the chosen movies (which are separated by "|||").
    
    We also create the labels for the chosen response tokens, masking out the prompt tokens.
    """
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    # assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    # Split the movies and tokenize each movie individually
    movies = [movie.strip() for movie in chosen.split("|||") if movie.strip()]
    
    tokenized_movies = []
    for movie in movies:
        if not movie:
            print(f"Empty movie string in batch element while tokenizing: {prompt} ||| {chosen}")
            continue
        movie_tokens = tokenizer(movie, add_special_tokens=False)
        if not movie_tokens['input_ids']:
            continue
        tokenized_movies.append(movie_tokens)

    # Append the EOS token to the last movie in the list
    # tokenized_movies[-1]['input_ids'].append(tokenizer.eos_token_id)
    # tokenized_movies[-1]['attention_mask'].append(1)

    separator_token = tokenizer.convert_tokens_to_ids(",")  # or 00000
    input_ids, attention_mask, movie_ids = [], [], []
    for idx, movie in enumerate(tokenized_movies):
        input_ids.extend(movie['input_ids'])        
        attention_mask.extend(movie['attention_mask']) 
        movie_ids.extend([idx] * len(movie['input_ids']))

        if idx == len(tokenized_movies) - 1:
            input_ids.append(tokenizer.eos_token_id)
        else:
            input_ids.append(separator_token)

        attention_mask.append(1)
        movie_ids.append(-1)  # Separator token has no real movie ID    

    # Combine all tokenized movies into a single sequence
    res_tokens = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'movie_ids': movie_ids
    }

    # Truncate if combined sequence is too long
    if len(prompt_tokens['input_ids']) > max_prompt_length or len(prompt_tokens['input_ids']) + len(res_tokens['input_ids']) > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')
        
    if len(prompt_tokens['input_ids']) + len(res_tokens['input_ids']) > max_length:
        remaining_length = max_length - len(prompt_tokens['input_ids'])
        res_tokens = {k: v[:remaining_length] for k, v in res_tokens.items()}
        
    # Create the combined sequence and labels (masking the prompt tokens)
    res_sequence_tokens = {
        k: prompt_tokens.get(k, []) + res_tokens[k] if k != 'movie_ids' else [-1] * len(prompt_tokens['input_ids']) + res_tokens[k]
        for k in res_tokens
    }

    num_prompt_tokens = len(prompt_tokens['input_ids'])
    res_sequence_tokens['labels'] = res_sequence_tokens['input_ids'][:]
    res_sequence_tokens['labels'][:num_prompt_tokens] = [-100] * num_prompt_tokens

    batch = {
        'prompt': prompt,
        'response': prompt + chosen,
        'response_only': chosen,
        'movies': movies
    }

    for k, toks in {'response': res_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.

       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""

    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                elif k.endswith('_movie_ids'):
                    padding_value = -1
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    return collate_fn
