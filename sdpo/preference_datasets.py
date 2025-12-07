import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict, Counter
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import pandas as pd
import re
from rapidfuzz import process
from itertools import chain

def clean_prompt(prompt: str) -> str:
    return re.sub(r"\[.*?\]", "", prompt).strip()

def normalize_movie_title(title: str) -> str:
    if not title:
        return title
    title = re.sub(r'\s*\(\d{4}\)\s*', '', title)
    title = re.sub(r'[^\w\s]', '', title.lower())
    title = ' '.join(title.split())
    
    return title if title else None

def get_reddit(csv_path: str, max_movies: int = 20000, silent: bool = False, split: str = 'train', num_negative: int = 1) -> Dict[str, Dict[str, List[str]]]:
    """
    Load and preprocess the Reddit dataset, and return a dict of prompts and responses
    containing movies and their rankings.
    """
    reddit_posts_train = pd.read_csv(csv_path)
    if split == 'eval':
        reddit_posts_train = reddit_posts_train.sample(frac=0.01, random_state=42)
    else:
        reddit_posts_train = reddit_posts_train.sample(frac=0.99, random_state=42)
        
    eligible_movies = [eval(i) for i in reddit_posts_train['movies']]
    frequency = Counter(list(chain.from_iterable(eligible_movies)))
    unique_movie_names = sorted(i[0] for i in frequency.most_common(max_movies))
    unique_movie_names = [movie for movie in unique_movie_names if movie.strip()]

    unique_movie_freqs = [frequency[movie] for movie in unique_movie_names]
    unique_movie_freqs = np.log(np.asarray(unique_movie_freqs) * 0.75)
    unique_movie_freqs = (unique_movie_freqs - np.min(unique_movie_freqs)) / (np.max(unique_movie_freqs) - np.min(unique_movie_freqs))
    unique_movie_freqs = unique_movie_freqs / np.sum(unique_movie_freqs)

    movie2id = {movie: idx for idx, movie in enumerate(unique_movie_names)}

    positive_samples = [
        list(dict.fromkeys([movie for movie in post if movie in movie2id])) 
        for post in eligible_movies
    ]
    if 'llm_rec_extracted_gpt-3.5-turbo-0125' in reddit_posts_train:
        llm_recommendations = [
            list(dict.fromkeys(eval(rec))) if isinstance(rec, str) else [] 
            for rec in reddit_posts_train['llm_rec_extracted_gpt-3.5-turbo-0125']
        ]
    else:
        llm_recommendations = [[] for i in reddit_posts_train['movies']]

    data = defaultdict(lambda: defaultdict(list))
    contexts = list(reddit_posts_train['full_situation'])
    if "resp" in reddit_posts_train:
        random_samples = [eval(item) for item in list(reddit_posts_train["resp"])]
    else:
        random_samples = []
    num_gt, num_gpt, num_mv = 3, 5, 20
    for i in tqdm.tqdm(range(len(positive_samples)), desc='Processing Reddit Data', disable=silent):
        if not positive_samples[i] and not llm_recommendations[i]:
            continue

        context = clean_prompt(contexts[i]) # String of post
        ground_truth = positive_samples[i] # list of gt movies in order
        llm_rec = llm_recommendations[i] # list of llm movies in order
        normalized_ground_truth = [normalize_movie_title(gt_movie) for gt_movie in ground_truth]

        filtered_llm_rec = []
        for llm_movie in llm_rec:
            normalized_llm_movie = normalize_movie_title(llm_movie)
            # Filter out llm recommendations that are too similar to the ground truth
            match = process.extractOne(normalized_llm_movie, normalized_ground_truth)
            if match is None or match[1] < 85:
                filtered_llm_rec.append(llm_movie)

        ground_truth = [movie for movie in ground_truth if movie.strip()]     
        ground_truth = ground_truth[:num_gt]
        ground_truth = sorted(ground_truth, key=lambda x: frequency[x], reverse=True)

        filtered_llm_rec = [movie for movie in filtered_llm_rec if movie not in ground_truth]
        filtered_llm_rec = filtered_llm_rec[:num_gpt]

        sampled_movies = list(random_samples[i].keys()) if len(random_samples) > 0 else []
        num_rnd = num_mv - len(ground_truth) - len(filtered_llm_rec) - len(sampled_movies)
        random_movies = []
        if num_rnd > 0:
            # we want more movies
            random_movies = np.random.choice(unique_movie_names, size=20, replace=False, p=unique_movie_freqs)
            # random_movies = random.sample(unique_movie_names, rnd_rec_required)
            random_movies = [movie for movie in random_movies if movie not in ground_truth and movie not in filtered_llm_rec]
            random_movies = random_movies[:num_rnd]

        combined = ground_truth + filtered_llm_rec + random_movies
        combined = combined[:num_mv]
        chosen = ",".join(combined)

        if combined:
            prompt = f"\nUser: {context}\nSystem: "
            data[prompt]["chosen"].append(chosen)
            cnt = 0  
            for _ in range(num_negative):
                shuffled_combined = combined.copy()
                random.shuffle(shuffled_combined)
                rejected = ",".join(shuffled_combined)
                cnt += 1
                data[prompt][f"rejected{cnt}"].append(rejected)
        else:
            print("Combined tokens in dataloader", combined)
            raise NotImplementedError
    return data

def get_nlp(csv_path: str, max_movies: int = 20000, silent: bool = False, split: str = 'train', num_negative: int = 1) -> Dict[str, Dict[str, Union[str, List[Tuple[int, int]]]]]:
    """
    Load and preprocess the Reddit dataset, and return a dict of prompts and responses
    containing movies and their rankings.

    Args:
        csv_path: Path to the Reddit dataset CSV.
        max_movies: Maximum number of unique movies to consider.
        silent: Whether to suppress progress bar.
        split: Whether to use the full dataset ('train') or only 1% of the data ('test').

    Returns:
        Dictionary mapping prompts to their movie lists and relevance rankings.
    """
    reddit_posts_train = pd.read_csv(csv_path)
    # reddit_posts_train = reddit_posts_train[:5000]
    if split == 'test':
        reddit_posts_train = reddit_posts_train.sample(frac=0.20, random_state=42)
    else:
        reddit_posts_train = reddit_posts_train.sample(frac=0.80, random_state=42)
        
    data = defaultdict(lambda: defaultdict(list))
    if "relevant list" in reddit_posts_train.columns:
        relevant_items = [eval(i) for i in reddit_posts_train['relevant list']]
        irrelevant_items = [eval(i) for i in reddit_posts_train['irelevant list']]
    else:
        relevant_items = [eval(i) for i in reddit_posts_train['preferred']]
        irrelevant_items = [eval(i) for i in reddit_posts_train['rejected']]
    contexts = list(reddit_posts_train['full_situation'])

    num_gt, num_rnd, num_mv = 3, 10, 10
    for i in tqdm.tqdm(range(len(contexts)), desc='Processing Reddit Data', disable=silent):
        context = contexts[i] # String of post
        ground_truth = relevant_items[i] # list of gt movies in order
        random_movies = irrelevant_items[i]

        if len(ground_truth) == 0 or len(random_movies) == 0:
            continue

        ground_truth = [str(movie) for movie in ground_truth if movie]     
        ground_truth = ground_truth[:num_gt]

        random_movies = [str(movie) for movie in random_movies if movie not in ground_truth]
        random_movies = random_movies[:num_rnd]

        current_list = ground_truth + random_movies

        if num_mv > len(current_list):
            extra_movies = random.choices(random_movies, k=num_mv - len(current_list))
            random_movies = random_movies + extra_movies

        combined = ground_truth + random_movies
        combined = combined[:num_mv]
        chosen = ",".join(combined)

        if combined:
            prompt = f"\nUser: {context}\nSystem: "
            data[prompt]["chosen"].append(chosen)
            cnt = 0  
            for _ in range(num_negative):
                shuffled_combined = combined.copy()
                random.shuffle(shuffled_combined)
                rejected = ",".join(shuffled_combined)
                cnt += 1
                data[prompt][f"rejected{cnt}"].append(rejected)
        else:
            print("Combined tokens in dataloader", combined)
            raise NotImplementedError
    return data

def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, embed_dir: str = None, num_negative: int = 1):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'reddit':
        data = get_reddit(embed_dir, 20000, split=split, silent=silent)
    elif name == 'nlp':
        data = get_nlp(embed_dir, 20000, split=split, silent=silent)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    return data

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
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn

def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    rejected_tokens = {}
    for key in rejected:
        rejected_tokens[key] = tokenizer(rejected[key], add_special_tokens=False)
        
    # assert tokenizer.eos_token_id not in prompt_tokens["input_ids"], f"Prompt contains EOS token: {prompt}"
    # assert (
    #     tokenizer.eos_token_id not in chosen_tokens["input_ids"]
    # ), f"Chosen response contains EOS token: {chosen}"
    # assert (
    #     all([tokenizer.eos_token_id not in rejected_tokens[key]["input_ids"] for key in rejected_tokens])
    # ), f"Rejected response contains EOS token: {rejected}"

    chosen_tokens["input_ids"].append(tokenizer.eos_token_id)
    chosen_tokens["attention_mask"].append(1)
    for key in rejected_tokens:
        rejected_tokens[key]["input_ids"].append(tokenizer.eos_token_id)
        rejected_tokens[key]["attention_mask"].append(1)
    max_rejected_len = max([len(rejected_tokens[key]["input_ids"]) for key in rejected_tokens])
    longer_response_length = max(len(chosen_tokens["input_ids"]), max_rejected_len)

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
        if truncation_mode == "keep_start":
            prompt_tokens = {k: v[: max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == "keep_end":
            prompt_tokens = {k: v[-max_prompt_length :] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f"Unknown truncation mode: {truncation_mode}")

    # if that's still too long, truncate the response
    if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
        chosen_tokens = {k: v[: max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[: max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {}
    # rejected_tokens: Dict[str, Dict]

    for key in rejected_tokens:
        rejected_sequence_tokens[key] = {k: prompt_tokens[k] + rejected_tokens[key][k] for k in rejected_tokens[key]}
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(
        prompt_tokens["input_ids"]
    )
    for key in rejected_sequence_tokens:
        rejected_sequence_tokens[key]["labels"] = rejected_sequence_tokens[key]["input_ids"][:]
        rejected_sequence_tokens[key]["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(
            prompt_tokens["input_ids"]
        )

    batch = {}

    batch["prompt"] = prompt
    batch["chosen"] = prompt + chosen
    for key in rejected:
        batch[key] = prompt + rejected[key]
    batch["chosen_response_only"] = chosen
    for key in rejected:
        batch[f"{key}_response_only"] = rejected[key]

    for k, toks in {
        "chosen": chosen_sequence_tokens,
        # "rejected": rejected_sequence_tokens,
        "prompt": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}_{type_key}"] = tokens
    # rejected_sequence_tokens: Dict[str, Dict]
    for k, toks in rejected_sequence_tokens.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}_{type_key}"] = tokens
    
    return batch

def get_batch_iterator(names: List[str],
                       tokenizer,
                       prompt_template: str,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       embed_dir: Optional[str] = None,
                       num_negative: int = 1) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            dataset = get_dataset(name, split, silent=silent, cache_dir=cache_dir, embed_dir=embed_dir, num_negative=num_negative)
            for prompt, data in dataset.items():
                # Handle the new data structure with "chosen" and "rejected{N}" keys
                chosen = data["chosen"][0]
                rejected = {}
                # Collect all rejected responses
                for i in range(1, num_negative + 1):
                    key = f"rejected{i}"
                    if key in data and data[key]:
                        rejected[key] = data[key][0]
                
                flat_data.append((prompt, chosen, rejected, truncation_mode))

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, chosen, rejected, truncation_mode in flat_data:
            if done:
                break
            if prompt_template.strip():
                formatted_prompt = prompt_template.format(prompt).strip()
            else:
                formatted_prompt = prompt.strip()  
            batch_element = tokenize_batch_element(formatted_prompt, chosen, rejected, truncation_mode, tokenizer, max_length, max_prompt_length)
            batch.append(batch_element)
            example_idx += 1
            if len(batch) == batch_size:
                yield collate_fn(batch)
                if n_examples is not None and example_idx >= n_examples:
                    if not silent:
                        print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                    done = True
                batch = []
                
        if done:
            break

        epoch_idx += 1