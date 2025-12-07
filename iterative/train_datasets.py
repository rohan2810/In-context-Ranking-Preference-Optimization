import datasets
import torch
from torch.utils.data import DataLoader
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import json
import random
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import pandas as pd
from collections import Counter
from itertools import chain
import re
from typing import Dict
from rapidfuzz import process
import random

def clean_prompt(prompt: str) -> str:
    return re.sub(r"\[.*?\]", "", prompt).strip()


def normalize_movie_title(title: str) -> str:
    if not title:
        return title
    title = re.sub(r'\s*\(\d{4}\)\s*', '', title)
    title = re.sub(r'[^\w\s]', '', title.lower())
    title = ' '.join(title.split())
    
    return title if title else None

def remove_year_from_movie_name(movie_name: str) -> str:
    return re.sub(r'\s*\(\d{4}\)\s*', '', movie_name)

def get_reddit(csv_path: str, max_movies: int = 20000, silent: bool = False, split: str = 'train') -> Dict[str, Dict[str, Union[str, List[Tuple[int, int]]]]]:
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

    data = defaultdict(dict)
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
        # if not filtered_llm_rec:
        #     continue

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
        
        # Relevance  = 3 for gt movies and ranking is the index of the movie in the list
        ground_truth_ranked = [
            (movie, (2, idx + 1))  for idx, movie in enumerate(ground_truth)
        ]
        # Relevance = 2 for llm rec movies and ranking is the index of the movie in the list + len(ground_truth)
        start_rank = len(ground_truth) + 1
        llm_rec_ranked = [
            (movie, (1, start_rank + idx))  for idx, movie in enumerate(filtered_llm_rec)
        ]
        # Relevance = 1 for random movies and ranking is the index of the movie in the list + len(ground_truth) + len(llm_rec)
        start_rank = len(ground_truth) + len(filtered_llm_rec) + 1
        self_ranked = [
            (movie, (0, start_rank + idx))  for idx, movie in enumerate(sampled_movies)
        ]
        # Relevance = 1 for random movies and ranking is the index of the movie in the list + len(ground_truth) + len(llm_rec)
        start_rank = len(ground_truth) + len(filtered_llm_rec) + len(self_ranked) + 1
        random_ranked = [
            (movie, (0, start_rank + idx))  for idx, movie in enumerate(random_movies)
        ]

        combined = ground_truth_ranked + llm_rec_ranked + self_ranked + random_ranked
        combined = combined[:num_mv]
        random.shuffle(combined)

        # combined = [(movie, rank) for movie, rank in combined if movie.strip()]
        if combined:
            combined_movies, combined_ranking = zip(*combined)
            # prompt = f"Human: {context}\n\nAssistant:"
            prompt = f"\nUser: {context}\nSystem: "
            data[prompt].update({
                "movies": "|||".join(combined_movies),
                "relevance_rank": list(combined_ranking)
            })
        else:
            print("Combined tokens in dataloader", combined)
            raise NotImplementedError
    return data

def get_nlp(csv_path: str, max_movies: int = 20000, silent: bool = False, split: str = 'train') -> Dict[str, Dict[str, Union[str, List[Tuple[int, int]]]]]:
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
        
    data = defaultdict(dict)

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

        # Relevance  = 3 for gt movies and ranking is the index of the movie in the list
        ground_truth_ranked = [
            (movie, (1, idx + 1))  for idx, movie in enumerate(ground_truth)
        ]
        # Relevance = 2 for llm rec movies and ranking is the index of the movie in the list + len(ground_truth)
        start_rank = len(ground_truth) + 1
        random_ranked = [
            (movie, (0, start_rank + idx))  for idx, movie in enumerate(random_movies)
        ]

        combined = ground_truth_ranked + random_ranked
        combined = combined[:num_mv]
        random.shuffle(combined)

        # combined = [(movie, rank) for movie, rank in combined if movie.strip()]
        if combined:
            combined_movies, combined_ranking = zip(*combined)
            # prompt = f"Human: {context}\n\nAssistant:"
            prompt = f"\nUser: {context} {combined_movies}\nSystem: "
            data[prompt].update({
                "movies": "|||".join(combined_movies),
                "relevance_rank": list(combined_ranking)
            })
        else:
            print("Combined tokens in dataloader", combined)
            raise NotImplementedError
    return data

def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, embed_dir: str = None):
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

def get_batch_iterator(names: List[str],
                       prompt_template: str,
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed: int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       embed_dir: Optional[str] = None) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
        embed_dir: Directory to embeddings in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2 ** 32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_start'
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir, embed_dir=embed_dir).items():
                flat_data.append((prompt, data['movies'], data['relevance_rank'], truncation_mode))
                # [('Human: Shows like The IT Crowd, Black Books, The Mighty Boosh..\n\nAssistant:', 'Green Wing; Black Book (2006); Spaced; Man to Man (2005); The in Crowd (2000); Brass Eye; The Day Today; The Wrong Mans; Tower of London (1962); Derek (2008)', 
                # [(1, 8), (2, 1), (1, 10), (2, 4), (2, 3), (1, 6), (1, 7), (1, 9), (2, 2), (2, 5)], 'keep_start')]
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
            # with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, movies, relevance_rank, truncation_mode in flat_data:
            if done:
                break

            if prompt_template.strip():
                formatted_prompt = prompt_template.format(prompt).strip()
            else:
                formatted_prompt = prompt.strip()  
     
            batch_element = tokenize_batch_element(
                    prompt=formatted_prompt,
                    chosen=movies,
                    truncation_mode=truncation_mode,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    max_prompt_length=max_prompt_length
                )
            
            batch_element['relevance_rank'] = relevance_rank
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