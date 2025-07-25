"""
This script is used to pretokenize the dataset and generate the vocabulary of tokens.
"""

from collections import defaultdict
from itertools import pairwise

import time
import regex as re
from tqdm import tqdm
from cs336_basics.utils import get_file_path, save_file, find_chunk_boundaries

SPECIAL_TOKENS = ['<|endoftext|>', '<|system|>']
MAX_VOCAB_SIZE = 1000
NUM_WORKERS = 4
DATASET_NAME = 'TinyStoriesV2-GPT4-train.txt'
# DATASET_NAME = 'small.txt'
# TEST_DATASET_NAME = 'tinystories_sample_5M.txt'
TEST_DATASET_NAME = 'text.txt'
TinyStoriesV2_GPT4_VOCAB_NAME = 'TinyStoriesV2-GPT4-vocab.pkl'
TinyStoriesV2_GPT4_MERGES_NAME= 'TinyStoriesV2-GPT4-merges.pkl'
TEST_VOCAB_NAME = 'text-vocab.pkl'
TEST_MERGES_NAME = 'text-merges.pkl'
    

def pretokenize(file_path, special_tokens, num_processes):
    """
    pretokenize the dataset

    Args:
        file_path (str): file path of the dataset
        num_processes (int, optional): number of processes to use.
    Returns:
        dict: pretoken_cnt
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # regex tips: 
    # 1. use '|' to join multiple patterns
    # 2. re.escape to avoid special characters
    # 3. use () to remain patterns in the result 
    # special_tokens_for_split = '(' +'|'.join(map(re.escape, special_tokens)) + ')'
    special_tokens_for_split = '|'.join(map(re.escape, special_tokens))


    print("special_tokens_for_split are: ", special_tokens_for_split)

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        print(f"boundaries size {len(boundaries)}, boundaries are: {boundaries}")
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        print(f"Starting pretokenization ...")

        pretoken_cnt = defaultdict(int)
        for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), desc="Reading and Pretokenizing", total=len(boundaries)-1):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            sub_chunks = re.split(special_tokens_for_split, chunk)
            # print("sub_chunks are: ", sub_chunks)
            for sub_chunk in sub_chunks:
                # if sub_chunk in special_tokens:
                #     continue
                for match in re.finditer(PAT, sub_chunk):
                    key = tuple(map(lambda x: x.encode('utf-8'), match.group()))
                    # if b'\xe2' in key or b'\x80' in key:
                    #     print(key)
                    #     print("match is: ",  match.group())
                        
                    pretoken_cnt[key] += 1

    print("pretoken_cnt size: ", len(pretoken_cnt))
    return pretoken_cnt


def print_sorted_pretoken_cnt(pretoken_cnt):
    """print the sorted pretoken_cnt

    Args:
        pretoken_cnt (dict): pretoken_cnt
    """
    sorted_tokens = sorted([(v, k) for k, v in pretoken_cnt.items()], reverse=True)
    print(sorted_tokens[:30])


def init_vocabulary(special_tokens):
    """initialize the vocabulary

    Args:
        special_tokens (list[str]): the list of special tokens

    Returns:
        tuple: vocabulary, reversed_vocab
    """
    # map token id to byte
    print("Initializing vocabulary...")
    vocabulary = defaultdict(bytes)
    # map byte to byte
    reversed_vocab = defaultdict(int)

    vocab_size = 256
    for i in range(vocab_size):
        reversed_vocab[bytes([i])] = i
        vocabulary[i] = bytes([i])

    for t in special_tokens:
        reversed_vocab[t.encode('utf-8')] = vocab_size
        vocabulary[vocab_size] = t.encode('utf-8')
        vocab_size += 1

    return vocabulary, reversed_vocab

# check if (x1, x2) is in token_list, if so, merge them
def check_and_update_key(token_list, x1, x2):
    """
    check if (x1, x2) is in token_list, if so, merge them and update the token_list accordingly

    Args:
        token_list (list[bytes]): the list of tokens
        x1 (bytes): the first token
        x2 (bytes): the second token

    Returns:
        tuple: the merged token
    """
    ret = []
    i = 0
    while i < len(token_list) - 1:
        if token_list[i] == x1 and token_list[i + 1] == x2:
            ret.append(x1 + x2)
            i += 2
        else:
            ret.append(token_list[i])
            i += 1

    if i == len(token_list) - 1:
        ret.append(token_list[i])
    return tuple(ret)


# get the most frequent pair
def merge_pretokens(pretoken_cnt, special_tokens, max_tokens):
    """
    merge the pretokens based on the most frequent pair

    Args:
        pretoken_cnt (dict): pretoken_cnt
        vocabulary (dict): vocabulary
        reversed_vocab (list[bytes]): reversed_vocab
        max_tokens (int): max_tokens
    """
    vocabulary, reversed_vocab = init_vocabulary(special_tokens)
    vocab_size = len(vocabulary)
    merged_pairs = []

    with tqdm(total=max_tokens, desc="Merging tokens") as pbar:
        pbar.update(vocab_size)
        while vocab_size < max_tokens:
            max_v = 0
            max_pair = None
            pair_counter = defaultdict(int)

            for k, v in pretoken_cnt.items():
                for pair in pairwise(k):
                    # if b'\xe2' in pair or b'\x80' in pair:
                    #     print(pair)
                    #     print("key is: ",  k)
                    pair_counter[pair] += v
                    if max_pair is None or (pair_counter[pair], pair) > (max_v, max_pair):
                        max_v = pair_counter[pair]
                        max_pair = pair

            if max_pair is None:
                break

            # add new token to vocalbulary
            merged_pairs.append(max_pair)
            
            new_token = max_pair[0] + max_pair[1]
            # if new_token in reversed_vocab:
            #     print(f"Exists!!!! reversed_vocab[{new_token}] = {reversed_vocab[new_token]}")

            vocabulary[vocab_size] = new_token
            reversed_vocab[new_token] = vocab_size

            vocab_size += 1
            
            tmp = pretoken_cnt.copy()
            pretoken_cnt = defaultdict(int)
            # update pretookenizer based on max_pair
            for k, v in tmp.items():
                new_key = check_and_update_key(k, *max_pair)
                pretoken_cnt[new_key] = v

            pbar.update(1)
        
    return vocabulary, reversed_vocab, merged_pairs
    
def train_bpe(input_path, vocab_size, special_tokens):
    """
    generate the bpe token vocabulary

    Args:
        input_path (str): input path of the dataset
        vocab_size (int): vocabulary size
        special_tokens (list[str]): special tokens
    """

    pretoken_cnt = pretokenize(input_path, SPECIAL_TOKENS, NUM_WORKERS)
    vocabulary, reversed_vocab, merged_pairs = merge_pretokens(pretoken_cnt, special_tokens, vocab_size)
    print("vocabulary size: ", len(vocabulary))
    print("token_to_byte size: ", len(reversed_vocab))
    print("merged_pairs size: ", len(merged_pairs))

    # print(reversed_vocab[b'\xe2'])
    # print(reversed_vocab[b'\x80'])
    # print("vocabulary: ", vocabulary)
    # print("reversed_vocab: ", reversed_vocab)
    # print("merged_pairs: ", merged_pairs)
    return vocabulary, merged_pairs


def save_vocabulary_and_merges(vocabulary, merged_pairs, vocab_path, merges_path):
    """
    save the vocabulary and merged_pairs to a file

    Args:
        vocabulary (dict): vocabulary
        merged_pairs (list[tuple]): merged_pairs
        vocab_path (str): path to save the vocabulary
        merges_path (str): path to save the merged_pairs
    """
    save_file(vocabulary, vocab_path)
    save_file(merged_pairs, merges_path)
    

def get_longest_token(vocabulary):
    """
    get the longest token
    """
    return max(vocabulary.values(), key=len)

def print_dict(vocabulary, reversed_vocab, merged_pairs, n):
    """
    print the dictionary

    Args:
        vocabulary (dict): vocabulary
        reversed_vocab (list[bytes]): reversed_vocab
        n (int): num of elements to print
    """

    print('-----vocabulary token to byte-----')
    print_num = n
    for voc in vocabulary.items():
        if print_num == 0:
            break
        print(voc)
        print_num -= 1

    print_num = n
    print('-----byte_to_token-----')
    for rv in reversed_vocab.items():
        if print_num == 0:
            break
        print(rv)
        print_num -= 1
    
    print_num = n
    print('-----merges-----')
    for mp in merged_pairs:
        if print_num == 0:
            break
        print(mp)
        print_num -= 1

if __name__ == "__main__":
    # file_path = get_file_path(DATASET_NAME)
    # vocab_path = get_file_path(TinyStoriesV2_GPT4_VOCAB_NAME)
    # merges_path = get_file_path(TinyStoriesV2_GPT4_MERGES_NAME)
    file_path = get_file_path(TEST_DATASET_NAME)
    vocab_path = get_file_path(TEST_VOCAB_NAME)
    merges_path = get_file_path(TEST_MERGES_NAME)

    # issue: (b'\xe2', b'\x80') not in merges

    print('-----start to generate vocabulary-----')
    start_time = time.time()
    try:
        vocabulary, merged_pairs = train_bpe(file_path, MAX_VOCAB_SIZE, SPECIAL_TOKENS)
        save_vocabulary_and_merges(vocabulary, merged_pairs, vocab_path, merges_path)

    except Exception as e:
        print(f"Error: {e}")
    end_time = time.time()

    print(f'Time taken: {end_time - start_time} seconds')
    print("longest token: ", get_longest_token(vocabulary))
    print('-----vocabulary generation done -----')
    # vocab1 = load_file(vocab_path)
    # merges1 = load_file(merges_path)
    # assert vocab1 == vocabulary
    # assert merges1 == merged_pairs

    print('Done')
