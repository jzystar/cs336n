from itertools import pairwise
from math import inf
from typing import Iterable, Iterator
import regex as re
from tqdm import tqdm
from cs336_basics.utils import get_file_path, load_file, find_chunk_boundaries

class BPETokenizer:
    # Declaration of instance variables
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None
    
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        self.special_tokens_patterns = f"({'|'.join(map(re.escape, self.special_tokens))})"
        # add new special tokens to the vocab and reversed_vocab if not exist
        token_id = len(self.vocab)
        for st in self.special_tokens:
            st = st.encode('utf-8')
            if st not in self.reversed_vocab:
                self.reversed_vocab[st] = token_id
                self.vocab[token_id] = st
                token_id += 1

        self.merges_maps = {pair: i for i, pair in enumerate(self.merges)}
        # self.special_tokens = list(map(lambda x: x.encode('utf-8'), self.special_tokens))

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens


        Args:
            vocab_filepath (str): path of the vocabulary file
            merges_filepath (str): path of the merges file
            special_tokens (list[str], optional): list of special tokens. Defaults to None.

        Returns:
            BPETokenizer: return a Tokenizer instance
        """
        vocabulary = load_file(vocab_filepath)
        merges = load_file(merges_filepath)

        return cls(vocabulary, merges, special_tokens)
    

    def _pretokenize(self, text):
        """
        Pretokenize an input text.

        Args:
            text (str): input text

        Returns:
            list[str]: list of pretokens
        """

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        sub_chunks = [text]
        if self.special_tokens:
            print("special_tokens: ", self.special_tokens)
            print("special_tokens_patterns: ", self.special_tokens_patterns)
            sub_chunks = re.split(self.special_tokens_patterns, text)
            # print("sub_chunks: ", sub_chunks)

        pretokens = []
        
        for sub_chunk in sub_chunks:
            if sub_chunk in self.special_tokens:
                pretokens.append(sub_chunk)
            else:
                for match in re.finditer(PAT, sub_chunk):
                    key = tuple(map(lambda x: bytes([x]), match.group().encode('utf-8')))
                    pretokens.append(key)
        print("pretoken size: ", len(pretokens))
        # print("pretokens: ", pretokens)
        return pretokens

    def encode_from_file(self, file_path, num_processes):
        """Encode a file into a sequence of token IDs.

        Args:
            file_path (str): path to the file
            num_processes (int): number of processes

        Returns:
            list[int]: sequence of token IDs
        """
        with open(file_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_processes, "<|endoftext|>".encode("utf-8"))
            print(f"boundaries size {len(boundaries)}, boundaries are: {boundaries}")
            tokens = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                pretokens_of_chunk = self._pretokenize(chunk)
                tokens.extend(self._eoncode_of_pretokens(pretokens_of_chunk))
            
        return tokens

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.

        Args:
            text (str): input text

        Returns:
            list[int]: sequence of token IDs
        """
        pretokens = self._pretokenize(text)
        return self._eoncode_of_pretokens(pretokens)
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable ofstrings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into memory.

        Args:
            iterable (Iterable[str]): iterable of strings

        Yields:
            Iterator[int]: generator of token IDs
        """
        for text in iterable:
            for token in self.encode(text):
                yield token
    
    def decode(self, ids: list[int]) -> str:
        text = b""
        for id in ids:
            text += self.vocab[id]
        return text.decode('utf-8', errors='replace')

    def _eoncode_of_pretokens(self, pretokens):
        """
        The process of encoding a list of pretokens into a list of token IDs.

        Args:
            pretokens (list[str]): list of pretokens

        Returns:
            list[int]: list of token IDs
        """
        tokens = []
        try:
            for pt in tqdm(pretokens, desc="Encoding"):
                # print("tokens are :", tokens)
                if pt in self.special_tokens:
                    tokens.append(self.reversed_vocab[pt.encode('utf-8')])
                else:
                    # print("not special token: ", pt)
                    max_it = 20
                    while True:
                        if max_it == 0:
                            break
                        max_it -= 1

                        min_id = inf
                        min_pair = None
                        for pair in pairwise(pt):
                            if pair not in self.merges_maps:
                                continue
                            cur_pair_id = self.merges_maps[pair]
                            if cur_pair_id < min_id:
                                min_id = cur_pair_id
                                min_pair = pair
                        if min_pair is None:
                            break
                       
                        # update pretoken to new one
                        new_pt = []
                        i = 0
                        while i < len(pt) - 1:
                            if tuple(pt[i:i+2]) == min_pair:
                                new_pt.append(pt[i] + pt[i+1])
                                i += 2
                            else:
                                new_pt.append(pt[i])
                                i += 1

                        if i == len(pt) - 1:
                            new_pt.append(pt[i])
                        pt = new_pt

                    for token in pt:
                        if token in self.reversed_vocab:
                            tokens.append(self.reversed_vocab[token])
                        else:
                            print("token not exits in vocabulary: ", token)
            return tokens
        except KeyError as e:
            print("KeyError: ", e)
            print("key not exits in vocabulary: ", token)
        except Exception as e:
            print("Error: ", e)
    


DATASET_NAME = 'TinyStoriesV2-GPT4-train.txt'
# DATASET_NAME = 'small.txt'
TEST_DATASET_NAME = 'text.txt'
TinyStoriesV2_GPT4_VOCAB_NAME = 'TinyStoriesV2-GPT4-vocab.pkl'
TinyStoriesV2_GPT4_MERGES_NAME= 'TinyStoriesV2-GPT4-merges.pkl'
TEST_VOCAB_NAME = 'text-vocab.pkl'
TEST_MERGES_NAME = 'text-merges.pkl'

if __name__ == "__main__":
    vocab_path = get_file_path(TEST_VOCAB_NAME)
    merges_path = get_file_path(TEST_MERGES_NAME)
    special_tokens = ['<|user|>', '<|endoftext|>', '<|system|>', '<|startoftext|>']

    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)
    print(f"tokenizer vocab is {tokenizer.vocab}")
    print(f"tokenizer reversed_vocab is {tokenizer.reversed_vocab}")
    print(f"tokenizer merges is {tokenizer.merges}")

    text = "<|startoftext|>lower is lowest, <|user|>newest west east <|endoftext|>"
    ids = tokenizer.encode(text)
    print("ids: ", ids)
    decoded_text = tokenizer.decode(ids)
    assert decoded_text == text, "decoded_text is not equal to text"
    print("decoded_text: ", decoded_text)

    