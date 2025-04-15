import os
import time
import regex as re

from typing import BinaryIO
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from functools import wraps


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{func.__name__}] took {elapsed:.2f} seconds.")
        return result
    return wrapper


@timed
def simple_pretokenizer(text):
    pre_tokens = text.strip().split()
    return pre_tokens


# Uses PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def regex_pretokenize(text, regex=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""):
    # Return a list of matches instead of just printing
    return [match.group() for match in re.finditer(regex, text)]


def strip_and_pretokenize(text, special_tokens):
    split_pattern = "|".join(re.escape(token) for token in special_tokens)
    docs = text.split(text, split_pattern)

    return docs


def decode_token(token):
    if isinstance(token, int):
        # Single-byte integer token
        return bytes([token]).decode("utf-8", errors='replace')
    elif isinstance(token, tuple):
        # Tuple token, decode recursively
        return "".join(decode_token(st) for st in token)
    else:
        # Special tokens, represented as strings
        return str(token)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def calc_pair_counts_naive(tokens):
    pair_counts = defaultdict(int)
    for token in tokens:
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_counts[pair] += 1

    return pair_counts


@timed
def load_dataset_stream(filepath, split_token="<|endoftext|>"):
    docs = []
    current_doc = []
    split_token = split_token.strip()

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line == split_token:
                if current_doc:
                    docs.append("\n".join(current_doc))
                    current_doc = []
            else:
                current_doc.append(line)
    if current_doc:
        docs.append("\n".join(current_doc))

    return docs


@timed
def merge_naive(
    tokens: list[bytes],
    vocab: dict,
    vocab_size: int,
    max_merges: int = None,
):

    merges = []
    next_token_id = len(vocab)
    num_merges = 0

    while len(vocab) < vocab_size:
        pair_counts = calc_pair_counts_naive(tokens)
        if not pair_counts:
            break

        most_freq_pair = max(pair_counts, key=lambda p: (pair_counts[p], decode_token(p)))

        vocab[most_freq_pair] = next_token_id
        next_token_id += 1

        merges.append(most_freq_pair)

        new_tokens = []
        for t in tokens:
            new_t = []
            i = 0
            while i < len(t):
                if i < len(t) - 1 and (t[i], t[i + 1]) == most_freq_pair:
                    new_t.append(most_freq_pair)
                    i += 2
                else:
                    new_t.append(t[i])
                    i += 1
            new_tokens.append(new_t)

        tokens = new_tokens

        num_merges += 1
        if max_merges is not None and num_merges >= max_merges:
            break

    return tokens, vocab, merges


@timed
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = None,
    max_merges: int = None,  # add optional control
):

    """
    Train BPE tokenizer.

    Args:
        input_path (str): Path to your text file for BPE tokenizer training.
        vocab_size (int): Desired final vocabulary size (256 byte tokens + special tokens + merges).
        special_tokens (list[str]): Special tokens explicitly added to vocab.
        num_processes (int): Number of processes for parallel pre-tokenization. Defaults to CPU count.
        max_merges (int): For optional direct control of number of merges.

    Returns:
        vocab: dict[bytes or tuple, int] (token → id)
        merges: list[tuple[bytes, bytes]] (ordered list of merges performed)
    """

    import os
    from multiprocessing import Pool, cpu_count

    vocab = {bytes([i]): i for i in range(256)}

    for token in special_tokens:
        vocab[token] = len(vocab)

    # Find chunk boundaries (for multiprocessing)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes or cpu_count(), "<|endoftext|>".encode("utf-8"))

    def pretokenize_chunk(args):
        start, end, filepath = args
        with open(filepath, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="replace")
        # Split on special tokens explicitly, then pre-tokenize each document separately
        docs = strip_and_pretokenize(chunk, special_tokens=special_tokens)
        return docs

    chunk_args = [(boundaries[i], boundaries[i+1], input_path) for i in range(len(boundaries) - 1)]

    if num_processes is None:
        num_processes = cpu_count()

    with Pool(num_processes) as pool:
        docs_chunks = pool.map(pretokenize_chunk, chunk_args)

    # Flatten
    docs = [doc for chunk in docs_chunks for doc in chunk]
    token_seq = [doc.encode("utf-8") for doc in docs]

    tokens, vocab, merges = merge_naive(
        tokens=tokens,
        vocab=vocab,
        vocab_size=vocab_size,
        max_merges=max_merges
    )

    return vocab, merges
