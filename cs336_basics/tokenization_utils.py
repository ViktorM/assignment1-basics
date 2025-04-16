import os
import time
import heapq
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
    docs = re.split(split_pattern, text)
    docs = [doc.strip() for doc in docs if doc.strip()]

    return docs


def decode_token_old(token):
    if isinstance(token, int):
        # Represents a single byte value within a tuple during recursion
        return bytes([token]).decode("utf-8", errors='replace')
    elif isinstance(token, tuple):
        # Tuple token, decode recursively
        return "".join(decode_token(st) for st in token)
    elif isinstance(token, bytes):
        # Explicitly handle bytes keys (initial vocab, possibly merged tokens)
        return token.decode("utf-8", errors='replace')
    elif isinstance(token, str):
        # Handle special tokens (which are strings)
        return token
    else:
        # Should not happen with current vocab structure
        raise TypeError(f"Unexpected token type in vocab key: {type(token)}")


def decode_token(token_id, vocab):
    """
    Decode a token ID to its byte representation using the vocabulary.

    Args:
        token_id (int): The integer token ID to decode.
        vocab (dict[int, bytes]): The vocabulary mapping token IDs to bytes.

    Returns:
        str: Decoded token as a UTF-8 string, with errors replaced.
    """
    token_bytes = vocab[token_id]
    return token_bytes.decode('utf-8', errors='replace')


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


def calc_pair_counts(tokens: list[list[int]]):
    pair_counts = defaultdict(int)
    for token in tokens:
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])  # pairs of token IDs
            pair_counts[pair] += 1
    return pair_counts


def calc_pair_positions(tokens: list[list[int]]):
    pair_positions = defaultdict(set)
    for token_idx, token in enumerate(tokens):
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            pair_positions[pair].add((token_idx, i))

    return pair_positions


def calc_pair_counts_positions(tokens: list[list[int]]):
    pair_counts = defaultdict(int)
    pair_positions = defaultdict(set)

    for token_idx, token in enumerate(tokens):
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            pair_counts[pair] += 1
            pair_positions[pair].add((token_idx, i))

    return pair_counts, pair_positions


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
    tokens: list[list[int]],  # each token is a list of int IDs
    vocab: dict[int, bytes],  # vocab maps int IDs → bytes
    vocab_size: int,
    max_merges: int = None,
):

    merges = []  # store merges as tuples of bytes
    next_token_id = max(vocab.keys()) + 1  # next available token ID

    num_merges = 0

    while len(vocab) < vocab_size:
        # Count pair occurrences explicitly (by token IDs)
        pair_counts = calc_pair_counts(tokens)
        if not pair_counts:
            break  # no more pairs to merge explicitly

        # Select the most frequent pair with deterministic tie-breaking
        most_freq_pair = max(pair_counts, key=lambda p: (pair_counts[p], decode_token(p[0], vocab) + decode_token(p[1], vocab)))

        # Define merged bytes (concatenate bytes from vocab)
        merged_bytes = vocab[most_freq_pair[0]] + vocab[most_freq_pair[1]]

        # Update vocab explicitly: token ID → merged bytes
        vocab[next_token_id] = merged_bytes

        # Append merged pair to merges as tuple of bytes
        merges.append((vocab[most_freq_pair[0]], vocab[most_freq_pair[1]]))

        # Replace merged token pairs
        new_tokens = []
        for t in tokens:
            new_t = []
            i = 0
            while i < len(t):
                if (
                    i < len(t) - 1
                    and (t[i], t[i + 1]) == most_freq_pair
                ):
                    # Replace pair by new merged token ID
                    new_t.append(next_token_id)
                    i += 2
                else:
                    # Keep existing token
                    new_t.append(t[i])
                    i += 1
            new_tokens.append(new_t)

        tokens = new_tokens

        next_token_id += 1
        num_merges += 1

        # Check max merges limit
        if max_merges is not None and num_merges >= max_merges:
            break

    return tokens, vocab, merges


@timed
def merge(
    tokens: list[list[int]],
    vocab: dict[int, bytes],
    vocab_size: int,
    max_merges: int = None,
):

    merges = []
    next_token_id = max(vocab.keys()) + 1
    num_merges = 0

    pair_counts, pair_positions = calc_pair_counts_positions(tokens)

    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    while len(vocab) < vocab_size and heap:
        neg_count, most_freq_pair = heapq.heappop(heap)
        freq = -neg_count

        if pair_counts.get(most_freq_pair, 0) != freq or freq < 1:
            continue

        # Perform merge
        merged_token = next_token_id
        next_token_id += 1
        merges.append((vocab[most_freq_pair[0]], vocab[most_freq_pair[1]]))
        vocab[merged_token] = vocab[most_freq_pair[0]] + vocab[most_freq_pair[1]]

        positions = pair_positions.pop(most_freq_pair, set()).copy()
        pair_counts.pop(most_freq_pair, None)

        for token_idx, pos in positions:
            token = tokens[token_idx]

            # Verify position validity explicitly
            if pos >= len(token) - 1 or (token[pos], token[pos+1]) != most_freq_pair:
                continue

            # Identify neighboring pairs
            prev_pair = (token[pos - 1], token[pos]) if pos > 0 else None
            next_pair = (token[pos + 1], token[pos + 2]) if pos + 2 < len(token) else None

            # Perform token merge at this position
            token[pos:pos + 2] = [merged_token]

            # Update positions for affected pairs
            if prev_pair:
                pair_positions[prev_pair].discard((token_idx, pos - 1))
                pair_counts[prev_pair] -= 1

                new_prev_pair = (prev_pair[0], merged_token)
                pair_positions[new_prev_pair].add((token_idx, pos - 1))
                pair_counts[new_prev_pair] += 1
                heapq.heappush(heap, (-pair_counts[new_prev_pair], new_prev_pair))

            if next_pair:
                pair_positions[next_pair].discard((token_idx, pos + 1))
                pair_counts[next_pair] -= 1

                new_next_pair = (merged_token, next_pair[1])
                pair_positions[new_next_pair].add((token_idx, pos))
                pair_counts[new_next_pair] += 1
                heapq.heappush(heap, (-pair_counts[new_next_pair], new_next_pair))

        num_merges += 1

        if max_merges is not None and num_merges >= max_merges:
            break

    return tokens, vocab, merges


def pretokenize_chunk(args):
    start, end, filepath, special_tokens = args
    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="replace")
    docs = strip_and_pretokenize(chunk, special_tokens=special_tokens)
    return docs


@timed
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = None,
    max_merges: int = None
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
        vocab: dict[int, bytes] (token ID → bytes)
        merges: list[tuple[bytes, bytes]] (ordered list of merges performed)
    """
    from multiprocessing import Pool, cpu_count

    # Initialize vocabulary with single-byte tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # Find chunk boundaries for parallel processing
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes or cpu_count(), "<|endoftext|>".encode("utf-8"))

    chunk_args = [(boundaries[i], boundaries[i+1], input_path, special_tokens)
                  for i in range(len(boundaries)-1)]

    if num_processes is None:
        num_processes = cpu_count()

    with Pool(num_processes) as pool:
        docs_chunks = pool.map(pretokenize_chunk, chunk_args)

    # Flatten and convert pre-tokens to sequences of byte IDs
    docs = [doc for chunk in docs_chunks for doc in chunk]
    tokens = [list(doc.encode("utf-8")) for doc in docs]

    tokens, vocab, merges = merge(
        tokens=tokens,
        vocab=vocab,
        vocab_size=vocab_size,
        max_merges=max_merges
    )

    return vocab, merges
