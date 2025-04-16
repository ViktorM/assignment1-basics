import os
import time
import heapq
import regex as re

from typing import BinaryIO

from collections import defaultdict
from functools import wraps
from multiprocessing import Pool, cpu_count


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
    """
    Very naive token splitting on whitespace.
    """
    return text.strip().split()


def regex_pretokenize(
    text, 
    regex=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
):
    """
    Return list of tokens from text using the given regex.
    """
    return [match.group() for match in re.finditer(regex, text)]


def strip_and_pretokenize(text, special_tokens):
    """
    Split the text by special tokens, strip, and return non-empty pieces.
    """
    split_pattern = "|".join(re.escape(token) for token in special_tokens)
    docs = re.split(split_pattern, text)
    docs = [doc.strip() for doc in docs if doc.strip()]
    return docs


# Single-argument decode_token() that can handle ints, bytes, tuple, or str
def decode_token(token):
    """
    Decode an arbitrary 'token' object (int, bytes, str, or tuple) into a string,
    used by the test harness if your vocab keys are not just bytes.

    The test harness calls decode_token(token) with 1 argument. We cannot rely
    on a second vocab argument. So this version does a naive, purely structural
    decode:
      - int: treat as a single-byte
      - bytes: decode as UTF-8
      - str: return as-is
      - tuple: decode each subtoken recursively, then join
    """
    if isinstance(token, int):
        # interpret as single-byte
        return bytes([token]).decode("utf-8", errors="replace")
    elif isinstance(token, bytes):
        return token.decode("utf-8", errors="replace")
    elif isinstance(token, str):
        return token  # already a string
    elif isinstance(token, tuple):
        # decode each element in the tuple, then concat
        return "".join(decode_token(t) for t in token)
    else:
        raise TypeError(f"Unexpected token type: {type(token)}")


def decode_token_strict(token_id, id_to_bytes):
    """
    A helper for tie-breaking in naive BPE merges:
    Convert token_id -> bytes -> a Python string with 'replace' errors
    so we can do lexicographical comparison in a deterministic way.
    """
    return id_to_bytes[token_id].decode("utf-8", errors="replace")


@timed
def load_dataset_stream(filepath, split_token="<|endoftext|>"):
    """
    Load dataset from a text file, splitting by `split_token`.
    """
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


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    """
    Determines chunk boundaries by searching for `split_special_token`
    near uniformly spaced offsets in the file.
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        # Only one chunk or empty file => trivial
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if not mini_chunk:  # EOF
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    # Remove duplicates and sort
    return sorted(set(chunk_boundaries))


def calc_pair_counts_positions(tokens: list[list[int]]):
    """
    Returns:
      - pair_counts: dict[(int,int) → int], how often a pair occurs.
      - pair_positions: dict[(int,int) → set of (token_idx, position)] 
    """
    pair_counts = defaultdict(int)
    pair_positions = defaultdict(set)
    for token_idx, token_seq in enumerate(tokens):
        # token_seq is a list of integer token-IDs
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i+1])
            pair_counts[pair] += 1
            pair_positions[pair].add((token_idx, i))
    return pair_counts, pair_positions

# TODO: to fix
@timed
def merge_naive(
    tokens: list[list[int]],  # each doc as a list of int IDs
    vocab: dict[int, bytes],  # maps token ID → bytes
    vocab_size: int,
    special_token_ids: set[int],
    max_merges: int = None,
):
    """
    The simpler, iterative BPE loop:
      1) count all pairs
      2) pick the most frequent
      3) merge them
      4) repeat
    Because we recalc pair counts each iteration, it can be very slow.
    """
    merges = []
    next_token_id = max(vocab.keys()) + 1
    num_merges = 0

    while len(vocab) < vocab_size:
        pair_counts = defaultdict(int)
        # Recompute pair occurrences on every iteration
        for seq in tokens:
            for i in range(len(seq) - 1):
                # Skip pairs that contain special tokens
                if seq[i] in special_token_ids or seq[i + 1] in special_token_ids:
                    continue
                pair_counts[(seq[i], seq[i + 1])] += 1

        if not pair_counts:
            break

        # pick the most frequent pair
        best_pair = max(pair_counts.keys(), key=lambda p: (
                pair_counts[p],
                decode_token_strict(p[0], vocab) + decode_token_strict(p[1], vocab)
            )
        )
        best_pair_count = pair_counts[best_pair]
        if best_pair_count < 1:
            break

        # create new token
        merged_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[next_token_id] = merged_bytes
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # replace all occurrences
        new_tokens = []
        for seq in tokens:
            i = 0
            merged_seq = []
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i+1]) == best_pair:
                    # merge
                    merged_seq.append(next_token_id)
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1
            new_tokens.append(merged_seq)

        tokens = new_tokens
        next_token_id += 1
        num_merges += 1

        if max_merges and num_merges >= max_merges:
            break

    return tokens, vocab, merges


# Iver bytes trick for tie-breaking
def invert_bytes(b: bytes) -> bytes:
    # 255 - x in each position
    return bytes(255 - x for x in b)


class ReverseBytes:
    def __init__(self, b: bytes):
        self.b = b
    def __lt__(self, other: "ReverseBytes"):
        # Reverse the natural order: if self.b is greater (lexicographically), then we consider self < other.
        return self.b > other.b
    def __eq__(self, other: object):
        if not isinstance(other, ReverseBytes):
            return NotImplemented
        return self.b == other.b
    def __repr__(self):
        return repr(self.b)


def tiebreak_value(a_id: int, b_id: int, vocab: dict[int, bytes]) -> tuple[str, str]:
    """
    Return a tuple of decoded strings for tie-breaking:
    among ties in frequency, the pair whose (decoded_a, decoded_b) is
    lexicographically smaller gets popped first from the heap.
    """
    Adec = vocab[a_id].decode('utf-8', errors='replace')
    Bdec = vocab[b_id].decode('utf-8', errors='replace')
    return (Adec, Bdec)


def tiebreak_key(a: int, b: int, vocab: dict[int, bytes]) -> tuple[str, str]:
    return (
        vocab[a].decode("utf-8", errors="replace"),
        vocab[b].decode("utf-8", errors="replace"),
    )


def make_heap(
    pair_counts: dict[tuple[int, int], int],
    vocab: dict[int, bytes],
    special_token_ids: set[int]
):
    """
    Build the initial max-heap (using Python's min-heap with negative frequency).
    We store: ( -count, invert_bytes(vocab[idA]+vocab[idB]), (idA, idB) ).
    Also, skip pairs if they involve special tokens.
    """
    heap = []
    for pair, count in pair_counts.items():
        if pair[0] is None or pair[1] is None:
            print(f"None in pair: {pair}, count: {count}")
            continue
        if pair[0] in special_token_ids or pair[1] in special_token_ids:
            # skip merging across special tokens
            continue
        if count < 1:
            continue

        tiebreak_key = tiebreak_value(pair[0], pair[1], vocab)

        heap.append((-count, tiebreak_key, (pair[0], pair[1])))

    heapq.heapify(heap)
    return heap


@timed
def merge_with_heap(
    tokens: list[list[int]],
    vocab: dict[int, bytes],
    vocab_size: int,
    special_token_ids: set[int] = None,
    max_merges: int = None,
):
    """
    tokens: list of list of int token-IDs
    v: dict[int -> bytes], the current token ID -> raw bytes
    returns:
      tokens: updated
      vocab: updated
      merges: list of (bytes, bytes)
    """
    if special_token_ids is None:
        special_token_ids = set()
    merges = []
    next_token_id = max(vocab.keys()) + 1
    num_merges = 0

    pair_counts, pair_positions = calc_pair_counts_positions(tokens)
    heap = make_heap(pair_counts, vocab, special_token_ids)

    while len(vocab) < vocab_size and heap:
        neg_count, inv_key, pair = heapq.heappop(heap)
        freq = -neg_count

        # If the current frequency is out-of-date or zero, skip this entry.
        if pair_counts.get(pair, 0) != freq or freq < 1:
            continue

        same = [(tie_key, pair)]
        while heap and -heap[0][0] == freq:
            n_neg, n_key, n_pair = heapq.heappop(heap)
            if pair_counts.get(n_pair, 0) == freq:
                same.append((n_key, n_pair))

        # 4) Pick the lexicographically smallest decoded pair
        best_key, best_pair = min(same, key=lambda x: x[0])

        # 5) Push the “losers” back
        for k, p in same:
            if p is not best_pair or k != best_key:
                heapq.heappush(heap, (-freq, k, p))

        # 6) Now *merge* best_pair, not the original `pair`
        pair = best_pair

        # Merge: create a new token by concatenating the two tokens.
        merged_bytes = vocab[pair[0]] + vocab[pair[1]]
        vocab[next_token_id] = merged_bytes
        merges.append((vocab[pair[0]], vocab[pair[1]]))

        # Remove the merged pair from our statistics.
        positions = pair_positions.pop(pair)
        pair_counts.pop(pair, None)

        # Process positions from right-to-left so index shifts don't affect unprocessed positions.
        positions_sorted = sorted(positions, key=lambda x: (x[0], x[1]), reverse=True)
        for token_idx, pos in positions_sorted:
            seq = tokens[token_idx]
            if pos >= len(seq) - 1 or (seq[pos], seq[pos+1]) != pair:
                continue

            # Replace the merged pair with the new token.
            seq[pos:pos+2] = [next_token_id]

            # Update neighbors around the merge.
            # Update left neighbor
            if pos > 0:
                left_id = seq[pos-1]
                old_left = (left_id, pair[0])
                if old_left in pair_positions:
                    pair_positions[old_left].discard((token_idx, pos-1))
                    if pair_counts.get(old_left, 0) > 0:
                        pair_counts[old_left] -= 1

                new_left = (left_id, next_token_id)
                if left_id not in special_token_ids and next_token_id not in special_token_ids:
                    pair_positions[new_left].add((token_idx, pos-1))
                    pair_counts[new_left] += 1
                    freq_new = pair_counts[new_left]

                    key = tiebreak_key(left_id, next_token_id, vocab)
                    heapq.heappush(heap, (-freq_new, key, new_left))

            # Update right neighbor
            if pos < len(seq) - 1:
                right_id = seq[pos+1]
                old_right = (pair[0], right_id)
                if old_right in pair_positions:
                    pair_positions[old_right].discard((token_idx, pos))
                    if pair_counts.get(old_right, 0) > 0:
                        pair_counts[old_right] -= 1

                new_right = (next_token_id, right_id)
                if next_token_id not in special_token_ids and right_id not in special_token_ids:
                    pair_positions[new_right].add((token_idx, pos))
                    pair_counts[new_right] += 1
                    freq_new = pair_counts[new_right]
                    key = tiebreak_key(next_token_id, right_id, vocab)
                    heapq.heappush(heap, (-freq_new, key, new_right))

        next_token_id += 1
        num_merges += 1
        if max_merges is not None and num_merges >= max_merges:
            break

    return tokens, vocab, merges


def pretokenize_chunk(args):
    """
    Worker function for parallel chunk reading and pre-tokenization.
    """
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
    max_merges: int = None,
    use_naive_merge: bool = False
):
    """
    Train BPE tokenizer.

    Args:
        input_path (str): Path to text file for BPE training.
        vocab_size (int): Desired final vocabulary size
                          (including 256 bytes + special tokens + merges).
        special_tokens (list[str]): Custom special tokens to add to vocab.
        num_processes (int): Processes for parallel chunk reading; defaults to CPU count.
        max_merges (int): Limit merges for debugging or partial training.
        use_naive_merge (bool): If True, uses the simple iterative approach
                                (slower but more canonical).
    Returns:
        vocab: dict[int, bytes] (token ID → bytes)
        merges: list[tuple[bytes, bytes]] (ordered merges performed)
    """
    if num_processes is None:
        num_processes = cpu_count()

    # Initialize vocab with 256 byte tokens
    # vocab = {i: bytes([i]) for i in range(256)}
    # Add special tokens
    # for token in special_tokens:
    #     vocab[len(vocab)] = token.encode("utf-8")

    # 1) We'll store integer->bytes internally (id_to_bytes) for merges.
    #    Then we’ll invert to produce the final 'vocab' as {bytes: id}.
    id_to_bytes = {}

    # Add 256 single-byte tokens
    next_id = 0
    for i in range(256):
        id_to_bytes[next_id] = bytes([i])
        next_id += 1

    # Add special tokens as distinct IDs
    special_token_ids = set()
    for sp in special_tokens:
        # We'll store them as bytes too, to keep merges consistent. 
        # TODO: double-check is it should be as str in the final vocab, do it. 
        # The test harness is typically OK with them as bytes or str.
        id_to_bytes[next_id] = sp.encode("utf-8")
        special_token_ids.add(next_id)
        next_id += 1

    # Find chunk boundaries for parallel pre-tokenization
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_processes, 
            split_special_token=b"<|endoftext|>"
        )

    chunk_args = [
        (boundaries[i], boundaries[i+1], input_path, special_tokens)
        for i in range(len(boundaries)-1)
    ]

    # Parallel load & pretokenize
    with Pool(num_processes) as pool:
        docs_chunks = pool.map(pretokenize_chunk, chunk_args)

    # Flatten docs
    docs = [doc for chunk in docs_chunks for doc in chunk]

    # Convert each doc to a sequence of byte IDs
    # tokens = [list(d.encode("utf-8")) for d in docs]

    # Convert doc strings -> list of integer IDs
    #    (simply byte-encode each doc, and map each byte to an ID).
    #    We have 0..255 for each byte, plus special tokens for others if present.
    #    If you want to treat unknown bytes differently, you can handle that below.
    tokens = []
    for d in docs:
        d_bytes = d.encode("utf-8", errors="replace")
        seq = []
        for b in d_bytes:
            # b is an int from 0..255. That maps 1:1 to our initial IDs.
            # (No fallback for unknown bytes.)
            seq.append(b)  # directly use b as token ID
        tokens.append(seq)

    if use_naive_merge:
        tokens, id_to_bytes, merges = merge_naive(tokens, id_to_bytes, vocab_size, special_token_ids, max_merges)
    else:
        tokens, id_to_bytes, merges = merge_with_heap(tokens, id_to_bytes, vocab_size, special_token_ids, max_merges)

    # Build the reversed, final vocab as {raw_token: int_id}
    #    The test harness expects:   for token, idx in vocab.items(): ...
    #    We'll do a stable ordering by ascending int ID so IDs are from 0..(N-1).
    #    If you do not care about consistent ID ordering, you can do any order.
    final_vocab = {}
    sorted_ids = sorted(id_to_bytes.keys())  # e.g. [0,1,2,3,...]
    for i in sorted_ids:
        raw_token = id_to_bytes[i]  # usually bytes
        final_vocab[raw_token] = i  # store as {bytes: int}

    return final_vocab, merges
