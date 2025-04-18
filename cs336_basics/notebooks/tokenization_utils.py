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


def regex_pretokenize(
    text,
    regex=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
):
    return [match.group() for match in re.finditer(regex, text)]


def strip_and_pretokenize(text, special_tokens):
    split_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    parts = re.split(split_pattern, text)
    return [p.strip() for p in parts if p.strip()]


def decode_token(token):
    if isinstance(token, int):
        return bytes([token]).decode('utf-8', errors='replace')
    elif isinstance(token, bytes):
        return token.decode('utf-8', errors='replace')
    elif isinstance(token, str):
        return token
    elif isinstance(token, tuple):
        return ''.join(decode_token(t) for t in token)
    else:
        raise TypeError(f"Unexpected token type: {type(token)}")


def decode_token_strict(token_id, id_to_bytes):
    return id_to_bytes[token_id].decode('utf-8', errors='replace')


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if desired_num_chunks <= 1 or size == 0:
        return [0, size]
    chunk_size = size // desired_num_chunks
    bounds = [i * chunk_size for i in range(desired_num_chunks + 1)]
    bounds[-1] = size
    mini = 4096
    for idx in range(1, len(bounds) - 1):
        pos = bounds[idx]
        file.seek(pos)
        while True:
            data = file.read(mini)
            if not data:
                bounds[idx] = size
                break
            found = data.find(split_special_token)
            if found != -1:
                bounds[idx] = pos + found
                break
            pos += mini
    return sorted(set(bounds))


def calc_pair_counts_positions(tokens: list[list[int]]):
    pair_counts = defaultdict(int)
    pair_positions = defaultdict(set)
    for ti, seq in enumerate(tokens):
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            pair_counts[pair] += 1
            pair_positions[pair].add((ti, i))
    return pair_counts, pair_positions


def invert_bytes(b: bytes) -> bytes:
    """
    Byte‑wise invert for lexicographic tie‑breaking:
    makes the lexicographically largest bytestring have the smallest key.
    """
    return bytes(255 - x for x in b)


def lex_key(b: bytes) -> bytes:
    # no inversion, just raw bytes for natural lex ordering
    return b


@timed
def merge_naive(
    tokens: list[list[int]],
    vocab: dict[int, bytes],
    vocab_size: int,
    special_token_ids: set[int],
    max_merges: int = None,
):
    merges = []
    next_id = max(vocab.keys()) + 1
    num_merges = 0

    while len(vocab) < vocab_size:
        counts = defaultdict(int)
        for seq in tokens:
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i+1]
                if a in special_token_ids or b in special_token_ids:
                    continue
                counts[(a, b)] += 1

        if not counts:
            break
        max_freq = max(counts.values())
        if max_freq < 1:
            break

        # tie‑break by lexicographically greatest raw bytes
        candidates = [p for p, c in counts.items() if c == max_freq]
        best_pair = max(
            candidates,
            key=lambda p: lex_key(vocab[p[0]] + vocab[p[1]])
        )

        # merge best_pair
        a, b = best_pair
        merged_bytes = vocab[a] + vocab[b]
        vocab[next_id] = merged_bytes
        merges.append((vocab[a], vocab[b]))

        # rebuild tokens with this merge
        new_tokens = []
        for seq in tokens:
            out, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i+1]) == (a, b):
                    out.append(next_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            new_tokens.append(out)
        tokens = new_tokens

        next_id += 1
        num_merges += 1
        if max_merges and num_merges >= max_merges:
            break

    return tokens, vocab, merges


@timed
def merge_with_heap(
    tokens: list[list[int]],
    vocab: dict[int, bytes],
    vocab_size: int,
    special_token_ids: set[int],
    max_merges: int = None,
):
    merges = []
    next_id = max(vocab.keys()) + 1

    # initial counts & positions
    pair_counts, pair_positions = calc_pair_counts_positions(tokens)

    # build heap with raw‑bytes tie‑break (invert_bytes makes largest raw bytes smallest key)
    heap = [
        (-cnt, vocab[a] + vocab[b], a, b)
        for (a, b), cnt in pair_counts.items()
        if cnt > 0 and a not in special_token_ids and b not in special_token_ids
    ]
    heapq.heapify(heap)

    num_merges = 0
    while len(vocab) < vocab_size and heap:
        neg_cnt, _, a, b = heapq.heappop(heap)
        freq = -neg_cnt

        # skip stale or zero
        if pair_counts.get((a, b), 0) != freq or freq < 1:
            continue

        # record merge
        merged_bytes = vocab[a] + vocab[b]
        vocab[next_id] = merged_bytes
        merges.append((vocab[a], vocab[b]))

        # remove old pair
        positions = pair_positions.pop((a, b), set())
        pair_counts.pop((a, b), None)

        # apply merge in each token, update neighbors
        for ti, pos in sorted(positions, reverse=True):
            seq = tokens[ti]
            if pos < len(seq)-1 and (seq[pos], seq[pos+1]) == (a, b):
                seq[pos:pos+2] = [next_id]

                # left neighbor (L, a) → (L, next_id)
                if pos > 0:
                    L = seq[pos-1]
                    old = (L, a)
                    if old in pair_positions:
                        pair_positions[old].discard((ti, pos-1))

                    # only decrement once, and never below zero
                    if old in pair_counts and pair_counts[old] > 0:
                        pair_counts[old] -= 1

                    new = (L, next_id)

                    if L not in special_token_ids and next_id not in special_token_ids:
                        pair_positions[new].add((ti, pos-1))
                        pair_counts[new] = pair_counts.get(new, 0) + 1
                        heapq.heappush(
                            heap,
                            (
                                -pair_counts[new],
                                vocab[L] + vocab[next_id],
                                L,
                                next_id
                            )
                        )

                # right neighbor (b, R) → (next_id, R)
                if pos < len(seq) - 1:
                    R = seq[pos+1]
                    old = (b, R)
                    if old in pair_positions:
                        pair_positions[old].discard((ti, pos))

                    pair_counts[old] = pair_counts.get(old, 0) - 1
                    if old in pair_counts and pair_counts[old] > 0:
                        pair_counts[old] -= 1

                    new = (next_id, R)

                    if next_id not in special_token_ids and R not in special_token_ids:
                        pair_positions[new].add((ti, pos))
                        pair_counts[new] = pair_counts.get(new, 0) + 1
                        heapq.heappush(
                            heap,
                            (
                                -pair_counts[new],
                                lex_key(vocab[next_id] + vocab[R]),
                                next_id,
                                R
                            )
                        )

        next_id += 1
        num_merges += 1
        if max_merges and num_merges >= max_merges:
            break

    return tokens, vocab, merges


@timed
def load_dataset_stream(filepath, split_token="<|endoftext|>"):
    docs, curr = [], []
    st = split_token.strip()
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            l = line.strip()
            if l == st:
                if curr:
                    docs.append("\n".join(curr))
                    curr = []
            else:
                curr.append(l)
    if curr:
        docs.append("\n".join(curr))
    return docs


def pretokenize_chunk(args):
    start, end, filepath, special_tokens = args
    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end-start).decode('utf-8', errors='replace')
    docs = strip_and_pretokenize(chunk, special_tokens)
    tokens = []
    for doc in docs:
        if doc in special_tokens:
            continue
        for pt in regex_pretokenize(doc):
            tokens.append(list(pt.encode("utf-8", errors="replace")))
    return tokens


@timed
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = None,
    max_merges: int = None,
    use_naive_merge: bool = False
):
    if num_processes is None:
        num_processes = cpu_count()

    id_to_bytes = {i: bytes([i]) for i in range(256)}
    next_id = 256
    special_token_ids = set()
    for tok in special_tokens:
        id_to_bytes[next_id] = tok.encode('utf-8')
        special_token_ids.add(next_id)
        next_id += 1

    with open(input_path, 'rb') as f:
        bounds = find_chunk_boundaries(f, num_processes, special_tokens[0].encode('utf-8'))

    args = [(bounds[i], bounds[i+1], input_path, special_tokens)
            for i in range(len(bounds)-1)]
    with Pool(num_processes) as pool:
        chunks = pool.map(pretokenize_chunk, args)

    tokens = [token for chunk in chunks for token in chunk]

    if use_naive_merge:
        tokens, id_to_bytes, merges = merge_naive(
            tokens, id_to_bytes, vocab_size, special_token_ids, max_merges)
    else:
        tokens, id_to_bytes, merges = merge_with_heap(
            tokens, id_to_bytes, vocab_size, special_token_ids, max_merges)

    final_vocab = {id_to_bytes[i]: i for i in sorted(id_to_bytes)}

    return final_vocab, merges
