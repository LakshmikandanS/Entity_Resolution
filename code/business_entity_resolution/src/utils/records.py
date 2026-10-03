"""Fixed-width per-record arrays (265 bytes/record) consumed by the GPU feature kernels.

Strings are reduced once, in stage 03, to padded uint8 character arrays and int32 token hashes, so
the pair stage never touches Python strings. Hash 0 is reserved for padding.
"""
import zlib

import numpy as np

CORE_CHARS = 32
ADDR_CHARS = 64
CORE_TOK = 6
NAME_TOK = 8
ADDR_TOK = 16
HNUM = 4
SKEY = 3

REC_DTYPE = np.dtype([
    ("core_chars", np.uint8, (CORE_CHARS,)),
    ("addr_chars", np.uint8, (ADDR_CHARS,)),   # address tokens sorted alphabetically (order-free)
    ("core_tok", np.int32, (CORE_TOK,)),
    ("name_tok", np.int32, (NAME_TOK,)),
    ("addr_tok", np.int32, (ADDR_TOK,)),
    ("hnum", np.int32, (HNUM,)),
    ("skey", np.int32, (SKEY,)),
    ("core_concat", np.int32),
    ("state", np.int32),
    ("legal", np.int32),
    ("n_core", np.uint8),
    ("n_name", np.uint8),
    ("n_addr", np.uint8),
    ("n_hnum", np.uint8),
    ("n_skey", np.uint8),
    ("core_len", np.uint8),
    ("addr_len", np.uint8),
    ("flags", np.uint16),
])
assert REC_DTYPE.itemsize == 265, REC_DTYPE.itemsize


def h32(text):
    """Stable 32-bit token hash; never 0 (0 = padding)."""
    if not text:
        return 0
    v = zlib.crc32(text.encode("ascii", "ignore"))
    return v or 1


def _hash_matrix(lists, width):
    out = np.zeros((len(lists), width), dtype=np.uint32)
    for i, toks in enumerate(lists):
        for j, t in enumerate(toks[:width]):
            out[i, j] = h32(t)
    return out.view(np.int32)


def _char_matrix(strings, width):
    buf = b"".join(s.encode("ascii", "ignore")[:width].ljust(width, b"\0") for s in strings)
    return np.frombuffer(buf, dtype=np.uint8).reshape(len(strings), width)


def build_record_block(rows):
    """rows: list of dicts with keys canon, core, legal, tokens, hnums, skeys, state, sorted_str,
    flags. Returns a structured array of len(rows)."""
    n = len(rows)
    rec = np.zeros(n, dtype=REC_DTYPE)
    core_str = [" ".join(r["core"]) for r in rows]
    rec["core_chars"] = _char_matrix(core_str, CORE_CHARS)
    rec["addr_chars"] = _char_matrix([r["sorted_str"] for r in rows], ADDR_CHARS)
    rec["core_tok"] = _hash_matrix([list(dict.fromkeys(r["core"])) for r in rows], CORE_TOK)
    rec["name_tok"] = _hash_matrix([list(dict.fromkeys(r["canon"])) for r in rows], NAME_TOK)
    rec["addr_tok"] = _hash_matrix([r["tokens"] for r in rows], ADDR_TOK)
    rec["hnum"] = _hash_matrix([r["hnums"] for r in rows], HNUM)
    rec["skey"] = _hash_matrix([r["skeys"] for r in rows], SKEY)
    u32 = lambda vals: np.array(vals, dtype=np.uint32).view(np.int32)
    rec["core_concat"] = u32([h32("".join(r["core"])) for r in rows])
    rec["state"] = u32([h32(r["state"]) for r in rows])
    rec["legal"] = u32([h32(r["legal"]) for r in rows])
    clip = lambda vals: np.minimum(np.array(vals, dtype=np.int64), 255).astype(np.uint8)
    rec["n_core"] = clip([len(set(r["core"])) for r in rows])
    rec["n_name"] = clip([len(set(r["canon"])) for r in rows])
    rec["n_addr"] = clip([len(r["tokens"]) for r in rows])
    rec["n_hnum"] = clip([len(r["hnums"]) for r in rows])
    rec["n_skey"] = clip([len(r["skeys"]) for r in rows])
    rec["core_len"] = clip([min(len(s), CORE_CHARS) for s in core_str])
    rec["addr_len"] = clip([min(len(r["sorted_str"]), ADDR_CHARS) for r in rows])
    rec["flags"] = np.array([r["flags"] for r in rows], dtype=np.uint16)
    return rec
