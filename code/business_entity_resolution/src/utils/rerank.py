"""GPU re-ranking of blocked candidates before the top-K cut (stage 04).

Why: ranking by summed key IDF left true links whose only shared key is a name key (address empty
or noisy) below rank 32. On a 20k-S1 training sample, recall@32 was 0.925 with IDF ranking; it
rose to 0.965 when every aggregated candidate was rescored with this cheap similarity (reachable
ceiling with the same keys 0.976).

score = 0.4 * max(core-token Jaccard, core trigram Dice) + 0.4 * address-token Jaccard
        + 0.2 * house-number Jaccard + 0.02 * cos,   cos = sum IDF / target_key_norm ** 0.25
The weights were picked on that training sample; no test data is involved.

Compact per-record arrays live on the GPU for the whole stage and pairs are gathered there:
34 int32 per record -> train targets 10.3M x 136 B = 1.40 GB, S1 2.2M x 136 B = 0.30 GB, plus
<= 0.5 GB of batch temporaries at RERANK_BATCH = 131,072 pairs (measured ~8M pairs/s, 2.2 GB peak).
"""
import numpy as np
import torch

from .features import _dice, _set_stats, _trigrams

W_NAME, W_ADDR, W_HNUM, W_COS = 0.4, 0.4, 0.2, 0.02
COS_POWER = 0.25
RERANK_BATCH = 131_072
COMPACT_COLS = 34   # core_tok 6 | addr_tok 16 | hnum 4 | core_chars 32 bytes = 8 int32


def compact_records(rec, step=1_000_000):
    out = np.empty((len(rec), COMPACT_COLS), np.int32)
    for a in range(0, len(rec), step):
        b = rec[a:a + step]
        out[a:a + len(b)] = np.concatenate(
            [b["core_tok"], b["addr_tok"], b["hnum"], np.ascontiguousarray(b["core_chars"]).view(np.int32)], 1)
    return out


def target_key_norm(index, n_targets, step=2_000_000):
    """Sum of IDF over each target's (kept) keys, from the CSR index."""
    norm = np.zeros(n_targets, np.float64)
    offs = np.asarray(index.offs)
    for a in range(0, len(index.keys), step):
        b = min(len(index.keys), a + step)
        df = np.diff(offs[a:b + 1])
        np.add.at(norm, np.asarray(index.post[offs[a]:offs[b]]), np.repeat(np.asarray(index.idf[a:b], np.float64), df))
    return norm.astype(np.float32)


def _jacc(i, na, nb):
    u = na + nb - i
    return torch.where(u > 0, i.float() / u.clamp(min=1).float(), torch.zeros_like(i, dtype=torch.float32))


class GPUReranker:
    def __init__(self, rec_s1, rec_t, tnorm, device):
        self.device = device
        self.s1 = torch.from_numpy(compact_records(rec_s1)).to(device)
        self.t = torch.from_numpy(compact_records(rec_t)).to(device)
        self.tnorm = tnorm

    def vram_gb(self):
        return (self.s1.numel() + self.t.numel()) * 4 / 2**30

    @torch.no_grad()
    def score(self, s1, tgt, idf_sum):
        cos = idf_sum / np.maximum(self.tnorm[tgt], 1e-6) ** COS_POWER
        out = np.empty(len(s1), np.float32)
        for a in range(0, len(s1), RERANK_BATCH):
            si = torch.from_numpy(s1[a:a + RERANK_BATCH].astype(np.int64)).to(self.device)
            ti = torch.from_numpy(tgt[a:a + RERANK_BATCH].astype(np.int64)).to(self.device)
            A, B = self.s1[si], self.t[ti]
            core = _jacc(*_set_stats(A[:, :6], B[:, :6]))
            addr = _jacc(*_set_stats(A[:, 6:22], B[:, 6:22]))
            hnum = _jacc(*_set_stats(A[:, 22:26], B[:, 22:26]))
            tri = torch.nan_to_num(_dice(_trigrams(A[:, 26:].contiguous().view(torch.uint8)),
                                         _trigrams(B[:, 26:].contiguous().view(torch.uint8))), nan=0.0)
            sc = W_NAME * torch.maximum(core, tri) + W_ADDR * addr + W_HNUM * hnum
            out[a:a + len(si)] = sc.cpu().numpy()
        return out + W_COS * cos.astype(np.float32)
