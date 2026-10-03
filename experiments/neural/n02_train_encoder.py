"""n02: fine-tune the bi-encoder (multilingual-e5-small, 384 -> 128 projection) with InfoNCE.

Group = 1 encoder-split S1 + 1 positive target + `hard_negs` hard negatives sampled from its baseline
candidates. Every row of the batch is scored against all candidates in the batch (in-batch + hard negatives).

Three losses, one shared encoder:
  comb : normalize(name ⊕ addr); masks only other true links of the same S1.
  name : additionally masks candidates whose baseline core-name key equals the anchor's (twins share names).
  addr : additionally masks candidates with the anchor's address key or an empty address; rows whose positive
         has an empty address are skipped.

Estimated cost (encoder split, defaults): ~763k groups/epoch, batch 64 groups = 320 records = 640 sequences.
  VRAM  ~2.2-2.8 GB with gradient checkpointing (weights 0.47 GB fp32, grads+AdamW 0.26 GB for 21.7M trainable
        params, activations ~0.6 GB, CUDA context/slack ~0.8 GB); capped at vram_fraction of the card.
  RAM   ~1.5 GB (encoder data as Arrow + model)
  disk  ~1.4 GB per checkpoint (model + optimiser), final model ~0.47 GB
  time  ~35-70 min per epoch on an RTX 5060 Laptop (estimate, not measured)
Resumable: checkpoints every `ckpt_every` steps to work/encoder/ckpt_last.pt; rerun the same command to resume.
On CUDA OOM the batch is halved (down to min_groups_per_batch) and the step retried.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from common import Monitor, add_config_arg, atomic_write_json, limit_vram, load_config, pick_device
from model import DualFieldEncoder, tokenize


class EncoderData:
    def __init__(self, d):
        s1 = pq.read_table(d / "s1.parquet")
        tg = pq.read_table(d / "targets.parquet")
        g = pq.read_table(d / "groups.parquet")
        self.s1_name = s1.column("name").combine_chunks()
        self.s1_addr = s1.column("addr").combine_chunks()
        self.s1_nkey = s1.column("name_key").to_numpy()
        self.s1_akey = s1.column("addr_key").to_numpy()
        self.t_name = tg.column("name").combine_chunks()
        self.t_addr = tg.column("addr").combine_chunks()
        self.t_nkey = tg.column("name_key").to_numpy()
        self.t_akey = tg.column("addr_key").to_numpy()
        self.t_empty = tg.column("addr_empty").to_numpy(zero_copy_only=False).astype(bool)
        self.t_owner = tg.column("owner_s1").to_numpy()
        self.g_s1 = g.column("s1").to_numpy()
        self.g_pos = g.column("pos").to_numpy()
        self.g_val = g.column("is_val").to_numpy(zero_copy_only=False).astype(bool)
        self.hardneg = np.load(d / "hardneg.npy")

    @staticmethod
    def take(arr, idx):
        return arr.take(idx).to_pylist()


def build_batch(D: EncoderData, gidx: np.ndarray, n_hn: int, rng, all_pool: bool = False):
    a, p = D.g_s1[gidx], D.g_pos[gidx]
    pool = D.hardneg[a]
    if all_pool:
        hn = pool
    else:
        score = np.where(pool >= 0, rng.random(pool.shape), -1.0)
        pick = np.argsort(-score, axis=1)[:, :n_hn]          # random valid slots first, then the -1 padding
        hn = np.take_along_axis(pool, pick, axis=1)
    cand = np.concatenate([p, hn.reshape(-1)])
    valid = cand >= 0
    cs = np.where(valid, cand, 0)
    return a, p, hn, cand, valid, cs


def embed_fields(model, tok, D, a, cs, mc, device):
    names = D.take(D.s1_name, a) + D.take(D.t_name, cs)
    addrs = D.take(D.s1_addr, a) + D.take(D.t_addr, cs)
    ids, am = tokenize(tok, [mc["prefix_name"] + x for x in names], mc["max_len_name"], device)
    en = model(ids, am)
    ids, am = tokenize(tok, [mc["prefix_addr"] + x for x in addrs], mc["max_len_addr"], device)
    ea = model(ids, am)
    B = len(a)
    return en[:B], en[B:], ea[:B], ea[B:]


def masked_ce(logits, mask, row_ok):
    logits = logits.masked_fill(mask, float("-inf"))
    tgt = torch.arange(logits.shape[0], device=logits.device)
    loss = F.cross_entropy(logits[row_ok], tgt[row_ok]) if row_ok.any() else logits.sum() * 0
    acc = (logits.argmax(1) == tgt)[row_ok].float().mean() if row_ok.any() else torch.tensor(0.0)
    return loss, acc


def contrastive_losses(D, a, p, cand, valid, cs, en_a, en_c, ea_a, ea_c, tau, device):
    B, C = len(a), len(cand)
    t = lambda x: torch.as_tensor(x, device=device)
    is_pos_col = t(np.arange(C)[None, :] == np.arange(B)[:, None])
    invalid = t(~valid)[None, :].expand(B, C)
    same_entity = t(D.t_owner[cs][None, :] == a[:, None]) & ~is_pos_col
    base = invalid | same_entity
    name_twin = t(D.t_nkey[cs][None, :] == D.s1_nkey[a][:, None]) & ~is_pos_col
    addr_twin = t((D.t_akey[cs][None, :] == D.s1_akey[a][:, None]) | D.t_empty[cs][None, :]) & ~is_pos_col
    comb_a = F.normalize(torch.cat([en_a, ea_a], 1), dim=-1)
    comb_c = F.normalize(torch.cat([en_c, ea_c], 1), dim=-1)
    all_rows = torch.ones(B, dtype=torch.bool, device=device)
    addr_rows = t(~D.t_empty[p])
    lc, acc_c = masked_ce(comb_a @ comb_c.T / tau, base, all_rows)
    ln, acc_n = masked_ce(en_a @ en_c.T / tau, base | name_twin, all_rows)
    la, acc_a = masked_ce(ea_a @ ea_c.T / tau, base | addr_twin, addr_rows)
    return (lc, ln, la), (acc_c, acc_n, acc_a)


@torch.no_grad()
def evaluate(model, tok, D, val_idx, cfg, device, max_batches):
    """Recall@1 of the positive against the S1's own hard-negative pool (comb / name / addr)."""
    model.eval()
    mc, B = cfg["model"], cfg["train"]["groups_per_batch"]
    hits = {"comb": [], "name": [], "addr": []}
    rng = np.random.default_rng(0)
    for bi, s in enumerate(range(0, len(val_idx), B)):
        if bi >= max_batches:
            break
        a, p, hn, cand, valid, cs = build_batch(D, val_idx[s:s + B], 0, rng, all_pool=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            en_a, en_c, ea_a, ea_c = embed_fields(model, tok, D, a, cs, mc, device)
        n = len(a)
        pool = hn.shape[1]
        hv = torch.as_tensor(valid[n:].reshape(n, pool), device=device)
        for key, (xa, xc) in {"name": (en_a, en_c), "addr": (ea_a, ea_c),
                              "comb": (F.normalize(torch.cat([en_a, ea_a], 1), dim=-1),
                                       F.normalize(torch.cat([en_c, ea_c], 1), dim=-1))}.items():
            sp = (xa * xc[:n]).sum(1)
            sh = (xa[:, None, :] * xc[n:].reshape(n, pool, -1)).sum(-1).masked_fill(~hv, -2.0)
            hits[key].append((sp > sh.max(1).values).float().cpu())
    model.train()
    return {f"val_recall@1_{k}": float(torch.cat(v).mean()) if v else float("nan") for k, v in hits.items()}


def main():
    ap = add_config_arg(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after this many optimiser steps with a checkpoint (testing / time-boxed runs)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc, mc = cfg["train"], cfg["model"]
    data_dir = cfg.work / "encoder_data"
    if not (data_dir / "DONE").exists():
        raise SystemExit("run n01_build_encoder_data.py first")
    out_dir = cfg.wpath("encoder", "x").parent
    final_dir = out_dir / "final"
    if (final_dir / "meta.json").exists():
        print(f"{final_dir} exists; delete it to retrain")
        return
    mon = Monitor(cfg, "n02_train_encoder")
    device = pick_device()
    limit_vram(tc["vram_fraction"])
    torch.manual_seed(tc["seed"])

    with mon.stage("load"):
        D = EncoderData(data_dir)
        train_idx = np.flatnonzero(~D.g_val)
        val_idx = np.flatnonzero(D.g_val)
        tok = AutoTokenizer.from_pretrained(mc["name_or_path"])
        model = DualFieldEncoder(mc["name_or_path"], mc["proj_dim"], mc["freeze_word_embeddings"],
                                 tc["grad_checkpointing"]).to(device)
        model.train()
        opt = torch.optim.AdamW(model.param_groups(tc["lr_backbone"], tc["lr_proj"], tc["weight_decay"]))
        base_lrs = [g["lr"] for g in opt.param_groups]
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mon.note("params", {"total": sum(p.numel() for p in model.parameters()), "trainable": n_trainable})
        print(f"groups train {len(train_idx):,} val {len(val_idx):,}; trainable params {n_trainable:,}")

    ckpt = out_dir / "ckpt_last.pt"
    # frozen weights (the 96M-parameter word-embedding table) never change: keep them out of checkpoints,
    # which saves ~0.4 GB of disk and of host RAM while saving
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    st = {"epoch": 0, "consumed": 0, "step": 0, "B": tc["groups_per_batch"], "best": -1.0, "history": []}
    if ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(blob["model"], strict=False)
        if unexpected or set(missing) - frozen:
            raise SystemExit(f"{ckpt} does not match the model (missing {missing}, unexpected {unexpected})")
        opt.load_state_dict(blob["opt"])
        st = blob["state"]
        del blob
        print(f"resumed at epoch {st['epoch']} group {st['consumed']:,} step {st['step']}")

    def save_ckpt():
        tmp = ckpt.with_suffix(".tmp")
        state = {k: v for k, v in model.state_dict().items() if k not in frozen}
        torch.save({"model": state, "opt": opt.state_dict(), "state": st}, tmp)
        os.replace(tmp, ckpt)

    total = tc["epochs"] * len(train_idx)
    warm = max(1, int(tc["warmup_frac"] * total))
    tau = tc["temperature"]
    w = (tc["loss_w_comb"], tc["loss_w_name"], tc["loss_w_addr"])
    t_log = time.time()
    with mon.stage("train"):
        while st["epoch"] < tc["epochs"]:
            perm = np.random.default_rng(tc["seed"] + st["epoch"]).permutation(train_idx)
            rng = np.random.default_rng(tc["seed"] * 1000 + st["step"])
            while st["consumed"] < len(perm):
                gidx = perm[st["consumed"]: st["consumed"] + st["B"]]
                done = st["epoch"] * len(train_idx) + st["consumed"]
                frac = done / warm if done < warm else max(0.0, (total - done) / max(1, total - warm))
                for g, lr in zip(opt.param_groups, base_lrs):
                    g["lr"] = lr * frac
                try:
                    a, p, hn, cand, valid, cs = build_batch(D, gidx, tc["hard_negs"], rng)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=tc["bf16"] and device.type == "cuda"):
                        en_a, en_c, ea_a, ea_c = embed_fields(model, tok, D, a, cs, mc, device)
                    (lc, ln, la), (ac, an, aa) = contrastive_losses(D, a, p, cand, valid, cs,
                                                                    en_a, en_c, ea_a, ea_c, tau, device)
                    loss = w[0] * lc + w[1] * ln + w[2] * la
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad],
                                                   tc["grad_clip"])
                    opt.step()
                except torch.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if st["B"] <= tc["min_groups_per_batch"]:
                        raise
                    st["B"] = max(tc["min_groups_per_batch"], st["B"] // 2)
                    print(f"CUDA OOM -> groups_per_batch={st['B']}", flush=True)
                    continue
                st["consumed"] += len(gidx)
                st["step"] += 1
                if st["step"] % tc["log_every"] == 0:
                    dt = time.time() - t_log
                    t_log = time.time()
                    rec = {"step": st["step"], "epoch": st["epoch"], "groups": st["consumed"],
                           "loss": round(loss.item(), 4), "l_comb": round(lc.item(), 4),
                           "l_name": round(ln.item(), 4), "l_addr": round(la.item(), 4),
                           "acc_comb": round(ac.item(), 3), "acc_name": round(an.item(), 3),
                           "acc_addr": round(aa.item(), 3), "B": st["B"],
                           "groups_per_s": round(tc["log_every"] * st["B"] / max(dt, 1e-6), 1),
                           "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                           if device.type == "cuda" else 0}
                    print(rec, flush=True)
                if st["step"] % tc["eval_every"] == 0:
                    ev = evaluate(model, tok, D, val_idx, cfg, device, tc["val_batches"])
                    ev["step"] = st["step"]
                    st["history"].append(ev)
                    print(ev, flush=True)
                    if ev["val_recall@1_comb"] > st["best"]:
                        st["best"] = ev["val_recall@1_comb"]
                        model.save(out_dir / "best", tok, {**mc, "step": st["step"], **ev})
                if st["step"] % tc["ckpt_every"] == 0:
                    save_ckpt()
                if args.max_steps and st["step"] >= args.max_steps:
                    save_ckpt()
                    print(f"stopped at step {st['step']} (--max-steps); rerun to resume", flush=True)
                    mon.close()
                    return
            st["epoch"] += 1
            st["consumed"] = 0
            save_ckpt()

    with mon.stage("final_eval"):
        # same validation subset as the periodic evals, so the best-checkpoint choice compares like with like
        ev = evaluate(model, tok, D, val_idx, cfg, device, tc["val_batches"])
        ev["step"] = st["step"]
        st["history"].append(ev)
        print(ev, flush=True)
        if ev["val_recall@1_comb"] >= st["best"]:
            st["best"] = ev["val_recall@1_comb"]
            model.save(out_dir / "best", tok, {**mc, "step": st["step"], **ev})
        os.replace(out_dir / "best", final_dir)
        ckpt.unlink(missing_ok=True)  # final model kept; optimiser checkpoint no longer needed
        atomic_write_json(out_dir / "train_history.json", st["history"])
        mon.note("final_val", ev)
    mon.close()


if __name__ == "__main__":
    main()
