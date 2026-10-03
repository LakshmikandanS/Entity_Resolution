"""End-to-end smoke test on a tiny synthetic dataset (same schema as the competition data):

  1. the REAL baseline stages (train 01-06, test 01/03/04/05) run on the synthetic data, unmodified, into
     <out>/bwork and <out>/bart (they never touch the repo's work/ or artifacts/);
  2. n01..n06 run against them through baseline_api.py, with a random-init 2-layer BERT and a character
     tokenizer built offline instead of multilingual-e5-small.

Takes ~2 minutes; touches neither the real dataset nor the internet. Numbers it prints are meaningless.

    python experiments/neural/tests/smoke_test.py --out <scratch dir>
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
NEURAL = HERE.parent
sys.path.insert(0, str(NEURAL))


def check_group_rank_margin():
    import torch
    from n04_pair_similarity import group_rank_margin

    rng = np.random.default_rng(0)
    g = rng.integers(0, 50, 2000)
    s = rng.random(2000).astype(np.float32)
    r, m = group_rank_margin(torch.from_numpy(g), torch.from_numpy(s))
    for k in np.unique(g):
        idx = np.flatnonzero(g == k)
        order = idx[np.argsort(-s[idx], kind="stable")]
        assert np.array_equal(r.numpy()[order], np.arange(1, idx.size + 1)), k
        assert np.allclose(m.numpy()[idx], s[idx].max() - s[idx]), k
    print("group_rank_margin: OK")


def check_clean():
    from common import clean_address

    assert clean_address("251 EL PAULO CT, NULL, ST. LOUIS, MO") == "251 EL PAULO CT, ST. LOUIS, MO"
    assert clean_address("Trivandrum, Thiruvananthapuram, KL") == "Trivandrum, Thiruvananthapuram, KL"
    assert clean_address("T-2, null, Chennai, TN") == "T-2, Chennai, TN"
    assert clean_address("") == ""
    print("clean_address: OK")


def run(script, cfg_path, *extra):
    cmd = [sys.executable, str(NEURAL / script), "--config", str(cfg_path), *extra]
    print("\n$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)


def run_baseline(src: Path, out: Path):
    common = ["--data-dir", str(out / "dataset"), "--work-dir", str(out / "bwork"),
              "--artifacts-dir", str(out / "bart"), "--force"]  # --force: skip RAM checks on a busy laptop
    steps = [("01_normalize.py", ["--split", "train"]), ("02_build_rewrite_map.py", []),
             ("03_build_blocking_indexes.py", ["--split", "train"]),
             ("04_generate_candidates.py", ["--split", "train", "--min-recall", "0"]),
             ("05_build_features.py", ["--split", "train"]), ("06_build_training_set.py", []),
             ("01_normalize.py", ["--split", "test"]), ("03_build_blocking_indexes.py", ["--split", "test"]),
             ("04_generate_candidates.py", ["--split", "test"]), ("05_build_features.py", ["--split", "test"])]
    for script, extra in steps:
        print(f"\n$ baseline {script} {' '.join(extra)}", flush=True)
        r = subprocess.run([sys.executable, str(src / "training" / script), *common, *extra],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        print("\n".join((r.stdout + r.stderr).strip().splitlines()[-2:]), flush=True)
        r.check_returncode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    check_clean()
    check_group_rank_margin()

    from synthetic import make_dataset, make_tiny_model

    make_dataset(out / "dataset")
    make_tiny_model(out / "tiny_model", out / "dataset")

    cfg = yaml.safe_load((NEURAL / "config.yaml").read_text(encoding="utf-8"))
    repo = (NEURAL / cfg["paths"]["root"]).resolve()
    baseline_src = repo / cfg["baseline"]["src"]
    run_baseline(baseline_src, out)

    cfg["paths"] = {"root": ".", "dataset": "dataset", "work": "work"}
    cfg["baseline"].update({"src": str(baseline_src), "work_dir": "bwork", "artifacts_dir": "bart"})
    cfg["split"]["encoder_frac"] = 0.3
    cfg["split"]["encoder_val_frac"] = 0.2
    cfg["model"]["name_or_path"] = str(out / "tiny_model")
    cfg["model"]["prefix_name"], cfg["model"]["prefix_addr"] = "name: ", "address: "
    cfg["encoder_data"]["batch_rows"] = 500
    # high learning rate so a random-init tiny model visibly learns in a few seconds
    cfg["train"].update({"groups_per_batch": 16, "log_every": 5, "eval_every": 10, "ckpt_every": 10,
                         "val_batches": 5, "epochs": 8, "lr_backbone": 2e-3, "lr_proj": 5e-3})
    cfg["embed"].update({"csv_block_bytes": 8192, "batch_name": 64, "batch_addr": 64, "flush_every_blocks": 1})
    cfg["similarity"].update({"target_block_rows": 150, "pair_chunk": 300, "pass2_chunk": 400})
    cfg["compare"].update({"predict_chunk": 700, "hist_bins": 20, "max_train_rows": 100_000})
    cfg["predict"].update({"chunk": 700, "out_dir": "work/output_test"})
    cfg_path = out / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    run("n01_build_encoder_data.py", cfg_path)
    run("n02_train_encoder.py", cfg_path, "--max-steps", "15")   # interrupted run with a checkpoint ...
    run("n02_train_encoder.py", cfg_path)                         # ... resumed to the end
    run("n03_embed.py", cfg_path, "--split", "train", "--sources", "1")   # partial, then resume the rest
    run("n03_embed.py", cfg_path, "--split", "train")
    run("n04_pair_similarity.py", cfg_path, "--split", "train")
    run("n04_pair_similarity.py", cfg_path, "--split", "train")             # rerun: everything skipped
    run("n05_train_compare.py", cfg_path)
    run("n03_embed.py", cfg_path, "--split", "test")
    run("n04_pair_similarity.py", cfg_path, "--split", "test")
    run("n06_predict_test.py", cfg_path)

    # ---- assertions ----------------------------------------------------------------------------------
    work = out / "work"
    hist = yaml.safe_load((work / "encoder" / "train_history.json").read_text(encoding="utf-8"))
    print("encoder val recall@1 (comb) by eval:", [round(h["val_recall@1_comb"], 3) for h in hist])
    assert max(h["val_recall@1_comb"] for h in hist) > hist[0]["val_recall@1_comb"], "encoder did not learn"
    feats = np.load(work / "feats" / "train_neural.npy")
    assert feats[:, 2].astype(np.float32).std() > 0.01, "cosines collapsed to a constant"
    assert not np.isnan(feats[:, [0, 1, 2, 4, 5, 6, 7]]).any(), "unexpected NaN in neural features"
    assert (feats[:, 4] >= 1).all() and (feats[:, 6] >= 1).all()
    assert np.allclose(feats[:, 2], (feats[:, 0].astype(np.float32) + feats[:, 1]) / 2, atol=2e-3)
    emb = np.load(work / "emb" / "train" / "s2_name.npy")
    assert np.allclose(np.linalg.norm(emb.astype(np.float32), axis=1), 1, atol=1e-2)
    rep = yaml.safe_load((work / "compare" / "ab_report.json").read_text(encoding="utf-8"))
    assert set(rep["arms"]) == {"A", "B"}
    assert rep["arms"]["A"]["features"] + rep["setup"]["neural_features"] == rep["arms"]["B"]["features"]
    for arm in ("A", "B"):
        a = rep["arms"][arm]
        assert abs(a["macro_f05"] - a["macro_f05_check"]) < 1e-9, "per-entity scores disagree with the sweep"
        assert {m["train_rows"] for m in a["models"].values()} == \
               {m["train_rows"] for m in rep["arms"]["A"]["models"].values()}, "arms trained on different rows"
    # the eval split's training rows exclude every encoder-split S1
    import pyarrow.parquet as pq
    from common import encoder_split_mask, load_config
    c = load_config(cfg_path)
    ids = pq.read_table(work / "emb" / "train" / "s1_ids.parquet").column("entity_id").to_pylist()
    enc = encoder_split_mask(ids, c)
    for p in (work / "compare" / "trainset").glob("part-*.parquet"):
        assert not enc[pq.read_table(p, columns=["s1"]).column(0).to_numpy()].any(), "encoder S1 in A/B rows"
    n_test_s1 = sum(1 for _ in open(out / "dataset" / "test" / "test_source1.tsv", encoding="utf-8")) - 1
    res = (work / "output_test" / "arm_B" / "matching_results.tsv").read_text(encoding="utf-8").splitlines()
    assert len(res) == n_test_s1 + 1
    print("\nA/B (synthetic, meaningless numbers):",
          {a: round(v["macro_f05"], 4) for a, v in rep["arms"].items()})
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
