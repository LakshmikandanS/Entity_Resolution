"""Run the whole neural / hybrid experiment end to end: encoder data -> encoder -> embeddings ->
pair similarities -> A/B report -> test predictions.

Each stage runs in its own Python process (RAM and VRAM go back to the OS between stages). Output is
shown live and written to a log file. The run stops at the first failing stage; every stage resumes or
skips finished work, so rerunning the same command continues where it stopped.

Before any neural stage, the baseline stages it reads are checked. With --with-baseline the missing
baseline stages are run first through code/business_entity_resolution/run_pipeline.py (unchanged).

Examples (from the repository root)
  python experiments/neural/run_neural.py --with-baseline     # baseline prerequisites + n01 .. n06
  python experiments/neural/run_neural.py                     # n01 .. n06 (baseline already done)
  python experiments/neural/run_neural.py --dry-run           # print the commands only
  python experiments/neural/run_neural.py --train-only        # up to the A/B report (n05)
  python experiments/neural/run_neural.py --test-only         # test embeddings, similarities, n06
  python experiments/neural/run_neural.py --from n03-train --to n05
  python experiments/neural/run_neural.py --stage-args n02="--max-steps 2000"
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASELINE_RUNNER = os.path.join(ROOT, "code", "business_entity_resolution", "run_pipeline.py")

# (stage id, script, split or None)
STAGES = [
    ("n01", "n01_build_encoder_data.py", None),
    ("n02", "n02_train_encoder.py", None),
    ("n03-train", "n03_embed.py", "train"),
    ("n04-train", "n04_pair_similarity.py", "train"),
    ("n05", "n05_train_compare.py", None),
    ("n03-test", "n03_embed.py", "test"),
    ("n04-test", "n04_pair_similarity.py", "test"),
    ("n06", "n06_predict_test.py", None),
]
IDS = [s[0] for s in STAGES]
TRAIN_IDS = IDS[:IDS.index("n05") + 1]
TEST_IDS = IDS[IDS.index("n03-test"):]
# baseline stages each neural stage reads (run_pipeline.py stage ids)
BASELINE_NEEDS = {
    "n01": ["01-train", "02", "03-train", "04-train", "05-train"],
    "n04-train": ["01-train", "03-train", "04-train", "05-train"],
    "n05": ["01-train", "02", "03-train", "04-train", "05-train", "06"],
    "n04-test": ["01-test", "03-test", "04-test", "05-test"],
    "n06": ["01-test", "03-test", "04-test", "05-test"],
}


def parse_stage_args(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            sys.exit(f"--stage-args expects STAGE=\"ARGS\", got {item!r}")
        sid, args = item.split("=", 1)
        if sid not in IDS:
            sys.exit(f"unknown stage {sid!r}; valid: {', '.join(IDS)}")
        out.setdefault(sid, []).extend(shlex.split(args, posix=(os.name != "nt")))
    return out


def select(args):
    if args.only:
        bad = [s for s in args.only if s not in IDS]
        if bad:
            sys.exit(f"unknown stage(s) {bad}; valid: {', '.join(IDS)}")
        return [s for s in IDS if s in args.only]
    ids = TRAIN_IDS if args.train_only else TEST_IDS if args.test_only else IDS
    lo = IDS.index(args.start) if args.start else 0
    hi = IDS.index(args.end) if args.end else len(IDS) - 1
    return [s for s in ids if lo <= IDS.index(s) <= hi]


def missing_baseline(cfg_path, needed):
    """Baseline stage ids (run_pipeline.py naming) whose _MANIFEST.json is missing."""
    sys.path.insert(0, HERE)
    from common import load_config
    import baseline_api
    api = baseline_api.load(load_config(cfg_path))
    out = []
    for st in needed:
        try:
            api.require(st)
        except SystemExit:
            out.append(st)
    return out


def run(cmd, log):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
    proc.wait()
    log.flush()
    return proc.returncode


def main():
    for stream in (sys.stdout, sys.stderr):   # Indic names / progress bars must not crash a cp1252 console
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--train-only", action="store_true", help="n01 .. n05 (ends with the A/B report)")
    g.add_argument("--test-only", action="store_true", help="n03-test, n04-test, n06 (needs n02 and n05)")
    g.add_argument("--only", nargs="+", metavar="STAGE", help="run exactly these stages")
    ap.add_argument("--from", dest="start", choices=IDS, help="first stage to run")
    ap.add_argument("--to", dest="end", choices=IDS, help="last stage to run")
    ap.add_argument("--with-baseline", action="store_true",
                    help="first run any missing baseline stages via code/business_entity_resolution/run_pipeline.py")
    ap.add_argument("--arm", choices=["A", "B"], default="B", help="n06: which arm's models predict test")
    ap.add_argument("--force-n01", action="store_true", help="rebuild the encoder data")
    ap.add_argument("--stage-args", action="append", metavar='STAGE="ARGS"',
                    help='extra arguments for one stage, e.g. n02="--max-steps 2000"; repeatable')
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    extra = parse_stage_args(args.stage_args)
    selected = select(args)
    if not selected:
        sys.exit("no stages selected")

    commands = []
    for sid, script, split in STAGES:
        if sid not in selected:
            continue
        cmd = [sys.executable, "-u", os.path.join(HERE, script), "--config", args.config]
        if split:
            cmd += ["--split", split]
        if sid == "n01" and args.force_n01:
            cmd.append("--force")
        if sid == "n06":
            cmd += ["--arm", args.arm]
        commands.append((sid, cmd + extra.get(sid, [])))

    needed = sorted({b for s in selected for b in BASELINE_NEEDS.get(s, [])},
                    key=lambda b: (b.endswith("-test"), b))
    missing = missing_baseline(args.config, needed) if needed else []
    if missing:
        if not args.with_baseline and args.dry_run:
            print(f"NOTE: baseline stages not finished yet: {', '.join(missing)} (add --with-baseline)\n")
        elif not args.with_baseline:
            sys.exit(f"\nBaseline stages not finished yet: {', '.join(missing)}\n"
                     f"Rerun with --with-baseline to run them first, or run:\n"
                     f"    python {os.path.relpath(BASELINE_RUNNER, ROOT)} --only {' '.join(missing)}\n")
        else:
            commands.insert(0, ("baseline", [sys.executable, "-u", BASELINE_RUNNER, "--skip-resource-check",
                                         "--only", *missing]))

    if args.dry_run:
        for sid, cmd in commands:
            print(f"[{sid}] " + " ".join(shlex.quote(c) for c in cmd))
        return

    sys.path.insert(0, HERE)
    from common import load_config
    work = load_config(args.config).work
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = args.log_file or os.path.join(work, "logs", f"neural_{stamp}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    summary = {"started": stamp, "config": args.config, "stages": [], "log": log_path}
    t_all = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        head = f"Neural pipeline: {' -> '.join(s for s, _ in commands)}\nLog: {log_path}\n"
        print(head)
        log.write(head)
        for sid, cmd in commands:
            banner = f"\n{'=' * 78}\n[{sid}] {os.path.basename(cmd[2])}  ({time.strftime('%H:%M:%S')})\n{'=' * 78}\n"
            sys.stdout.write(banner)
            log.write(banner + " ".join(cmd) + "\n")
            t0 = time.time()
            rc = run(cmd, log)
            dt = time.time() - t0
            summary["stages"].append({"stage": sid, "returncode": rc, "seconds": round(dt, 1)})
            if rc != 0:
                msg = (f"\nStage {sid} FAILED (exit code {rc}) after {dt / 60:.1f} min.\n"
                       f"Fix the cause, then rerun the same command: finished work is skipped.\n")
                sys.stdout.write(msg)
                log.write(msg)
                break
            done = f"[{sid}] done in {dt / 60:.1f} min\n"
            sys.stdout.write(done)
            log.write(done)
            log.flush()
    summary["seconds_total"] = round(time.time() - t_all, 1)
    summary["ok"] = len(summary["stages"]) == len(commands) and all(s["returncode"] == 0 for s in summary["stages"])
    os.makedirs(os.path.join(work, "logs"), exist_ok=True)
    with open(os.path.join(work, "logs", f"run_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    lines = ["", "Stage times:"]
    lines += [f"  {s['stage']:10s} {'ok' if s['returncode'] == 0 else 'FAILED':7s} {s['seconds'] / 60:7.1f} min"
              for s in summary["stages"]]
    lines.append(f"Total {summary['seconds_total'] / 60:.1f} min. "
                 f"{'All stages succeeded.' if summary['ok'] else 'Stopped early.'}")
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    with open(log_path, "a", encoding="utf-8") as log:    # the summary also lands in the log file
        log.write(text)
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
