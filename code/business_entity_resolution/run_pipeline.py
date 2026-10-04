"""Run the whole pipeline end to end: train stages, test stages, submission files.

Each stage runs in its own Python process, so all of its RAM and VRAM is returned to the OS before
the next stage starts (important on a 16 GB / 8 GB laptop). Output is shown live and also written
to a log file. The run stops at the first failing stage. Because every stage skips work it has
already finished, rerunning the same command resumes where it stopped.

Examples
  python run_pipeline.py                         # everything: 00 -> 11
  python run_pipeline.py --dry-run               # print the commands only
  python run_pipeline.py --train-only            # 00 .. 10
  python run_pipeline.py --from 05-train --to 09 # a slice
  python run_pipeline.py --only 04-train --stage-args 04-train="--k 40 --min-recall 0.98"
  python run_pipeline.py --force-stages 07 08 09 # rebuild some stages, keep the rest
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.join(HERE, "src", "training")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# (stage id, script, split or None, takes --split)
STAGES = [
    ("00", "00_resource_check.py", None),
    ("01-train", "01_normalize.py", "train"),
    ("02", "02_build_rewrite_map.py", None),
    ("03-train", "03_build_blocking_indexes.py", "train"),
    ("04-train", "04_generate_candidates.py", "train"),
    ("05-train", "05_build_features.py", "train"),
    ("06", "06_build_training_set.py", None),
    ("07", "07_train.py", None),
    ("08", "08_generate_oof.py", None),
    ("09", "09_tune_threshold.py", None),
    ("10", "10_train_final.py", None),
    ("01-test", "01_normalize.py", "test"),
    ("03-test", "03_build_blocking_indexes.py", "test"),
    ("04-test", "04_generate_candidates.py", "test"),
    ("05-test", "05_build_features.py", "test"),
    ("11", "11_predict_test.py", None),
]
IDS = [s[0] for s in STAGES]
TRAIN_IDS = IDS[:IDS.index("10") + 1]
TEST_IDS = IDS[IDS.index("01-test"):]
NO_FORCE = {"09", "10", "11"}   # these always recompute; --force is not a flag they need


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
    ids = TRAIN_IDS if args.train_only else ["00"] + TEST_IDS if args.test_only else IDS
    if args.skip_resource_check:
        ids = [s for s in ids if s != "00"]
    lo = IDS.index(args.start) if args.start else 0
    hi = IDS.index(args.end) if args.end else len(IDS) - 1
    return [s for s in ids if lo <= IDS.index(s) <= hi]


def build_command(stage, args, extra):
    sid, script, split = stage
    cmd = [sys.executable, "-u", os.path.join(TRAINING, script),
           "--data-dir", args.data_dir, "--work-dir", args.work_dir,
           "--artifacts-dir", args.artifacts_dir, "--n-threads", str(args.n_threads)]
    if split:
        cmd += ["--split", split]
    if sid == "00" and args.probe_xgboost:
        cmd.append("--probe-xgboost")
    if sid == "07" and args.backend != "auto":
        cmd += ["--backend", args.backend]
    if sid == "10" and args.refit_full:
        cmd.append("--refit-full")
    if sid == "11" and args.model_set != "folds":
        cmd += ["--model-set", args.model_set]
    if sid in args.force_stages or (args.force_all and sid not in NO_FORCE):
        cmd.append("--force")
    cmd += extra.get(sid, [])
    return cmd


def run(cmd, log):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:   # flush every line so the console and the log file are always current
        sys.stdout.write(line)
        sys.stdout.flush()
        log.write(line)
        log.flush()
    proc.wait()
    log.flush()
    return proc.returncode


def main():
    # Stage logs contain Indic and accented text; never let a narrow console encoding kill the run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--work-dir", default=os.path.join(ROOT, "work"))
    ap.add_argument("--artifacts-dir", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--n-threads", type=int, default=int(os.environ.get("N_THREADS", 4)))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--train-only", action="store_true", help="stages 00 .. 10")
    g.add_argument("--test-only", action="store_true", help="stage 00 + test stages + 11 (needs a trained bundle)")
    g.add_argument("--only", nargs="+", metavar="STAGE", help="run exactly these stages")
    ap.add_argument("--from", dest="start", choices=IDS, help="first stage to run")
    ap.add_argument("--to", dest="end", choices=IDS, help="last stage to run")
    ap.add_argument("--skip-resource-check", action="store_true")
    ap.add_argument("--probe-xgboost", action="store_true", help="stage 00: test-train a tiny model on CUDA")
    ap.add_argument("--backend", choices=["auto", "xgboost", "hgb"], default="auto", help="stage 07")
    ap.add_argument("--refit-full", action="store_true", help="stage 10: also train one model on all data")
    ap.add_argument("--model-set", choices=["folds", "full"], default="folds", help="stage 11")
    ap.add_argument("--force-stages", nargs="+", default=[], metavar="STAGE",
                    help="pass --force to these stages (rebuild them)")
    ap.add_argument("--force-all", action="store_true", help="rebuild every selected stage")
    ap.add_argument("--stage-args", action="append", metavar='STAGE="ARGS"',
                    help='extra arguments for one stage, e.g. 04-train="--k 40"; repeatable')
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()
    bad = [s for s in args.force_stages if s not in IDS]
    if bad:
        sys.exit(f"unknown stage(s) in --force-stages: {bad}")
    extra = parse_stage_args(args.stage_args)
    selected = select(args)
    if not selected:
        sys.exit("no stages selected")
    stages = [s for s in STAGES if s[0] in selected]
    commands = [(s[0], build_command(s, args, extra)) for s in stages]

    if args.dry_run:
        for sid, cmd in commands:
            print(f"[{sid}] " + " ".join(shlex.quote(c) for c in cmd))
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = args.log_file or os.path.join(args.work_dir, "logs", f"pipeline_{stamp}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    summary = {"started": stamp, "stages": [], "log": log_path}
    t_all = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        def emit(text, to_log=True):
            """Console and log file get the same runner messages, flushed immediately."""
            sys.stdout.write(text)
            sys.stdout.flush()
            if to_log:
                log.write(text)
                log.flush()

        emit(f"Pipeline: {' -> '.join(selected)}\nLog: {log_path}\n\n")
        for sid, cmd in commands:
            emit(f"\n{'=' * 78}\n[{sid}] {' '.join(cmd[2:3])}  ({time.strftime('%H:%M:%S')})\n{'=' * 78}\n")
            log.write(" ".join(cmd) + "\n")
            t0 = time.time()
            rc = run(cmd, log)
            dt = time.time() - t0
            summary["stages"].append({"stage": sid, "returncode": rc, "seconds": round(dt, 1)})
            if rc != 0:
                emit(f"\nStage {sid} FAILED (exit code {rc}) after {dt / 60:.1f} min.\n"
                     f"Fix the cause, then rerun the same command: finished stages are skipped.\n"
                     f"To restart from this stage only: python run_pipeline.py --from {sid}\n")
                break
            emit(f"[{sid}] done in {dt / 60:.1f} min\n")
        summary["seconds_total"] = round(time.time() - t_all, 1)
        summary["ok"] = all(s["returncode"] == 0 for s in summary["stages"]) and len(summary["stages"]) == len(commands)
        lines = ["", "Stage times:"]
        lines += [f"  {s['stage']:9s} {'ok' if s['returncode'] == 0 else 'FAILED':7s} {s['seconds'] / 60:7.1f} min"
                  for s in summary["stages"]]
        lines.append(f"Total {summary['seconds_total'] / 60:.1f} min. "
                     f"{'PIPELINE COMPLETE: all stages succeeded.' if summary['ok'] else 'PIPELINE STOPPED EARLY.'}"
                     f"  ({time.strftime('%H:%M:%S')})")
        emit("\n".join(lines) + "\n")
    os.makedirs(os.path.join(args.artifacts_dir, "pipeline_runs"), exist_ok=True)
    with open(os.path.join(args.artifacts_dir, "pipeline_runs", f"run_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
