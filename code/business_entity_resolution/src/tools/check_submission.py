"""Local format check for output/matching_results.tsv and output/candidate_pairs.tsv.

Mirrors the README rules (the official utils/validate_submission.py is not in this checkout):
  * header names; tab-separated; exactly one row per test S1 id, no extra ids
  * list entries are S2-/S3- ids that exist in the test files, no duplicates within a list
  * every matched id also appears in that S1's candidate list
Ids are held as sorted numpy arrays (not Python sets) to stay within a few hundred MB.
"""
import argparse
import os
import sys

import numpy as np


def read_ids(path, prefix):
    out = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            eid = line.split("\t", 1)[0]
            if not eid.startswith(prefix):
                raise SystemExit(f"{path}: unexpected id {eid!r}")
            out.append(int(eid[3:]))
    a = np.array(out, dtype=np.int64)
    a.sort()
    return a


def contains(sorted_ids, values):
    if len(values) == 0:
        return np.ones(0, bool)
    pos = np.minimum(np.searchsorted(sorted_ids, values), len(sorted_ids) - 1)
    return sorted_ids[pos] == values


def parse_list(field):
    return [x for x in field.split(",") if x] if field else []


def check(path, header2, s1_ids, s2_ids, s3_ids, problems, keep_lists=False):
    seen = np.zeros(len(s1_ids), bool)
    lists = {} if keep_lists else None
    rows = links = 0
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["source1_entity_id", header2]:
            problems.append(f"{os.path.basename(path)}: header {header}")
        for ln, line in enumerate(f, 2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                problems.append(f"{os.path.basename(path)}:{ln}: expected 2 tab-separated columns")
                continue
            s1, lst = parts
            if not s1.startswith("S1-"):
                problems.append(f"{os.path.basename(path)}:{ln}: bad S1 id {s1!r}")
                continue
            k = np.searchsorted(s1_ids, int(s1[3:]))
            if k >= len(s1_ids) or s1_ids[k] != int(s1[3:]):
                problems.append(f"{os.path.basename(path)}:{ln}: {s1} is not a test S1 id")
                continue
            if seen[k]:
                problems.append(f"{os.path.basename(path)}:{ln}: duplicate row for {s1}")
            seen[k] = True
            ids = parse_list(lst)
            if len(set(ids)) != len(ids):
                problems.append(f"{os.path.basename(path)}:{ln}: duplicate ids in list for {s1}")
            s2 = np.array([int(x[3:]) for x in ids if x.startswith("S2-")], np.int64)
            s3 = np.array([int(x[3:]) for x in ids if x.startswith("S3-")], np.int64)
            if len(s2) + len(s3) != len(ids):
                problems.append(f"{os.path.basename(path)}:{ln}: non S2/S3 id in list for {s1}")
            if not contains(s2_ids, s2).all() or not contains(s3_ids, s3).all():
                problems.append(f"{os.path.basename(path)}:{ln}: unknown S2/S3 id in list for {s1}")
            if keep_lists and ids:
                lists[s1] = set(ids)
            rows += 1
            links += len(ids)
            if len(problems) > 50:
                break
    missing = int((~seen).sum())
    if missing:
        problems.append(f"{os.path.basename(path)}: {missing} test S1 ids have no row")
    return rows, links, lists


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()
    s1 = read_ids(os.path.join(args.test_dir, "test_source1.tsv"), "S1-")
    s2 = read_ids(os.path.join(args.test_dir, "test_source2.tsv"), "S2-")
    s3 = read_ids(os.path.join(args.test_dir, "test_source3.tsv"), "S3-")
    problems = []
    m_rows, m_links, m_lists = check(args.matching, "matched_entity_ids", s1, s2, s3, problems, keep_lists=True)
    c_rows, c_links, _ = check(args.candidate, "candidate_entity_ids", s1, s2, s3, problems)
    # subset check: stream the candidate file once more, only for S1s that have matches
    not_subset = 0
    with open(args.candidate, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1_id, lst = line.rstrip("\n").split("\t")
            if s1_id in m_lists:
                if not m_lists[s1_id] <= set(parse_list(lst)):
                    not_subset += 1
    if not_subset:
        problems.append(f"{not_subset} S1 rows have matched ids that are not in their candidate list")
    print(f"matching_results.tsv: {m_rows:,} rows, {m_links:,} links; candidate_pairs.tsv: {c_rows:,} rows, "
          f"{c_links:,} candidates; test S1 {len(s1):,}")
    if problems:
        print("FAIL")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
