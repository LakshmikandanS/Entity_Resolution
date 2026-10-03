"""Target exclusivity (problem.md decision D6).

Ground truth: an S2/S3 record belongs to at most one S1 (0 of 7,638,365 links violate this).
Decision rule, applied identically to OOF predictions (tuning) and test predictions:
  1. for each target, collect the S1s that have it as a candidate, with model probability p;
  2. keep only the highest-p S1 (ties -> lowest S1 row index, deterministic);
  3. the threshold tau is applied afterwards, so a target is assigned to at most one S1 and only
     when that S1's probability clears tau.
Implemented as two streaming passes with arrays sized by #targets (no global sort of all pairs).
"""
import numpy as np


class Exclusivity:
    def __init__(self, n_targets):
        self.best_p = np.full(n_targets, -1.0, np.float32)
        self.best_s1 = np.full(n_targets, np.iinfo(np.int32).max, np.int32)

    def pass1(self, tgt, p):
        np.maximum.at(self.best_p, tgt, p.astype(np.float32))

    def pass1b(self, s1, tgt, p):
        tie = p.astype(np.float32) == self.best_p[tgt]
        np.minimum.at(self.best_s1, tgt[tie], s1[tie])

    def winners(self, s1, tgt, p):
        """Boolean mask of rows that win their target."""
        return (p.astype(np.float32) == self.best_p[tgt]) & (s1 == self.best_s1[tgt])

    def check(self, tgt_assigned):
        """Assert no target is assigned twice."""
        if len(tgt_assigned) and np.bincount(tgt_assigned).max() > 1:
            raise AssertionError("a target was assigned to more than one S1")
