"""wdb_gridwalk_maint -- append-tail maintenance for the gridwalk structure.

gridwalk (#1) builds the heavy filled-cell bulk once from a full group-by and reads it. This makes
inserts cheap: instead of re-running the 100M-row group-by, an insert touches only the affected cell.

Three pieces of launch state make incremental inserts EXACT (identical to a full rebuild):
  - base  : the heavy cells (count>=2) -- gid (sorted) + count. The bulk from #1.
  - ones  : the count==1 gids at launch. Needed so a singleton crossing 1->2 is counted exactly;
            without it an ex-singleton would undercount by its launch occurrence forever.
  - tail  : a small mutable map gid->count for pairs that are new or promoted since launch.

An insert of pair P (gid = codeA*Vb + codeB):
  - P in base   -> bump its count in place (base is the reserved-slot store for count growth).
  - P in tail   -> bump the tail count.
  - P in ones   -> promote: it was 1 at launch, now 1 + inserted.
  - otherwise   -> brand new: tail count = inserted.

topk merges base (bumped) with the heavy (count>=2) tail entries. recompact folds the tail back into a
fresh sorted base + ones (preserving untouched launch singletons) and clears the tail -- the
recompact-on-restart step; the tail degrades slowly between recompactions (most inserts hit existing
heavy cells, which bump in place rather than growing the tail).

This is the in-memory maintenance machinery, validated exactly against a full rebuild on 100M-row
cb25db. Compact at-rest persistence of ones (gid-only gap codec) and the tail, and anchor-based
point-lookup that avoids materializing the base flat, are the next increments.
"""
import numpy as np


class GwMaint:
    """Maintains a gridwalk heavy-cell structure under inserts. Working state is flat (base_gid sorted,
    base_cnt mutable int64, ones_gid sorted) plus a tail dict; counts stay exact vs a full rebuild."""

    def __init__(self, base_gid, base_cnt, ones_gid):
        self.bg = np.ascontiguousarray(base_gid, dtype=np.int64)        # heavy gids, sorted ascending
        self.bc = np.ascontiguousarray(base_cnt, dtype=np.int64)        # heavy counts, mutable (bumped)
        self.og = np.ascontiguousarray(ones_gid, dtype=np.int64)        # launch count==1 gids, sorted
        self.tail = {}                                                  # gid -> current count (new/promoted)

    # ---- construction ----
    @classmethod
    def from_codes(cls, codesA, codesB, Vb):
        """Build launch state from two code columns (the same group-by gridwalk does, but keeping the
        count==1 cells as `ones` rather than discarding them)."""
        ca = np.asarray(codesA, dtype=np.int64); cb = np.asarray(codesB, dtype=np.int64)
        gid, cnt = np.unique(ca * Vb + cb, return_counts=True)
        heavy = cnt >= 2
        return cls(gid[heavy], cnt[heavy], gid[cnt == 1]), int(Vb)

    @classmethod
    def from_segment(cls, seg, cols):
        a, b = sorted(cols)
        ca = seg._raw_codes(a).astype(np.int64); cb = seg._raw_codes(b).astype(np.int64)
        Vb = int(cb.max()) + 1
        m, _ = cls.from_codes(ca, cb, Vb)
        return m, Vb

    # ---- mutation ----
    def insert(self, gids):
        """Apply a batch of inserted grid-ids (one entry per occurrence; multiplicity is fine)."""
        gids = np.asarray(gids, dtype=np.int64)
        if gids.size == 0:
            return
        ug, uc = np.unique(gids, return_counts=True)
        if self.bg.size:
            pos = np.clip(np.searchsorted(self.bg, ug), 0, self.bg.size - 1)
            in_base = self.bg[pos] == ug
        else:
            pos = np.zeros(ug.size, dtype=np.intp)
            in_base = np.zeros(ug.size, dtype=bool)
        np.add.at(self.bc, pos[in_base], uc[in_base])          # bump existing heavy cells in place
        ng = ug[~in_base]; nc = uc[~in_base]
        if ng.size:
            if self.og.size:
                p1 = np.clip(np.searchsorted(self.og, ng), 0, self.og.size - 1)
                was_one = self.og[p1] == ng
            else:
                was_one = np.zeros(ng.size, dtype=bool)
            tail = self.tail
            for g, c, w in zip(ng.tolist(), nc.tolist(), was_one.tolist()):
                if g in tail:
                    tail[g] += c
                else:
                    tail[g] = c + (1 if w else 0)              # +1 if it was a launch singleton

    def insert_codes(self, codesA, codesB, Vb):
        ca = np.asarray(codesA, dtype=np.int64); cb = np.asarray(codesB, dtype=np.int64)
        self.insert(ca * Vb + cb)

    # ---- read ----
    def _merged(self):
        if not self.tail:
            return self.bg, self.bc
        tg = np.fromiter(self.tail.keys(), dtype=np.int64, count=len(self.tail))
        tc = np.fromiter(self.tail.values(), dtype=np.int64, count=len(self.tail))
        h = tc >= 2
        return np.concatenate([self.bg, tg[h]]), np.concatenate([self.bc, tc[h]])

    def topk(self, lim):
        """Top-lim cells by current count (base bumped + heavy tail). Returns (gids, counts), count
        descending. Exact vs a full rebuild on the same data."""
        allg, allc = self._merged()
        if lim <= 0 or allg.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        k = min(lim, allg.size)
        idx = np.argpartition(allc, -k)[-k:]
        idx = idx[np.argsort(allc[idx], kind='stable')[::-1]]
        return allg[idx[:lim]], allc[idx[:lim]]

    # ---- maintenance ----
    def recompact(self):
        """Fold the tail back into a fresh sorted base + ones and clear the tail. Untouched launch
        singletons are preserved; promoted ones move to the tail-derived cells. Restores the tight
        launch-style state so the tail (and its read overhead) resets to empty."""
        if self.tail:
            tg = np.fromiter(self.tail.keys(), dtype=np.int64, count=len(self.tail))
            tc = np.fromiter(self.tail.values(), dtype=np.int64, count=len(self.tail))
        else:
            tg = np.empty(0, np.int64); tc = np.empty(0, np.int64)
        if tg.size and self.og.size:
            p = np.clip(np.searchsorted(self.og, tg), 0, self.og.size - 1)
            promoted = self.og[p] == tg
            keep = np.ones(self.og.size, bool); keep[p[promoted]] = False
            untouched_ones = self.og[keep]
        else:
            untouched_ones = self.og
        allg = np.concatenate([self.bg, tg, untouched_ones])
        allc = np.concatenate([self.bc, tc, np.ones(untouched_ones.size, np.int64)])
        order = np.argsort(allg, kind='stable')
        allg = allg[order]; allc = allc[order]
        heavy = allc >= 2
        self.bg = np.ascontiguousarray(allg[heavy]); self.bc = np.ascontiguousarray(allc[heavy])
        self.og = np.ascontiguousarray(allg[allc == 1]); self.tail = {}

    # ---- introspection ----
    def tail_size(self):
        return len(self.tail)

    def heavy_size(self):
        return int(self.bg.size + sum(1 for c in self.tail.values() if c >= 2))
