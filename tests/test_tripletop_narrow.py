"""tripletop's two-key walk with a NARROW person (2026-10-15): a yes/no or tiny column's codes are uint8; the
packed (person << 23 | phrase) key must widen before the shift, or every person's phrase rows merge into one
(found by the Q16 composition tests: bin x word answered (0, 'b', 10053) where the truth is 5057 + 4996).
With so narrow a person the bar is empty, and tripletop now hands the query on (the composed pair count)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import controller
import test_gridwalk_compose as F


def test_narrow_person_against_pandas():
    db, df, _d = F._fixture()
    for a, b in (('bin', 'word'), ('word', 'bin'), ('tiny', 'word'), ('word', 'tiny')):
        for lim in (5, 10):
            q = 'SELECT %s, %s, COUNT(*) FROM t GROUP BY %s, %s ORDER BY COUNT(*) DESC LIMIT %d' % (a, b, a, b, lim)
            got = F._rows(db.run(q))
            want = df.groupby([a, b]).size().sort_values(ascending=False).head(lim)
            assert sorted(int(r[-1]) for r in got) == sorted(want.tolist()), (a, b, lim, controller._SERVED[0], got)
            truth = {(str(k[0]), str(k[1])): int(v) for k, v in df.groupby([a, b]).size().items()}
            for r in got:
                assert truth[(str(r[0]), str(r[1]))] == int(r[2]), (a, b, r)
            # a person with fewer than k empty-phrase groups leaves tripletop's bar at 0: it hands the query on
            person = a if b == 'word' else b
            if df[person].nunique() < lim:
                assert controller._SERVED[0] != 'tripletop', (a, b, lim)
