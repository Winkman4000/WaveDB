"""
wdb_colresult -- the columnar treaty's carrier.

Above the gate, fast paths hand back columns instead of transcribing them into
~200ns-a-piece python tuples (28M objects for a 9M-row answer; the paperwork was
costing as much as the war). ColRows is list-compatible: len() is free, iteration
or indexing materializes real tuples once, lazily -- so every existing consumer
(suite, users, S-kind validators) sees exactly the rows it always saw, while
count-only consumers (the board's N kind) never pay transcription at all.

Columns are either ('arr', numpy_or_list) -- materialized by tolist -- or
('lazy', codes_array, decode_fn) -- decoded only if someone actually reads rows.
"""
import numpy as np

COL_GATE = 1_000_000     # outputs above this ride columns; below, tuples as ever


class ColRows:
    __slots__ = ('_cols', '_n', '_mat')

    def __init__(self, cols, n):
        self._cols = cols
        self._n = int(n)
        self._mat = None

    def _materialize(self):
        if self._mat is None:
            outs = []
            for c in self._cols:
                if c[0] == 'lazy':
                    outs.append(c[2](c[1]))
                else:
                    v = c[1]
                    outs.append(v.tolist() if isinstance(v, np.ndarray) else v)
            self._mat = list(zip(*outs))
            self._cols = None
        return self._mat

    def __len__(self):
        return self._n

    def __iter__(self):
        return iter(self._materialize())

    def __getitem__(self, i):
        return self._materialize()[i]

    def __repr__(self):
        return '<ColRows n=%d %s>' % (self._n,
                                      'materialized' if self._mat is not None else 'columnar')
