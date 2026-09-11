"""THE WORKING-SET GOVERNOR: transient memory is budgeted like resident memory.
Before an operator materialises something N-scale it asks; under the budget it
proceeds, over it STREAMS (chunks the caller pulls, freed behind), and if it
cannot stream it DECLINES BY NAME. The engine never walks into the OOM killer.

    WDB_WORK_MB   budget for one query's transient working set
                  (default 25% of physical RAM or of the cgroup limit)
    WDB_BLOCK_ROWS rows per streamed block (default 4M)
"""
import os


def _physical_bytes():
    try:
        b = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except Exception:
        b = 16 << 30
    try:
        with open('/sys/fs/cgroup/memory.max') as f:
            v = f.read().strip()
            if v.isdigit(): b = min(b, int(v))
    except Exception:
        pass
    return b


def budget_bytes():
    v = os.environ.get('WDB_WORK_MB')
    if v:
        try: return int(float(v)) << 20
        except Exception: pass
    return _physical_bytes() // 4


def block_rows():
    try:
        return int(float(os.environ.get('WDB_BLOCK_ROWS', '4000000')))
    except Exception:
        return 4_000_000


# measured on CPython 3.11: a row tuple of C cells costs ~56 + 8*C bytes plus the cells
# (an int ~28-32, a float 24, a short string ~50-60); 80 bytes per cell is the honest mean
BYTES_PER_CELL = 80
BYTES_PER_ROW = 64


def rows_bytes(nrows, ncols):
    return int(nrows) * (BYTES_PER_ROW + BYTES_PER_CELL * int(ncols))


class WorkingSetExceeded(NotImplementedError):
    """A named decline: the caller can add LIMIT, ask for columns, or stream."""


def ask(nrows, ncols, what='result'):
    """Raise WorkingSetExceeded if materialising nrows x ncols Python cells would exceed
    the budget. Callers that can stream should call may_stream() instead."""
    need = rows_bytes(nrows, ncols)
    cap = budget_bytes()
    if need > cap:
        raise WorkingSetExceeded('%s of %s rows x %d cols (~%.1f GB as Python rows) exceeds the working-set budget '
                                 '(%.1f GB): add a LIMIT, ask for columns (columnar=True), or stream (db.stream)' % (
                                     what, format(int(nrows), ','), int(ncols), need / 1e9, cap / 1e9))
    return need


def fits(nrows, ncols):
    return rows_bytes(nrows, ncols) <= budget_bytes()


class Peak:
    """Cheap peak-RSS watermark for a query (RUSAGE deltas), for the bill."""
    def __init__(self):
        import resource
        self._r = resource
        self.start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    def gb(self):
        return self._r.getrusage(self._r.RUSAGE_SELF).ru_maxrss / 1e6
