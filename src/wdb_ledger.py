"""wdb_ledger -- the routing ledger (the future router's memory).

Every routed query appends one JSON line: when, what shape (sql hash +
prefix), which path served it, how long it took, and any per-stage bills
a lane volunteers. Disk-resident JSONL beside the catalog; no RAM state;
append-only; failures never disturb the query. The training set for the
learned router accumulates as a side effect of simply running.
"""
import json
import os
import time

_ENV_OFF = 'WDB_LEDGER_OFF'

_STAGES = [{}]                                   # the current query's itemized bill


def reset_stages():
    _STAGES[0] = {}


def stage(name, ms):
    """A lane volunteers where its milliseconds went. Never raises."""
    try:
        _STAGES[0][name] = _STAGES[0].get(name, 0.0) + float(ms)
    except Exception:
        pass



def _path_for(seg):
    try:
        return os.path.join(os.path.dirname(seg.path), 'routing_ledger.jsonl')
    except Exception:
        return None


def log(seg, sql, path_name, ms, stages=None, n_rows=None):
    """One routed query, one line. Never raises."""
    if os.environ.get(_ENV_OFF):
        return
    try:
        p = _path_for(seg)
        if p is None:
            return
        import wdb_sidecar
        if not wdb_sidecar.births_on(os.path.dirname(p)):
            return                               # THE SWITCH: a vanilla realm writes nothing, not even its diary
        rec = {
            't': round(time.time(), 3),
            'q': hash(sql) & 0xFFFFFFFFFFFF,
            'sql': sql[:160],
            'path': path_name,
            'ms': round(float(ms), 3),
        }
        if n_rows is not None:
            rec['rows'] = int(n_rows)
        stages = stages or (_STAGES[0] or None)
        if stages:
            rec['stages'] = {k: round(float(v), 3) for k, v in stages.items()}
        # THE OPEN HANDLE: open/close per line was 2 ms of a 6 ms query on the network volume; the
        # handle stays open per ledger path, line-buffered, so a line costs a write
        f = _OPEN.get(p)
        if f is None or f.closed:
            f = _OPEN[p] = open(p, 'a', buffering=1)
        f.write(json.dumps(rec) + '\n')
    except Exception:
        pass                                     # the ledger never hurts the query


_OPEN = {}
