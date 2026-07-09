"""Turn a query into a total-order variant so a non-deterministic LIMIT answer becomes deterministic.

Ties/unordered LIMITs are the reason DuckDB's exact hash false-fails correct WaveDB answers. Appending
the non-aggregate output columns to ORDER BY forces exactly one legal answer; if WaveDB and DuckDB then
agree, WaveDB's original answer was a valid one. The group/output keys uniquely identify each row, so
they are a sufficient tiebreak (no need to also order by the aggregates)."""
import sqlglot
from sqlglot import expressions as E


def total_order_sql(q):
    """Return q with every non-aggregate projection appended to ORDER BY, or None if not applicable
    (parse failure, SELECT *, or an all-aggregate projection with nothing to order by)."""
    try:
        t = sqlglot.parse_one(q)
    except Exception:
        return None
    if not isinstance(t, E.Select):
        return None
    projs = t.expressions
    if any(isinstance(p, E.Star) for p in projs):   # bare `SELECT *`: tiebreak on the full hits schema
        HITS = ('AdvEngineID,ClientIP,CounterID,DontCountHits,EventDate,EventTime,IsDownload,IsLink,'
                'IsRefresh,MobilePhone,MobilePhoneModel,Referer,RefererHash,RegionID,ResolutionWidth,'
                'SearchEngineID,SearchPhrase,Title,TraficSourceID,URL,URLHash,UserID,WatchID,'
                'WindowClientHeight,WindowClientWidth').split(',')
        keys = [E.Ordered(this=E.column(c)) for c in HITS]
        t.set('expressions', [E.column(c) for c in HITS])   # explicit columns: identical order in both engines
        order = t.args.get('order')
        if order is None:
            t.set('order', E.Order(expressions=keys))
        else:
            order.set('expressions', list(order.expressions) + keys)
        return t.sql()
    keys = []
    for p in projs:
        e = p.this if isinstance(p, E.Alias) else p
        if e.find(E.AggFunc):                     # skip aggregates; keys alone give a total order
            continue
        keys.append(E.Ordered(this=e.copy()))
    if not keys:
        return None
    order = t.args.get('order')
    if order is None:
        t.set('order', E.Order(expressions=keys))
    else:
        order.set('expressions', list(order.expressions) + keys)
    return t.sql()
