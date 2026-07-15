"""megaboard: the full-surface benchmark -- every family, every distinct shape, one board.

Compare kinds:
  H  full-result hash via total_order_sql (fully deterministic)
  C  single value (COUNT/scalar)
  M  ordered-column multiset + row count (LIMIT-tie tolerant)
  S  full unordered set compare (no LIMIT)
  N  row-count only (ROWS-frame tie order is engine-arbitrary; values legitimately differ)
"""

QUERIES = [
    # ---- scans & simple aggregates (5) ----
    ('scan-count',      'C', "SELECT COUNT(*) FROM hits"),
    ('scan-sum',        'C', "SELECT SUM(AdvEngineID) FROM hits"),
    ('scan-avg',        'C', "SELECT AVG(ResolutionWidth) FROM hits"),
    ('scan-minmax',     'C', "SELECT MIN(EventDate), MAX(EventDate) FROM hits"),
    ('scan-cd',         'C', "SELECT COUNT(DISTINCT SearchPhrase) FROM hits"),
    # ---- filters: every predicate kind (12) ----
    ('f-eq',            'C', "SELECT COUNT(*) FROM hits WHERE CounterID = 62"),
    ('f-neq',           'C', "SELECT COUNT(*) FROM hits WHERE SearchPhrase <> ''"),
    ('f-range',         'C', "SELECT COUNT(*) FROM hits WHERE ResolutionWidth BETWEEN 1000 AND 1400"),
    ('f-and3',          'C', "SELECT COUNT(*) FROM hits WHERE CounterID = 62 AND AdvEngineID = 0 AND RegionID = 229"),
    ('f-or',            'C', "SELECT COUNT(*) FROM hits WHERE CounterID = 62 OR CounterID = 155"),
    ('f-in',            'C', "SELECT COUNT(*) FROM hits WHERE RegionID IN (229, 2, 208, 169)"),
    ('f-notin',         'C', "SELECT COUNT(*) FROM hits WHERE SearchEngineID NOT IN (0, 2)"),
    ('f-like-contains', 'C', "SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%'"),
    ('f-like-prefix',   'C', "SELECT COUNT(*) FROM hits WHERE URL LIKE 'http://holodilnik%'"),
    ('f-like-multi',    'C', "SELECT COUNT(*) FROM hits WHERE SearchPhrase LIKE '%карт%мир%'"),
    ('f-like-under',    'C', "SELECT COUNT(*) FROM hits WHERE Title LIKE '_оскв%'"),
    ('f-scalarpred',    'C', "SELECT COUNT(*) FROM hits WHERE LENGTH(URL) > 100"),
    # ---- group by: key shapes (10) ----
    ('g-1key',          'M', "SELECT RegionID, COUNT(*) AS c FROM hits GROUP BY RegionID ORDER BY c DESC LIMIT 10"),
    ('g-2key',          'M', "SELECT RegionID, SearchEngineID, COUNT(*) AS c FROM hits GROUP BY RegionID, SearchEngineID ORDER BY c DESC LIMIT 10"),
    ('g-3key',          'M', "SELECT RegionID, SearchEngineID, AdvEngineID, COUNT(*) AS c FROM hits GROUP BY RegionID, SearchEngineID, AdvEngineID ORDER BY c DESC LIMIT 10"),
    ('g-str',           'M', "SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10"),
    ('g-bigdict',       'M', "SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10"),
    ('g-exhour',        'M', "SELECT extract(hour FROM EventTime) AS h, COUNT(*) AS c FROM hits GROUP BY h ORDER BY c DESC LIMIT 24"),
    ('g-exyear2',       'M', "SELECT extract(year FROM EventTime) AS y, SearchEngineID, COUNT(*) AS c FROM hits GROUP BY y, SearchEngineID ORDER BY c DESC LIMIT 10"),
    ('g-arith',         'M', "SELECT ClientIP - 1 AS c1, COUNT(*) AS c FROM hits GROUP BY c1 ORDER BY c DESC LIMIT 10"),
    ('g-len',           'M', "SELECT LENGTH(Referer) AS n, COUNT(*) AS c FROM hits WHERE CounterID = 62 GROUP BY n ORDER BY c DESC LIMIT 10"),
    ('g-where',         'M', "SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE CounterID = 62 AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10"),
    # ---- aggregates over groups (6) ----
    ('a-sum',           'M', "SELECT RegionID, SUM(ResolutionWidth) AS t FROM hits GROUP BY RegionID ORDER BY t DESC LIMIT 10"),
    ('a-avg',           'M', "SELECT RegionID, AVG(ResolutionWidth) AS a FROM hits GROUP BY RegionID ORDER BY a DESC LIMIT 10"),
    ('a-minmax',        'M', "SELECT SearchEngineID, MIN(ResolutionWidth) AS mn, MAX(ResolutionWidth) AS mx FROM hits GROUP BY SearchEngineID ORDER BY mx DESC LIMIT 10"),
    ('a-cd-grp',        'M', "SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10"),
    ('a-multi',         'M', "SELECT SearchEngineID, COUNT(*) AS c, SUM(ResolutionWidth) AS t, AVG(ResolutionWidth) AS a FROM hits GROUP BY SearchEngineID ORDER BY c DESC LIMIT 10"),
    ('a-datemin',       'M', "SELECT RegionID, MIN(EventDate) AS d0, MAX(EventDate) AS d1 FROM hits GROUP BY RegionID ORDER BY RegionID LIMIT 10"),
    # ---- HAVING (3) ----
    ('h-count',         'M', "SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase HAVING COUNT(*) > 5000 ORDER BY c DESC LIMIT 20"),
    ('h-sum',           'M', "SELECT RegionID, SUM(AdvEngineID) AS t, COUNT(*) AS c FROM hits GROUP BY RegionID HAVING SUM(AdvEngineID) > 5000 ORDER BY c DESC LIMIT 10"),
    ('h-avg-multi',     'M', "SELECT SearchEngineID, AVG(ResolutionWidth) AS a, COUNT(*) AS c FROM hits GROUP BY SearchEngineID HAVING AVG(ResolutionWidth) >= 1200 AND COUNT(*) > 100000 ORDER BY c DESC LIMIT 10"),
    # ---- DISTINCT (3) ----
    ('d-mid',           'S', "SELECT DISTINCT RegionID, SearchEngineID FROM hits"),
    ('d-scalar-lim',    'N', "SELECT DISTINCT extract(hour FROM EventTime), CounterID FROM hits LIMIT 50"),
    ('d-small',         'S', "SELECT DISTINCT SearchEngineID, AdvEngineID FROM hits"),
    # ---- ORDER/LIMIT shapes (4) ----
    ('o-cluster',       'H', "SELECT EventTime, UserID FROM hits ORDER BY EventTime LIMIT 100"),
    ('o-value',         'M', "SELECT UserID, ResolutionWidth FROM hits ORDER BY ResolutionWidth DESC LIMIT 100"),
    ('o-offset',        'M', "SELECT RegionID, COUNT(*) AS c FROM hits GROUP BY RegionID ORDER BY c DESC LIMIT 10 OFFSET 20"),
    ('o-2col',          'M', "SELECT SearchEngineID, RegionID, COUNT(*) AS c FROM hits GROUP BY SearchEngineID, RegionID ORDER BY c DESC, RegionID LIMIT 15"),
    # ---- set operations (4) ----
    ('u-union',         'S', "SELECT RegionID FROM hits WHERE CounterID = 62 UNION SELECT RegionID FROM hits WHERE CounterID = 155"),
    ('u-unionall', 'M', "SELECT SearchEngineID, COUNT(*) AS c FROM hits WHERE AdvEngineID = 2 GROUP BY SearchEngineID UNION ALL SELECT SearchEngineID, COUNT(*) AS c FROM hits WHERE AdvEngineID = 62 GROUP BY SearchEngineID ORDER BY c DESC LIMIT 10"),
    ('u-intersect',     'S', "SELECT DISTINCT SearchEngineID FROM hits WHERE AdvEngineID = 2 INTERSECT SELECT DISTINCT SearchEngineID FROM hits WHERE AdvEngineID = 62"),
    ('u-except',        'S', "SELECT DISTINCT SearchEngineID FROM hits WHERE AdvEngineID = 2 EXCEPT SELECT DISTINCT SearchEngineID FROM hits WHERE AdvEngineID = 62"),
    # ---- subqueries: uncorrelated (5) ----
    ('sq-scalar',       'C', "SELECT COUNT(*) FROM hits WHERE ResolutionWidth > (SELECT AVG(ResolutionWidth) FROM hits)"),
    ('sq-in',           'C', "SELECT COUNT(*) FROM hits WHERE SearchPhrase IN (SELECT SearchPhrase FROM hits WHERE CounterID = 62 AND SearchPhrase <> '')"),
    ('sq-notin',        'C', "SELECT COUNT(*) FROM hits WHERE SearchPhrase <> '' AND SearchPhrase NOT IN (SELECT SearchPhrase FROM hits WHERE CounterID = 62 AND SearchPhrase <> '')"),
    ('sq-in-grp',       'M', "SELECT RegionID, COUNT(*) AS c FROM hits WHERE SearchPhrase IN (SELECT SearchPhrase FROM hits WHERE CounterID = 62 AND SearchPhrase <> '') GROUP BY RegionID ORDER BY c DESC LIMIT 10"),
    ('sq-nested',       'C', "SELECT COUNT(*) FROM hits WHERE ResolutionWidth > (SELECT AVG(ResolutionWidth) FROM hits WHERE RegionID IN (SELECT RegionID FROM hits WHERE CounterID = 62))"),
    # ---- correlated (3) ----
    ('cq-exists',       'C', "SELECT COUNT(*) FROM hits h1 WHERE EXISTS (SELECT 1 FROM hits h2 WHERE h2.SearchPhrase = h1.SearchPhrase AND h2.CounterID = 62)"),
    ('cq-notexists',    'C', "SELECT COUNT(*) FROM hits h1 WHERE NOT EXISTS (SELECT 1 FROM hits h2 WHERE h2.SearchPhrase = h1.SearchPhrase AND h2.CounterID = 62)"),
    ('cq-aboveavg',     'S', "SELECT RegionID, ResolutionWidth FROM hits WHERE ResolutionWidth > (SELECT AVG(h2.ResolutionWidth) FROM hits h2 WHERE h2.RegionID = hits.RegionID)"),
    # ---- CTEs (3) ----
    ('cte-view',        'M', "WITH v AS (SELECT SearchPhrase, RegionID FROM hits WHERE CounterID = 62) SELECT SearchPhrase, COUNT(*) AS c FROM v WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10"),
    ('cte-agg',         'M', "WITH t AS (SELECT RegionID, COUNT(*) AS c FROM hits GROUP BY RegionID) SELECT RegionID, c FROM t WHERE c > 5000000 ORDER BY c DESC LIMIT 10"),
    ('cte-chain',       'C', "WITH a AS (SELECT SearchPhrase, RegionID FROM hits WHERE CounterID = 62), b AS (SELECT SearchPhrase FROM a WHERE RegionID = 229) SELECT COUNT(*) FROM b WHERE SearchPhrase <> ''"),
    # ---- windows (12) ----
    ('w-rn-top2',       'N', "SELECT UserID, EventTime, ROW_NUMBER() OVER (PARTITION BY UserID ORDER BY EventTime) AS rn FROM hits QUALIFY rn <= 2"),
    ('w-partcount',     'S', "SELECT RegionID, COUNT(*) OVER (PARTITION BY RegionID) AS n FROM hits QUALIFY n > 5000000"),
    ('w-runsum',        'S', "SELECT SearchEngineID, ResolutionWidth, SUM(ResolutionWidth) OVER (PARTITION BY SearchEngineID ORDER BY EventTime) AS rs FROM hits QUALIFY rs > 40000000000"),
    ('w-runavg',        'N', "SELECT RegionID, AVG(ResolutionWidth) OVER (PARTITION BY RegionID ORDER BY EventTime) AS ra FROM hits QUALIFY ra > 1600"),
    ('w-runmin',        'N', "SELECT RegionID, MIN(ResolutionWidth) OVER (PARTITION BY RegionID ORDER BY EventTime) AS rm FROM hits QUALIFY rm = 0"),
    ('w-partavg',       'S', "SELECT SearchEngineID, AVG(ResolutionWidth) OVER (PARTITION BY SearchEngineID) AS aw FROM hits QUALIFY aw > 1500"),
    ('w-desc-last',     'N', "SELECT UserID, EventTime, ROW_NUMBER() OVER (PARTITION BY UserID ORDER BY EventTime DESC) AS rn FROM hits QUALIFY rn = 1"),
    ('w-multipart',     'S', "SELECT CounterID, RegionID, COUNT(*) OVER (PARTITION BY CounterID, RegionID) AS n FROM hits QUALIFY n > 1000000"),
    ('w-frame-avg',     'N', "SELECT UserID, AVG(ResolutionWidth) OVER (PARTITION BY UserID ORDER BY EventTime ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS ma FROM hits QUALIFY ma > 2000"),
    ('w-frame-max',     'N', "SELECT RegionID, MAX(ResolutionWidth) OVER (PARTITION BY RegionID ORDER BY EventTime ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS sx FROM hits QUALIFY sx = 0"),
    ('w-q-mixed',       'S', "SELECT RegionID, ResolutionWidth, COUNT(*) OVER (PARTITION BY RegionID) AS n FROM hits QUALIFY n > 5000000 AND ResolutionWidth > 1900"),
    # ---- grouping sets (3) ----
    ('gs-rollup',       'M', "SELECT RegionID, SearchEngineID, COUNT(*) AS c FROM hits GROUP BY ROLLUP(RegionID, SearchEngineID) ORDER BY c DESC LIMIT 15"),
    ('gs-cube',         'M', "SELECT RegionID, AdvEngineID, COUNT(*) AS c, SUM(ResolutionWidth) AS t FROM hits WHERE CounterID = 62 GROUP BY CUBE(RegionID, AdvEngineID) ORDER BY c DESC LIMIT 12"),
    ('gs-sets',         'M', "SELECT RegionID, SearchEngineID, COUNT(*) AS c FROM hits GROUP BY GROUPING SETS ((RegionID), (SearchEngineID), ()) ORDER BY c DESC LIMIT 12"),
    # ---- joins (7) ----
    ('j-dim-grp',       'M', "SELECT d.zone, COUNT(*) AS c FROM hits h JOIN rdim d ON h.RegionID = d.rid GROUP BY d.zone ORDER BY c DESC LIMIT 10"),
    ('j-dim-filter',    'M', "SELECT d.tier, COUNT(*) AS c, SUM(h.ResolutionWidth) AS t FROM hits h JOIN rdim d ON h.RegionID = d.rid WHERE d.zone = 'z07' GROUP BY d.tier ORDER BY c DESC LIMIT 3"),
    ('j-semi',          'C', "SELECT COUNT(*) FROM hits h JOIN rdim d ON h.RegionID = d.rid WHERE d.tier = 'gold'"),
    ('j-giant',         'M', "SELECT g.seg, COUNT(*) AS c FROM hits h JOIN gdim g ON h.UserID = g.uid GROUP BY g.seg ORDER BY c DESC LIMIT 3"),
    ('j-giant-filter',  'M', "SELECT g.cohort, COUNT(*) AS c, SUM(h.ResolutionWidth) AS t FROM hits h JOIN gdim g ON h.UserID = g.uid WHERE g.seg = 'vip' GROUP BY g.cohort ORDER BY c DESC LIMIT 10"),
    ('j-left',          'M', "SELECT d.zone, COUNT(*) AS c FROM hits h LEFT JOIN rdim d ON h.RegionID = d.rid GROUP BY d.zone ORDER BY c DESC LIMIT 10"),
    ('j-mixed-grp',     'M', "SELECT d.zone, h.SearchEngineID, COUNT(*) AS c FROM hits h JOIN rdim d ON h.RegionID = d.rid WHERE h.CounterID = 62 GROUP BY d.zone, h.SearchEngineID ORDER BY c DESC LIMIT 10"),
    # ---- datetime & misc (5) ----
    ('t-datewindow',    'C', "SELECT COUNT(*) FROM hits WHERE EventDate >= '2013-07-14' AND EventDate <= '2013-07-16'"),
    ('t-minute',        'M', "SELECT RegionID, extract(minute FROM EventTime) AS m, COUNT(*) AS c FROM hits WHERE CounterID = 62 GROUP BY RegionID, m ORDER BY c DESC LIMIT 10"),
    ('t-null',          'C', "SELECT COUNT(*) FROM hits WHERE DontCountHits IS NULL"),
    ('t-bool',          'C', "SELECT COUNT(*) FROM hits WHERE IsMobile = 1 AND SearchPhrase <> ''"),
    ('t-bigint',        'M', "SELECT UserID, COUNT(*) AS c FROM hits GROUP BY UserID ORDER BY c DESC LIMIT 10"),
    # ---- window fns: remaining distinct shapes (5) ----
    ('w-rank',          'N', "SELECT RegionID, RANK() OVER (PARTITION BY RegionID ORDER BY EventDate) AS r FROM hits QUALIFY r = 1"),
    ('w-dense',         'N', "SELECT SearchEngineID, DENSE_RANK() OVER (PARTITION BY SearchEngineID ORDER BY EventDate) AS r FROM hits QUALIFY r = 2"),
    ('w-lag',           'N', "SELECT UserID, LAG(ResolutionWidth) OVER (PARTITION BY UserID ORDER BY EventTime) AS pw FROM hits LIMIT 1000"),
    ('w-lead',          'N', "SELECT UserID, LEAD(ResolutionWidth) OVER (PARTITION BY UserID ORDER BY EventTime) AS nw FROM hits LIMIT 1000"),
    ('w-dictord',       'N', "SELECT RegionID, SearchPhrase, ROW_NUMBER() OVER (PARTITION BY RegionID ORDER BY SearchPhrase) AS rn FROM hits QUALIFY rn = 1"),
    # ---- more predicate shapes (6) ----
    ('f-notlike',       'C', "SELECT COUNT(*) FROM hits WHERE URL NOT LIKE '%google%' AND CounterID = 62"),
    ('f-eqlike',        'C', "SELECT COUNT(*) FROM hits WHERE CounterID = 62 AND URL LIKE '%google%'"),
    ('f-isnotnull',     'C', "SELECT COUNT(*) FROM hits WHERE DontCountHits IS NOT NULL"),
    ('f-in-str',        'C', "SELECT COUNT(*) FROM hits WHERE SearchPhrase IN ('', 'погода')"),
    ('f-or-mixed',      'C', "SELECT COUNT(*) FROM hits WHERE CounterID = 62 AND (URL LIKE '%google%' OR SearchPhrase <> '')"),
    ('f-scalar-eq',     'C', "SELECT COUNT(*) FROM hits WHERE LENGTH(SearchPhrase) = 12"),
    # ---- scalar group keys: remaining kinds (3) ----
    ('g-lower',         'M', "SELECT LOWER(SearchPhrase) AS l, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY l ORDER BY c DESC LIMIT 10"),
    ('g-substr',        'M', "SELECT SUBSTR(URL, 1, 20) AS p, COUNT(*) AS c FROM hits GROUP BY p ORDER BY c DESC LIMIT 10"),
    ('g-1key-sum-ord',  'M', "SELECT ClientIP, SUM(ResolutionWidth) AS t FROM hits GROUP BY ClientIP ORDER BY t DESC LIMIT 10"),
    # ---- set-op & subquery extras (3) ----
    ('u-order',         'M', "SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE CounterID = 62 AND SearchPhrase <> '' GROUP BY SearchPhrase UNION ALL SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE CounterID = 155 AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10"),
    ('sq-in-drive',     'M', "SELECT SearchEngineID, COUNT(*) AS c FROM hits WHERE SearchPhrase IN (SELECT SearchPhrase FROM hits WHERE CounterID = 155 AND SearchPhrase <> '') GROUP BY SearchEngineID ORDER BY c DESC LIMIT 10"),
    ('sq-scalar-min',   'C', "SELECT COUNT(*) FROM hits WHERE EventDate = (SELECT MIN(EventDate) FROM hits)"),
    # ---- join extras (2) ----
    ('j-dump',          'N', "SELECT h.SearchPhrase, d.zone FROM hits h JOIN rdim d ON h.RegionID = d.rid WHERE d.tier = 'gold' AND h.SearchPhrase <> '' LIMIT 1000"),
    ('j-left-dump',     'N', "SELECT h.RegionID, d.zone FROM hits h LEFT JOIN rdim d ON h.RegionID = d.rid WHERE h.CounterID = 62 LIMIT 1000"),
]