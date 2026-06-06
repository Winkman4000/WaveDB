# WaveDB — SQL Capabilities

What WaveDB can and cannot do at the atomic SQL level, vs a general-purpose
database (SQLite as the reference). This is the source of truth for capability.
A feature is marked supported ONLY if it runs on the real path (`db.run`).

Legend:  ✓ supported   ~ partial   ✗ not yet

**Rule: this file is updated in the same commit that adds or changes a feature
on the main path. If it's ✓ here, `db.run` does it.**

---

## Coverage at a glance

| Area | WaveDB | SQLite |
|---|:-:|:-:|
| Read / analytics (filter, group, join, aggregate, distinct) | strong | yes |
| General SQL surface (subqueries, CTE, window, CASE, functions) | mostly ✗ | yes |
| Write / DML (insert, update, delete) | basic | full |
| Schema / DDL (create, alter, drop) | basic | full |
| Transactions, constraints, views, triggers | ✗ | yes |

Rough feature coverage vs SQLite's surface: **~35–40%** — deep on the analytics
read path, shallow on the rest of the language.

---

## SELECT — shape

| Feature | WaveDB | SQLite |
|---|:-:|:-:|
| SELECT column list | ✓ | ✓ |
| SELECT * | ✓ | ✓ |
| Column alias (AS) in output | ✓ | ✓ |
| Table alias | ✓ | ✓ |
| SELECT DISTINCT | ✓ | ✓ |
| Expression in output list (e.g. a+b as non-aggregate) | ✗ | ✓ |
| Literal / constant in output list | ✗ | ✓ |
| WHERE | ✓ | ✓ |
| GROUP BY (single column) | ✓ | ✓ |
| GROUP BY (multi-column) | ✓ | ✓ |
| GROUP BY expression | ✗ | ✓ |
| HAVING | ✓ | ✓ |
| ORDER BY (ASC / DESC) | ✓ | ✓ |
| ORDER BY multiple keys | ✓ | ✓ |
| LIMIT | ✓ | ✓ |
| OFFSET | ✗ | ✓ |

## WHERE — operators

| Operator | WaveDB | SQLite |
|---|:-:|:-:|
| = , != / <> | ✓ | ✓ |
| < , > , <= , >= | ✓ | ✓ |
| BETWEEN | ✓ | ✓ |
| IN (value list) | ✓ | ✓ |
| IN (subquery) | ✗ | ✓ |
| LIKE | ✓ | ✓ |
| ILIKE | ✓ | ✓ (via NOCASE) |
| GLOB / REGEXP | ✗ | ✓ |
| IS NULL / IS NOT NULL | ✓ | ✓ |
| AND / OR / NOT / parentheses | ✓ | ✓ |
| EXISTS | ✗ | ✓ |
| Predicate on expression (e.g. WHERE a+b > 5) | ✗ | ✓ |

## Aggregates

| Feature | WaveDB | SQLite |
|---|:-:|:-:|
| COUNT(*) | ✓ | ✓ |
| COUNT(col) | ✓ | ✓ |
| COUNT(DISTINCT col) | ✓ | ✓ |
| SUM / AVG / MIN / MAX | ✓ | ✓ |
| Arithmetic inside aggregate (SUM(a*b)) | ✓ | ✓ |
| COUNT(DISTINCT) over multiple cols / expr | ✗ | ✓ |
| GROUP_CONCAT / string_agg | ✗ | ✓ |
| TOTAL | ✗ | ✓ |
| Aggregate FILTER (WHERE ...) clause | ✗ | ✓ |

## JOINs

| Feature | WaveDB | SQLite |
|---|:-:|:-:|
| INNER JOIN (2 tables) | ✓ | ✓ |
| INNER JOIN (3+ tables / FK chain) | ✓ | ✓ |
| LEFT / RIGHT / FULL OUTER JOIN | ✗ | ✓ |
| CROSS JOIN | ✗ | ✓ |
| Self-join | ✗ | ✓ |
| USING clause | ✗ | ✓ |
| Non-equi join (ON a < b) | ✗ | ✓ |

## Not yet built — the big general-SQL gaps (SQLite has all of these)

| Feature | WaveDB | SQLite |
|---|:-:|:-:|
| Subqueries — scalar (SELECT (SELECT ...)) | ✗ | ✓ |
| Subqueries — in FROM (derived table) | ✗ | ✓ |
| Subqueries — correlated | ✗ | ✓ |
| CTE / WITH | ✗ | ✓ |
| Recursive CTE (WITH RECURSIVE) | ✗ | ✓ |
| Window functions (OVER / PARTITION BY) | ✗ | ✓ |
| ROW_NUMBER / RANK / DENSE_RANK | ✗ | ✓ |
| LAG / LEAD / running totals | ✗ | ✓ |
| CASE WHEN ... THEN ... END | ✗ | ✓ |
| COALESCE / NULLIF / IFNULL | ✗ | ✓ |
| CAST / type conversion | ✗ | ✓ |
| String functions (substr, upper, lower, trim, replace, length, ‖) | ✗ | ✓ |
| Math functions (abs, round, ceil, floor, %) | ✗ | ✓ |
| Date/time functions (date, strftime, julianday) | ✗ | ✓ |
| Set ops: UNION / UNION ALL | ✗ | ✓ |
| Set ops: INTERSECT / EXCEPT | ✗ | ✓ |

## DML — writing data

| Feature | WaveDB | SQLite |
|---|:-:|:-:|
| INSERT ... VALUES (one or more rows) | ✓ | ✓ |
| INSERT ... SELECT | ✗ | ✓ |
| UPSERT / ON CONFLICT | ✗ | ✓ |
| UPDATE ... SET (literal / simple expr) | ~ | ✓ |
| UPDATE with subquery / joined update | ✗ | ✓ |
| DELETE (with WHERE) | ✓ | ✓ |
| RETURNING clause | ✗ | ✓ |

## DDL — schema

| Feature | WaveDB | SQLite |
|---|:-:|:-:|
| CREATE TABLE (typed columns) | ✓ | ✓ |
| ALTER TABLE ADD COLUMN | ✓ | ✓ |
| ALTER TABLE RENAME COLUMN | ✓ | ✓ |
| ALTER TABLE DROP COLUMN | ✗ | ✓ |
| DROP TABLE | ✓ | ✓ |
| Constraints (PRIMARY KEY / UNIQUE / CHECK / FK) enforced | ✗ | ✓ |
| GENERATED / computed columns | ✗ | ✓ |
| Views | ✗ | ✓ |
| Triggers | ✗ | ✓ |
| User-defined indexes (CREATE INDEX) | ✗ | ✓ |

## Engine / runtime

| Feature | WaveDB | SQLite |
|---|:-:|:-:|
| Transactions (BEGIN / COMMIT / ROLLBACK) | ✗ | ✓ |
| Multiple tables in one database | ✓ | ✓ |
| Lossless columnar storage (smaller than parquet) | ✓ | n/a |
| Beats DuckDB on analytics (median ~3x, 21/30 sample) | ✓ | n/a |

---

_Last updated: commit 270623d. Datasets used for measurement: TPC-H sf=1
(30-query scoreboard, examples/report.md) and ClickBench 10M-row slice
(examples/clickbench.md)._
