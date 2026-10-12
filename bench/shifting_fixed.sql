# The shifting board's fixed set: variants that once exposed a problem. One per line:
# base query index (0-based, queries.sql) <TAB> what changed <TAB> SQL
# Seeded from the ClickBench review of WaveDB (#2424, 2026-10-12). Append every new finding.
18	review: extract hour instead of minute	SELECT UserID, extract(hour FROM EventTime) AS m, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10
42	review: DATE_TRUNC hour instead of minute (OFFSET 10: 48 hours fit no OFFSET 1000)	SELECT DATE_TRUNC('hour', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('hour', EventTime) ORDER BY DATE_TRUNC('hour', EventTime) LIMIT 10 OFFSET 10
27	review: Title instead of URL	SELECT CounterID, AVG(length(Title)) AS l, COUNT(*) AS c FROM hits WHERE Title <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25
20	review: yandex instead of google	SELECT COUNT(*) FROM hits WHERE URL LIKE '%yandex%'
28	review: the regex on URL instead of Referer	SELECT REGEXP_REPLACE(URL, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, AVG(length(URL)) AS l, COUNT(*) AS c, MIN(URL) FROM hits WHERE URL <> '' GROUP BY k HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25
13	review: Title instead of SearchPhrase (was WRONG, fixed 7f86628)	SELECT Title, COUNT(DISTINCT UserID) AS u FROM hits WHERE Title <> '' GROUP BY Title ORDER BY u DESC LIMIT 10
