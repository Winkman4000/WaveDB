# Benchmark crosswalk

Maps canonical corpus shapes (`docs/corpus.md`) to their source benchmark queries. `M#` = our TPC-H matrix (`bench/catalog.py`). `Q#` = ClickBench (`bench/clickbench_queries.sql`, Q0..Q42).

## Canonical → source

| canonical | benches | shape | matrix | clickbench |
|---|---|---|---|---|
| C01 | both | whole COUNT(*) | M1 | Q0 |
| C02 | both | whole SUM | M2 | Q29 |
| C03 | ours | whole multi-agg | M3 | — |
| C04 | clickbench | AVG | — | Q3 |
| C05 | clickbench | AVG+COUNT+SUM | — | Q2 |
| C06 | clickbench | MAX+MIN | — | Q6 |
| C07 | both | WHERE numeric > | M11, M14, M15 | Q1, Q20 |
| C08 | ours | WHERE BETWEEN + agg | M12, M13, M16 | — |
| C09 | clickbench | projection +WHERE | — | Q19 |
| C10 | ours | GROUP BY 2-col (Q1) | M7 | — |
| C11 | ours | GROUP BY K3 count | M4, M8, M10 | — |
| C12 | ours | GROUP BY K3 sum | M5, M9 | — |
| C13 | ours | GROUP BY K7 avg | M6 | — |
| C14 | ours | HAVING | M25 | — |
| C15 | ours | ORDER BY + LIMIT | M24 | — |
| C16 | clickbench | AVG+COUNT+SUM, GROUP BY 2+ +topK | — | Q32 |
| C17 | clickbench | COUNT, GROUP BY 1 +topK | — | Q15, Q33 |
| C18 | clickbench | COUNT, GROUP BY 2+ +topK | — | Q16, Q17, Q18, Q34, Q35 |
| C19 | ours | WHERE + GROUP BY | M17 | — |
| C20 | clickbench | AVG+COUNT+MIN, GROUP BY 1 +WHERE +topK +HAVING | — | Q28 |
| C21 | clickbench | AVG+COUNT+SUM, GROUP BY 2+ +WHERE +topK | — | Q30, Q31 |
| C22 | clickbench | AVG+COUNT, GROUP BY 1 +WHERE +topK +HAVING | — | Q27 |
| C23 | clickbench | COUNT+MIN, GROUP BY 1 +WHERE +topK | — | Q21 |
| C24 | clickbench | COUNT, GROUP BY 1 +WHERE +topK | — | Q7, Q12, Q36, Q37, Q38, Q42 |
| C25 | clickbench | COUNT, GROUP BY 2+ +WHERE +topK | — | Q14, Q39, Q40, Q41 |
| C26 | both | COUNT(DISTINCT) low | M21, M22 | Q4, Q5 |
| C27 | ours | DISTINCT 1-col | M18, M19, M20 | — |
| C28 | ours | grouped COUNT(DISTINCT) | M23 | — |
| C29 | clickbench | COUNT(DISTINCT) AVG+COUNT+SUM, GROUP BY 1 +topK | — | Q9 |
| C30 | clickbench | COUNT(DISTINCT) COUNT+MIN, GROUP BY 1 +WHERE +topK | — | Q22 |
| C31 | clickbench | COUNT(DISTINCT) COUNT, GROUP BY 1 +topK | — | Q8 |
| C32 | clickbench | COUNT(DISTINCT) COUNT, GROUP BY 1 +WHERE +topK | — | Q10, Q13 |
| C33 | clickbench | COUNT(DISTINCT) COUNT, GROUP BY 2+ +WHERE +topK | — | Q11 |
| C34 | clickbench | projection +WHERE +topK | — | Q23, Q24, Q25, Q26 |
| C35 | ours | JOIN + WHERE | M29 | — |
| C36 | ours | JOIN group child-key | M27, M28, M30 | — |
| C37 | ours | JOIN group parent-key | M26 | — |

## Matrix (M#) → canonical

| M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 | M11 | M12 | M13 | M14 | M15 | M16 | M17 | M18 | M19 | M20 | M21 | M22 | M23 | M24 | M25 | M26 | M27 | M28 | M29 | M30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | C02 | C03 | C11 | C12 | C13 | C10 | C11 | C12 | C11 | C07 | C08 | C08 | C07 | C07 | C08 | C19 | C27 | C27 | C27 | C26 | C26 | C28 | C15 | C14 | C37 | C36 | C36 | C35 | C36 |

## ClickBench (Q#) → canonical

| Q0 | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 | Q10 | Q11 | Q12 | Q13 | Q14 | Q15 | Q16 | Q17 | Q18 | Q19 | Q20 | Q21 | Q22 | Q23 | Q24 | Q25 | Q26 | Q27 | Q28 | Q29 | Q30 | Q31 | Q32 | Q33 | Q34 | Q35 | Q36 | Q37 | Q38 | Q39 | Q40 | Q41 | Q42 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | C07 | C05 | C04 | C26 | C26 | C06 | C24 | C31 | C29 | C32 | C33 | C24 | C32 | C25 | C17 | C18 | C18 | C18 | C09 | C07 | C23 | C30 | C34 | C34 | C34 | C34 | C22 | C20 | C02 | C21 | C21 | C16 | C17 | C18 | C18 | C24 | C24 | C24 | C25 | C25 | C25 | C24 |
