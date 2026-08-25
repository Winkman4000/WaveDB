# THE ENGINE CODEX (seed — populated from code, never from memory)
Jackson's ruling: every part of the codebase accounted for and known for
where it adds contribution, so changes target a NAMED LAYER and tests
reflect consequences of that layer.

## THE LAYERS (taxonomy to populate, one module at a time, by READING each)
1. ENCODING LAYER — wdb_encode: modes (0..6+), dresses (enc-0..13), elections, seals, frame law
2. STORAGE/METADATA — wdb_engine.Segment (passport: V, N, BR, dt, stairs), wdb_catalog, sidecars (plists, jptr pointers, gbc/gbp)
3. READS LAYER — read_methods.py (named!), engine reads: codes/codes_band/codes_at/_raw_codes/_e13_band, blockstats
4. QUERY IDENTIFICATION / ROUTING — controller.py (the consult chain), commands.py
5. PLANNING / COURTS — per-shape lanes: funnel, join (chains/hash/sidecars/potency), lenagg, sampletop, compound (SURVIVOR-SET — already exists!), coscan, countpos, clustertopk, bsi, affine*, cdgroup, cte...
6. EMIT/RESULTS — colresult, sql tail (_apply_order/having), top-k emits
7. REFEREE/GATES — tests/ (1692), megaboard, board_clickbench, board_tpch, duck referees

## KERNELS ARE NOT A LAYER (Jackson's correction)
Kernels are a SPEED CLASS, not a place: a kernel can be a read, a fold,
a sort -- it belongs to whatever layer's stage it compiles. wdb_kernels,
wdb_exprjit, wdb_radix, bsi_kernels are SHARED ARMORIES (cross-cutting
infrastructure, accounted for in the census but not pipeline rungs).
Every organ's codex entry must state WHICH of its stages run at kernel
speed and which run in interpreted glue -- that attribute is the era's
recurring diagnosis (lenagg: 4ms fold in 320ms glue; Q3: 62ms kernel in
2000ms glue). A new kernel is never an architectural change by itself;
the architectural fact is which layer's stage got compiled.

## MEASURED-VERDICTS TABLE (append-only; warm wall, commit, date)
| query | before | after | commit | note |
|---|---|---|---|---|
| TPCH Q3 | 23.0s | 2.00s | 202c59b | arc: uniq-meta, jptr sidecars, sorted-run, mono-acc; kernel=62ms; REMAINING: emit 475 (topk gate), assemble 360, prework 133, ~950ms UNMEASURED in db.run dispatch |
| TPCH Q3 | 2.00s | 1.34s | 63441ba | BALANCED BILL era: setup bracket found 885ms never measured; leading-run court killed the 60M co-key gathers (2003->1820); numeric-emit + top-k gate fired, emit 473->32 (1820->1340). Bill now: setup 733, assemble 350, kernel 60, emit 32, prework 131. Next: sub-bracket setup; assemble LUTs. |

## STANDING RULES (the anti-slip laws)
- BALANCE LAW: no strike while the bill's stages don't sum to ~the wall.
- SCOPE LAW: a principle names WHICH LAYER it restructures before code is touched ("changes which rows exist" vs "which test runs first").
- CENSUS LAW: scope existing organs (this codex + census) before building anything.
- VERDICT LAW: the only progress number is warm wall before vs after, written here.


## VERDICTS 2026-08-24 (the cascade session)
| query | before | after | commit | note |
|---|---|---|---|---|
| TPCH Q3 | 1.34s | 1.004s | 30b34bf | THE SURVIVOR CASCADE LANDED: 59,986,052 -> 302,114 rows printed on the bill (duck-identical survivor count); kernel 2ms, emit 21ms. Remaining: cascade passes 540 (kernelize first full-width LUT pass), setup 180, pre-work 130. |
| megaboard | 3.57x | 3.63x RECORD | 30b34bf | 103/103, zero wrong, 89 wins, cascade aboard |
| TPCH board | 1 wrong | ZERO WRONG | 30b34bf | Q18 fail-open FIXED (3 sites): subquery-IN became isin([])=all-False=silent empty. LAW ENGRAVED: fail closed means fail LOUD, never empty. Q18 now an honest hole. |
| TPCH Q3 | 1.004s | 0.819s | 33db9e3 | THE HOME-TABLE LAW: predicates judged at home in PARALLEL, verdicts flow FK roads down, fact pays one gather per road. Cascade 540->360. Remaining: cascade 360 (fuse fact pass), setup 178, assemble 143, prework 130. |
| REGRESSION FLAG | - | - | - | Q1 0.44->0.83s and Q6 0.31->0.50 vs board v1 (different pod; duck moved 0.44->0.37 only). INVESTIGATE next session: single-table fused_agg era changes. |

## NEXT SESSION — THE SURVIVOR CASCADE (single task, Jackson-ratified — DONE, kept for history)
SCOPE: restructures the join court pipeline — changes WHICH ROWS EXIST
after the filter stage. Not a trim.
1. FILTER FIRST, code space, potency-descending: shipdate keep-LUT over
   sequential codes -> gather pointers ONLY for survivors -> orderdate
   LUT -> 2-hop mktsegment -> ~3M survivor indices. No values decoded.
2. THEN setup at survivor scale: group-key codes sliced at survivors,
   gids on survivors, slots as V-LUTs (decode spec).
3. Kernel on survivors, pred spent. Emit decodes k.
Gates: suite + megaboard + TPCH board, exactness vs duck, warm-wall
verdict vs 1.34s written to the table. Projected band: 300-500ms.


## NEXT SESSION DOCKET (post home-table-law, Jackson-ratified)
1. THE RUN-ROAD LAW: monotone pointer roads read as RUNS, not
   addresses -- the cascade fact pass becomes repeat(verdicts,
   run_lengths): zero gathers, zero pointer bytes. (~90ms of Q3)
2. POINTER WIDTH ELECTION: u16/u32 by parent size at sidecar birth
   (naked int64 = 2-3x bytes); elegant end-state: .jptr as a
   dressed wdb column under the encoder own elections.
3. setup 178 + assemble 143 sub-bills; prework 130.
4. REGRESSION FLAG (standing): Q1 0.44->0.83, Q6 0.31->0.50 vs
   board v1 -- bill single-table fused_agg BEFORE new blades.
VERDICT 2026-08-25: Q1 regression CURED 869->503 exact (643f0e3, the
marginal-bound gate: exact census product decides cascade firing);
Q3 788 unharmed.
Baselines: Q3 0.819s (33db9e3) | megaboard 90 wins zero-wrong |
TPCH zero-wrong, 3 ok, 11 honest holes.


## VERDICTS 2026-08-25 (morning)
| item | verdict |
|---|---|
| Q1 regression | CURED 869->503 exact (643f0e3): THE MARGINAL-BOUND GATE -- expected survivors from exact census product decides cascade firing (fires iff <=50% survive). Q3 unharmed. |
| run-road law | TRIED 3 FORMS, MEASURED 819 -> 1002/883/1042, REVERTED (e35240a). At Q3 shape the 60M-wide verdict-repeat loses to the 32M survivor-scale gather. Shelved with numbers; revisit only for multi-road low-survivor shapes. |
| width election | REVERTED (8f0112b): numpy indexes with intp -- narrow pointer arrays pay a cast-copy per gather (+50ms/query for 240MB disk). Index arrays are intp by law. |
Baselines now: Q3 0.832s | Q1 0.503s | both exact.


## VERDICTS 2026-08-25 (afternoon -- the stamp session)
| item | verdict |
|---|---|
| SURVIVOR-SCALE READS (52b8c96) | Q3 832 -> 699ms EXACT, REALM RECORD. Within-home potency cascade (first conjunct sequential, later ones codes_at at survivors); gkeys+slots fetch at survivor scale; K dict-wide from passport. Setup 178->38 (plain), ->1ms (stamped). |
| THE 255-SLICE SUM STAMP (Jackson) | BUILT+PROVEN: per-order sums, 255 quantile slices, u8 stamp = 25MB on disk (0.4B/row). Stamped top-10 == duck EXACT; STOP RULE PROVEN on real data (#10 rev 415,367 >= ceiling[254] 391,935 -- one slice provably suffices). Cascade reaches 1,960 rows, setup 1ms, kernel 0ms. |
| stamp speed, honest | 808 vs plain 699: the stamp own 60M pass (~150-200ms u8 decode) outweighs savings while downstream is already cheap. PAYS WHEN: (a) u8 scan kernelized (~40ms), (b) the DESCENT COURT wired (auto stamp injection + stop rule + slice descent), (c) month-prune shrinks the fine pass. All docketed. |
Baselines: Q3 0.699 EXACT (record) | Q1 0.503 | stamp aboard lineitem, ceilings in lineitem.sumslice.json, backup lineitem_0.wdb.bak intact.
NEXT: descent court + stamp-scan kernel + month-prune; boards re-gate first (launched at session end -- read /tmp/megaboard_v4_6.log + /tmp/tpchboard8.log).


## THE POTENCY LAW, COMPLETED (Jackson, 2026-08-25)
The aggregation is always the filter target. A filter has potency
only while (downstream work saved) > (its own cost) -- prune rate
alone is meaningless. When the protected work is already cheap,
NO filter earns a place. Cascade MEMBERSHIP and ORDER both follow.
Today proofs: 13ms stamp = potent vs an 18ms sum; 150ms stamp
pass = cargo; run-road repeat = negative potency; marginal-bound
gate = this law at the cascade gate.

## NEXT SESSION (single task): THE DESCENT COURT
Wire the measured 129ms floor as the join court lane: stamp scan
(13) -> confirm at survivors (98) -> survivor sum + top-k (18) ->
STOP-RULE legality check (descend a slice on failure). Engine
target 180-250 vs duck 265. Floor probe + numbers above. Gates:
boards zero-wrong first (logs /tmp/megaboard_v4_6.log,
/tmp/tpchboard8.log from launched runs).
