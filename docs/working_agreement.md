# Working agreement

## Rule 1 — No scattered work. Finish the job.

Anything we start in isolation ends in exactly one of two states:

1. **Deleted** — we tried it, it isn't worth keeping, it leaves the tree.
2. **Integrated** — it is wired into the system so the system *just works* with it:
   the normal execution path uses it, the suite covers it, no manual setup step is
   required to get its benefit.

There is no third "sitting off to the side as an island" state. No exceptions.

**Why.** Islands don't compound. A capability the engine doesn't reach on its own
(a planner nothing consults, an index nothing auto-builds, a benchmark that needs a
manual setup the engine doesn't do) will be forgotten, mismeasured, or silently
regressed. We already paid for this once: joins were reported as a 0.06–0.30x *loss*
because the FK pointers they depend on weren't declared — the capability existed but
was not integrated, so the measurement lied.

**Test for "integrated".** Could a new person, or a future session with no memory of
this one, get the benefit without knowing the trick? If the answer needs a manual
incantation, it is not integrated yet.

**Before starting anything new**, the things already started must each be driven to
delete-or-integrate. We do not open a new front while islands are open.
