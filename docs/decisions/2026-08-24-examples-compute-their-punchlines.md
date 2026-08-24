# Example notebooks compute their punchlines

Status: adopted

## Context

A full review of the example notebooks found that the curriculum's teaching
claims and its executable evidence had drifted apart in three graded ways.
Some labs computed the right result but never surfaced it — tracked
notebooks are stored output-free, so a bare trailing expression shows a
repository reader nothing. Some typed the number that carried the lesson
directly into a fixture: judge verdicts stored as literal booleans, a
cost/quality table of hand-written floats. And in three places the shipped
fixture actively contradicted the narrated lesson: the model-selection TCO
section claimed the cheaper-per-token model cost more per session while its
own fixtures computed the opposite; the RAG workshop's four-way retrieval
ablation produced byte-identical quality metrics in every mode, so its
capstone rejected the change on a hard-coded simulated latency table and its
release artifact was dead code; the fine-tuning course's default evaluation
wrote report files its own promotion notebook refused to read, making the
adopt path structurally unreachable.

Three alternatives lost:

**Committing executed outputs** was rejected. The output-free contract is
deliberate and enforced: it keeps diffs reviewable, prevents stale or
environment-specific results from masquerading as current evidence, and
keeps accidental capture of local paths or identity out of Git. The fix is
to print the punchline from the cell, not to freeze one.

**Relaxing the one-teaching-step cell cap** so demonstrations could grow
inside notebooks was rejected. The cap is what keeps a lesson readable; the
two uncapped labs proved demonstrations fit the discipline. Mechanics belong
in the tested support modules and generators, split into capped steps, with
the notebook calling them and printing the result.

**Labelling fabricated numbers as illustrative and keeping them** was
rejected. A labelled fabrication still teaches the wrong epistemics — the
curriculum's own thesis is that a plausible number is not evidence — and it
drifts: the TCO section's narration was falsified by its own fixture
precisely because nothing computed the claim.

## Decision

Every example-notebook demonstration must:

1. **Compute the number that carries its lesson** from a fixture the reader
   can inspect and modify. A quality score, agreement rate, cost, or metric
   delta is derived by running the fixture through the same scorers and
   policies the lesson teaches — never typed in as a literal.
2. **Plant the failure it warns about and show the gate catching it.** A
   check that can only ever pass is decoration; the leak, the overfit
   challenger, the forged record, the broken stamp, the unsafe statement is
   constructed in the lab and visibly rejected.
3. **Print its punchline.** Tracked notebooks are output-free, so the cell
   itself prints the before/after contrast or caught refusal it exists to
   show.
4. **Produce the narrated outcome from the shipped fixture.** When a
   comparison motivates a decision — an ablation, a session-economics
   inversion, a promotion — the committed data must actually yield that
   outcome when run, and a simulated operational metric (such as fixture
   latency) may inform but never decide the release outcome of the default
   offline path. Simulated values stay labelled `simulated_offline_fixture`.

Tests pin the computed consequences (agreement values, gate failures,
optimizer train-versus-holdout gaps, divergent retrieval metrics) rather
than the fixture literals, so a fixture edit that silently breaks a
demonstration fails CI.

## Consequences

A reader who runs any lab now sees the claim proven, and a reader browsing
the source sees the printed contrast next to the code that computed it. The
support modules carry more logic, which is the intended trade: it is logic
under test.

What becomes forbidden: reintroducing a typed-in verdict, score, or cost
where a fixture could compute one; adding a demonstration whose failure
branch is unreachable; and editing fixture data in a way that flattens a
demonstrated divergence or inversion — the pinned tests exist so that such
an edit is a visible decision, not an accident. A metric that genuinely
cannot be computed offline is labelled and excluded from decisions, not
invented.
