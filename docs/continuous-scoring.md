# Continuous scoring: reading the judge's uncertainty

An experimental scoring path for AgentKit that reads the **top logprobs at
the score position** instead of parsing the label a judge emits. It runs
beside the discrete judges — never instead of them — and is report-only
until the sweep below settles its configuration.

## The problem it addresses

A discrete judge collapses its verdict into one emitted label. Two answers
of clearly different quality tie whenever they round to the same label,
which makes discrete judges nearly useless for *ranking* — exactly the
operation best-of-N selection and fine-grained comparisons need. The
continuous path scores from the judge's full first-token distribution: a
judge that emits "R" at 80% confidence scores differently from one that
emits "R" at 55%.

## How a score is produced

1. The verifier is prompted for a **single-token score label** — uppercase
   letters (`A` = worst upward), never digits, because every common
   tokenizer emits one letter as one token while multi-digit scores split
   across tokens.
2. The response's first generated position is read with
   `logprobs`/`top_logprobs`; `exp()` of each returned logprob gives the
   alternatives' probabilities.
3. Probabilities are **filtered to valid score tokens** (case and
   surrounding whitespace collapse into one label) and renormalized by the
   retained sum — the **normalization mass**.
4. The score is the probability-weighted average of the label values,
   mapped to `[0, 1]`.

The normalization mass is the calibration signal. A call whose retained
mass falls below the configured threshold (default 0.5) is flagged: the
prompt is not reliably steering the model to a score token, and the
weighted average is then a summary of noise. Every run reports the mean
and minimum mass, the low-mass rate, and the rate of calls that produced
no score token at all.

Three refinements from the verification literature apply on top:

- **Criteria decomposition.** Each row is judged per criterion (factual
  agreement, coverage, no fabrication) and the criterion scores are
  averaged, rather than one "is this good" call. The criteria are
  versioned platform assets in `aai_core.agentkit.continuous`; projects
  select the scorer, never redefine them.
- **Repeated evaluation.** Each judgment runs `repeats` times and
  averages.
- **Positional alternation.** The pairwise comparator
  (`ContinuousVerifier.compare`) alternates which candidate sits in the A
  slot across repeats and mirrors the swapped scores, so positional bias
  cancels instead of accumulating.

## The hard constraint: the backend must return logprobs

The verifier makes its own OpenAI-compatible chat calls — MLflow's
built-in judges never expose logprobs. That constrains the model choice:

- **Works:** an Azure OpenAI deployment (`provider: azure_apim`), or a
  Databricks-served model whose serving endpoint supports
  `top_logprobs`.
- **Never works:** the Anthropic API — it exposes no top logprobs — and
  any gateway that drops the parameter.

Support is **probed at runtime** with one tiny call before any row is
scored. A backend that refuses the parameters (HTTP 400/404/422) or
returns a well-formed response without logprobs makes the run **fall back
to the existing discrete path** with a clear warning and the
`aai.continuous_scoring: fallback-discrete` tag; authentication,
permission, rate-limit, and server failures propagate as the failures
they are.

## Configuration

```yaml
scorers:
  continuous:
    enabled: true        # off by default
    granularity: 20      # 2..26 score letters; sweep 5/10/20 before pinning
    repeats: 1           # K repeated judgments per criterion, averaged
    judge_model: verifier-model   # optional; defaults to scorers.judge_model
    low_mass_threshold: 0.5
```

`judge_model` names any `providers.models` entry in `aai-platform.yml`,
so the discrete judges can stay on a non-logprob endpoint while the
verifier points at a logprob-capable deployment. The continuous scorer
has the correctness contract — every row needs
`expectations.expected_facts` or `expectations.expected_response` — and a
dataset that cannot satisfy it skips the path with the reason printed.

Verifier calls are real judge spend: the plan prints them before the run
(`rows × criteria × repeats + 1 probe`) and `budget.max_judge_calls`
covers them together with the integrity re-scoring calls.

## What lands on the MLflow run

| Field | Meaning |
|---|---|
| `correctness_continuous/mean` | the continuous score (per-row samples persist like any scorer's) |
| `correctness/mean` | the discrete path's score, unchanged, for comparison |
| `correctness_continuous/tie_rate`, `correctness/tie_rate` | fraction of row pairs each instrument cannot distinguish |
| `continuous/normalization_mass_mean`, `.../normalization_mass_min` | retained probability mass before renormalizing |
| `continuous/low_mass_rate`, `continuous/invalid_rate` | flagged and score-token-free calls |
| `continuous/judge_calls`, `continuous/input_tokens`, `continuous/output_tokens` | what the instrument cost |
| `continuous/fallback` | 1.0 when the backend forced the discrete fallback |
| params `continuous_granularity`, `continuous_repeats`, … | the instrument configuration, recorded per run |

All of it is report-only: no thresholds reference these metrics, the
scorer is not in the shared registry, and baselines do not record it. A
continuous scorer that *crashes* mid-run still fails the gate through
`correctness_continuous/error_count`, like any scorer — instrument
failures stay loud; only the detected-upfront capability gap degrades.

## The sweep: settling granularity and repeats empirically

The paper behind this path uses a 20-point scale; the wider literature
argues coarse scales align better with humans. The decision belongs to
data, not to a default:

```bash
# plumbing check + report format, no credentials, no spend
python scripts/sweep_continuous_scoring.py --simulate

# the real experiment
python scripts/sweep_continuous_scoring.py \
    --dataset evals/data/golden_cases.json --model judge-model \
    --granularities 5,10,20 --repeats 1,2,4 --output sweep.json
```

For every gold row the script builds **four graded candidates with a
known ordering** — the expected answer verbatim, a deterministic
paraphrase, a degraded half-answer with a fabricated rider, and another
row's answer — and reports per (granularity, K) combination the mean
Kendall tau-b against that ordering plus the tie rate, for the continuous
score *and* for the discrete parse of the same calls. Ranking agreement
tells you which configuration is accurate; the tie-rate gap tells you
what the logprob weighting buys over parsing the label. Normalization
mass and token counts complete the picture. Pick the cheapest
configuration whose tau stops improving, then pin it in `agentkit.yaml`.

The pivot-tournament best-of-N selection from the same framework is
deliberately not implemented yet: the scorer is validated first, the
selection loop second.

## Related documents

- `docs/agent-evaluation.md` — the comparison-first evaluation pipeline
  this path plugs into
- `docs/decisions/2026-08-30-continuous-logprob-scoring.md` — why it is
  built this way
