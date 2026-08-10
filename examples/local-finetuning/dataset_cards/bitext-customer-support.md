# Dataset card: Bitext customer-support structured output

## Identity and rights

| Field | Verified value |
|---|---|
| Name | Bitext Gen AI Chatbot Customer Support Dataset |
| Curriculum version | `bitext-customer-support-curated-v1` |
| Kaggle version | 1 |
| Owner | Bitext |
| Source | <https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset> |
| Accessible | Yes, public dataset API and download responded |
| Date accessed | 2026-07-31 |
| Source last updated | 2024-03-18T20:25:00.117Z |
| License | Community Data License Agreement – Sharing – Version 1.0 (`CDLA-Sharing-1.0`) |
| Permitted use | Computational use and publication subject to the agreement |
| Redistribution | Published Data and Enhanced Data remain under CDLA-Sharing-1.0 with required notices |
| Curriculum posture | Learning input only; not automatically cleared for commercial, enterprise, or production use |

The current license value came from Kaggle's dataset API and was checked against
the official CDLA agreement. It was not inferred from a tag or copied from a
third-party mirror.

## Source contents

The Kaggle archive contains one UTF-8 CSV:

`Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`

| Property | Value |
|---|---:|
| CSV bytes | 19,202,474 |
| Archive bytes observed | 3,007,665 |
| Records | 26,872 |
| File format | CSV |
| CSV SHA-256 | `6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434` |
| Archive SHA-256 | `2388c8303786e3ba2909e1bcddd9a9708fcd06a6e1c4bce43f126123fe7b8e51` |
| Language | English |
| Intents | 27 |
| Categories in the CSV | 11 |
| Missing values in five source columns | 0 |

The Kaggle description says 10 categories, but the current CSV contains 11:
`ACCOUNT`, `CANCEL`, `CONTACT`, `DELIVERY`, `FEEDBACK`, `INVOICE`, `ORDER`,
`PAYMENT`, `REFUND`, `SHIPPING`, and `SUBSCRIPTION`. The curriculum follows the
file it hashes and records this mismatch rather than silently rewriting source
labels.

## Columns

| Column | Meaning | Curriculum use |
|---|---|---|
| `flags` | Bitext language-generation variation tags | Difficulty and language-style slices |
| `instruction` | Customer-support utterance | Model input |
| `category` | Broad source category | Normalized lowercase target field |
| `intent` | One of 27 intent labels | Principal target |
| `response` | Example virtual-assistant response | Aggregate source-quality audit only; excluded from training targets |

`requires_escalation` is not a source column. It is a versioned curriculum
policy-derived label, not a fact asserted by Bitext.

The assistant response target is also curriculum-derived. Policy
`bitext-safe-response-v1` produces a brief intent-specific acknowledgement or,
for the four escalation intents, an explicit support-specialist handoff. This
keeps the first lab deterministic and measurable. It does not claim that the
source replies or generated wording are reviewed production support policy.

## Provenance and label quality

Bitext describes a hybrid process: natural texts supplied seeds, NLP extracted
those seeds, NLG expanded them, and computational linguists curated the steps.
The records must therefore be treated as mixed/hybrid rather than purely human
or purely synthetic. Responses appear generated or templated and are not
reviewed production support policy.

Intent labels are dense and nearly balanced (roughly 950–1,000 records per
intent), with no observed normalized-instruction duplicate group carrying
conflicting intent or category labels. Limitations include overlapping intents,
template artifacts, source-description/category drift, unresolved slots, and no
native escalation or unsupported-intent labels.

## Quality findings from the pinned CSV

| Finding | Result |
|---|---:|
| Exact full-row duplicates | 0 |
| Normalized duplicate rows beyond first occurrence | 2,501 (9.31%) |
| Normalized duplicate groups | 1,187 |
| Records belonging to duplicate groups | 3,785 (14.09%) |
| Largest duplicate group | 12 |
| Normalized-instruction duplicate groups with label conflicts | 0 |
| Audited near-duplicate pairs at threshold 0.90 | 29,419 |
| Near-duplicate clusters after split repair | 2,774 |
| Near/template clusters with conflicting labels (excluded) | 127 clusters / 3,254 records |
| Instructions, character length | min 6, median 48, max 92 |
| Records with Bitext `W` offensive-language flag | 1,288 |
| Records with `Z` error/typo flag | 5,286 |
| Records with `Q` colloquial flag | 8,968 |
| Records with `{{...}}` placeholder syntax | 13,041 |

The reproducible audit also reports inferred template families, near-duplicate
clusters, label and flag distributions, record/token-length proxies, invalid
records, sensitive-pattern counts, split integrity, and a dataset fingerprint.

## Sensitive-information review

The dataset is presented as generated/hybrid customer-support content, not as a
collection of customer records. Even so, pattern screening found 55 rows with
email-like strings, five with URL-like strings, one with a phone-like string,
and extensive named placeholder fields. These counts do not establish whether
the values identify real people. Processing masks email, URL, and phone-like
values before portable training records are written. Exploration reports counts
and masked samples only; it does not display large raw-content tables.

Offensive-language variants are retained only as an explicit evaluation slice.
Learners should decide whether their own policy requires additional filtering.

## Filtering and split methodology

The project deterministically selects 60 records per intent when the source
supports them: 40 training, 10 validation, and 10 frozen test records per
intent. That produces 1,080 training, 270 validation, and 270 test examples.

Selection uses stable content-derived IDs and seed 42. Exact normalized
utterances and inferred template families cannot cross splits. A near-duplicate
check runs after selection and fails preparation if a prohibited cross-split
pair remains. The split gate also rejects target leakage and any test/evaluation
ID reused in few-shot demonstrations. The CSV exposes no conversation ID,
customer/account ID, source document, template ID, or creation time; inferred
template grouping is explicitly a limitation, not a claimed source field.

The test membership is written to a versioned manifest with hashes. Any change
to frozen IDs creates a new curriculum dataset version.

## Framework-neutral transformation

Each selected record becomes one single-line chat JSON object with:

- stable `example_id`, source name, and source version;
- system, user, and assistant messages;
- intent, category, flags, split group, and policy metadata;
- a strict JSON assistant target.

MLX-LM reads the chat files directly. A future TRL/PEFT implementation can use
the same logical records. Source rows remain immutable; every mask,
normalization, derived label, selection, and split is implemented in code.

## Purpose, limitations, and prohibited uses

Original intended use: intent detection, fine-tuning, domain adaptation, and
virtual-assistant experimentation as described by the dataset owner.

Curriculum use: local structured-output classification, LoRA, prompt baselines,
schema enforcement, deterministic evaluation, slice analysis, response-policy
review, and MLflow evidence.

Known limitations:

- English only.
- Hybrid generated/curated provenance; not observed live support traffic.
- No native escalation, unsupported-intent, customer, conversation, template,
  timestamp, or source-document grouping fields.
- Example source responses may contain templates, policy gaps, or unsafe
  assumptions; they are audited in aggregate but excluded from model targets.
- No out-of-domain examples. Unsupported-intent evaluation detects
  out-of-vocabulary labels emitted on the in-domain frozen set; a future
  version should add a separately reviewed out-of-domain routing set.
- Kaggle metadata and the file disagree about category count.

Do not use this dataset to impersonate customer-support policy, make automated
high-impact decisions, handle real credentials, or claim production readiness.
Do not present public Kaggle availability as a commercial rights clearance.
Redistribution must follow CDLA-Sharing-1.0 and the notice in `DATA_LICENSE.md`.

## Maintainer and change history

Maintainer: AAI AI/ML Platform Team.

- 2026-07-31: Version 1 card created from live Kaggle metadata and direct CSV
  inspection; pinned source hashes and local-first split policy recorded.
