# Verified curriculum dataset catalog

This catalog records what was visible from the current public Kaggle metadata
and file listings on 2026-07-31. Only Bitext is used by the required runnable
lab. Optional datasets remain unsuitable for a lab until their files receive
the same full inspection, rights review, quality audit, and dataset card.

| Phase | Current title and owner | Version; size; files | Kaggle license | Status |
|---|---|---|---|---|
| 1 | [Bitext Gen AI Chatbot Customer Support Dataset](https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset), Bitext | v1; 19,202,474 bytes; one CSV | CDLA-Sharing-1.0 | Required; verified, inspected, and pinned |
| 2 | [Customer IT Support - Ticket Dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets), Tobias Bueck | v14; 61,248,472 bytes; five CSV files | CC BY 4.0 | Optional until current files are inspected |
| Advanced | [Invoice NER Dataset](https://www.kaggle.com/datasets/nikitpatel/invoice-ner-dataset), NikitPatel | v1; 24,062 bytes; one XLSX | CC BY-NC-SA 4.0 | Optional, non-commercial terms; text-only warning applies |
| Advanced | [Prompt Engineering Dataset](https://www.kaggle.com/datasets/austinfairbanks/prompt-engineering-dataset), Austin Fairbanks | v3; 5,899,721 bytes; five CSV files | MIT in Kaggle metadata | Optional; current files differ from the original v1 description |
| Advanced | [Dataset for fine-tuning LLM to generate MiniZinc](https://www.kaggle.com/datasets/robertopenco/dataset-for-fine-tuning-llm-to-generate-minizinc), Roberto Penco | v3; 345,596 bytes; three XLSX and two README files | CC BY 4.0 | Optional; requires local MiniZinc execution evaluation |

## Phase 2 guardrails

The current ticket page describes queue, priority, language, subject, body,
answer, type, business type, and tags, but the five current CSV files have not
been imported into this project. A future adapter must confirm their exact
columns and languages rather than copying the prose schema. It should start with
English-only queue, priority, type, and human-review routing. It must group on
only dimensions the files actually contain; no customer, source, thread,
template, duplicate-cluster, or time grouping may be claimed without evidence.

Multilingual inference, response generation, and multi-label tags remain
advanced extensions after the English task works.

## Invoice guardrails

The current file is an XLSX described as containing `Input` raw extracted text
and `Final_Output` structured JSON. No image or bounding-box file appeared in
the public file listing. A text-only model cannot process invoice images; an OCR
or multimodal stage would be a separate advanced module. The non-commercial and
share-alike license, sensitive invoice content, field normalization, missing
fields, hallucinations, and provenance all require review before use.

## Prompt-transformation guardrails

The current Kaggle release is version 3 and exposes five CSV files rather than
only the original 1,000-row file described in version 1. Kaggle labels the data
synthetic. Open-ended transformations need rubric and human calibration; they
are not an introductory deterministic-scoring project.

## MiniZinc guardrails

The current release uses XLSX source files. Evaluation must parse, compile, and
execute generated MiniZinc against test cases, then measure constraint
satisfaction, objective value, runtime, and invalid code. Exact text match or an
LLM judge alone is insufficient, and the task may exceed a 300M–600M model.
