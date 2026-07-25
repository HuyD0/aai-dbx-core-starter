from __future__ import annotations

import json
from pathlib import Path

from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, Safety

from aai_core import bootstrap
from aai_core.agents import AgentRequest
from aai_core.evaluation import EvaluationSuite, QualityThreshold
from app.agent import RAGAgent

ROOT = Path(__file__).resolve().parents[1]


def predict_fn(question: str) -> str:
    response = RAGAgent().invoke(
        AgentRequest(messages=[{"role": "user", "content": question}])
    )
    return response.content


def main() -> None:
    context = bootstrap(ROOT / "aai-platform.yml")
    cases = json.loads((ROOT / "evals/data/release_cases.json").read_text())
    suite = EvaluationSuite(
        scorers=[RetrievalGroundedness(), RelevanceToQuery(), Safety()],
        thresholds=[
            QualityThreshold(
                metric="relevance_to_query/mean",
                direction="higher",
                required=0.8,
                max_regression=0.05,
            )
        ],
    )
    report = suite.run(data=cases, predict_fn=predict_fn)
    report.require_passed()
    print(
        {
            "application": context.tags.application,
            "release": context.tags.release,
            "metrics": report.metrics,
        }
    )


if __name__ == "__main__":
    main()
