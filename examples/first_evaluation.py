from mlflow.genai.scorers import RelevanceToQuery

from aai_core.evaluation import EvaluationSuite, QualityThreshold

suite = EvaluationSuite(
    scorers=[RelevanceToQuery()],
    thresholds=[
        QualityThreshold(
            metric="relevance_to_query/mean",
            direction="higher",
            required=0.8,
        )
    ],
)

report = suite.run(
    data=[{"inputs": {"question": "What is the approved process?"}}],
    predict_fn=lambda question: "Replace with the application under test.",
)
report.require_passed()
