from aai_core import bootstrap
from aai_core.experiments import ExperimentManager


def main() -> None:
    ctx = bootstrap()
    experiments = ExperimentManager(
        experiment_name=ctx.settings.effective_experiment_name,
        context=ctx.tags,
    )

    with experiments.run(
        run_name="first-comparison",
        parameters={"candidate": "baseline"},
    ):
        print("Log the candidate's metrics and artifacts here.")


if __name__ == "__main__":
    main()
