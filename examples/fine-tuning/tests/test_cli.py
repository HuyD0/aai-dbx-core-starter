import json

import pytest

from aai_fine_tuning.cli import main


def test_memory_command_prints_a_complete_estimate(capsys):
    assert main(["memory", "--parameters-billions", "70"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["parameters_billions"] == 70.0
    assert record["weights_gb"] == pytest.approx(140.0)
    assert record["total_gb"] == pytest.approx(
        record["weights_gb"]
        + record["gradients_gb"]
        + record["optimizer_gb"]
        + record["activations_gb"]
    )
    assert 0 < record["weights_share"] < 1


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        main([])
