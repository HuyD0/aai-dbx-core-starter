"""Full-gate output tests — failure triage must not echo case content."""

from types import SimpleNamespace

from app.triage import print_failure_triage


class Frame:
    def __init__(self, rows):
        self.rows = rows
        self.columns = sorted({column for row in rows for column in row})

    def iterrows(self):
        return enumerate(self.rows)


def test_failure_triage_hides_details_and_raw_content_by_default(capsys):
    result_df = Frame(
        [
            {
                "inputs/question": "private customer question",
                "outputs": "private model answer",
                "domain_policy/value": "no",
                "domain_policy/rationale": (
                    "The response disclosed personal contact information."
                ),
            },
            {
                "inputs/question": "another private question",
                "outputs": "another private answer",
                "domain_policy/value": "yes",
                "domain_policy/rationale": "The response followed policy.",
            },
        ]
    )
    report = SimpleNamespace(result_df=result_df)

    print_failure_triage(report)

    output = capsys.readouterr().out
    assert "row 1: domain_policy" in output
    assert "details omitted" in output
    assert "disclosed personal contact information" not in output
    assert "private customer question" not in output
    assert "private model answer" not in output
    assert "another private question" not in output
    assert "another private answer" not in output


def test_failure_triage_details_are_explicit_opt_in(capsys):
    result_df = Frame(
        [
            {
                "domain_policy/value": "no",
                "domain_policy/rationale": "A governed diagnostic rationale.",
            }
        ]
    )
    report = SimpleNamespace(result_df=result_df)

    print_failure_triage(report, include_details=True)

    output = capsys.readouterr().out
    assert "details enabled" in output
    assert "A governed diagnostic rationale." in output


def test_failure_triage_bounds_error_details_and_item_count(capsys):
    result_df = Frame(
        [
            {"domain_policy/error_message": "private\n" + "x" * 300},
            {"domain_policy/value": False},
        ]
    )

    print_failure_triage(
        SimpleNamespace(result_df=result_df),
        max_items=1,
        include_details=True,
    )

    output = capsys.readouterr().out
    assert "private x" in output
    assert "..." in output
    assert "1 additional failure(s) omitted" in output


def test_failure_triage_handles_missing_or_clean_result_frames(capsys):
    print_failure_triage(SimpleNamespace())
    print_failure_triage(SimpleNamespace(result_df=Frame([{"judge/value": "yes"}])))

    output = capsys.readouterr().out
    assert "unavailable (no result dataframe)" in output
    assert "no explicit scorer failures" in output
