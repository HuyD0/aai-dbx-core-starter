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
