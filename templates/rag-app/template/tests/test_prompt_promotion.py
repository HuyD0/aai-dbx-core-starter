"""The release helper must carry persisted evidence into guarded promotion."""

from types import SimpleNamespace

import scripts.promote_prompt as promotion


def test_promotion_passes_the_exact_decision_run(monkeypatch):
    calls = []
    context = SimpleNamespace(
        prompts=SimpleNamespace(
            promote=lambda *args, **kwargs: calls.append((args, kwargs))
        )
    )
    monkeypatch.setattr(promotion, "bootstrap", lambda _: context)
    monkeypatch.setattr(
        "sys.argv",
        [
            "promote_prompt.py",
            "--version",
            "7",
            "--decision-run-id",
            "decision-run-7",
            "--to",
            "validation",
        ],
    )

    promotion.main()

    assert calls == [
        (
            ("agent-system",),
            {
                "alias": "validation",
                "version": 7,
                "decision_run_id": "decision-run-7",
            },
        )
    ]
