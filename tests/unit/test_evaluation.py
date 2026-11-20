from pathlib import Path
from unittest.mock import Mock

import pytest
from typing_extensions import Self

from app.core.config import Settings
from app.observability.evaluation import score_response, summarize_results
from scripts.evaluate_chat import run_evaluation


def test_score_response_requires_answer_phrase_and_citation() -> None:
    """Mark a response successful when phrase and citation checks pass."""
    case = {
        "id": "capital",
        "expected_answer_phrases": ["Prague"],
        "expected_citation_sources": ["cities.txt"],
    }
    response = {
        "answer": "Prague is the capital. [S1]",
        "citations": [{"source": "data/cities.txt"}],
    }

    result = score_response(case, response, index=1)

    assert result == {
        "id": "capital",
        "answer_match": True,
        "citation_match": True,
        "passed": True,
    }


def test_score_response_fails_when_citation_is_missing() -> None:
    """Reject uncited answers even when expected phrases are present."""
    case = {
        "expected_answer_phrases": ["Prague"],
        "expected_citation_sources": [],
    }

    result = score_response(
        case,
        {"answer": "Prague is the capital.", "citations": []},
        index=3,
    )

    assert result["id"] == "case-3"
    assert result["answer_match"] is True
    assert result["citation_match"] is False
    assert result["passed"] is False


def test_score_response_rejects_cases_without_expected_phrases() -> None:
    """Reject evaluation cases that cannot measure answer relevance."""
    with pytest.raises(ValueError):
        score_response(
            {"expected_answer_phrases": []},
            {"answer": "text", "citations": []},
            index=1,
        )


def test_summarize_results_reports_answer_citation_and_pass_rates() -> None:
    """Aggregate answer match, citation, and overall pass metrics."""
    results = [
        {"passed": True, "answer_match": True, "citation_match": True},
        {"passed": False, "answer_match": True, "citation_match": False},
    ]

    summary = summarize_results(results)

    assert summary["cases_total"] == 2
    assert summary["cases_passed"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["answer_match_rate"] == 1.0
    assert summary["citation_coverage"] == 0.5


def test_evaluation_runner_translates_wrong_typed_case_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Turn malformed evaluation fields into a handled runner error."""
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(
        '[{"question":"question",'
        '"expected_answer_phrases":"not-a-list"}]',
        encoding="utf-8",
    )
    response = Mock()
    response.json.return_value = {"answer": "answer", "citations": []}
    response.raise_for_status.return_value = None

    class FakeClient:
        """Return a fixed response without making an HTTP request."""

        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def post(self, url: str, json: dict[str, str]) -> Mock:
            return response

    monkeypatch.setattr("scripts.evaluate_chat.httpx.Client", FakeClient)

    with pytest.raises(RuntimeError, match="Case 1 request failed"):
        run_evaluation(
            Settings(evaluation_dataset_path=str(dataset_path))
        )
