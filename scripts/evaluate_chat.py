"""Run a deterministic question and citation benchmark against the chat API."""

import json
import sys
from pathlib import Path
from typing import Any

import httpx

from app.core.config import (
    EVALUATION_CASE_ID_PREFIX,
    EVALUATION_DATASET_ERROR,
    EVALUATION_ERROR_EXIT_CODE,
    EVALUATION_FAILURE_EXIT_CODE,
    EVALUATION_HTTP_CLIENT_ERROR,
    EVALUATION_HTTP_ERROR_TEMPLATE,
    EVALUATION_ID_FIELD,
    EVALUATION_JSON_DECODE_ERROR,
    EVALUATION_JSON_INDENT,
    EVALUATION_QUESTION_FIELD,
    EVALUATION_RESULT_FIELD,
    EVALUATION_RESULTS_FIELD,
    EVALUATION_SUCCESS_EXIT_CODE,
    EVALUATION_SUMMARY_FIELD,
    Settings,
    get_settings,
)
from app.observability.evaluation import score_response, summarize_results


def load_cases(dataset_path: Path) -> list[dict[str, Any]]:
    """Load evaluation cases from a JSON file."""
    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(EVALUATION_JSON_DECODE_ERROR) from exc
    if not isinstance(dataset, list) or not dataset:
        raise ValueError(EVALUATION_DATASET_ERROR)
    if any(not isinstance(case, dict) for case in dataset):
        raise ValueError(EVALUATION_DATASET_ERROR)
    return dataset


def run_evaluation(settings: Settings) -> dict[str, Any]:
    """Query the configured chat API and score every evaluation case."""
    cases = load_cases(Path(settings.evaluation_dataset_path))
    results: list[dict[str, Any]] = []
    case_id: Any = EVALUATION_CASE_ID_PREFIX
    try:
        with httpx.Client(timeout=settings.evaluation_timeout_seconds) as client:
            for index, case in enumerate(cases, start=1):
                case_id = case.get(EVALUATION_ID_FIELD, index)
                question = case.get(EVALUATION_QUESTION_FIELD)
                if not isinstance(question, str) or not question:
                    raise ValueError(EVALUATION_DATASET_ERROR)
                response = client.post(
                    settings.evaluation_endpoint,
                    json={EVALUATION_QUESTION_FIELD: question},
                )
                response.raise_for_status()
                results.append(
                    score_response(case, response.json(), index)
                )
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        raise RuntimeError(EVALUATION_HTTP_ERROR_TEMPLATE % (case_id, exc)) from exc
    return {
        EVALUATION_RESULTS_FIELD: results,
        EVALUATION_SUMMARY_FIELD: summarize_results(results),
    }


def main() -> int:
    """Run the configured benchmark and print its JSON report."""
    settings = get_settings()
    try:
        report = run_evaluation(settings)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"{EVALUATION_HTTP_CLIENT_ERROR}: {exc}", file=sys.stderr)
        return EVALUATION_ERROR_EXIT_CODE
    print(json.dumps(report, indent=EVALUATION_JSON_INDENT))
    all_passed = all(
        result[EVALUATION_RESULT_FIELD]
        for result in report[EVALUATION_RESULTS_FIELD]
    )
    return (
        EVALUATION_SUCCESS_EXIT_CODE
        if all_passed
        else EVALUATION_FAILURE_EXIT_CODE
    )


if __name__ == "__main__":
    raise SystemExit(main())
