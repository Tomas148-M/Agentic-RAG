"""Deterministic answer and citation checks for a small RAG evaluation set."""

from typing import Any

from app.core.config import (
    EVALUATION_ANSWER_FIELD,
    EVALUATION_ANSWER_KEY,
    EVALUATION_ANSWER_MATCH_RATE_FIELD,
    EVALUATION_CASE_ID_PREFIX,
    EVALUATION_CASE_INVALID_ERROR,
    EVALUATION_CASES_PASSED_FIELD,
    EVALUATION_CASES_TOTAL_FIELD,
    EVALUATION_CITATION_COVERAGE_FIELD,
    EVALUATION_CITATION_KEY,
    EVALUATION_CITATION_SOURCE_FIELD,
    EVALUATION_CITATIONS_FIELD,
    EVALUATION_EXPECTED_PHRASES_FIELD,
    EVALUATION_EXPECTED_SOURCES_FIELD,
    EVALUATION_ID_FIELD,
    EVALUATION_PASS_RATE_KEY,
    EVALUATION_RESPONSE_INVALID_ERROR,
    EVALUATION_RESULT_FIELD,
)


def score_response(
    case: dict[str, Any], response: dict[str, Any], index: int
) -> dict[str, Any]:
    """Score one chat response against required phrases and citations.

    Args:
        case: Evaluation case containing expected answer phrases and sources.
        response: JSON-compatible response from ``POST /api/v1/chat``.
        index: One-based dataset index used when no case ID is supplied.

    Returns:
        A stable result object containing component metrics and pass status.

    Raises:
        TypeError: If a case or response field has the wrong type.
        ValueError: If a case or response field has an invalid value.
    """
    if not isinstance(case, dict):
        raise TypeError(EVALUATION_CASE_INVALID_ERROR)
    if not isinstance(response, dict):
        raise TypeError(EVALUATION_RESPONSE_INVALID_ERROR)
    answer = response.get(EVALUATION_ANSWER_FIELD)
    citations = response.get(EVALUATION_CITATIONS_FIELD)
    phrases = case.get(EVALUATION_EXPECTED_PHRASES_FIELD)
    expected_sources = case.get(EVALUATION_EXPECTED_SOURCES_FIELD, [])
    if not isinstance(answer, str) or not isinstance(citations, list):
        raise TypeError(EVALUATION_RESPONSE_INVALID_ERROR)
    if not isinstance(phrases, list):
        raise TypeError(EVALUATION_CASE_INVALID_ERROR)
    if not phrases:
        raise ValueError(EVALUATION_CASE_INVALID_ERROR)
    if any(not isinstance(phrase, str) for phrase in phrases):
        raise TypeError(EVALUATION_CASE_INVALID_ERROR)
    if any(not phrase for phrase in phrases):
        raise ValueError(EVALUATION_CASE_INVALID_ERROR)
    if not isinstance(expected_sources, list):
        raise TypeError(EVALUATION_CASE_INVALID_ERROR)
    if any(not isinstance(source, str) for source in expected_sources):
        raise TypeError(EVALUATION_CASE_INVALID_ERROR)
    if any(not source for source in expected_sources):
        raise ValueError(EVALUATION_CASE_INVALID_ERROR)

    answer_match = all(phrase.casefold() in answer.casefold() for phrase in phrases)
    citation_sources = [
        citation.get(EVALUATION_CITATION_SOURCE_FIELD)
        for citation in citations
        if isinstance(citation, dict)
        and isinstance(citation.get(EVALUATION_CITATION_SOURCE_FIELD), str)
    ]
    citation_match = bool(citation_sources)
    if expected_sources:
        citation_match = citation_match and all(
            any(expected.casefold() in source.casefold() for source in citation_sources)
            for expected in expected_sources
        )
    identifier = case.get(EVALUATION_ID_FIELD)
    if not isinstance(identifier, str) or not identifier:
        identifier = f"{EVALUATION_CASE_ID_PREFIX}{index}"
    return {
        EVALUATION_ID_FIELD: identifier,
        EVALUATION_ANSWER_KEY: answer_match,
        EVALUATION_CITATION_KEY: citation_match,
        EVALUATION_RESULT_FIELD: answer_match and citation_match,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, int | float]:
    """Calculate pass and citation coverage metrics for an evaluation run."""
    total = len(results)
    passed = sum(bool(result[EVALUATION_RESULT_FIELD]) for result in results)
    citation_count = sum(
        bool(result[EVALUATION_CITATION_KEY]) for result in results
    )
    answer_count = sum(bool(result[EVALUATION_ANSWER_KEY]) for result in results)
    return {
        EVALUATION_CASES_TOTAL_FIELD: total,
        EVALUATION_CASES_PASSED_FIELD: passed,
        EVALUATION_PASS_RATE_KEY: passed / total if total else 0.0,
        EVALUATION_CITATION_COVERAGE_FIELD: (
            citation_count / total if total else 0.0
        ),
        EVALUATION_ANSWER_MATCH_RATE_FIELD: (
            answer_count / total if total else 0.0
        ),
    }
