from unittest.mock import Mock

import langgraph._internal._config as langgraph_config
import pytest

from app.agents.graph import build_agentic_rag_graph
from app.core.config import Settings
from app.integrations.llm.base import LLMProviderError
from app.rag.retriever import RetrievedChunk


@pytest.fixture
def retrieved_chunk() -> RetrievedChunk:
    """Return one source chunk for deterministic workflow tests."""
    return RetrievedChunk("The capital is Prague.", "cities.txt", 0, 0.1)


def test_graph_routes_unsupported_request_without_retrieval() -> None:
    """Out-of-scope requests get a fixed response without searching documents."""
    llm = Mock()
    llm.generate.return_value = "UNSUPPORTED"
    retriever = Mock()

    result = build_agentic_rag_graph(retriever, llm, max_retries=1)("Hello")

    assert result["citations"] == []
    assert "configured sources" in result["answer"]
    assert "citations_valid" not in result
    assert "answer_grounded" not in result
    retriever.retrieve.assert_not_called()
    llm.generate.assert_called_once()


def test_graph_rewrites_grades_and_returns_evaluated_citations(
    retrieved_chunk: RetrievedChunk,
) -> None:
    """A cited answer is returned only after the evaluator approves it."""
    retriever = Mock()
    retriever.retrieve.return_value = [retrieved_chunk]
    llm = Mock()
    llm.generate.side_effect = [
        "RETRIEVE",
        "capital of Czechia",
        "YES",
        "Prague is the capital. [S1]",
        "PASS",
    ]

    result = build_agentic_rag_graph(retriever, llm, max_retries=1)(
        "What is the capital?"
    )

    assert result["answer"] == "Prague is the capital. [S1]"
    assert result["citations"][0].source == "cities.txt"
    retriever.retrieve.assert_called_once_with("capital of Czechia")
    assert llm.generate.call_count == 5


def test_graph_retries_retrieval_then_returns_no_context() -> None:
    """Unhelpful excerpts trigger bounded query rewrites and no false answer."""
    retriever = Mock()
    retriever.retrieve.return_value = [
        RetrievedChunk("Unrelated content", "other.txt", 0, 0.8)
    ]
    llm = Mock()
    llm.generate.side_effect = [
        "RETRIEVE",
        "first search",
        "NO",
        "second search",
        "NO",
    ]

    result = build_agentic_rag_graph(retriever, llm, max_retries=1)("Question")

    assert result["answer"] == "No relevant documents were found."
    assert result["citations"] == []
    assert "citations_valid" not in result
    assert "answer_grounded" not in result
    assert [call.args[0] for call in retriever.retrieve.call_args_list] == [
        "first search",
        "second search",
    ]


def test_graph_fails_after_answer_evaluation_retry_limit(
    retrieved_chunk: RetrievedChunk,
) -> None:
    """Repeated evaluator rejection becomes a provider error, not an answer."""
    retriever = Mock()
    retriever.retrieve.return_value = [retrieved_chunk]
    llm = Mock()
    llm.generate.side_effect = [
        "RETRIEVE",
        "capital",
        "YES",
        "Prague is the capital. [S1]",
        "FAIL",
        "Prague is the capital. [S1]",
        "FAIL",
    ]

    with pytest.raises(LLMProviderError):
        build_agentic_rag_graph(retriever, llm, max_retries=1)("Question")

    assert llm.generate.call_count == 7


def test_graph_retries_supported_but_off_topic_answer(
    retrieved_chunk: RetrievedChunk,
) -> None:
    """Evaluator rejection of an off-topic answer produces a focused retry."""
    question = "What is the capital of Czechia?"
    retriever = Mock()
    retriever.retrieve.return_value = [retrieved_chunk]
    llm = Mock()
    llm.generate.side_effect = [
        "RETRIEVE",
        "Czechia capital",
        "YES",
        "The excerpt discusses a European city. [S1]",
        "FAIL",
        "Prague is the capital of Czechia. [S1]",
        "PASS",
    ]

    result = build_agentic_rag_graph(retriever, llm, max_retries=1)(question)

    assert result["answer"] == "Prague is the capital of Czechia. [S1]"
    evaluator_prompt = llm.generate.call_args_list[4].args[0]
    assert question in evaluator_prompt
    assert "Reject answers that are supported but off-topic" in evaluator_prompt
    retry_prompt = llm.generate.call_args_list[5].args[0]
    assert "Answer the original question directly" in retry_prompt


def test_graph_max_retry_path_overrides_low_langgraph_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Five retrieval retries finish even when LangGraph's default limit is low."""
    monkeypatch.setattr(langgraph_config, "DEFAULT_RECURSION_LIMIT", 2)
    retriever = Mock()
    retriever.retrieve.return_value = [
        RetrievedChunk("Unrelated content", "other.txt", 0, 0.8)
    ]
    llm = Mock()
    llm.generate.side_effect = [
        "RETRIEVE",
        *(item for _ in range(6) for item in ("search query", "NO")),
    ]

    result = build_agentic_rag_graph(retriever, llm, max_retries=5)("Question")

    assert result["answer"] == "No relevant documents were found."
    assert retriever.retrieve.call_count == 6


def test_agentic_settings_validate_retry_limit() -> None:
    """Agent retry count is selectable and bounded by application settings."""
    settings = Settings(
        rag_mode="agentic", agentic_max_retries=3, _env_file=None
    )
    assert settings.rag_mode == "agentic"
    assert settings.agentic_max_retries == 3
    with pytest.raises(ValueError):
        Settings(agentic_max_retries=6, _env_file=None)


@pytest.mark.parametrize("max_retries", [0, 1, 2])
def test_graph_retrieval_limit_counts_retries_after_initial_search(
    max_retries: int,
) -> None:
    """Each retry limit permits the initial search plus that many retries."""
    retriever = Mock()
    retriever.retrieve.return_value = [
        RetrievedChunk("Unrelated content", "other.txt", 0, 0.8)
    ]
    llm = Mock()
    llm.generate.side_effect = [
        "RETRIEVE",
        *(item for _ in range(max_retries + 1) for item in ("search", "NO")),
    ]

    result = build_agentic_rag_graph(retriever, llm, max_retries)("Question")

    assert result["answer"] == "No relevant documents were found."
    assert retriever.retrieve.call_count == max_retries + 1


@pytest.mark.parametrize("max_retries", [0, 1, 2])
def test_graph_generation_limit_includes_initial_attempt(
    max_retries: int,
    retrieved_chunk: RetrievedChunk,
) -> None:
    """Each generation limit permits one initial attempt plus its retries."""
    retriever = Mock()
    retriever.retrieve.return_value = [retrieved_chunk]
    llm = Mock()
    llm.generate.side_effect = [
        "RETRIEVE",
        "capital",
        "YES",
        *(
            item
            for _ in range(max_retries + 1)
            for item in ("Prague is the capital. [S1]", "FAIL")
        ),
    ]

    with pytest.raises(LLMProviderError):
        build_agentic_rag_graph(retriever, llm, max_retries)("Question")

    assert retriever.retrieve.call_count == 1
    assert llm.generate.call_count == 3 + (2 * (max_retries + 1))


@pytest.mark.parametrize("max_retries", [0, 1, 2])
def test_graph_can_pass_evaluation_on_last_generation_attempt(
    max_retries: int,
    retrieved_chunk: RetrievedChunk,
) -> None:
    """A passing answer on the final permitted generation is returned."""
    retriever = Mock()
    retriever.retrieve.return_value = [retrieved_chunk]
    llm = Mock()
    llm.generate.side_effect = [
        "RETRIEVE",
        "capital",
        "YES",
        *(
            item
            for attempt in range(max_retries + 1)
            for item in (
                "Prague is the capital. [S1]",
                "PASS" if attempt == max_retries else "FAIL",
            )
        ),
    ]

    result = build_agentic_rag_graph(retriever, llm, max_retries)("Question")

    assert result["answer"] == "Prague is the capital. [S1]"
    assert llm.generate.call_count == 3 + (2 * (max_retries + 1))


@pytest.mark.parametrize("max_retries", [0, 1, 2])
def test_graph_retrieval_and_generation_limits_are_independent(
    max_retries: int,
    retrieved_chunk: RetrievedChunk,
) -> None:
    """Both workflows can reach their final allowed attempt independently."""
    retriever = Mock()
    retriever.retrieve.side_effect = [
        *(
            [RetrievedChunk("Unrelated content", "other.txt", 0, 0.8)]
            for _ in range(max_retries)
        ),
        [retrieved_chunk],
    ]
    llm = Mock()
    llm.generate.side_effect = [
        "RETRIEVE",
        *(
            item
            for attempt in range(max_retries + 1)
            for item in (
                f"search {attempt}",
                "NO" if attempt < max_retries else "YES",
            )
        ),
        *(
            item
            for attempt in range(max_retries + 1)
            for item in (
                "Prague is the capital. [S1]",
                "PASS" if attempt == max_retries else "FAIL",
            )
        ),
    ]

    result = build_agentic_rag_graph(retriever, llm, max_retries)("Question")

    assert result["answer"] == "Prague is the capital. [S1]"
    assert result["retrieval_retries"] == max_retries
    assert result["generation_attempts"] == max_retries + 1
    assert retriever.retrieve.call_count == max_retries + 1
