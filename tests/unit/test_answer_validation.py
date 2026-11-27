from unittest.mock import Mock

import pytest
from langgraph.graph import END, START, StateGraph

from app.agents.edges.routing import route_after_evaluation
from app.agents.nodes.evaluate import create_evaluate_node
from app.agents.nodes.generate import create_generate_node
from app.agents.state import AgentState
from app.core.config import END_NODE, FAIL_NODE, GENERATE_NODE
from app.rag.retriever import RetrievedChunk


@pytest.fixture
def chunk() -> RetrievedChunk:
    """Return a source passage with stable citation metadata."""
    return RetrievedChunk("Prague is the capital.", "cities.txt", 0, 0.1)


def test_evaluator_skips_llm_when_citations_are_invalid(
    chunk: RetrievedChunk,
) -> None:
    """Invalid citation syntax or IDs cannot be approved by the evaluator."""
    llm = Mock()
    evaluate = create_evaluate_node(llm)

    result = evaluate(
        {
            "question": "What is the capital?",
            "answer": "Prague [S2]",
            "relevant_chunks": [chunk],
            "citations_valid": False,
        }
    )

    assert result == {"answer_grounded": False}
    llm.generate.assert_not_called()


@pytest.mark.parametrize(
    ("evaluation", "expected"), [("FAIL", False), ("PASS", True)]
)
def test_evaluator_sets_grounding_from_model_result(
    chunk: RetrievedChunk,
    evaluation: str,
    expected: bool,
) -> None:
    """A valid citation set is grounded only when the evaluator passes it."""
    llm = Mock()
    llm.generate.return_value = evaluation
    evaluate = create_evaluate_node(llm)

    result = evaluate(
        {
            "question": "What is the capital?",
            "answer": "Prague [S1]",
            "relevant_chunks": [chunk],
            "citations_valid": True,
        }
    )

    assert result["answer_grounded"] is expected
    llm.generate.assert_called_once()


@pytest.mark.parametrize(
    ("answer", "expected_citations_valid"),
    [("Prague [S1]", True), ("Prague [S2]", False)],
)
def test_generation_resets_stale_grounding_in_merged_graph_state(
    chunk: RetrievedChunk,
    answer: str,
    expected_citations_valid: bool,
) -> None:
    """Every new answer clears prior grounding approval in merged graph state."""
    llm = Mock()
    llm.generate.return_value = answer
    workflow = StateGraph(AgentState)
    workflow.add_node("generate", create_generate_node(llm))
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", END)
    graph = workflow.compile()

    result = graph.invoke(
        {
            "question": "What is the capital?",
            "relevant_chunks": [chunk],
            "answer_grounded": True,
        }
    )

    assert result["citations_valid"] is expected_citations_valid
    assert result["answer_grounded"] is False


@pytest.mark.parametrize(
    ("grounded", "expected"),
    [(True, END_NODE), (False, GENERATE_NODE)],
)
def test_evaluation_routing_uses_grounding_flag(
    grounded: bool,
    expected: str,
) -> None:
    """Only evaluator approval ends the graph successfully."""
    assert route_after_evaluation(
        {"answer_grounded": grounded, "generation_attempts": 0}, 0
    ) == expected


def test_evaluation_routing_fails_after_grounding_retries_are_exhausted() -> None:
    """An ungrounded answer fails after the final generation attempt."""
    assert route_after_evaluation(
        {"answer_grounded": False, "generation_attempts": 1}, 0
    ) == FAIL_NODE
