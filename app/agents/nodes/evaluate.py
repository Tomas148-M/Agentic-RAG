from collections.abc import Callable

from app.agents.prompts.evaluator import (
    EVALUATION_RETRY_FEEDBACK,
    build_answer_evaluator_prompt,
)
from app.agents.state import AgentState
from app.core.config import EVALUATION_PASS
from app.integrations.llm.base import LLMProvider


def create_evaluate_node(
    llm: LLMProvider,
) -> Callable[[AgentState], AgentState]:
    """Create a node that checks answer grounding and citation validity.

    Args:
        llm (LLMProvider): Model used to evaluate answer support.

    Returns:
        Callable[[AgentState], AgentState]: LangGraph answer evaluation node.
    """

    def evaluate(state: AgentState) -> AgentState:
        chunks = state.get("relevant_chunks", [])
        excerpts = [
            {"id": f"S{index}", "text": chunk.text}
            for index, chunk in enumerate(chunks, start=1)
        ]
        if not state.get("answer_valid", False):
            return {"answer_valid": False}
        result = llm.generate(
            build_answer_evaluator_prompt(
                state["question"], state.get("answer", ""), excerpts,
                state.get("feedback", ""),
            )
        )
        passed = result.strip().upper() == EVALUATION_PASS
        return {
            "answer_valid": passed,
            "generation_feedback": (
                "" if passed else EVALUATION_RETRY_FEEDBACK
            ),
        }

    return evaluate
