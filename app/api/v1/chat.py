from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_chat_service
from app.integrations.llm.base import LLMProviderError
from app.rag.retriever import RetrievalError
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    """Answer a question from retrieved documents with source citations.

    Args:
        request (ChatRequest): The question to answer.
        service (ChatService): Configured document question-answering service.

    Returns:
        ChatResponse: Answer and its referenced sources.
    """
    try:
        return service.answer(request.question)
    except RetrievalError as exc:
        raise HTTPException(
            status_code=503, detail="Document retrieval unavailable"
        ) from exc
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=502, detail="Unable to generate a cited answer"
        ) from exc
