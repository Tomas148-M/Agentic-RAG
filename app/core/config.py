from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROUTE_RETRIEVE = "retrieve"
ROUTE_UNSUPPORTED = "unsupported"
ROUTE_NODE = "route"
REWRITE_NODE = "rewrite_query"
RETRIEVE_NODE = "retrieve"
GRADE_CONTEXT_NODE = "grade_context"
GENERATE_NODE = "generate"
EVALUATE_NODE = "evaluate"
RETRY_QUERY_NODE = "retry_query"
NO_CONTEXT_NODE = "no_context"
UNSUPPORTED_NODE = "unsupported"
FAIL_NODE = "fail"
END_NODE = "__end__"
EVALUATION_PASS = "PASS"
CONTEXT_RELEVANT = "YES"
RAG_MODE_BASIC = "basic"
RAG_MODE_AGENTIC = "agentic"
DEFAULT_AGENTIC_MAX_RETRIES = 2
LANGGRAPH_RECURSION_LIMIT_BASE = 7
LANGGRAPH_RECURSION_LIMIT_PER_RETRY = 6
MAX_QUERY_LENGTH = 4000
CITATION_MARKER_PATTERN = r"\[(?:[A-Za-z]\d+|[Ss][\s+-]+\d+)[^\]]*\]"
VALID_CITATION_PATTERN = r"\[S([1-9]\d*)\]"
NO_DOCUMENTS_ANSWER = "No relevant documents were found."
UNSUPPORTED_REQUEST_ANSWER = (
    "I can only answer questions using the ingested documents."
)
ANSWER_ERROR_MESSAGE = "Unable to produce a supported answer"


class Settings(BaseSettings):
    """Application settings loaded from the environment and an optional .env file.

    Attributes:
        app_name (str): Application name exposed by the API.
        debug (bool): Whether FastAPI debug mode is enabled.
        llm_provider (str): Selected text generation provider.
        ollama_base_url (str): Ollama server address.
        ollama_model (str): Installed Ollama model name.
        codex_executable (str): Codex CLI executable on the server.
        codex_model (str | None): Optional Codex model override.
        llm_timeout_seconds (float): Maximum wait for provider responses.
        embedding_model (str): Sentence Transformers model name or local path.
        chroma_path (str): Directory for persistent Chroma storage.
        chroma_collection (str): Collection used for document chunks.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of shared characters between chunks.
        rag_mode (str): Chat orchestration mode, basic or agentic.
        agentic_max_retries (int): Maximum recovery attempts in the agent graph.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENTIC_RAG_", extra="ignore")

    app_name: str = "Agentic RAG"
    debug: bool = False
    llm_provider: Literal["ollama", "codex"] = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    codex_executable: str = "codex"
    codex_model: str | None = None
    llm_timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chroma_path: str = "data/chroma"
    chroma_collection: str = "documents"
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    rag_mode: Literal["basic", "agentic"] = "basic"
    agentic_max_retries: int = Field(
        default=DEFAULT_AGENTIC_MAX_RETRIES, ge=0, le=5
    )

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        """Validate the relationship between chunk size and overlap.

        Returns:
            Settings: The validated settings instance.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings for this process.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()


def get_agentic_recursion_limit(max_retries: int) -> int:
    """Calculate the maximum LangGraph node executions for both retry loops.

    Args:
        max_retries (int): Maximum retrieval and answer retries.

    Returns:
        int: LangGraph recursion limit with room for final nodes.
    """
    return (
        LANGGRAPH_RECURSION_LIMIT_BASE
        + LANGGRAPH_RECURSION_LIMIT_PER_RETRY * max_retries
    )
