from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROUTE_RETRIEVE = "retrieve"
ROUTE_UNSUPPORTED = "unsupported"
ROUTE_DOCUMENTS = "DOCUMENTS"
ROUTE_UNSUPPORTED_TOKEN = "UNSUPPORTED"
ROUTE_WEB = "WEB"
ROUTE_MCP = "MCP"
ROUTE_HYBRID = "HYBRID"
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
    "I can only answer information requests using the configured sources."
)
ANSWER_ERROR_MESSAGE = "Unable to produce a supported answer"
DEFAULT_WEB_SEARCH_ENDPOINT = "https://api.tavily.com/search"
DEFAULT_WEB_SEARCH_RESULTS = 5
MCP_PROTOCOL_VERSION = "2025-03-26"
MCP_CLIENT_NAME = "agentic-rag"
MCP_CLIENT_VERSION = "0.1.0"
MCP_JSONRPC_VERSION = "2.0"
MCP_JSONRPC_KEY = "jsonrpc"
MCP_METHOD_KEY = "method"
MCP_PARAMS_KEY = "params"
MCP_PROTOCOL_VERSION_KEY = "protocolVersion"
MCP_CAPABILITIES_KEY = "capabilities"
MCP_CLIENT_INFO_KEY = "clientInfo"
MCP_NAME_KEY = "name"
MCP_VERSION_KEY = "version"
MCP_TOOL_NAME_KEY = "name"
MCP_IS_ERROR_KEY = "isError"
MCP_ARGUMENTS_KEY = "arguments"
MCP_STRUCTURED_CONTENT_KEY = "structuredContent"
MCP_CONTENT_KEY = "content"
MCP_CONTENT_BLOCK_TYPE_KEY = "type"
MCP_CONTENT_BLOCK_TEXT_TYPE = "text"
MCP_CONTENT_BLOCK_TEXT_KEY = "text"
MCP_REQUEST_ID = 1
MCP_INITIALIZE_METHOD = "initialize"
MCP_INITIALIZED_METHOD = "notifications/initialized"
MCP_LIST_TOOLS_METHOD = "tools/list"
MCP_CALL_TOOL_METHOD = "tools/call"
HTTP_JSON_CONTENT_TYPE = "application/json"
MCP_SEARCH_SOURCE_PREFIX = "mcp://"
DEFAULT_TOOL_RESULT_CHARS = 12000
MCP_SESSION_HEADER = "Mcp-Session-Id"
MCP_PROTOCOL_HEADER = "MCP-Protocol-Version"
MCP_AUTHORIZATION_HEADER = "Authorization"
MCP_BEARER_PREFIX = "Bearer "
MCP_ACCEPT_HEADER = "Accept"
MCP_ACCEPT_VALUE = "application/json, text/event-stream"
MCP_CONTENT_TYPE_HEADER = "content-type"
MCP_SSE_CONTENT_TYPE = "text/event-stream"
MCP_SSE_DATA_PREFIX = "data:"
MCP_CURSOR_PARAM = "cursor"
MCP_CONTENT_TYPE_HEADER_NAME = "Content-Type"
HTTP_STATUS_ACCEPTED = 202
MCP_JSONRPC_ID_KEY = "id"
MCP_JSONRPC_ERROR_KEY = "error"
MCP_JSONRPC_RESULT_KEY = "result"
MCP_TOOLS_KEY = "tools"
MCP_NEXT_CURSOR_KEY = "nextCursor"
MCP_INPUT_SCHEMA_KEY = "inputSchema"
MCP_SCHEMA_PROPERTIES_KEY = "properties"
MCP_SCHEMA_REQUIRED_KEY = "required"
MCP_SCHEMA_ADDITIONAL_PROPERTIES_KEY = "additionalProperties"
MCP_SCHEMA_TYPE_KEY = "type"
MCP_SCHEMA_NULLABLE_KEY = "nullable"
MCP_SCHEMA_OBJECT_TYPE = "object"
ROUTE_SOURCE_OPTIONS = (ROUTE_DOCUMENTS, ROUTE_WEB, ROUTE_MCP)
WEB_SEARCH_API_KEY_FIELD = "api_key"
WEB_SEARCH_QUERY_KEY = "query"
WEB_SEARCH_MAX_RESULTS_FIELD = "max_results"
WEB_SEARCH_INCLUDE_ANSWER_FIELD = "include_answer"
WEB_SEARCH_RESULTS_KEY = "results"
WEB_SEARCH_TITLE_KEY = "title"
WEB_SEARCH_URL_KEY = "url"
WEB_SEARCH_CONTENT_KEY = "content"
MCP_ALLOWED_ARGUMENT_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}
HYBRID_SOURCES_KEY = "sources"
SOURCE_SELECTION_ROUTE_KEY = "route"
SOURCE_SELECTION_MCP_TOOL_KEY = "mcp_tool"
SOURCE_SELECTION_MCP_ARGUMENTS_KEY = "mcp_arguments"
MCP_UNAVAILABLE_ERROR = "MCP server is unavailable"
MCP_URL_MISSING_ERROR = "MCP server URL is not configured"
MCP_INVALID_RESULT_ERROR = "MCP server returned an invalid result"
MCP_INVALID_INITIALIZATION_ERROR = "MCP server returned invalid initialization data"
MCP_REQUEST_FAILED_ERROR = "MCP server request failed"
MCP_INVALID_JSONRPC_ERROR = "MCP server returned invalid JSON-RPC data"
MCP_JSONRPC_ERROR = "MCP server returned a JSON-RPC error"
MCP_MALFORMED_JSON_ERROR = "MCP server returned malformed JSON"
MCP_MALFORMED_SSE_ERROR = "MCP server returned malformed SSE"
MCP_NO_SSE_RESPONSE_ERROR = "MCP server returned no matching event data"
MCP_NOTIFICATION_FAILED_ERROR = "MCP initialization notification failed"
MCP_TOOL_NOT_AVAILABLE_ERROR = "Requested MCP tool is not allowed or available"
MCP_ARGUMENTS_MUST_BE_OBJECT_ERROR = "MCP tool arguments must be an object"
MCP_ARGUMENTS_INVALID_ERROR = "MCP tool arguments do not match its input schema"
MCP_CURSOR_LOOP_ERROR = "MCP server returned a repeated tools cursor"
WEB_SEARCH_DISABLED_ERROR = "Web search is disabled"
WEB_SEARCH_KEY_MISSING_ERROR = "Web search API key is not configured"
WEB_SEARCH_FAILED_ERROR = "Web search request failed"
RETRIEVAL_WEB_UNAVAILABLE_ERROR = "Web search is unavailable"
RETRIEVAL_MCP_FAILED_ERROR = "MCP tool execution failed"
RETRIEVAL_MCP_TOOL_MISSING_ERROR = "No MCP tool was selected"
RETRIEVAL_HYBRID_SELECTION_REQUIRED_ERROR = (
    "Hybrid retrieval requires an explicit source selection"
)


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
        web_search_enabled (bool): Whether web search may be used by the agent.
        web_search_api_key (str | None): API key for the Tavily search API.
        web_search_endpoint (str): Tavily-compatible search endpoint.
        web_search_max_results (int): Maximum results returned by web search.
        mcp_server_url (str | None): Streamable HTTP MCP server endpoint.
        mcp_auth_token (str | None): Optional bearer token for the MCP server.
        mcp_allowed_tools (list[str]): MCP tools explicitly allowed to run.
        tool_timeout_seconds (float): Timeout for external tool requests.
        tool_result_max_chars (int): Maximum text retained for each tool result.
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
    web_search_enabled: bool = False
    web_search_api_key: str | None = None
    web_search_endpoint: str = DEFAULT_WEB_SEARCH_ENDPOINT
    web_search_max_results: int = Field(
        default=DEFAULT_WEB_SEARCH_RESULTS, ge=1, le=20
    )
    mcp_server_url: str | None = None
    mcp_auth_token: str | None = None
    mcp_allowed_tools: list[str] = Field(default_factory=list)
    tool_timeout_seconds: float = Field(default=20.0, gt=0, allow_inf_nan=False)
    tool_result_max_chars: int = Field(default=DEFAULT_TOOL_RESULT_CHARS, gt=0)

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
