import json
from typing import Any

import httpx

from app.core.config import (
    HTTP_JSON_CONTENT_TYPE,
    HTTP_STATUS_ACCEPTED,
    MCP_ACCEPT_HEADER,
    MCP_ACCEPT_VALUE,
    MCP_ALLOWED_ARGUMENT_TYPES,
    MCP_ARGUMENTS_INVALID_ERROR,
    MCP_ARGUMENTS_KEY,
    MCP_ARGUMENTS_MUST_BE_OBJECT_ERROR,
    MCP_AUTHORIZATION_HEADER,
    MCP_BEARER_PREFIX,
    MCP_CALL_TOOL_METHOD,
    MCP_CAPABILITIES_KEY,
    MCP_CLIENT_INFO_KEY,
    MCP_CLIENT_NAME,
    MCP_CLIENT_VERSION,
    MCP_CONTENT_TYPE_HEADER,
    MCP_CONTENT_TYPE_HEADER_NAME,
    MCP_CURSOR_LOOP_ERROR,
    MCP_CURSOR_PARAM,
    MCP_INITIALIZE_METHOD,
    MCP_INITIALIZED_METHOD,
    MCP_INVALID_INITIALIZATION_ERROR,
    MCP_INVALID_JSONRPC_ERROR,
    MCP_INVALID_RESULT_ERROR,
    MCP_IS_ERROR_KEY,
    MCP_JSONRPC_ERROR,
    MCP_JSONRPC_ERROR_KEY,
    MCP_JSONRPC_ID_KEY,
    MCP_JSONRPC_KEY,
    MCP_JSONRPC_RESULT_KEY,
    MCP_JSONRPC_VERSION,
    MCP_LIST_TOOLS_METHOD,
    MCP_MALFORMED_JSON_ERROR,
    MCP_MALFORMED_SSE_ERROR,
    MCP_METHOD_KEY,
    MCP_NAME_KEY,
    MCP_NEXT_CURSOR_KEY,
    MCP_NO_SSE_RESPONSE_ERROR,
    MCP_NOTIFICATION_FAILED_ERROR,
    MCP_PARAMS_KEY,
    MCP_PROTOCOL_HEADER,
    MCP_PROTOCOL_VERSION,
    MCP_PROTOCOL_VERSION_KEY,
    MCP_REQUEST_FAILED_ERROR,
    MCP_REQUEST_ID,
    MCP_SCHEMA_ADDITIONAL_PROPERTIES_KEY,
    MCP_SCHEMA_NULLABLE_KEY,
    MCP_SCHEMA_OBJECT_TYPE,
    MCP_SCHEMA_PROPERTIES_KEY,
    MCP_SCHEMA_REQUIRED_KEY,
    MCP_SCHEMA_TYPE_KEY,
    MCP_SESSION_HEADER,
    MCP_SSE_CONTENT_TYPE,
    MCP_SSE_DATA_PREFIX,
    MCP_TOOLS_KEY,
    MCP_URL_MISSING_ERROR,
    MCP_VERSION_KEY,
    Settings,
)


class MCPError(Exception):
    """Raised when the MCP server request fails."""


class MCPClient:
    """Minimal synchronous MCP Streamable HTTP client."""

    def __init__(self, settings: Settings) -> None:
        if not settings.mcp_server_url:
            raise MCPError(MCP_URL_MISSING_ERROR)
        self.url = settings.mcp_server_url
        self.timeout = settings.tool_timeout_seconds
        self.auth_token = settings.mcp_auth_token
        self.session_id: str | None = None
        self.initialized = False
        self.request_id = MCP_REQUEST_ID - 1

    def list_tools(self) -> list[dict[str, Any]]:
        """Initialize the connection and return every tool page."""
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            params = {MCP_CURSOR_PARAM: cursor} if cursor is not None else {}
            payload = self._request(MCP_LIST_TOOLS_METHOD, params)
            page_tools = payload.get(MCP_TOOLS_KEY, [])
            if isinstance(page_tools, list):
                tools.extend(tool for tool in page_tools if isinstance(tool, dict))
            next_cursor = payload.get(MCP_NEXT_CURSOR_KEY)
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            if next_cursor in seen_cursors:
                raise MCPError(MCP_CURSOR_LOOP_ERROR)
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        return tools

    def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call an advertised server tool and return its result payload."""
        payload = self._request(
            MCP_CALL_TOOL_METHOD,
            {MCP_NAME_KEY: name, MCP_ARGUMENTS_KEY: arguments},
        )
        if payload.get(MCP_IS_ERROR_KEY):
            raise MCPError(MCP_JSONRPC_ERROR)
        return payload

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.initialized:
            self._initialize()
        result = self._send(method, params)
        if not isinstance(result, dict):
            raise MCPError(MCP_INVALID_RESULT_ERROR)
        return result

    def _initialize(self) -> None:
        payload = self._send(
            MCP_INITIALIZE_METHOD,
            {
                MCP_PROTOCOL_VERSION_KEY: MCP_PROTOCOL_VERSION,
                MCP_CAPABILITIES_KEY: {},
                MCP_CLIENT_INFO_KEY: {
                    MCP_NAME_KEY: MCP_CLIENT_NAME,
                    MCP_VERSION_KEY: MCP_CLIENT_VERSION,
                },
            },
        )
        if not isinstance(payload, dict):
            raise MCPError(MCP_INVALID_INITIALIZATION_ERROR)
        self.initialized = True
        self._send_notification(MCP_INITIALIZED_METHOD)

    def _send(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any] | list[Any] | None:
        self.request_id += 1
        request_id = self.request_id
        body = {
            MCP_JSONRPC_KEY: MCP_JSONRPC_VERSION,
            MCP_JSONRPC_ID_KEY: request_id,
            MCP_METHOD_KEY: method,
            MCP_PARAMS_KEY: params,
        }
        headers = self._headers()
        try:
            response = httpx.post(
                self.url,
                json=body,
                headers=headers,
                timeout=self.timeout,
            )
            if response.headers.get(MCP_SESSION_HEADER):
                self.session_id = response.headers[MCP_SESSION_HEADER]
            response.raise_for_status()
        except (httpx.HTTPError, RuntimeError) as exc:
            raise MCPError(MCP_REQUEST_FAILED_ERROR) from exc
        if response.status_code == HTTP_STATUS_ACCEPTED or not response.content:
            return None
        response_data = self._parse_response(response, request_id)
        if not isinstance(response_data, dict):
            raise MCPError(MCP_INVALID_JSONRPC_ERROR)
        if response_data.get(MCP_JSONRPC_ID_KEY) != request_id:
            raise MCPError(MCP_INVALID_JSONRPC_ERROR)
        error = response_data.get(MCP_JSONRPC_ERROR_KEY)
        if error:
            raise MCPError(MCP_JSONRPC_ERROR)
        return response_data.get(MCP_JSONRPC_RESULT_KEY)

    def _send_notification(self, method: str) -> None:
        body = {
            MCP_JSONRPC_KEY: MCP_JSONRPC_VERSION,
            MCP_METHOD_KEY: method,
        }
        try:
            response = httpx.post(
                self.url,
                json=body,
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except (httpx.HTTPError, RuntimeError) as exc:
            raise MCPError(MCP_NOTIFICATION_FAILED_ERROR) from exc

    def _headers(self) -> dict[str, str]:
        headers = {
            MCP_ACCEPT_HEADER: MCP_ACCEPT_VALUE,
            MCP_CONTENT_TYPE_HEADER_NAME: HTTP_JSON_CONTENT_TYPE,
            MCP_PROTOCOL_HEADER: MCP_PROTOCOL_VERSION,
        }
        if self.session_id:
            headers[MCP_SESSION_HEADER] = self.session_id
        if self.auth_token:
            headers[MCP_AUTHORIZATION_HEADER] = f"{MCP_BEARER_PREFIX}{self.auth_token}"
        return headers

    @staticmethod
    def _parse_response(response: httpx.Response, request_id: int) -> Any:
        content_type = response.headers.get(MCP_CONTENT_TYPE_HEADER, "")
        if MCP_SSE_CONTENT_TYPE not in content_type:
            try:
                response_data = response.json()
            except ValueError as exc:
                raise MCPError(MCP_MALFORMED_JSON_ERROR) from exc
            return response_data

        event_lines: list[str] = []
        events: list[str] = []
        for line in response.text.splitlines():
            if line:
                event_lines.append(line)
            elif event_lines:
                events.append("\n".join(event_lines))
                event_lines = []
        if event_lines:
            events.append("\n".join(event_lines))

        for event in events:
            data_lines = [
                line[len(MCP_SSE_DATA_PREFIX):].lstrip()
                for line in event.splitlines()
                if line.startswith(MCP_SSE_DATA_PREFIX)
            ]
            if not data_lines:
                continue
            try:
                event_data = json.loads("\n".join(data_lines))
            except ValueError as exc:
                raise MCPError(MCP_MALFORMED_SSE_ERROR) from exc
            if (
                isinstance(event_data, dict)
                and event_data.get(MCP_JSONRPC_ID_KEY) == request_id
            ):
                return event_data
        raise MCPError(MCP_NO_SSE_RESPONSE_ERROR)


def validate_tool_arguments(
    arguments: dict[str, Any], schema: dict[str, Any]
) -> None:
    """Validate simple JSON Schema object constraints used by MCP tools."""
    if not isinstance(arguments, dict):
        raise TypeError(MCP_ARGUMENTS_MUST_BE_OBJECT_ERROR)
    if schema.get(MCP_SCHEMA_TYPE_KEY, MCP_SCHEMA_OBJECT_TYPE) != MCP_SCHEMA_OBJECT_TYPE:
        raise TypeError(MCP_ARGUMENTS_INVALID_ERROR)
    properties = schema.get(MCP_SCHEMA_PROPERTIES_KEY, {})
    required = schema.get(MCP_SCHEMA_REQUIRED_KEY, [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise TypeError(MCP_ARGUMENTS_INVALID_ERROR)
    if any(key not in arguments for key in required):
        raise ValueError(MCP_ARGUMENTS_INVALID_ERROR)
    if schema.get(MCP_SCHEMA_ADDITIONAL_PROPERTIES_KEY) is False and any(
        key not in properties for key in arguments
    ):
        raise ValueError(MCP_ARGUMENTS_INVALID_ERROR)
    for key, value in arguments.items():
        property_schema = properties.get(key)
        if not isinstance(property_schema, dict):
            continue
        schema_types = property_schema.get(MCP_SCHEMA_TYPE_KEY)
        if isinstance(schema_types, str):
            schema_types = [schema_types]
        if not isinstance(schema_types, list):
            continue
        if value is None:
            if "null" in schema_types or property_schema.get(
                MCP_SCHEMA_NULLABLE_KEY
            ) is True:
                continue
            raise ValueError(MCP_ARGUMENTS_INVALID_ERROR)
        expected_types: list[type] = []
        for schema_type in schema_types:
            if not isinstance(schema_type, str):
                continue
            if schema_type == "null":
                continue
            expected_type = MCP_ALLOWED_ARGUMENT_TYPES.get(schema_type)
            if isinstance(expected_type, tuple):
                expected_types.extend(expected_type)
            elif expected_type is not None:
                expected_types.append(expected_type)
        if not expected_types:
            raise ValueError(MCP_ARGUMENTS_INVALID_ERROR)
        if isinstance(value, bool) and (
            bool not in expected_types
            and (int in expected_types or float in expected_types)
        ):
            raise ValueError(MCP_ARGUMENTS_INVALID_ERROR)
        if not isinstance(value, tuple(expected_types)):
            raise TypeError(MCP_ARGUMENTS_INVALID_ERROR)
