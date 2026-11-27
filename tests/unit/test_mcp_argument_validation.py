from typing import Any

import pytest

from app.core.config import (
    MCP_ARGUMENTS_INVALID_ERROR,
    MCP_ARGUMENTS_MUST_BE_OBJECT_ERROR,
)
from app.integrations.mcp.client import validate_tool_arguments


@pytest.mark.parametrize(
    ("arguments", "schema", "error", "message"),
    [
        (
            [],
            {"type": "array"},
            TypeError,
            MCP_ARGUMENTS_MUST_BE_OBJECT_ERROR,
        ),
        (
            {},
            {"type": "array"},
            TypeError,
            MCP_ARGUMENTS_INVALID_ERROR,
        ),
        (
            {},
            {"properties": []},
            TypeError,
            MCP_ARGUMENTS_INVALID_ERROR,
        ),
        (
            {},
            {"required": "query"},
            TypeError,
            MCP_ARGUMENTS_INVALID_ERROR,
        ),
        (
            {},
            {"required": ["query"]},
            ValueError,
            MCP_ARGUMENTS_INVALID_ERROR,
        ),
        (
            {"extra": 1},
            {"properties": {}, "additionalProperties": False},
            ValueError,
            MCP_ARGUMENTS_INVALID_ERROR,
        ),
    ],
)
def test_object_constraints_raise_expected_errors(
    arguments: Any,
    schema: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    """Object shape, required keys, and extra keys retain exact errors."""
    with pytest.raises(error, match=f"^{message}$"):
        validate_tool_arguments(arguments, schema)


@pytest.mark.parametrize(
    ("value", "property_schema"),
    [
        (None, {"type": ["string", "null"]}),
        (None, {"type": ["null"]}),
        (None, {"type": "string", "nullable": True}),
    ],
)
def test_nullable_values_are_accepted(
    value: Any, property_schema: dict[str, Any]
) -> None:
    """Null is accepted through a null union or the nullable extension."""
    validate_tool_arguments({"value": value}, {"properties": {"value": property_schema}})


@pytest.mark.parametrize(
    ("value", "property_schema", "error"),
    [
        (None, {"type": "string", "nullable": False}, ValueError),
        (None, {"type": "string"}, ValueError),
        ("value", {"type": ["null"]}, ValueError),
        ("value", {"type": ["unknown"]}, ValueError),
        (3, {"type": "string"}, TypeError),
    ],
)
def test_invalid_or_nonnullable_values_preserve_error_types(
    value: Any,
    property_schema: dict[str, Any],
    error: type[Exception],
) -> None:
    """Invalid unions and mismatched supported types retain old semantics."""
    with pytest.raises(error, match=f"^{MCP_ARGUMENTS_INVALID_ERROR}$"):
        validate_tool_arguments({"value": value}, {"properties": {"value": property_schema}})


@pytest.mark.parametrize(
    ("value", "schema_type"),
    [
        ("hello", "string"),
        (3, "integer"),
        (3.5, "number"),
        (True, "boolean"),
        ([1], "array"),
        ({"x": 1}, "object"),
        (3, ["integer", "string"]),
        ("three", ["integer", "string"]),
    ],
)
def test_supported_scalar_and_union_types_are_accepted(
    value: Any, schema_type: str | list[str]
) -> None:
    """Supported scalar and union type declarations accept matching values."""
    validate_tool_arguments(
        {"value": value},
        {"properties": {"value": {"type": schema_type}}},
    )


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("schema_type", ["integer", "number", ["integer", "string"]])
def test_boolean_does_not_match_numeric_or_other_union_types(
    value: bool, schema_type: str | list[str]
) -> None:
    """Python bool's integer subclass relationship must not admit it as numeric."""
    with pytest.raises(ValueError, match=f"^{MCP_ARGUMENTS_INVALID_ERROR}$"):
        validate_tool_arguments(
            {"value": value},
            {"properties": {"value": {"type": schema_type}}},
        )


def test_boolean_matches_a_union_that_explicitly_allows_boolean() -> None:
    """A type union that includes boolean still accepts bool values."""
    validate_tool_arguments(
        {"value": True},
        {"properties": {"value": {"type": ["integer", "boolean"]}}},
    )


@pytest.mark.parametrize(
    "property_schema",
    [None, "string", {}, {"type": None}, {"type": 4}, {"type": {"x": 1}}],
)
def test_missing_or_malformed_property_type_remains_unchecked(
    property_schema: Any,
) -> None:
    """Unsupported or absent property metadata remains permissive."""
    validate_tool_arguments(
        {"value": object()}, {"properties": {"value": property_schema}}
    )


def test_unknown_types_are_ignored_when_a_supported_union_type_remains() -> None:
    """Unknown union entries do not disable a recognized type constraint."""
    validate_tool_arguments(
        {"value": "valid"},
        {"properties": {"value": {"type": [None, "unknown", "string"]}}},
    )
    with pytest.raises(TypeError, match=f"^{MCP_ARGUMENTS_INVALID_ERROR}$"):
        validate_tool_arguments(
            {"value": 1},
            {"properties": {"value": {"type": ["unknown", "string"]}}},
        )
