from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Question to answer from the ingested document collection."""

    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Strip the question and reject input containing only whitespace.

        Args:
            value (str): Submitted question.

        Returns:
            str: Non-empty question without surrounding whitespace.
        """
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank")
        return value


class Citation(BaseModel):
    """A referenced document chunk; source is a local file path."""

    id: str
    source: str
    chunk_index: int
    excerpt: str


class ChatResponse(BaseModel):
    """Generated answer and the retrieved sources cited in it."""

    answer: str
    citations: list[Citation]
