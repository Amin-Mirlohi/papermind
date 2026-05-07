"""Result type for structured-output calls.

A discriminated union over the four meaningfully different outcomes of
`client.responses.parse()`. Use Python's match statement to consume:

    match safe_extract(text, schema=Paper):
        case Ok(paper):       use(paper)
        case Refused(reason): log_refusal(reason)
        case Truncated():     retry_with_more_tokens()
        case Failed(err):     bubble_up(err)
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Ok(Generic[T]):
    """The model returned a valid, schema-conforming object."""
    value: T


@dataclass(frozen=True)
class Refused:
    """The safety system declined. `reason` is the model's refusal text."""
    reason: str


@dataclass(frozen=True)
class Truncated:
    """Output was cut off — usually max_output_tokens or context limit."""
    detail: str = ""


@dataclass(frozen=True)
class Failed:
    """Anything else: network error, parse error, unexpected shape."""
    error: str


# A single type alias for "any of the above" — useful in signatures.
Result = Ok[T] | Refused | Truncated | Failed
