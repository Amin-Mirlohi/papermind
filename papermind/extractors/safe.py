"""Phase 6: a robust wrapper that distinguishes Ok / Refused / Truncated / Failed."""

from typing import TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from papermind.result import Failed, Ok, Refused, Result, Truncated


T = TypeVar("T", bound=BaseModel)


def safe_parse(
    text: str,
    *,
    schema: type[T],
    system_prompt: str,
    model: str = "gpt-4o-2024-08-06",
    max_output_tokens: int | None = None,
) -> Result[T]:
    """Call responses.parse and return a Result, never raise.

    Translates each failure mode into a distinct Result variant so the caller
    can decide what to do.
    """
    client = OpenAI()

    try:
        response = client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            text_format=schema,
            max_output_tokens=max_output_tokens,
        )
    except OpenAIError as e:
        return Failed(f"OpenAI error: {e}")
    except ValidationError as e:
        return Failed(f"Validation error: {e}")
    except Exception as e:  # noqa: BLE001 — we genuinely want a catch-all here
        return Failed(f"Unexpected: {type(e).__name__}: {e}")

    # Truncation / content-filter / other "incomplete" outcomes
    if getattr(response, "status", None) == "incomplete":
        reason = getattr(
            getattr(response, "incomplete_details", None), "reason", "unknown"
        )
        if reason == "max_output_tokens":
            return Truncated(detail=reason)
        # content_filter and other incomplete reasons fall through to Failed
        return Failed(f"incomplete: {reason}")

    # Refusal: output_parsed is None and a content block carries a refusal string
    if response.output_parsed is None:
        for item in response.output:
            for content in getattr(item, "content", []) or []:
                refusal = getattr(content, "refusal", None)
                if refusal:
                    return Refused(refusal)
        return Failed("output_parsed is None but no refusal found")

    return Ok(response.output_parsed)
