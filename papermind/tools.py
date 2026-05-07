"""Phase 7: tools the model can call.

Tool argument schemas are Pydantic models — the same shape as response schemas.
That sameness IS the lesson: structured outputs and function calling use the
same JSON-Schema-with-strict mechanism, in different parts of the API call.
"""

import urllib.parse
import xml.etree.ElementTree as ET
from io import BytesIO

import httpx
from pydantic import BaseModel, Field
from pypdf import PdfReader


# ─── Tool argument schemas ─────────────────────────────────────────────

class SearchArxivArgs(BaseModel):
    """Arguments for the search_arxiv tool."""
    query: str = Field(
        description="Free-text search query. Use a phrase like 'transformer attention sequence model'."
    )
    max_results: int = Field(
        description="How many papers to return. 3-5 is usually plenty."
    )


class FetchArxivPaperArgs(BaseModel):
    """Arguments for the fetch_arxiv_paper tool."""
    arxiv_id: str = Field(
        description="The arXiv ID, e.g. '1706.03762' or '2005.14165'. No 'v1' suffix."
    )


# ─── Tool implementations ──────────────────────────────────────────────

ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


def search_arxiv(query: str, max_results: int) -> list[dict]:
    """Search arXiv. Returns a list of {id, title, summary}."""
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={urllib.parse.quote(query)}"
        f"&max_results={max_results}"
    )
    response = httpx.get(url, timeout=20.0)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    results = []
    for entry in root.findall("a:entry", ARXIV_NS):
        full_id = (entry.find("a:id", ARXIV_NS).text or "").strip()
        # full_id looks like 'http://arxiv.org/abs/1706.03762v5' — extract '1706.03762'
        arxiv_id = full_id.rsplit("/", 1)[-1].split("v")[0]
        title = (entry.find("a:title", ARXIV_NS).text or "").strip()
        summary = (entry.find("a:summary", ARXIV_NS).text or "").strip()
        results.append({"id": arxiv_id, "title": title, "summary": summary[:500]})
    return results


def fetch_arxiv_paper(arxiv_id: str) -> str:
    """Download a paper's PDF and return the first two pages of text."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    response = httpx.get(url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()

    reader = PdfReader(BytesIO(response.content))
    pages = reader.pages[:2]
    return "\n\n".join(page.extract_text() or "" for page in pages)


# ─── Tool defs (the JSON sent to the API) ──────────────────────────────

def _strictify(schema: dict) -> None:
    """Walk a JSON Schema in place; add additionalProperties:false to every object.

    Strict mode requires this on every object node. Pydantic doesn't add it by
    default, so we walk the tree ourselves. (The SDK does the same thing
    internally for text_format=PydanticModel.)
    """
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
    for value in list(schema.values()):
        if isinstance(value, dict):
            _strictify(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _strictify(item)


def to_tool_def(args_model: type[BaseModel], *, name: str, description: str) -> dict:
    """Build an OpenAI Responses tool definition from a Pydantic args model.

    The same schema-derivation a Paper response goes through, just plugged
    into a different slot of the API call.
    """
    schema = args_model.model_json_schema()
    schema.pop("title", None)
    _strictify(schema)
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": schema,
        "strict": True,
    }


TOOL_DEFS = [
    to_tool_def(
        SearchArxivArgs,
        name="search_arxiv",
        description=(
            "Search arXiv for papers matching a free-text query. Returns "
            "{id, title, summary}. Call this when you have a topic or rough "
            "title but don't know the exact arXiv ID."
        ),
    ),
    to_tool_def(
        FetchArxivPaperArgs,
        name="fetch_arxiv_paper",
        description=(
            "Download an arXiv paper by ID and return the first two pages of "
            "extracted text. Call AFTER you have a specific arXiv ID, to read "
            "enough of the paper to extract metadata."
        ),
    ),
]


TOOL_REGISTRY = {
    "search_arxiv": (SearchArxivArgs, search_arxiv),
    "fetch_arxiv_paper": (FetchArxivPaperArgs, fetch_arxiv_paper),
}
