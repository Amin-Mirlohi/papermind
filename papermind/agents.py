"""Phase 9: PydanticAI rewrites of the earlier-phase extractors.

Same Pydantic models as before — but the framework absorbs the agent loop,
tool dispatch, and (most importantly) validation retries.
"""

import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import BytesIO

import httpx
from pydantic_ai import Agent, ModelRetry, RunContext
from pypdf import PdfReader

from papermind.schemas import Paper


# ─── Slice A: the simplest agent ──────────────────────────────────────

PAPER_SYSTEM_PROMPT = (
    "You extract bibliographic metadata from academic papers. "
    "Read the provided text and return the structured fields. "
    "Be faithful to the source — do not invent authors or venues. "
    "If a field is unclear from the text, make a best-effort inference and keep it concise."
)

paper_agent = Agent(
    "openai:gpt-4o-2024-08-06",
    output_type=Paper,
    system_prompt=PAPER_SYSTEM_PROMPT,
)


# ─── Slice B: validation retries ──────────────────────────────────────
# The output_validator runs AFTER Pydantic parses the response. Raising
# ModelRetry feeds the error back to the model and asks it to try again.
# Use this for constraints strict mode can't express (counts, ranges,
# cross-field invariants).

@paper_agent.output_validator
def check_keyword_count(_: RunContext, paper: Paper) -> Paper:
    n = len(paper.keywords)
    if not (3 <= n <= 7):
        raise ModelRetry(
            f"You returned {n} keywords. The schema requires between 3 and 7 "
            f"keywords. Re-emit the full Paper with a corrected keyword list."
        )
    return paper


# ─── Slice C: tools + dependency injection ────────────────────────────

@dataclass
class ResearchDeps:
    """Per-run dependencies. Each agent.run(...) gets its own instance."""
    http_client: httpx.Client


RESEARCH_SYSTEM_PROMPT = (
    "You are a research librarian. Given a user query about a paper, find it "
    "and extract its metadata.\n\n"
    "Workflow:\n"
    "1. If the user describes a paper without an exact arXiv ID, call "
    "   search_arxiv to find candidates, then pick the best match.\n"
    "2. Once you have an arXiv ID, call fetch_arxiv_paper to read the actual "
    "   paper text.\n"
    "3. Use the fetched text to fill the Paper schema.\n\n"
    "Be efficient: skip search if the user already gave you a specific arXiv ID."
)

research_agent = Agent(
    "openai:gpt-4o-2024-08-06",
    output_type=Paper,
    deps_type=ResearchDeps,
    system_prompt=RESEARCH_SYSTEM_PROMPT,
)


_ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


@research_agent.tool
def search_arxiv(
    ctx: RunContext[ResearchDeps], query: str, max_results: int
) -> list[dict]:
    """Search arXiv for papers matching a free-text query.

    Returns a list of {id, title, summary}. Use this when you have a topic
    or rough title but don't know the exact arXiv ID.
    """
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query={urllib.parse.quote(query)}"
        f"&max_results={max_results}"
    )
    response = ctx.deps.http_client.get(url, timeout=20.0)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    results = []
    for entry in root.findall("a:entry", _ARXIV_NS):
        full_id = (entry.find("a:id", _ARXIV_NS).text or "").strip()
        arxiv_id = full_id.rsplit("/", 1)[-1].split("v")[0]
        title = (entry.find("a:title", _ARXIV_NS).text or "").strip()
        summary = (entry.find("a:summary", _ARXIV_NS).text or "").strip()
        results.append({"id": arxiv_id, "title": title, "summary": summary[:500]})
    return results


@research_agent.tool
def fetch_arxiv_paper(ctx: RunContext[ResearchDeps], arxiv_id: str) -> str:
    """Download an arXiv paper by ID and return the first two pages of text.

    Call AFTER you have a specific arXiv ID, to read enough of the paper to
    extract metadata.
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    response = ctx.deps.http_client.get(url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()

    reader = PdfReader(BytesIO(response.content))
    pages = reader.pages[:2]
    return "\n\n".join(page.extract_text() or "" for page in pages)


# ─── Demo runners ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv
    from rich import print

    from papermind.extractors.metadata import SAMPLE_TEXT

    load_dotenv()

    mode = sys.argv[1] if len(sys.argv) > 1 else "basic"

    if mode == "basic":
        # Slice A + B: simple extraction with the validator in play
        print("[bold cyan]── basic agent (Phase 1 equivalent) ──[/bold cyan]")
        result = paper_agent.run_sync(SAMPLE_TEXT)
        print(result.output)

    elif mode == "research":
        # Slice C: research agent with tools
        print("[bold cyan]── research agent (Phase 7 equivalent) ──[/bold cyan]")
        with httpx.Client() as client:
            deps = ResearchDeps(http_client=client)
            for query in [
                "Find me the original Transformer paper by Vaswani et al.",
                "Get the GPT-3 paper (arxiv 2005.14165) — just the metadata.",
            ]:
                print(f"\n[bold]Query:[/bold] {query}")
                result = research_agent.run_sync(query, deps=deps)
                print(result.output)

    else:
        print(f"[red]Unknown mode: {mode}. Try 'basic' or 'research'.[/red]")
