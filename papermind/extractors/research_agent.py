"""Phase 7: a research agent that uses tools to find and extract paper metadata."""

import json

from openai import OpenAI

from papermind.schemas import Paper
from papermind.tools import TOOL_DEFS, TOOL_REGISTRY


SYSTEM_PROMPT = (
    "You are a research librarian. Given a user query about a paper, find it "
    "and extract its metadata.\n\n"
    "Workflow:\n"
    "1. If the user describes a paper without an exact arXiv ID, call "
    "   search_arxiv to find candidates, then pick the best match.\n"
    "2. Once you have an arXiv ID, call fetch_arxiv_paper to read the actual "
    "   paper text.\n"
    "3. Use the fetched text to fill the Paper schema as your final answer.\n\n"
    "Be efficient: skip search if the user already gave you a specific arXiv ID."
)


MAX_TURNS = 8


def research_paper(query: str, *, model: str = "gpt-4o-2024-08-06") -> Paper | None:
    """Run the agent loop until the model produces a Paper or we hit MAX_TURNS."""
    client = OpenAI()

    input_items: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    previous_response_id: str | None = None

    for turn in range(MAX_TURNS):
        kwargs = {
            "model": model,
            "input": input_items,
            "tools": TOOL_DEFS,
            "text_format": Paper,
        }
        if previous_response_id:
            # Server-side conversation state — we only need to send NEW items.
            kwargs["previous_response_id"] = previous_response_id

        response = client.responses.parse(**kwargs)
        previous_response_id = response.id

        # Did the model produce a final structured answer?
        if response.output_parsed is not None:
            return response.output_parsed

        # Otherwise it emitted tool calls. Execute them and prep the next turn.
        tool_calls = [
            item for item in response.output
            if getattr(item, "type", None) == "function_call"
        ]
        if not tool_calls:
            return None  # no parsed output and no tool calls — stuck

        next_inputs: list[dict] = []
        for call in tool_calls:
            args_model, fn = TOOL_REGISTRY[call.name]
            try:
                args = args_model.model_validate_json(call.arguments)
                result = fn(**args.model_dump())
                output_payload = (
                    result if isinstance(result, str) else json.dumps(result)
                )
            except Exception as e:  # noqa: BLE001
                output_payload = f"ERROR: {type(e).__name__}: {e}"

            next_inputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output_payload,
            })
            print(f"  [tool] {call.name}({call.arguments}) → "
                  f"{output_payload[:120].replace(chr(10), ' ')}...")

        input_items = next_inputs

    return None  # MAX_TURNS exhausted


if __name__ == "__main__":
    from dotenv import load_dotenv
    from rich import print

    load_dotenv()

    queries = [
        "Find me the original Transformer paper by Vaswani et al. and extract its metadata.",
        "Get me the GPT-3 paper (arxiv 2005.14165) — I just want the metadata.",
        "There's a famous paper introducing BERT. Find it and give me the metadata.",
    ]

    for q in queries:
        print(f"\n[bold cyan]Query:[/bold cyan] {q}")
        paper = research_paper(q)
        print("\n[bold]Result:[/bold]")
        print(paper)
