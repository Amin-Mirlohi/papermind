"""Phase 3: extract structured claims from a paper, with optional fields."""

from openai import OpenAI

from papermind.schemas import Claim, ClaimsExtraction


SYSTEM_PROMPT = (
    "You extract the load-bearing claims from an academic paper. A claim is a "
    "declarative assertion the paper makes about the world, its method, its "
    "results, or its limitations.\n\n"
    "Rules:\n"
    "- Extract 5-15 claims, prioritizing the paper's core contributions and "
    "  empirical findings over throwaway statements.\n"
    "- Paraphrase each claim so it stands alone — do not quote.\n"
    "- For optional fields (confidence_hedge, page_reference, "
    "  supporting_citation): emit null when the information is not present "
    "  in the source. Do not guess. Null is a valid, informative answer."
)


def extract_claims(text: str, *, model: str = "gpt-4o-2024-08-06") -> list[Claim]:
    """Send paper text to the model; return a list of validated claims."""
    client = OpenAI()

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        text_format=ClaimsExtraction,
    )

    return response.output_parsed.claims


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from dotenv import load_dotenv
    from rich import print

    from papermind.extractors.metadata import SAMPLE_TEXT
    from papermind.pdf import read_pdf

    load_dotenv()

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        pdf_paths = sorted(target.glob("*.pdf")) if target.is_dir() else [target]
    else:
        pdf_paths = []

    if not pdf_paths:
        print("[dim]No PDFs found — using built-in sample.[/dim]")
        for claim in extract_claims(SAMPLE_TEXT):
            print(claim)
    else:
        for path in pdf_paths:
            print(f"\n[bold cyan]── {path.name} ──[/bold cyan]")
            text = read_pdf(path, max_pages=None)  # full paper for claim extraction
            claims = extract_claims(text)
            print(f"[dim]Extracted {len(claims)} claims[/dim]")
            for i, claim in enumerate(claims, 1):
                print(f"\n[bold]{i}.[/bold] {claim}")
