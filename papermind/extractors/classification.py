"""Phase 2: classify a paper's methodology and contribution type."""

from openai import OpenAI

from papermind.schemas import PaperClassification


SYSTEM_PROMPT = (
    "You are a research-paper analyst. Given the text of a paper, classify it "
    "by methodology (how the work is conducted) and contribution type (what is "
    "new about it). Prefer the most specific category that fits. Reserve 'other' "
    "for genuine edge cases — if you reach for 'other', you must briefly justify "
    "in the matching '_other' field."
)


def classify_paper(text: str, *, model: str = "gpt-4o-2024-08-06") -> PaperClassification:
    """Send paper text to the model; return a validated classification."""
    client = OpenAI()

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        text_format=PaperClassification,
    )

    return response.output_parsed


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
        print(classify_paper(SAMPLE_TEXT))
    else:
        for path in pdf_paths:
            print(f"\n[bold cyan]── {path.name} ──[/bold cyan]")
            text = read_pdf(path, max_pages=4)  # a bit more context for classification
            print(classify_paper(text))
