"""Phase 1: extract structured metadata from raw paper text."""

from openai import OpenAI

from papermind.schemas import Paper


SYSTEM_PROMPT = (
    "You extract bibliographic metadata from academic papers. "
    "Read the provided text and return the structured fields. "
    "Be faithful to the source — do not invent authors or venues. "
    "If a field is unclear from the text, make a best-effort inference and keep it concise."
)


def extract_metadata(text: str, *, model: str = "gpt-4o-2024-08-06") -> Paper:
    """Send raw paper text to the model and get back a validated Paper."""
    client = OpenAI()

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        text_format=Paper,
    )

    return response.output_parsed


# ─── Hardcoded test snippet (Attention Is All You Need, 2017) ──────────

SAMPLE_TEXT = """\
Attention Is All You Need

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

Google Brain / Google Research / University of Toronto

31st Conference on Neural Information Processing Systems (NIPS 2017),
Long Beach, CA, USA. arXiv:1706.03762

Abstract
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and
convolutions entirely. Experiments on two machine translation tasks show
these models to be superior in quality while being more parallelizable and
requiring significantly less time to train.
"""


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from dotenv import load_dotenv
    from rich import print

    from papermind.pdf import read_pdf

    load_dotenv()

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        pdf_paths = sorted(target.glob("*.pdf")) if target.is_dir() else [target]
    else:
        pdf_paths = []

    if not pdf_paths:
        print("[dim]No PDFs found — using built-in sample.[/dim]")
        print(extract_metadata(SAMPLE_TEXT))
    else:
        for path in pdf_paths:
            print(f"\n[bold cyan]── {path.name} ──[/bold cyan]")
            text = read_pdf(path)
            print(extract_metadata(text))
