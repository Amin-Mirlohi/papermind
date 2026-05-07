"""Phase 5: full paper assessment with reasoning-as-schema."""

from openai import OpenAI

from papermind.schemas import PaperAssessment


SYSTEM_PROMPT = (
    "You assess academic papers. The output schema requires you to walk "
    "through 3-6 reasoning steps BEFORE producing your final classifications "
    "and novelty score. Do this in earnest — the steps are how you arrive at "
    "the verdict, not a justification you write afterwards.\n\n"
    "For each reasoning step, ground 'observation' in something specific from "
    "the paper text. Vague observations make weak inferences.\n\n"
    "Novelty score guidance:\n"
    "  1-3:  incremental tweak; minor improvement on a known method.\n"
    "  4-6:  solid contribution; new method/dataset/application but in a "
    "        well-explored area.\n"
    "  7-9:  significant; opens up new directions or beats SOTA by a wide "
    "        margin.\n"
    "  10:   paradigm-defining; rare, reserve for landmark work."
)


def assess_paper(text: str, *, model: str = "gpt-4o-2024-08-06") -> PaperAssessment:
    """Send paper text to the model; return a validated assessment."""
    client = OpenAI()

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        text_format=PaperAssessment,
    )

    return response.output_parsed


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from dotenv import load_dotenv
    from rich import print
    from rich.panel import Panel

    from papermind.extractors.metadata import SAMPLE_TEXT
    from papermind.pdf import read_pdf

    load_dotenv()

    def render(assessment: PaperAssessment) -> None:
        print("[bold]Reasoning steps[/bold]")
        for i, step in enumerate(assessment.reasoning_steps, 1):
            print(f"  [dim]{i}.[/dim] [italic]obs:[/italic] {step.observation}")
            print(f"     [italic]inf:[/italic] {step.inference}")
        verdict = (
            f"methodology   = {assessment.methodology.value}\n"
            f"contribution  = {assessment.contribution.value}\n"
            f"novelty_score = {assessment.novelty_score}/10\n\n"
            f"{assessment.final_justification}"
        )
        print(Panel(verdict, title="Verdict", border_style="cyan"))

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        pdf_paths = sorted(target.glob("*.pdf")) if target.is_dir() else [target]
    else:
        pdf_paths = []

    if not pdf_paths:
        print("[dim]No PDFs found — using built-in sample.[/dim]")
        render(assess_paper(SAMPLE_TEXT))
    else:
        for path in pdf_paths:
            print(f"\n[bold cyan]── {path.name} ──[/bold cyan]")
            text = read_pdf(path, max_pages=None)
            render(assess_paper(text))
