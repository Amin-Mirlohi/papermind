"""Phase 8: cross-paper synthesis pipeline.

Composes the per-paper extractors from Phases 1+3 with a corpus-level
synthesis call. Each stage's output is typed, so building the next stage's
input is object construction, not string parsing.
"""

from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from papermind.extractors.claims import extract_claims
from papermind.extractors.metadata import extract_metadata
from papermind.pdf import read_pdf
from papermind.schemas import Claim, Paper, SynthesisReport


# ─── Per-paper bundle ────────────────────────────────────────────────

@dataclass(frozen=True)
class PaperBrief:
    """A paper plus its derived structured outputs, identified by a stable id."""
    id: str
    paper: Paper
    claims: list[Claim]


def build_brief(pdf_path: Path) -> PaperBrief:
    """Run Phase 1 + Phase 3 extractors on one PDF, return a typed bundle."""
    text_short = read_pdf(pdf_path, max_pages=2)
    text_full = read_pdf(pdf_path, max_pages=None)
    paper = extract_metadata(text_short)
    claims = extract_claims(text_full)
    return PaperBrief(id=pdf_path.stem, paper=paper, claims=claims)


# ─── Synthesis call ──────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You synthesize a corpus of academic papers. You are given each paper's "
    "ID, metadata, and key claims. Produce:\n\n"
    "1. A list of contradictions across papers (only real ones — empty list "
    "   if none exist; do NOT invent disagreements).\n"
    "2. A reading plan that orders papers from accessible entry points to more "
    "   advanced work, and identifies prerequisite relationships.\n"
    "3. A short summary tying the corpus together.\n\n"
    "Use the exact paper IDs given in the input. Do not introduce new IDs."
)


def _format_brief(brief: PaperBrief) -> str:
    paper = brief.paper
    claim_lines = "\n".join(
        f"  - [{c.evidence_type}] {c.statement}" for c in brief.claims
    )
    return (
        f"[id={brief.id}]\n"
        f"Title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors)}\n"
        f"Year: {paper.year} | Venue: {paper.venue}\n"
        f"Abstract: {paper.abstract}\n"
        f"Key claims:\n{claim_lines}"
    )


def synthesize(briefs: list[PaperBrief], *, model: str = "gpt-4o-2024-08-06") -> SynthesisReport:
    """Send N typed paper briefs to the model; get back a typed synthesis report."""
    if not briefs:
        raise ValueError("Need at least one brief to synthesize.")

    corpus_text = "\n\n---\n\n".join(_format_brief(b) for b in briefs)

    client = OpenAI()
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": corpus_text},
        ],
        text_format=SynthesisReport,
    )
    return response.output_parsed


# ─── Markdown rendering (the human-readable side) ─────────────────────

def render_markdown(briefs: list[PaperBrief], report: SynthesisReport) -> str:
    by_id = {b.id: b for b in briefs}
    lines: list[str] = ["# PaperMind synthesis\n"]

    lines.append("## Summary\n")
    lines.append(report.summary + "\n")

    lines.append("## Reading plan\n")
    for i, pid in enumerate(report.reading_plan.ordered_paper_ids, 1):
        title = by_id[pid].paper.title if pid in by_id else "(unknown id)"
        lines.append(f"{i}. **{pid}** — {title}")
    lines.append("")
    lines.append(f"_Rationale:_ {report.reading_plan.rationale}\n")

    if report.reading_plan.prerequisites:
        lines.append("### Prerequisites\n")
        for prereq in report.reading_plan.prerequisites:
            lines.append(
                f"- `{prereq.paper_id}` ← requires: "
                + ", ".join(f"`{p}`" for p in prereq.prerequisite_ids)
            )
        lines.append("")

    lines.append("## Contradictions\n")
    if not report.contradictions:
        lines.append("_No contradictions found across the corpus._\n")
    else:
        for c in report.contradictions:
            lines.append(f"### {c.paper_a_id} ↔ {c.paper_b_id} ({c.nature})\n")
            lines.append(f"- **{c.paper_a_id}:** {c.claim_a}")
            lines.append(f"- **{c.paper_b_id}:** {c.claim_b}")
            if c.resolution_hypothesis:
                lines.append(f"- _Possible reconciliation:_ {c.resolution_hypothesis}")
            lines.append("")

    return "\n".join(lines)


# ─── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv
    from rich import print

    load_dotenv()

    if len(sys.argv) < 2:
        print("[red]Usage: python -m papermind.pipeline <pdf-dir>[/red]")
        sys.exit(1)

    pdf_dir = Path(sys.argv[1])
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"[red]No PDFs found in {pdf_dir}[/red]")
        sys.exit(1)

    print(f"[dim]Processing {len(pdf_paths)} papers...[/dim]")
    briefs: list[PaperBrief] = []
    for path in pdf_paths:
        print(f"  [dim]→ {path.name}[/dim]")
        briefs.append(build_brief(path))

    print("[dim]Synthesizing...[/dim]")
    report = synthesize(briefs)

    md = render_markdown(briefs, report)
    out_path = pdf_dir.parent / "synthesis.md"
    out_path.write_text(md)
    print(f"\n[green]Wrote {out_path}[/green]\n")

    print(md)
