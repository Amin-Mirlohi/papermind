"""Phase 10: eval harness for PaperAssessment.

Computes per-field metrics over a hand-labeled ground-truth corpus.
Reuses Phase 6's safe_parse so failures (refusals, truncations) are
visible in the report instead of crashing the run.
"""

from dataclasses import dataclass, field
from pathlib import Path

from papermind.extractors.assessment import SYSTEM_PROMPT
from papermind.extractors.safe import safe_parse
from papermind.pdf import read_pdf
from papermind.result import Failed, Ok, Refused, Truncated
from papermind.schemas import (
    ContributionType,
    Methodology,
    PaperAssessment,
)


# ─── Ground truth ─────────────────────────────────────────────────────
# In a real project this would live in data/ground_truth/*.json, edited
# by hand. Inline here for clarity. Disagreement on edge cases is itself
# a finding — different labelers will pick different categories for
# papers that span methodologies. That's the eval signal you care about.

@dataclass(frozen=True)
class GroundTruth:
    methodology: Methodology
    contribution: ContributionType
    novelty_score: int
    novelty_tolerance: int = 1  # accept within ±N as "correct"
    notes: str = ""


GROUND_TRUTH: dict[str, GroundTruth] = {
    "attention.pdf": GroundTruth(
        methodology=Methodology.system_design,
        contribution=ContributionType.novel_algorithm,
        novelty_score=10,
        notes="Paradigm-defining architecture; both system_design and empirical valid.",
    ),
    "bert.pdf": GroundTruth(
        methodology=Methodology.system_design,
        contribution=ContributionType.novel_algorithm,
        novelty_score=9,
        notes="Pre-training paradigm; high novelty but builds on transformers.",
    ),
    "gpt3.pdf": GroundTruth(
        methodology=Methodology.empirical_study,
        contribution=ContributionType.improvement,
        novelty_score=9,
        novelty_tolerance=2,
        notes="Scaling study — methodology arguably system_design too. Wider tolerance.",
    ),
}


# ─── Result types ─────────────────────────────────────────────────────

@dataclass
class CaseResult:
    pdf_name: str
    expected: GroundTruth
    actual: PaperAssessment | None
    failure: str | None  # None if Ok
    field_correct: dict[str, bool] = field(default_factory=dict)
    novelty_error: int | None = None


@dataclass
class EvalReport:
    n_total: int
    n_ok: int
    n_refused: int
    n_truncated: int
    n_failed: int
    methodology_accuracy: float
    contribution_accuracy: float
    novelty_within_tolerance: float
    mean_novelty_abs_error: float
    cases: list[CaseResult]

    @property
    def schema_validity_rate(self) -> float:
        """Fraction of calls that produced a valid PaperAssessment (didn't refuse/truncate/fail)."""
        return self.n_ok / self.n_total if self.n_total else 0.0


# ─── Per-case comparison ─────────────────────────────────────────────

def compare(expected: GroundTruth, actual: PaperAssessment) -> tuple[dict[str, bool], int]:
    """Compare a model output against ground truth. Returns (per-field correctness, novelty error)."""
    novelty_err = abs(actual.novelty_score - expected.novelty_score)
    return (
        {
            "methodology": actual.methodology == expected.methodology,
            "contribution": actual.contribution == expected.contribution,
            "novelty_within_tolerance": novelty_err <= expected.novelty_tolerance,
        },
        novelty_err,
    )


# ─── Runner ──────────────────────────────────────────────────────────

def run_evals(pdf_dir: Path) -> EvalReport:
    cases: list[CaseResult] = []

    for pdf_name, gt in GROUND_TRUTH.items():
        pdf_path = pdf_dir / pdf_name
        if not pdf_path.exists():
            cases.append(CaseResult(pdf_name, gt, None, f"missing file: {pdf_path}"))
            continue

        text = read_pdf(pdf_path, max_pages=None)
        result = safe_parse(text, schema=PaperAssessment, system_prompt=SYSTEM_PROMPT)

        match result:
            case Ok(actual):
                field_correct, novelty_err = compare(gt, actual)
                cases.append(CaseResult(
                    pdf_name=pdf_name,
                    expected=gt,
                    actual=actual,
                    failure=None,
                    field_correct=field_correct,
                    novelty_error=novelty_err,
                ))
            case Refused(reason):
                cases.append(CaseResult(pdf_name, gt, None, f"refused: {reason}"))
            case Truncated(detail):
                cases.append(CaseResult(pdf_name, gt, None, f"truncated: {detail}"))
            case Failed(error):
                cases.append(CaseResult(pdf_name, gt, None, f"failed: {error}"))

    return _aggregate(cases)


def _aggregate(cases: list[CaseResult]) -> EvalReport:
    n_total = len(cases)
    ok_cases = [c for c in cases if c.actual is not None]
    n_ok = len(ok_cases)
    n_refused = sum(1 for c in cases if c.failure and c.failure.startswith("refused"))
    n_truncated = sum(1 for c in cases if c.failure and c.failure.startswith("truncated"))
    n_failed = n_total - n_ok - n_refused - n_truncated

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return EvalReport(
        n_total=n_total,
        n_ok=n_ok,
        n_refused=n_refused,
        n_truncated=n_truncated,
        n_failed=n_failed,
        methodology_accuracy=mean(
            [1.0 if c.field_correct.get("methodology") else 0.0 for c in ok_cases]
        ),
        contribution_accuracy=mean(
            [1.0 if c.field_correct.get("contribution") else 0.0 for c in ok_cases]
        ),
        novelty_within_tolerance=mean(
            [
                1.0 if c.field_correct.get("novelty_within_tolerance") else 0.0
                for c in ok_cases
            ]
        ),
        mean_novelty_abs_error=mean(
            [c.novelty_error for c in ok_cases if c.novelty_error is not None]
        ),
        cases=cases,
    )


# ─── Reporting ───────────────────────────────────────────────────────

def render(report: EvalReport) -> None:
    from rich import print
    from rich.table import Table

    summary = Table(title="Eval summary", show_header=False)
    summary.add_column("metric")
    summary.add_column("value", justify="right")

    summary.add_row("cases (total)", str(report.n_total))
    summary.add_row("schema validity rate", f"{report.schema_validity_rate:.0%}")
    summary.add_row("ok", str(report.n_ok))
    summary.add_row("refused", str(report.n_refused))
    summary.add_row("truncated", str(report.n_truncated))
    summary.add_row("failed", str(report.n_failed))
    summary.add_row("[bold]field accuracy[/bold]", "")
    summary.add_row("  methodology", f"{report.methodology_accuracy:.0%}")
    summary.add_row("  contribution", f"{report.contribution_accuracy:.0%}")
    summary.add_row("  novelty (within tolerance)", f"{report.novelty_within_tolerance:.0%}")
    summary.add_row("  novelty mean abs error", f"{report.mean_novelty_abs_error:.2f}")

    print(summary)

    detail = Table(title="Per-case detail")
    detail.add_column("pdf")
    detail.add_column("status")
    detail.add_column("methodology", justify="center")
    detail.add_column("contribution", justify="center")
    detail.add_column("novelty (gt → pred, err)")

    for c in report.cases:
        if c.actual is None:
            detail.add_row(c.pdf_name, f"[red]{c.failure}[/red]", "-", "-", "-")
            continue
        m = "✓" if c.field_correct["methodology"] else f"✗ ({c.actual.methodology.value})"
        ct = "✓" if c.field_correct["contribution"] else f"✗ ({c.actual.contribution.value})"
        nov = (
            f"{c.expected.novelty_score} → {c.actual.novelty_score} "
            f"(err={c.novelty_error}, "
            f"{'✓' if c.field_correct['novelty_within_tolerance'] else '✗'})"
        )
        detail.add_row(c.pdf_name, "[green]ok[/green]", m, ct, nov)

    print(detail)


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    pdf_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/papers")
    report = run_evals(pdf_dir)
    render(report)
