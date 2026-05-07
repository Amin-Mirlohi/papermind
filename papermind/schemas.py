"""Pydantic schemas for PaperMind, grouped by phase."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ─── Phase 1: Foundations ─────────────────────────────────────────────

class Paper(BaseModel):
    """Metadata extracted from a single academic paper."""

    title: str = Field(
        description="The full title of the paper, exactly as written on the first page."
    )
    authors: list[str] = Field(
        description="Authors in the order they appear, formatted as 'First Last'."
    )
    abstract: str = Field(
        description="The complete abstract text, with whitespace normalized."
    )
    year: int = Field(
        description="Publication year as a 4-digit integer (e.g. 2024)."
    )
    venue: str = Field(
        description="Conference, journal, or preprint server (e.g. 'NeurIPS 2024', 'arXiv')."
    )
    keywords: list[str] = Field(
        description="Topical keywords. Use author-provided keywords when available; otherwise infer 3-7 specific terms."
    )


# ─── Phase 2: Constrained Vocabularies ────────────────────────────────

class Methodology(str, Enum):
    """How the paper produces its findings."""
    empirical_study = "empirical_study"
    theoretical = "theoretical"
    survey = "survey"
    system_design = "system_design"
    benchmark = "benchmark"
    other = "other"


class ContributionType(str, Enum):
    """The primary kind of contribution the paper claims."""
    novel_algorithm = "novel_algorithm"
    novel_dataset = "novel_dataset"
    novel_application = "novel_application"
    improvement = "improvement"
    analysis = "analysis"
    other = "other"


class PaperClassification(BaseModel):
    """A coarse classification of a paper's methodology and contribution."""

    methodology: Methodology = Field(
        description=(
            "How the work is conducted. Use 'empirical_study' for experiments on "
            "data; 'theoretical' for proofs and formal analysis; 'survey' for "
            "literature reviews; 'system_design' for architecture/engineering "
            "papers; 'benchmark' for evaluation suites. Use 'other' only when "
            "none of these clearly applies."
        )
    )
    methodology_other: str | None = Field(
        description=(
            "If methodology == 'other', a short phrase describing the actual "
            "methodology. Otherwise null."
        )
    )
    contribution: ContributionType = Field(
        description=(
            "The paper's primary novelty. 'novel_algorithm' for a new method; "
            "'novel_dataset' for a released corpus/benchmark; 'novel_application' "
            "for applying known methods to a new domain; 'improvement' for an "
            "incremental refinement of an existing approach; 'analysis' for "
            "studying behavior of existing systems. Use 'other' sparingly."
        )
    )
    contribution_other: str | None = Field(
        description=(
            "If contribution == 'other', a short phrase describing the actual "
            "contribution. Otherwise null."
        )
    )


# ─── Phase 3: Optional Fields & Explicit Nullability ─────────────────

class Claim(BaseModel):
    """A single declarative claim made by a paper."""

    statement: str = Field(
        description=(
            "A standalone, paraphrased version of the claim — should make sense "
            "out of context. Avoid quoting; rephrase in clear, neutral prose."
        )
    )
    evidence_type: Literal["empirical", "theoretical", "anecdotal", "citation"] = Field(
        description=(
            "What backs the claim. 'empirical' = experiments/measurements; "
            "'theoretical' = proofs/derivations; 'anecdotal' = examples or "
            "intuition without rigor; 'citation' = supported by reference to "
            "prior work."
        )
    )
    confidence_hedge: str | None = Field(
        description=(
            "If the authors hedged (e.g. 'we believe', 'it is likely', "
            "'arguably'), quote the exact hedge phrase. Null if the claim is "
            "stated as fact."
        )
    )
    page_reference: int | None = Field(
        description=(
            "Page number where the claim appears, if visible in the source. "
            "Null if not determinable."
        )
    )
    supporting_citation: str | None = Field(
        description=(
            "If the claim is backed by a specific cited reference, the citation "
            "key or short label (e.g. 'Vaswani et al. 2017'). Null otherwise."
        )
    )


class ClaimsExtraction(BaseModel):
    """Wrapper: the model returns a list of claims for one paper."""
    claims: list[Claim]


# ─── Phase 4: Recursive Schemas (Claim Graph) ────────────────────────

class ClaimNode(BaseModel):
    """A node in a paper's argument tree.

    Recursive: each node may contain children of the same shape. Becomes
    a $ref:"#/$defs/ClaimNode" in the JSON Schema sent to the API.
    """

    statement: str = Field(
        description=(
            "A single, paraphrased claim. Standalone — should make sense "
            "without the parent."
        )
    )
    type: Literal["main_claim", "supporting", "assumption", "limitation"] = Field(
        description=(
            "Role of this node in the argument. 'main_claim' = the paper's "
            "headline assertion (use exactly one at the top level). "
            "'supporting' = evidence/sub-claim that backs its parent. "
            "'assumption' = a premise the parent relies on but does not prove. "
            "'limitation' = a constraint or caveat acknowledged by the authors."
        )
    )
    children: list["ClaimNode"] = Field(
        description=(
            "Sub-claims that decompose or support this node. Use sparingly: "
            "ONLY when a claim has genuine internal structure. Empty list "
            "is the right answer for atomic claims. Do not nest deeper than "
            "3 levels total."
        )
    )


ClaimNode.model_rebuild()


class ClaimGraph(BaseModel):
    """Top-level wrapper around a recursive claim tree.

    Why the wrapper? Strict-mode structured outputs require an object at the
    schema root, and a bare recursive type makes some validators unhappy.
    The wrapper also gives us a place to attach sibling metadata later.
    """

    root: ClaimNode = Field(
        description=(
            "The single root of the paper's argument — should be the "
            "headline main_claim that everything else supports."
        )
    )


# ─── Phase 5: Chain-of-Thought as Structured Output ──────────────────

class ReasoningStep(BaseModel):
    """One step of the model's reasoning before it commits to a verdict."""

    observation: str = Field(
        description=(
            "A specific observation drawn from the paper text — a method, "
            "a result, a phrase from the abstract. Concrete, not generic."
        )
    )
    inference: str = Field(
        description=(
            "What this observation implies about methodology, contribution, "
            "or novelty. One sentence."
        )
    )


class PaperAssessment(BaseModel):
    """Full assessment of a paper.

    Field order matters: reasoning_steps comes FIRST so the model commits to
    evidence before producing its verdict. final_justification comes last as
    a closing summary, not the basis for the decision.
    """

    reasoning_steps: list[ReasoningStep] = Field(
        description=(
            "3-6 reasoning steps. Walk through what you see in the paper "
            "BEFORE deciding the methodology, contribution, and score."
        )
    )
    methodology: Methodology = Field(
        description="Final methodology classification, justified by the steps above."
    )
    contribution: ContributionType = Field(
        description="Final contribution-type classification, justified by the steps above."
    )
    novelty_score: int = Field(
        ge=1,
        le=10,
        description=(
            "Subjective novelty rating from 1 (incremental tweak) to 10 "
            "(paradigm-defining). Note: strict mode does not enforce this "
            "range — Pydantic validates it on parse."
        ),
    )
    final_justification: str = Field(
        description=(
            "A 1-2 sentence summary tying the score and classifications back "
            "to the strongest reasoning steps."
        )
    )


# ─── Phase 8: Cross-Paper Synthesis ──────────────────────────────────

class Contradiction(BaseModel):
    """A specific point where two papers disagree."""

    paper_a_id: str = Field(description="ID of the first paper involved.")
    paper_b_id: str = Field(description="ID of the second paper involved.")
    claim_a: str = Field(description="What paper A asserts, paraphrased.")
    claim_b: str = Field(description="What paper B asserts, paraphrased.")
    nature: Literal["direct", "methodological", "scope"] = Field(
        description=(
            "Type of disagreement. 'direct' = they assert opposing facts. "
            "'methodological' = different methods lead to different conclusions. "
            "'scope' = one paper's claim doesn't generalize to the other's setting."
        )
    )
    resolution_hypothesis: str | None = Field(
        description=(
            "If you can plausibly explain how both papers can be partially right "
            "(e.g. different settings, datasets, assumptions), state it briefly. "
            "Null when no plausible reconciliation comes to mind."
        )
    )


class Prerequisite(BaseModel):
    """One node in the prerequisite graph.

    NOTE: This exists because strict-mode JSON Schema cannot express
    `dict[str, list[str]]`. Open-keyed objects require additionalProperties to
    carry a type, which strict mode forbids. Lifting the dict into a list of
    typed pairs is the canonical workaround.
    """
    paper_id: str = Field(description="Paper that has prerequisites.")
    prerequisite_ids: list[str] = Field(
        description="IDs of papers that should be read first to understand this one."
    )


class ReadingPlan(BaseModel):
    """A recommended reading order for a corpus of papers."""

    ordered_paper_ids: list[str] = Field(
        description=(
            "All paper IDs in the recommended reading order — earliest "
            "(easiest entry point) first, most advanced last."
        )
    )
    rationale: str = Field(
        description=(
            "2-4 sentences explaining the ordering: what concept builds on what, "
            "and why this sequence minimizes prerequisite gaps."
        )
    )
    prerequisites: list[Prerequisite] = Field(
        description=(
            "For each paper that has prerequisites, list the papers that should "
            "be read first. Papers without prerequisites can be omitted."
        )
    )


class SynthesisReport(BaseModel):
    """Top-level synthesis across N papers."""

    contradictions: list[Contradiction] = Field(
        description=(
            "Pairs of papers that disagree, if any. Empty list is the right "
            "answer when no genuine contradictions exist — do not invent them."
        )
    )
    reading_plan: ReadingPlan = Field(description="A recommended reading order.")
    summary: str = Field(
        description=(
            "A 4-6 sentence overview tying the corpus together: shared themes, "
            "trajectory of ideas, where the field stands."
        )
    )
