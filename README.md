# PaperMind

**Learn OpenAI Structured Outputs and PydanticAI by building a research-paper triage pipeline. One concept per phase, ten phases.**

PaperMind is a CLI tool that ingests a folder of academic PDFs and produces a structured research dossier — extracted metadata, methodology classification, claim graphs, contradiction detection, and a recommended reading order — all driven by typed LLM outputs. The output is consumable by humans (markdown report) and machines (validated JSON for downstream RAG).

Equally importantly, it's a **learning project**: every file in this repo was written to confront one specific concept from the OpenAI Structured Outputs and PydanticAI documentation. If the docs gesture at it, PaperMind makes you feel it.

> _Insert a screenshot of the synthesis output or eval table here. The `rich`-rendered terminal output looks great._

---

## What you'll learn, by phase

| Phase | Concept | What you'll feel |
|---|---|---|
| 1 | Schema as contract | The basic `text_format=Pydantic` flow; why every field is required under strict mode |
| 2 | Constrained vocabularies | `Enum` vs `Literal`; designing closed vocabularies and the "other" escape hatch |
| 3 | Optional fields & explicit nullability | Why `Optional[X]` becomes `["string", "null"]` and the model emits `null` rather than omitting keys |
| 4 | Recursive schemas | Self-referential Pydantic types, `model_rebuild()`, `$ref: "#"` in JSON Schema |
| 5 | Chain-of-thought as structured output | Forcing reasoning *inside* the response schema, before the verdict |
| 6 | Refusals & failure-mode taxonomy | A discriminated union (`Ok` / `Refused` / `Truncated` / `Failed`) that turns errors into data |
| 7 | Function calling vs `text_format` | Tool args are schemas in a different slot — same JSON-Schema-with-strict mechanism |
| 8 | Cross-paper synthesis pipeline | Composing typed calls; the `dict[str, list[str]]` problem and the lifting workaround |
| 9 | PydanticAI agents | `ModelRetry`, tool decorators, dependency injection — production ergonomics over the same primitives |
| 10 | Evals & observability | Schema validity ≠ correctness; per-field metrics; honest tolerances; Logfire tracing |

---

## Quickstart

Requires Python 3.11+ and an OpenAI API key.

```bash
git clone https://github.com/Amin-Mirlohi/papermind.git
cd papermind

# Install with uv (recommended) or pip
uv pip install -e .

# Configure the API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# Drop a few arXiv PDFs into the data folder
mkdir -p papermind/data/papers
curl -L -o papermind/data/papers/attention.pdf https://arxiv.org/pdf/1706.03762

# Run Phase 1 — extract structured metadata
uv run python -m papermind.extractors.metadata papermind/data/papers

# Run the full Phase 8 pipeline — synthesizes across all PDFs
uv run python -m papermind.pipeline papermind/data/papers
```

The pipeline writes a `synthesis.md` report (human-readable) and produces validated Pydantic objects (machine-readable) at every stage.

---

## Stack

- [`openai`](https://github.com/openai/openai-python) (Responses API)
- [`pydantic`](https://github.com/pydantic/pydantic) — schemas, validation
- [`pydantic-ai`](https://ai.pydantic.dev/) — agents, retries, tools, DI
- [`pypdf`](https://github.com/py-pdf/pypdf) — PDF text extraction
- [`httpx`](https://github.com/encode/httpx) — arXiv API calls in Phase 7
- [`rich`](https://github.com/Textualize/rich) — terminal rendering

---

## Project layout

```
papermind/
├── schemas.py              # All Pydantic models, grouped by phase
├── pdf.py                  # PDF text extraction
├── result.py               # Phase 6 — Result tagged union (Ok/Refused/Truncated/Failed)
├── tools.py                # Phase 7 — tool argument schemas + arXiv API tools
├── pipeline.py             # Phase 8 — cross-paper synthesis pipeline
├── agents.py               # Phase 9 — PydanticAI agents
├── extractors/
│   ├── metadata.py         # Phase 1 — Paper metadata extraction
│   ├── classification.py   # Phase 2 — methodology / contribution classification
│   ├── claims.py           # Phase 3 — claim extraction with optional fields
│   ├── claim_graph.py      # Phase 4 — recursive claim trees
│   ├── assessment.py       # Phase 5 — CoT-as-schema paper assessment
│   ├── safe.py             # Phase 6 — robust safe_parse wrapper
│   ├── safe_demo.py        # Phase 6 — demonstrates each Result variant
│   └── research_agent.py   # Phase 7 — manual agent loop with arXiv tools
└── evals/
    └── run.py              # Phase 10 — eval harness with hand-labeled ground truth
```

Each phase file is self-contained — has its own `__main__` block — so you can read and run them in isolation.

---

## Honest framing

This repo is built for learning, not production. Things that are deliberately omitted:

- No retries, rate-limiting, or circuit breakers around API calls (Phase 9 covers `ModelRetry` for validation; SDK-level retries are out of scope).
- No caching of LLM outputs — every run hits the API.
- The eval set is three papers. A real eval set is at least 50, hand-labeled, and ideally from a corpus you didn't see while writing the prompts.
- The system prompts are written for clarity, not for cost or latency.

That said, the **patterns** here — typed pipelines, `Result`-style failure handling, schema-first design, `ModelRetry` for soft constraints — are the same patterns you'd reach for in a production system.

---

## License

MIT — see [LICENSE](LICENSE) (add this if you intend to publish).

---

## Acknowledgments

Inspired by the OpenAI Structured Outputs documentation and the PydanticAI examples. Built as a personal study project.
