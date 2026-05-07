"""Phase 4: extract a recursive claim graph from a paper."""

from openai import OpenAI

from papermind.schemas import ClaimGraph, ClaimNode


SYSTEM_PROMPT = (
    "You build the argument tree of an academic paper. The output is a single "
    "tree rooted at the paper's main claim, with sub-claims, assumptions, and "
    "limitations as children.\n\n"
    "Rules:\n"
    "- The root must be exactly one node of type 'main_claim'.\n"
    "- Add children only when a claim has real internal structure: it is "
    "  decomposed by sub-claims, or rests on stated assumptions, or carries "
    "  acknowledged limitations.\n"
    "- Atomic claims have an empty children list. Empty is the right answer.\n"
    "- Maximum tree depth: 3 levels (root + 2 levels of nesting). Prefer "
    "  breadth (more siblings) over depth.\n"
    "- Be concrete: paraphrase the paper, don't invent structure that isn't "
    "  there."
)


def extract_claim_graph(text: str, *, model: str = "gpt-4o-2024-08-06") -> ClaimNode:
    """Send paper text to the model; return the root of the validated tree."""
    client = OpenAI()

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        text_format=ClaimGraph,
    )

    return response.output_parsed.root


def tree_depth(node: ClaimNode) -> int:
    """Compute the depth of a claim tree (1 = just the root)."""
    if not node.children:
        return 1
    return 1 + max(tree_depth(child) for child in node.children)


def count_nodes(node: ClaimNode) -> int:
    """Total number of nodes in the tree."""
    return 1 + sum(count_nodes(child) for child in node.children)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from dotenv import load_dotenv
    from rich import print
    from rich.tree import Tree

    from papermind.extractors.metadata import SAMPLE_TEXT
    from papermind.pdf import read_pdf

    load_dotenv()

    def render(node: ClaimNode, parent: Tree) -> None:
        label = f"[bold]{node.type}[/bold]: {node.statement}"
        branch = parent.add(label)
        for child in node.children:
            render(child, branch)

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        pdf_paths = sorted(target.glob("*.pdf")) if target.is_dir() else [target]
    else:
        pdf_paths = []

    if not pdf_paths:
        print("[dim]No PDFs found — using built-in sample.[/dim]")
        root = extract_claim_graph(SAMPLE_TEXT)
        tree = Tree("Argument tree")
        render(root, tree)
        print(tree)
    else:
        for path in pdf_paths:
            print(f"\n[bold cyan]── {path.name} ──[/bold cyan]")
            text = read_pdf(path, max_pages=None)
            root = extract_claim_graph(text)
            print(
                f"[dim]depth={tree_depth(root)}  "
                f"nodes={count_nodes(root)}[/dim]"
            )
            tree = Tree("Argument tree")
            render(root, tree)
            print(tree)
