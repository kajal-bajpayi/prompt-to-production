"""
UC-X app.py — Ask My Documents
Policy Q&A agent: answers from 3 CMC policy files with citations, or refuses exactly.
Run: python app.py
"""

import sys
from pathlib import Path

import anthropic

# ── File paths ────────────────────────────────────────────────────────────────

POLICY_FILES = {
    "policy_hr_leave.txt": Path(__file__).parent / "../data/policy-documents/policy_hr_leave.txt",
    "policy_it_acceptable_use.txt": Path(__file__).parent / "../data/policy-documents/policy_it_acceptable_use.txt",
    "policy_finance_reimbursement.txt": Path(__file__).parent / "../data/policy-documents/policy_finance_reimbursement.txt",
}

REFUSAL_TEMPLATE = (
    "This question is not covered in the available policy documents "
    "(policy_hr_leave.txt, policy_it_acceptable_use.txt, policy_finance_reimbursement.txt). "
    "Please contact [relevant team] for guidance."
)

SYSTEM_PROMPT = """You are a policy Q&A agent for City Municipal Corporation (CMC).
Your sole job is to answer employee questions from the three policy documents provided below.

STRICT ENFORCEMENT RULES — no exceptions:
1. Answer from ONE document only. Never combine claims from two different documents into a single answer.
2. If answering the question requires combining information from more than one document, use the refusal template instead.
3. Never use hedging phrases: "while not explicitly covered", "typically", "generally understood", "it is common practice", or similar.
4. Every factual claim must include a citation in this exact format: (Source: <filename>, section <X.Y>)
5. If the question is not answered within the documents, respond with EXACTLY this refusal template and nothing else:

{refusal}

POLICY DOCUMENTS:

{documents}"""


# ── Skill: retrieve_documents ─────────────────────────────────────────────────

def retrieve_documents() -> dict[str, str]:
    """Load all 3 policy files and return {filename: content}. Halts on any missing file."""
    documents = {}
    for name, path in POLICY_FILES.items():
        resolved = path.resolve()
        if not resolved.exists():
            sys.exit(f"ERROR: Policy file not found: {resolved}\nCheck that the data directory is in place.")
        try:
            documents[name] = resolved.read_text(encoding="utf-8")
        except OSError as e:
            sys.exit(f"ERROR: Could not read {resolved}: {e}")
    return documents


# ── Skill: answer_question ────────────────────────────────────────────────────

def answer_question(question: str, documents: dict[str, str], client: anthropic.Anthropic) -> str:
    """
    Search indexed documents for a single-source answer with citation.
    Returns the exact refusal template if the question is not covered or requires cross-document blending.
    """
    doc_block = "\n\n".join(
        f"=== {name} ===\n{content}" for name, content in documents.items()
    )
    system = SYSTEM_PROMPT.format(refusal=REFUSAL_TEMPLATE, documents=doc_block)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text.strip()


# ── Interactive CLI ───────────────────────────────────────────────────────────

def main():
    print("Loading policy documents...", flush=True)
    documents = retrieve_documents()
    print(f"Loaded: {', '.join(documents.keys())}\n")

    client = anthropic.Anthropic()

    print("CMC Policy Q&A — type your question, or 'quit' to exit.\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        answer = answer_question(question, documents, client)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
