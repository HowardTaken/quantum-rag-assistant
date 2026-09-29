"""Manual retrieval-quality eval.

Not part of the pytest suite on purpose: it calls the real Gemini embedding
API for each query, so it costs quota and needs a real GOOGLE_API_KEY / a
populated chroma_db/. Run it by hand after changing chunking, the retrieval
k, or the embedding model, to check retrieval quality didn't regress:

    python -m eval.retrieval_eval

It measures top-k hit rate: for each hand-written question (grounded in the
actual paper contents), does the known correct source document appear
anywhere in the top-k chunks retrieved for it? This is a proxy for "can the
agent even find the right passage," not a check of the generated answer.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import PurePath

from config import get_settings
from tools import load_vector_store

# Each question is grounded in a real sentence from the corresponding paper
# (see the papers/ PDFs) so this evolves with the actual document set instead
# of testing against made-up facts.
EVAL_SET = [
    (
        "What frequency range defines the terahertz band?",
        "nphoton.2007.166.pdf",
    ),
    (
        "What is the maximum tuning frequency reached by the tunable metasurface "
        "external-cavity quantum cascade laser?",
        "tunable-metasurface-external-cavity-quantum-cascade-lasers-up-to-5-74-thz.pdf",
    ),
    (
        "How is disorder introduced into the multi-mode VECSEL metasurface?",
        "oe-33-9-18891.pdf",
    ),
    (
        "What THz frequency range do the multi-mode quantum-cascade VECSELs based on "
        "disordered metasurfaces target?",
        "oe-33-9-18891.pdf",
    ),
    (
        "What historically has the terahertz frequency range lacked, in terms of "
        "technology?",
        "nphoton.2007.166.pdf",
    ),
]


@dataclass
class EvalResult:
    query: str
    expected_source: str
    retrieved_sources: list[str]

    @property
    def hit(self) -> bool:
        return self.expected_source in self.retrieved_sources


def run_eval() -> list[EvalResult]:
    settings = get_settings()
    db = load_vector_store()
    results = []
    for query, expected_source in EVAL_SET:
        docs = db.similarity_search(query, k=settings.retrieval_k)
        retrieved = [d.metadata.get("source", "unknown") for d in docs]
        results.append(EvalResult(query, expected_source, retrieved))
    return results


def main() -> int:
    results = run_eval()
    hits = sum(r.hit for r in results)
    for r in results:
        status = "HIT " if r.hit else "MISS"
        print(f"[{status}] {r.query}")
        print(f"        expected: {r.expected_source}")
        print(f"        got:      {r.retrieved_sources}")
    print(f"\nTop-k hit rate: {hits}/{len(results)} ({100 * hits / len(results):.0f}%)")
    return 0 if hits == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
