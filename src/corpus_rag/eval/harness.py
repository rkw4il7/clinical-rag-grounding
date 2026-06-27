"""Eval orchestration: Layer-1 retrieval metrics, Layer-2 reference-free,
Layer-3 auto-qrels (plan "Eval harness" step).

All model/DB access is injected as callables so the orchestration is unit-
testable offline:

- ``retrieve_fn(query) -> list[Document]`` — runs the retriever for a query.
- ``generate_fn(prompt) -> str`` — calls the LLM once, returns its reply.

``scripts/eval.py`` supplies the live implementations; tests pass fakes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from corpus_rag.eval.metrics import RetrievalMetrics, aggregate, per_query
from corpus_rag.eval.qrels import (
    EvalCase,
    RelevanceSpec,
    relevance_flags,
    relevance_gains,
    specs_covered,
)
from corpus_rag.prompts import ABSTENTION_ANSWER

if TYPE_CHECKING:
    from haystack import Document

RetrieveFn = Callable[[str], "list[Document]"]
GenerateFn = Callable[[str], str]


# --- Layer 1: labelled retrieval metrics --------------------------------


@dataclass(frozen=True)
class CaseResult:
    """One query's retrieval outcome (for per-query reporting)."""

    query: str
    metrics: RetrievalMetrics
    n_retrieved: int
    n_relevant: int
    covered: int


def evaluate_retrieval(
    cases: Sequence[EvalCase],
    retrieve_fn: RetrieveFn,
    *,
    k: int,
) -> tuple[RetrievalMetrics, list[CaseResult]]:
    """Run Layer-1 retrieval metrics over a gold qrels set.

    :returns: ``(macro_metrics, per_case_results)``.
    """
    if not cases:
        raise ValueError("no eval cases provided")
    per_case: list[CaseResult] = []
    for case in cases:
        docs = retrieve_fn(case.query)
        flags = relevance_flags(docs, case.relevant)
        gains = relevance_gains(docs, case.relevant)
        covered = specs_covered(docs[:k], case.relevant)
        m = per_query(flags, k=k, covered=covered, total_relevant=len(case.relevant), gains=gains)
        per_case.append(
            CaseResult(
                query=case.query,
                metrics=m,
                n_retrieved=len(docs),
                n_relevant=sum(flags),
                covered=covered,
            )
        )
    macro = aggregate([c.metrics for c in per_case])
    return macro, per_case


# --- Layer 2: reference-free metrics ------------------------------------


def abstention_rate(
    queries: Sequence[str],
    run_fn: Callable[[str], tuple[str, list[Document]]],
) -> float:
    """Fraction of queries the system abstained on (no/low grounding)."""
    if not queries:
        return 0.0
    n_abstain = sum(1 for q in queries if run_fn(q)[0] == ABSTENTION_ANSWER)
    return n_abstain / len(queries)


def citation_coverage(answer: str, docs: Sequence[Document]) -> float:
    """Lexical PROXY for grounding: fraction of the answer's content words that
    also appear in the union of retrieved chunk texts.

    NOT a substitute for the LLM-judge faithfulness check — a high overlap does
    not prove entailment, and a paraphrased-but-faithful answer scores low. Use
    as a cheap, model-free smoke signal only.
    """
    answer_words = {w for w in _content_words(answer)}
    if not answer_words:
        return 0.0
    source_words: set[str] = set()
    for d in docs:
        source_words.update(_content_words(d.content or ""))
    return len(answer_words & source_words) / len(answer_words)


def _content_words(text: str) -> set[str]:
    """Lowercase alphanumeric tokens of length >= 3 (drop trivial stopword-ish)."""
    import re

    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3}


FAITHFULNESS_JUDGE_TEMPLATE = """\
You are an audit judge for a clinical RAG system. Decide whether EVERY specific
clinical claim in the ANSWER is supported by the SOURCES below. Specific claims
are facts, values, doses, thresholds, drug names, or recommendations. General
non-specific reasoning does not need support.

Reply with EXACTLY one word: SUPPORTED or UNSUPPORTED.

SOURCES:
{sources}

ANSWER:
{answer}

VERDICT:"""


def judge_faithfulness(
    answer: str,
    docs: Sequence[Document],
    generate_fn: GenerateFn,
) -> bool | None:
    """LLM-judge: are all specific claims in ``answer`` entailed by ``docs``?

    Caveat: by default the judge runs on the SAME local model as the generator
    (``scripts/eval.py`` reuses one endpoint), so this is a self-grading loop, not
    an independent audit. For a real audit, pass a ``generate_fn`` backed by a
    different model.

    :returns: ``True`` (supported), ``False`` (unsupported), or ``None`` when the
        answer is an abstention (nothing to judge — excluded from the rate).
    """
    if answer == ABSTENTION_ANSWER or not answer.strip():
        return None
    sources = "\n\n".join(f"[Source {i}]\n{d.content}" for i, d in enumerate(docs, start=1))
    prompt = FAITHFULNESS_JUDGE_TEMPLATE.format(sources=sources, answer=answer)
    verdict = generate_fn(prompt).strip().upper()
    # Robust to extra prose: look for the tokens, prefer UNSUPPORTED if ambiguous.
    if "UNSUPPORTED" in verdict:
        return False
    if "SUPPORTED" in verdict:
        return True
    # Unparseable verdict → treat as unsupported (fail safe for a safety metric).
    return False


def faithfulness_rate(
    queries: Sequence[str],
    run_fn: Callable[[str], tuple[str, list[Document]]],
    generate_fn: GenerateFn,
) -> tuple[float | None, int]:
    """Fraction of non-abstaining answers judged fully grounded.

    :returns: ``(rate, n_judged)``. ``rate`` is ``None`` when nothing was
        judgeable (every answer abstained) — NOT 1.0, which would falsely read as
        "100% faithful". ``n_judged`` discloses the sample size.
    """
    verdicts = []
    for q in queries:
        answer, docs = run_fn(q)
        v = judge_faithfulness(answer, docs, generate_fn)
        if v is not None:
            verdicts.append(v)
    if not verdicts:
        return None, 0
    return sum(1 for v in verdicts if v) / len(verdicts), len(verdicts)


# --- Layer 3: auto-generated qrels --------------------------------------


QUESTION_GEN_TEMPLATE = """\
Read the clinical SOURCE passage below and write ONE specific question that this
passage directly and completely answers. The question must be answerable from
this passage alone. Output ONLY the question, nothing else.

SOURCE:
{passage}

QUESTION:"""


def _distinctive_snippet(content: str, length: int) -> str:
    """A trimmed, single-line snippet of a chunk, used as its ``contains`` key."""
    flat = " ".join((content or "").split())
    return flat[:length]


def auto_generate_qrels(
    docs: Sequence[Document],
    generate_fn: GenerateFn,
    *,
    n: int,
    snippet_len: int = 80,
) -> list[EvalCase]:
    """Synthesize qrels FROM the ingested corpus (Layer 3), no human labels.

    For up to ``n`` evenly-sampled chunks, ask the LLM to write a question the
    chunk answers; the chunk becomes that question's gold-relevant doc, matched by
    its first ``snippet_len`` characters.

    Caveats (do not oversell this):
    - **Self-retrieval circularity.** The question is generated FROM the chunk and
      then used to test retrieval of that SAME chunk, so this measures
      self-retrievability — a weak proxy that tends to inflate retrieval scores,
      not an independent gold set.
    - **Snippet fragility.** The gold key is a fixed-length content prefix; if a
      later re-chunk splits inside those characters the substring match can break.
    - Sampling is deterministic (evenly spaced), so the set is stable for a fixed
      corpus + model at temperature 0.
    """
    usable = [d for d in docs if (d.content or "").strip()]
    if not usable:
        return []
    n = min(n, len(usable))
    # Evenly spaced, deterministic indices across the corpus.
    step = len(usable) / n
    chosen = [usable[int(i * step)] for i in range(n)]

    cases: list[EvalCase] = []
    for doc in chosen:
        snippet = _distinctive_snippet(doc.content, snippet_len)
        if not snippet:
            continue
        question = generate_fn(QUESTION_GEN_TEMPLATE.format(passage=doc.content)).strip()
        if not question:
            continue
        cases.append(EvalCase(query=question, relevant=(RelevanceSpec(contains=snippet),)))
    return cases
