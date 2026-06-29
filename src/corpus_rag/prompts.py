"""RAG prompt template enforcing the §2A grounding contract.

The template is the Stage-2 instruction (root ``spec.md`` §2A.2): the model
receives the original query plus the verbatim text of the top-N retrieved chunks
and must answer ONLY from that retrieved content. Specific clinical facts must
originate from the sources; general non-specific concepts may act as connective
reasoning; the model abstains when grounding is insufficient.
"""

from __future__ import annotations

# Exact string the model emits (and run_query falls back to) when grounding is
# insufficient. Kept as a constant so tests and the UI can detect abstention.
ABSTENTION_ANSWER = "Insufficient grounding in the corpus to answer."

# Jinja2 template for Haystack PromptBuilder. `documents` is the retriever's
# ranked list; `query` is the original user question. The abstention sentence is
# interpolated once here so the constant remains the single source of truth.
RAG_PROMPT_TEMPLATE = """\
You are a clinical corpus assistant. Answer the QUESTION using ONLY the RETRIEVED
SOURCES below, plus basic, non-specific clinical concepts as connective reasoning.

Hard rules:
- Every specific clinical fact, value, dose, threshold, or claim MUST come from
  the retrieved sources. NEVER use specific clinical knowledge from your own
  training data.
- General, non-specific clinical reasoning (common definitions, basic concepts)
  is allowed only as connective tissue, never as the source of a specific claim.
- If the retrieved sources do not support a specific answer to the question,
  reply with exactly this sentence and nothing else:
  "__ABSTENTION__"

{% if documents %}
RETRIEVED SOURCES (most relevant first):
{% for doc in documents %}
[Source {{ loop.index }}]
{{ doc.content }}
{% endfor %}
{% else %}
RETRIEVED SOURCES: (none)
{% endif %}

QUESTION: {{ query }}

ANSWER:""".replace("__ABSTENTION__", ABSTENTION_ANSWER)


# Jinja2 template for Haystack PromptBuilder (NOT str.replace): rendering through
# the same engine as RAG_PROMPT_TEMPLATE inserts source/query/partial-answer text
# as literal variable VALUES, so corpus or query text that happens to contain a
# placeholder token cannot corrupt the prompt structure (no injection).
CONTINUE_RAG_PROMPT_TEMPLATE = """\
You are a clinical corpus assistant. Continue the PREVIOUS ANSWER using ONLY the
RETRIEVED SOURCES below, plus basic, non-specific clinical concepts as connective
reasoning.

Hard rules:
- Continue from exactly where the PREVIOUS ANSWER stopped. Do not restart the
  answer or repeat completed sections unless repetition is needed to complete the
  interrupted sentence.
- Every specific clinical fact, value, dose, threshold, or claim MUST come from
  the retrieved sources. NEVER use specific clinical knowledge from your own
  training data.
- Return only the continuation text.

RETRIEVED SOURCES (most relevant first):
{% for doc in documents %}
[Source {{ loop.index }}]
{{ doc.content }}
{% endfor %}

QUESTION: {{ query }}

PREVIOUS ANSWER:
{{ partial_answer }}

CONTINUATION:"""
