"""Generate the synthetic sample corpus PDF (``tests/data/sample-clinical-guideline.pdf``).

Reproducible source for the small, committed sample document so the ingest →
chunk → embed → retrieve path is always runnable on a fresh clone (no reliance on
a local/LAN corpus). The content is SYNTHETIC, non-PHI, common-knowledge reference
text with two headed sections — enough for Docling's HybridChunker to emit
multiple chunks with heading provenance.

Run:  uv run python scripts/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "tests" / "data" / "sample-clinical-guideline.pdf"

DISCLAIMER = (
    "Synthetic sample document for software testing only. It contains general, "
    "common-knowledge reference text and is not medical advice."
)

# Two logical entries, each a heading + two paragraphs (four paragraphs total).
SECTIONS = [
    (
        "Hand Hygiene in Clinical Settings",
        [
            "Hand hygiene is widely regarded as the single most effective routine "
            "measure for reducing the transmission of common pathogens between "
            "patients and staff. General guidance frames it around clear moments of "
            "care, such as before and after contact with a patient or their "
            "immediate surroundings.",
            "When hands are not visibly soiled, an alcohol-based hand rub is "
            "commonly used for speed and convenience. When hands are visibly soiled, "
            "washing with soap and water is the general reference practice. These are "
            "well-established, non-specific principles rather than situation-specific "
            "instructions.",
        ],
    ),
    (
        "Adult Vital Signs: General Reference Ranges",
        [
            "For a resting adult, commonly cited general reference ranges include a "
            "heart rate of approximately 60 to 100 beats per minute and a "
            "respiratory rate of about 12 to 20 breaths per minute. These figures are "
            "textbook reference values, not thresholds for any individual.",
            "Body temperature is often described around 36.5 to 37.5 degrees Celsius. "
            "Any interpretation for a specific person should come from a qualified "
            "clinician; the values here exist only to provide retrievable, "
            "non-sensitive sample text for testing the retrieval pipeline.",
        ],
    ),
]


def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(OUT), pagesize=LETTER, title="Sample Clinical Reference (Synthetic)"
    )
    flow = [
        Paragraph("Sample Clinical Reference (Synthetic)", styles["Title"]),
        Paragraph(DISCLAIMER, styles["Italic"]),
        Spacer(1, 18),
    ]
    for heading, paragraphs in SECTIONS:
        flow.append(Paragraph(heading, styles["Heading1"]))
        for para in paragraphs:
            flow.append(Paragraph(para, styles["BodyText"]))
            flow.append(Spacer(1, 6))
        flow.append(Spacer(1, 12))
    doc.build(flow)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
