"""
Does the parser tuned on 5 synthetic PDFs survive a real insurance policy?

Every retrieval number in this project comes from the 5 synthetic policies in
data/policy_pdfs/synthetic/. Three genuine IRDAI-filed policies sit in
data/policy_pdfs/reference/ and have only ever been text-extracted -- never
chunked into an index, never scored. This script parses both sets at
production settings and compares their structure.

There is no groundtruth for the real documents, so this cannot report P@3.
What it can report is whether the damage-class tagger -- which drives both
retrieval relevance and clause bucketing -- finds anything to hold onto.

The second pass matters as much as the first. A low tag rate could mean the
tagger is broken, or it could mean the document genuinely does not discuss
those damage types. So the script also greps the raw text for each damage
family independently of the tagger, and flags only the cases where a word is
present but untagged. (Run as of 2026-08-07 the sole flagged case was the
word "dental" matching a loose probe for "dent" -- the tagger was correct to
ignore it, and there is no tagger bug.)

    python scripts/check_real_policy_parse.py
"""
import re
from collections import Counter
from pathlib import Path

from src.retrieval.policy_parser import (
    DAMAGE_KEYWORDS,
    clean_text,
    extract_pdf_text,
    parse_policy_pdf,
)

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = ROOT / "data" / "policy_pdfs" / "synthetic"
REFERENCE_DIR = ROOT / "data" / "policy_pdfs" / "reference"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
DEDUP_THRESHOLD = 0.90

# Deliberately looser than the production keyword lists: this is a presence
# probe, not a tagger. Anything it catches that the tagger does not becomes a
# candidate tagger gap to inspect by hand.
PRESENCE_PROBE = {
    "dent": r"\bdent",
    "scratch": r"\bscratch",
    "crack": r"\bcrack",
    "broken_lamp": r"\blamp|\bheadlight|\btail ?light",
    "shattered_glass": r"\bglass|windscreen|windshield|\bwindow",
    "flat_tyre": r"\btyre|\btire|puncture",
}


def structure(pdf_path):
    chunks = parse_policy_pdf(pdf_path, CHUNK_SIZE, CHUNK_OVERLAP, DEDUP_THRESHOLD)
    if not chunks:
        return None
    n = len(chunks)
    clause_types = Counter(c["clause_type"] for c in chunks)
    classes_seen = {k for c in chunks for k in c["damage_classes"]}
    untagged = sum(1 for c in chunks if not c["damage_classes"])
    return {
        "n_chunks": n,
        "pct_untagged": round(100 * untagged / n),
        "pct_general": round(100 * clause_types.get("general", 0) / n),
        "classes_covered": len(classes_seen),
        "clause_types": dict(clause_types.most_common()),
    }


def tagger_gaps(pdf_path):
    """Damage families whose words appear in the raw text but that the
    production tagger does not fire on. Non-empty means inspect by hand."""
    text = clean_text(extract_pdf_text(pdf_path)).lower()
    present = {c for c, pat in PRESENCE_PROBE.items() if re.search(pat, text)}
    tagged = {c for c, pats in DAMAGE_KEYWORDS.items()
              if any(re.search(p, text) for p in pats)}
    return sorted(present), sorted(present - tagged)


def run(directory, label):
    print(f"\n--- {label} ---")
    for pdf_path in sorted(directory.glob("*.pdf")):
        s = structure(pdf_path)
        name = pdf_path.stem[:30]
        if s is None:
            print(f"  {name:32} PARSE FAILED / EMPTY")
            continue
        present, gaps = tagger_gaps(pdf_path)
        print(f"  {name:32} {s['n_chunks']:>4} chunks | untagged {s['pct_untagged']:>3}% | "
              f"general {s['pct_general']:>3}% | classes tagged {s['classes_covered']}/6 | "
              f"words present {len(present)}/6")
        print(f"  {'':32}   clause_type: {s['clause_types']}")
        if gaps:
            print(f"  {'':32}   ⚠ present but untagged: {gaps} -- inspect by hand")


def main():
    run(SYNTHETIC_DIR, "SYNTHETIC (what every reported number was tuned on)")
    run(REFERENCE_DIR, "REAL IRDAI-FILED (never indexed, never scored)")
    print("\nNote: 'words present' uses a deliberately loose probe, so it can exceed "
          "'classes tagged'\nlegitimately. Only the ⚠ line indicates a possible tagger gap.")


if __name__ == "__main__":
    main()
