"""
What P@3 cannot see about chunk size.

Sweep F in sweep_rag_params.py shows mean P@3 rising as chunks get larger --
0.9133 at 300 chars, 0.9467 at 500. Taken at face value that argues for
bigger chunks. It shouldn't, for two reasons this script measures directly.

First, relevance is defined as "the chunk's regex damage-class tags intersect
the incident's classes". A longer chunk contains more text, matches more
keyword families, and is therefore relevant to more incidents by
construction. Sweep F already reports the random-retrieval baseline rising
in step (0.163 -> 0.185), so the lift over random falls even as raw P@3
climbs. The chunks are not being retrieved better; they are being counted
relevant more often.

Second, and invisible to any retrieval metric: the downstream ClauseRetriever
sorts each chunk into exactly one bucket via a single-valued clause_type. A
chunk that contains both a coverage grant and the exclusion qualifying it
gets one label, and the other half of its content becomes unreachable from
the bucket it belongs in -- the precise failure the two-query
coverage/exclusion split exists to prevent. This script counts those
mixed-signal chunks, plus multi-damage-class chunks, which are what the
faithfulness eval already flags as soft "multi-class chunk" warnings.

    python scripts/chunk_quality_analysis.py
"""
import json
import re
from pathlib import Path

from src.retrieval.policy_parser import (
    CLAUSE_TYPE_KEYWORDS,
    chunk_document,
    clean_text,
    contextualize,
    dedup_chunks,
    extract_pdf_text,
    tag_clause_type,
    tag_damage_classes,
)

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "policy_pdfs" / "synthetic"
OUT_PATH = ROOT / "data" / "rag_outputs" / "eval" / "chunk_quality_analysis.json"

CHUNK_OVERLAP = 40
DEDUP_THRESHOLD = 0.90
SIZES = [150, 200, 250, 300, 400, 500, 700, 1000]


def matches(text, patterns):
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)


def analyse(chunk_size):
    """Chunk the corpus at one size and measure evidence precision, not retrieval."""
    pre_dedup = 0
    chunks = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        text = clean_text(extract_pdf_text(pdf_path))
        raw = chunk_document(text, chunk_size, CHUNK_OVERLAP)
        pre_dedup += len(raw)
        chunks.extend(dedup_chunks(raw, DEDUP_THRESHOLD))

    n = len(chunks)
    multi_class = 0
    total_classes = 0
    mixed_signal = 0
    clause_types = {}

    for heading, body in chunks:
        ctx = contextualize(heading, body)
        classes = tag_damage_classes(ctx)
        total_classes += len(classes)
        if len(classes) >= 2:
            multi_class += 1

        ct = tag_clause_type(heading, body)
        clause_types[ct] = clause_types.get(ct, 0) + 1

        # Both a coverage grant and an exclusion/limit living in one chunk,
        # which a single-valued clause_type must then collapse to one label.
        has_cov = matches(ctx, CLAUSE_TYPE_KEYWORDS["coverage"])
        has_exc = (matches(ctx, CLAUSE_TYPE_KEYWORDS["exclusion"])
                   or matches(ctx, CLAUSE_TYPE_KEYWORDS["sub_limit"]))
        if has_cov and has_exc:
            mixed_signal += 1

    return {
        "chunk_size": chunk_size,
        "n_chunks_pre_dedup": pre_dedup,
        "n_chunks": n,
        "chunks_removed_by_dedup": pre_dedup - n,
        "mean_damage_classes_per_chunk": round(total_classes / n, 3),
        "multi_class_chunks": multi_class,
        "multi_class_pct": round(100 * multi_class / n, 1),
        "mixed_coverage_exclusion_chunks": mixed_signal,
        "mixed_signal_pct": round(100 * mixed_signal / n, 1),
        "clause_type_distribution": dict(sorted(clause_types.items())),
    }


def main():
    rows = [analyse(size) for size in SIZES]

    header = (f"{'size':>5} {'chunks':>7} {'dedup_rm':>9} {'cls/chunk':>10} "
              f"{'multi-class':>13} {'mixed cov+exc':>15}")
    print(header)
    for r in rows:
        print(f"{r['chunk_size']:>5} {r['n_chunks']:>7} {r['chunks_removed_by_dedup']:>9} "
              f"{r['mean_damage_classes_per_chunk']:>10.3f} "
              f"{str(r['multi_class_chunks']) + ' (' + str(r['multi_class_pct']) + '%)':>13} "
              f"{str(r['mixed_coverage_exclusion_chunks']) + ' (' + str(r['mixed_signal_pct']) + '%)':>15}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"chunk_overlap": CHUNK_OVERLAP,
                   "dedup_threshold": DEDUP_THRESHOLD,
                   "configs": rows}, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
