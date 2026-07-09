import os
import re
import sys
import csv
import json
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Constants
DAMAGE_CLASSES = [
    "dent", "scratch", "crack", "broken_lamp", "shattered_glass", "flat_tyre"
]

# Keywords used to auto-tag chunks with damage classes
DAMAGE_KEYWORDS = {
    "dent":            ["dent", "dented", "denting", "panel damage", "body damage",
                        "sheet metal", "impact", "collision"],
    "scratch":         ["scratch", "scratched", "scratching", "surface damage",
                        "abrasion", "paint damage", "paintwork"],
    "crack":           ["crack", "cracked", "cracking", "fracture", "structural damage",
                        "body crack", "chassis","torn body"],
    "broken_lamp":     ["lamp", "light", "headlight", "taillight", "indicator",
                        "broken lamp", "lighting unit", "reflector"],
    "shattered_glass": ["glass", "windscreen", "windshield", "window", "shattered",
                        "broken glass", "glazing"],
    "flat_tyre":       ["tyre", "tire", "flat tyre", "puncture", "tube",
                        "rubber", "wheel", "burst tyre"],
}

# Clause type keywords for metadata tagging
CLAUSE_TYPE_KEYWORDS = {
    "coverage":     ["will indemnify", "shall pay", "covers", "covered", "insured against",
                     "compensation", "reimburse", "payable"],
    "exclusion":    ["shall not be liable", "not covered", "excluded", "does not cover",
                     "no claim", "not payable", "excluded from"],
    "sub_limit":    ["limited to", "maximum", "not exceeding", "upto", "up to",
                     "subject to a maximum", "capped at"],
    "condition":    ["provided that", "subject to", "on condition", "only if",
                     "condition precedent", "provided always"],
    "definition":   ["means", "shall mean", "defined as", "refers to", "herein referred"],
}

# Section heading patterns that act as hard chunk boundaries
SECTION_PATTERNS = [
    r"^SECTION\s+[IVX]+",
    r"^[A-Z][A-Z\s&/]+:$",
    r"^\d+\.\s+[A-Z]",
    r"^[a-z]\)\s",
    r"^ADD-ON\s+COVER",
    r"^GENERAL\s+EXCEPTION",
    r"^CONDITIONS?:",
    r"^EXCLUSIONS?:",
    r"^DEDUCTIBLE:",
    r"^CLAIM\s+PROCEDURE",
]

SECTION_RE = re.compile("|".join(SECTION_PATTERNS), re.IGNORECASE)

# PDF Extraction

def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from a PDF using pdfplumber.
    Applies a basic column-order correction for two-column layouts.
    """
    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                width  = page.width
                height = page.height

                # Try two-column split: if left-column text + right-column text
                # gives more words than full-page extraction, use it
                left   = page.within_bbox((0, 0, width / 2, height))
                right  = page.within_bbox((width / 2, 0, width, height))
                full   = page.extract_text() or ""
                l_text = left.extract_text() or ""
                r_text = right.extract_text() or ""

                if len(l_text.split()) + len(r_text.split()) > len(full.split()) * 1.05:
                    text = l_text.strip() + "\n\n" + r_text.strip()
                else:
                    text = full

                pages_text.append(text)
    except Exception as e:
        log.error("Failed to extract %s: %s", pdf_path.name, e)
        return ""

    return "\n\n".join(pages_text)


def clean_text(raw: str, doc_name: str) -> str:
    """
    Remove common PDF artefacts: page headers/footers, excessive whitespace,
    ligature substitutions, and encoding noise.
    """
    lines   = raw.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Drop page header/footer patterns
        if re.match(r"^Page\s+\d+\s+of\s+\d+", stripped, re.IGNORECASE):
            continue
        if re.match(r"^Policy Wordings", stripped, re.IGNORECASE):
            continue
        if re.match(r"^UIN:\s+IRDAN", stripped, re.IGNORECASE):
            continue
        if re.match(r"^\*+$", stripped):
            continue

        # Fix common ligature artefacts
        stripped = stripped.replace("ﬁ", "fi").replace("ﬂ", "fl")
        stripped = stripped.replace("\u2013", "-").replace("\u2014", " - ")
        stripped = stripped.replace("\u2018", "'").replace("\u2019", "'")
        stripped = stripped.replace("\u201c", '"').replace("\u201d", '"')

        # Collapse excessive spaces
        stripped = re.sub(r"  +", " ", stripped)

        if stripped:
            cleaned.append(stripped)

    return "\n".join(cleaned)


# Section-aware chunking

def split_into_sections(text: str) -> list:
    """
    Split text on section headings before applying the token splitter.
    Returns a list of section strings.
    """
    sections = []
    current  = []

    for line in text.splitlines():
        if SECTION_RE.match(line.strip()) and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    return [s for s in sections if len(s) > 20]


def chunk_document(text: str, chunk_size: int = 300, chunk_overlap: int = 40) -> list:
    """
    Section-aware chunking: split on headings first, then apply
    RecursiveCharacterTextSplitter within each section.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )

    sections = split_into_sections(text)
    chunks   = []
    for section in sections:
        sub_chunks = splitter.split_text(section)
        chunks.extend(sub_chunks)

    # Filter very short chunks (likely headers or page noise)
    return [c.strip() for c in chunks if len(c.strip()) > 40]


# Damage-class and clause-type tagging

def tag_damage_classes(chunk: str) -> list:
    """Return list of damage classes mentioned or implied in a chunk."""
    chunk_lower = chunk.lower()
    tags = []
    for cls, keywords in DAMAGE_KEYWORDS.items():
        if any(kw in chunk_lower for kw in keywords):
            tags.append(cls)
    return tags


def tag_clause_type(chunk: str) -> str:
    """Return the most likely clause type for a chunk."""
    chunk_lower = chunk.lower()
    scores = {ct: 0 for ct in CLAUSE_TYPE_KEYWORDS}
    for ct, keywords in CLAUSE_TYPE_KEYWORDS.items():
        scores[ct] = sum(1 for kw in keywords if kw in chunk_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# Deduplication

def dedup_chunks(chunks: list, threshold: float = 0.90) -> list:
    """
    Remove near-identical chunks based on normalised character overlap.
    Two chunks are considered near-duplicates if their Jaccard similarity
    of word trigrams exceeds the threshold.
    """
    def trigrams(text):
        words = text.lower().split()
        return set(zip(words, words[1:], words[2:])) if len(words) >= 3 else set(words)

    unique  = []
    seen_tg = []

    for chunk in chunks:
        tg = trigrams(chunk)
        is_dup = False
        for ref in seen_tg:
            if not tg or not ref:
                continue
            jaccard = len(tg & ref) / len(tg | ref)
            if jaccard >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(chunk)
            seen_tg.append(tg)

    return unique


# Embedding and indexing 

def embed_and_index(
    all_chunks: list,
    all_metadata: list,
    model: SentenceTransformer,
    collection,
):
    """Embed all chunks and add to ChromaDB collection in batches."""
    BATCH = 64
    total = len(all_chunks)
    log.info("Embedding %d chunks in batches of %d...", total, BATCH)

    for start in tqdm(range(0, total, BATCH), desc="Embedding + indexing", unit="batch"):
        batch_texts = all_chunks[start: start + BATCH]
        batch_meta  = all_metadata[start: start + BATCH]
        batch_ids   = [f"chunk_{start + i:05d}" for i in range(len(batch_texts))]

        embeddings  = model.encode(batch_texts, show_progress_bar=False).tolist()

        # ChromaDB metadata values must be str/int/float/bool
        safe_meta = []
        for m in batch_meta:
            safe_meta.append({
                "doc_id":        m.get("doc_id", ""),
                "damage_classes": ",".join(m.get("damage_classes", [])),
                "clause_type":   m.get("clause_type", "general"),
                "chunk_index":   int(m.get("chunk_index", 0)),
                "text_preview":  m.get("text", "")[:120],
            })

        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=safe_meta,
        )

    log.info("Indexed %d chunks total", collection.count())


# Ground-truth mapping

def build_groundtruth(all_chunks: list, all_metadata: list, output_dir: Path):
    """Save chunk_id → damage classes + clause type mapping as JSON."""
    gt = {}
    for i, (chunk, meta) in enumerate(zip(all_chunks, all_metadata)):
        chunk_id = f"chunk_{i:05d}"
        gt[chunk_id] = {
            "text_preview":   chunk[:150],
            "damage_classes": meta.get("damage_classes", []),
            "clause_type":    meta.get("clause_type", "general"),
            "doc_id":         meta.get("doc_id", ""),
        }

    out_path = output_dir / "clause_groundtruth.json"
    with open(out_path, "w") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    log.info("Saved clause_groundtruth.json (%d entries)", len(gt))
    return gt


# Synthetic incident descriptions

INCIDENT_TEMPLATES = [
    # dent
    ("img_{i:04d}", "dent",
     "While parking, the vehicle was struck by an adjacent car door, leaving a noticeable dent on the front left door panel."),
    ("img_{i:04d}", "dent",
     "A reversing vehicle backed into the front bumper in a parking lot, causing a significant dent."),
    ("img_{i:04d}", "dent",
     "The car sustained a dent on the rear quarter panel after a minor collision at a junction."),
    ("img_{i:04d}", "dent",
     "Hailstorm left multiple small dents across the bonnet and roof of the vehicle."),
    ("img_{i:04d}", "dent",
     "During a slow-speed collision on a highway slip road, the driver-side door received a deep dent."),
    ("img_{i:04d}", "dent",
     "A shopping trolley rolled into the vehicle in a supermarket car park, creating a dent on the rear door."),
    ("img_{i:04d}", "dent",
     "The vehicle was sideswiped by a truck, leaving a long dent along the passenger-side panels."),
    ("img_{i:04d}", "dent",
     "Heavy debris falling from a construction site overhead caused multiple dents on the vehicle roof."),

    # scratch
    ("img_{i:04d}", "scratch",
     "A vandal keyed the vehicle overnight in the apartment parking, leaving deep scratches on both side doors."),
    ("img_{i:04d}", "scratch",
     "The insured accidentally scraped a concrete pillar in an underground car park, scratching the front bumper."),
    ("img_{i:04d}", "scratch",
     "A motorcyclist clipped the vehicle while overtaking, leaving a paint scratch on the rear quarter panel."),
    ("img_{i:04d}", "scratch",
     "Tree branches during a storm dragged across the bonnet, leaving multiple surface scratches."),
    ("img_{i:04d}", "scratch",
     "Improper car wash brush caused fine circular scratches across the entire vehicle body."),
    ("img_{i:04d}", "scratch",
     "The vehicle was scratched on the door sill area when luggage was being loaded at an airport."),
    ("img_{i:04d}", "scratch",
     "A cyclist lost control and their handlebar scraped the passenger door, causing a paint scratch."),
    ("img_{i:04d}", "scratch",
     "During an attempted break-in, the lock area of the driver door was deeply scratched."),

    # crack
    ("img_{i:04d}", "crack",
     "A large stone thrown up by an oncoming lorry cracked the front bumper assembly."),
    ("img_{i:04d}", "crack",
     "The vehicle's plastic bumper cracked after a low-speed collision with a pavement edge."),
    ("img_{i:04d}", "crack",
     "Extreme temperature differential caused the dashboard trim to crack along its seam."),
    ("img_{i:04d}", "crack",
     "A rear-end shunt in traffic cracked the tail-light housing and the bumper below it."),
    ("img_{i:04d}", "crack",
     "A heavy object fell onto the vehicle from a building balcony, cracking the roof panel and the A-pillar trim."),
    ("img_{i:04d}", "crack",
     "Side impact from another vehicle cracked the door skin at the pressed crease line."),

    # broken lamp
    ("img_{i:04d}", "broken_lamp",
     "The headlight assembly was shattered when the vehicle was rear-ended at a red light."),
    ("img_{i:04d}", "broken_lamp",
     "A reversing incident in a parking lot broke the left indicator unit completely."),
    ("img_{i:04d}", "broken_lamp",
     "Vandals smashed both tail light assemblies while the vehicle was parked on a public road overnight."),
    ("img_{i:04d}", "broken_lamp",
     "The front fog light was struck and broken when the vehicle went over an unmarked speed breaker at speed."),
    ("img_{i:04d}", "broken_lamp",
     "A side collision broke the rear reflector and the adjacent tail lamp housing."),
    ("img_{i:04d}", "broken_lamp",
     "The right headlight cluster was completely destroyed when the vehicle clipped a concrete divider."),
    ("img_{i:04d}", "broken_lamp",
     "Flying debris from roadworks shattered the driver-side fog lamp during highway driving."),

    # shattered glass
    ("img_{i:04d}", "shattered_glass",
     "The windscreen shattered after a large stone was projected by a passing truck on the expressway."),
    ("img_{i:04d}", "shattered_glass",
     "Break-in attempt resulted in the driver-side window being completely smashed."),
    ("img_{i:04d}", "shattered_glass",
     "A high-speed impact with a road barrier shattered both the windscreen and the passenger window."),
    ("img_{i:04d}", "shattered_glass",
     "Hailstones cracked and shattered the rear windscreen while the vehicle was parked in the open."),
    ("img_{i:04d}", "shattered_glass",
     "A vehicle rollover on a mountain road caused the sunroof glass panel to shatter completely."),
    ("img_{i:04d}", "shattered_glass",
     "Vandals smashed the rear window of the vehicle during a public disturbance event."),
    ("img_{i:04d}", "shattered_glass",
     "A falling tree branch during a storm shattered the windscreen and cracked the driver-side glass."),

    # flat tyre
    ("img_{i:04d}", "flat_tyre",
     "A nail embedded in the road punctured the rear left tyre, causing a sudden blowout on the highway."),
    ("img_{i:04d}", "flat_tyre",
     "The right front tyre burst after the vehicle drove over sharp debris at a construction site."),
    ("img_{i:04d}", "flat_tyre",
     "Both tyres on the driver side were slashed by vandals while the vehicle was parked."),
    ("img_{i:04d}", "flat_tyre",
     "A deep pothole ruptured the inner sidewall of the rear tyre, resulting in a flat."),
    ("img_{i:04d}", "flat_tyre",
     "A slow puncture developed in the front left tyre after encountering loose gravel."),
    ("img_{i:04d}", "flat_tyre",
     "The tyre valve was damaged during a kerb strike, causing air loss and a flat."),
    ("img_{i:04d}", "flat_tyre",
     "Exposure to a sharp piece of metal on a rural road caused a blowout to the rear right tyre."),

    # mixed / multi-damage
    ("img_{i:04d}", "dent,scratch",
     "A side-swipe collision left a dent and paint scratches along the entire passenger-side door."),
    ("img_{i:04d}", "crack,broken_lamp",
     "A low-speed front collision cracked the front bumper and shattered the right headlight assembly."),
    ("img_{i:04d}", "shattered_glass,dent",
     "A vehicle rolled over a road divider at speed, shattering the windscreen and causing roof dents."),
    ("img_{i:04d}", "flat_tyre,scratch",
     "Driving through debris on a flooded road caused a tyre puncture and road rash scratches on the lower sill."),
    ("img_{i:04d}", "broken_lamp,crack,dent",
     "A three-vehicle pileup caused front dents, cracked bumper panels, and broken headlight units."),
    ("img_{i:04d}", "scratch,crack",
     "Someone backed into the parked vehicle, scratching the boot lid and cracking the rear bumper moulding."),
]


def generate_incident_descriptions(output_dir: Path, n: int = 50):
    """Generate n synthetic incident descriptions."""
    incidents = []
    for i in range(min(n, len(INCIDENT_TEMPLATES))):
        template_id, damage_class, text = INCIDENT_TEMPLATES[i % len(INCIDENT_TEMPLATES)]
        incidents.append({
            "incident_id":    f"INC_{i+1:04d}",
            "image_id":       template_id.format(i=i + 1),
            "damage_classes": damage_class.split(","),
            "incident_date":  f"2026-0{(i % 9) + 1}-{(i % 28) + 1:02d}",
            "location":       ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
                               "Kolkata", "Pune", "Jaipur", "Ahmedabad", "Lucknow"][i % 10],
            "description":    text,
        })

    # Pad to n if templates < n
    while len(incidents) < n:
        extra = incidents[len(incidents) % len(INCIDENT_TEMPLATES)].copy()
        extra["incident_id"] = f"INC_{len(incidents)+1:04d}"
        incidents.append(extra)

    out_dir = output_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "incident_descriptions.json"
    with open(out_path, "w") as f:
        json.dump(incidents[:n], f, indent=2, ensure_ascii=False)
    log.info("Saved %d incident descriptions → %s", n, out_path)
    return incidents[:n]


# Retrieval smoke test

def run_smoke_test(collection, model: SentenceTransformer, output_dir: Path):
    """Run one retrieval query per damage class and save results."""
    test_queries = {
        "dent":            "Is dent damage covered under accidental external means?",
        "scratch":         "Does the policy cover scratches and surface paint damage?",
        "crack":           "Is cracking of bumper or body panels covered?",
        "broken_lamp":     "Are broken headlamps and tail lights covered?",
        "shattered_glass": "Is windscreen and window glass damage covered?",
        "flat_tyre":       "Is a flat tyre or tyre blowout covered under the policy?",
    }

    results = {}
    for cls, query in test_queries.items():
        query_emb = model.encode(query).tolist()
        resp = collection.query(
            query_embeddings=[query_emb],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            resp["documents"][0],
            resp["metadatas"][0],
            resp["distances"][0],
        ):
            hits.append({
                "chunk_id":      meta.get("text_preview", "")[:60],
                "clause_type":   meta.get("clause_type", ""),
                "damage_classes": meta.get("damage_classes", ""),
                "distance":      round(dist, 4),
                "text_preview":  doc[:120],
            })
        results[cls] = {"query": query, "top3": hits}
        log.info("Query [%-18s]: top result clause_type='%s'",
                 cls, hits[0]["clause_type"] if hits else "NONE")

    out_path = output_dir / "eval" / "retrieval_smoke_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Saved retrieval_smoke_test.json")
    return results


# TSV export

def export_chunks_tsv(all_chunks: list, all_metadata: list, output_dir: Path):
    out_path = output_dir / "chunks_all.tsv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["chunk_id", "doc_id", "damage_classes", "clause_type",
                         "chunk_len", "text"])
        for i, (chunk, meta) in enumerate(zip(all_chunks, all_metadata)):
            writer.writerow([
                f"chunk_{i:05d}",
                meta.get("doc_id", ""),
                ",".join(meta.get("damage_classes", [])),
                meta.get("clause_type", "general"),
                len(chunk),
                chunk.replace("\n", " "),
            ])
    log.info("Saved chunks_all.tsv (%d rows)", len(all_chunks))


# Main

def main():
    parser = argparse.ArgumentParser(description="Policy PDF preprocessing pipeline for RAG")
    parser.add_argument("--pdf_dir",       required=True,
                        help="Directory containing synthetic policy PDFs")
    parser.add_argument("--reference_dir", default=None,
                        help="Directory containing reference PDFs (extracted as text only, not indexed)")
    parser.add_argument("--output_dir",    required=True,
                        help="Directory for all output artefacts")
    parser.add_argument("--db_path",       default="./data/chroma_db",
                        help="Path for persistent ChromaDB storage")
    parser.add_argument("--collection",    default="policy_clauses",
                        help="ChromaDB collection name")
    parser.add_argument("--chunk_size",    default=300,  type=int)
    parser.add_argument("--chunk_overlap", default=40,   type=int)
    parser.add_argument("--dedup_thresh",  default=0.90, type=float,
                        help="Jaccard trigram similarity threshold for chunk dedup")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--n_incidents",   default=50,   type=int,
                        help="Number of synthetic incident descriptions to generate")
    parser.add_argument("--reset_collection", action="store_true",
                        help="Delete and recreate the ChromaDB collection if it already exists")
    args = parser.parse_args()

    pdf_dir    = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reference_text").mkdir(exist_ok=True)

    if not pdf_dir.exists():
        log.error("pdf_dir not found: %s", pdf_dir)
        sys.exit(1)

    # Load embedding model 
    log.info("Loading embedding model: %s", args.embedding_model)
    model = SentenceTransformer(args.embedding_model)
    log.info("Model loaded. Embedding dimension: %d", model.get_sentence_embedding_dimension())

    # Connect to ChromaDB
    log.info("Connecting to ChromaDB at: %s", args.db_path)
    client = chromadb.PersistentClient(path=args.db_path)

    if args.reset_collection:
        try:
            client.delete_collection(args.collection)
            log.info("Deleted existing collection '%s'", args.collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )
    log.info("Collection '%s' — existing chunks: %d", args.collection, collection.count())

    # Process reference PDFs (text extraction only, not indexed)
    if args.reference_dir:
        ref_dir = Path(args.reference_dir)
        ref_pdfs = list(ref_dir.glob("*.pdf"))
        log.info("Processing %d reference PDFs (text extraction only)", len(ref_pdfs))
        for pdf_path in ref_pdfs:
            raw  = extract_pdf_text(pdf_path)
            text = clean_text(raw, pdf_path.name)
            out  = output_dir / "reference_text" / f"{pdf_path.stem}.txt"
            with open(out, "w", encoding="utf-8") as f:
                f.write(text)
            log.info("  Reference: %-45s  → %d chars", pdf_path.name, len(text))

    # Process synthetic PDFs
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        log.warning("No PDF files found in %s", pdf_dir)
        sys.exit(1)

    log.info("=" * 60)
    log.info("Processing %d synthetic policy PDFs", len(pdf_paths))
    log.info("=" * 60)

    all_chunks   = []
    all_metadata = []
    doc_stats    = []

    for pdf_path in pdf_paths:
        log.info("Processing: %s", pdf_path.name)

        # Extract and clean
        raw_text    = extract_pdf_text(pdf_path)
        clean       = clean_text(raw_text, pdf_path.name)

        # Chunk
        chunks_raw  = chunk_document(clean, args.chunk_size, args.chunk_overlap)
        chunks_dedup = dedup_chunks(chunks_raw, threshold=args.dedup_thresh)
        removed_dup  = len(chunks_raw) - len(chunks_dedup)

        log.info("  Pages: ~%d chars | Raw chunks: %d | After dedup: %d (removed %d)",
                 len(clean), len(chunks_raw), len(chunks_dedup), removed_dup)

        # Tag each chunk
        doc_id = pdf_path.stem
        for idx, chunk in enumerate(chunks_dedup):
            damage_cls  = tag_damage_classes(chunk)
            clause_type = tag_clause_type(chunk)
            all_chunks.append(chunk)
            all_metadata.append({
                "doc_id":         doc_id,
                "chunk_index":    len(all_chunks) - 1,
                "damage_classes": damage_cls,
                "clause_type":    clause_type,
                "text":           chunk,
            })

        doc_stats.append({
            "doc_id":       doc_id,
            "raw_chars":    len(clean),
            "chunks_raw":   len(chunks_raw),
            "chunks_kept":  len(chunks_dedup),
            "dups_removed": removed_dup,
        })

    log.info("Total chunks to index: %d", len(all_chunks))

    # Embed and index
    embed_and_index(all_chunks, all_metadata, model, collection)

    # Ground-truth mapping
    build_groundtruth(all_chunks, all_metadata, output_dir)

    # TSV export
    export_chunks_tsv(all_chunks, all_metadata, output_dir)

    # Synthetic incident descriptions
    generate_incident_descriptions(output_dir, n=args.n_incidents)

    # Retrieval smoke test
    smoke_results = run_smoke_test(collection, model, output_dir)

    # Summary JSON
    coverage_by_class = defaultdict(int)
    for meta in all_metadata:
        for cls in meta.get("damage_classes", []):
            coverage_by_class[cls] += 1

    clause_type_dist = defaultdict(int)
    for meta in all_metadata:
        clause_type_dist[meta.get("clause_type", "general")] += 1

    summary = {
        "generated_at":       datetime.now().isoformat(),
        "embedding_model":    args.embedding_model,
        "collection_name":    args.collection,
        "db_path":            args.db_path,
        "chunk_size":         args.chunk_size,
        "chunk_overlap":      args.chunk_overlap,
        "total_chunks_indexed": collection.count(),
        "documents":          doc_stats,
        "coverage_by_damage_class": dict(coverage_by_class),
        "clause_type_distribution": dict(clause_type_dist),
        "smoke_test_summary": {
            cls: data["top3"][0]["clause_type"] if data["top3"] else "no result"
            for cls, data in smoke_results.items()
        },
    }

    with open(output_dir / "processing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # summary
    print("\n" + "=" * 60)
    print("POLICY PDF PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"PDFs processed:            {len(pdf_paths)}")
    print(f"Total chunks indexed:      {collection.count()}")
    print(f"ChromaDB path:             {args.db_path}")
    print(f"Collection name:           {args.collection}")
    print(f"\nChunks per damage class:")
    for cls in DAMAGE_CLASSES:
        print(f"  {cls:<22}  {coverage_by_class.get(cls, 0)}")
    print(f"\nClause type distribution:")
    for ct, cnt in sorted(clause_type_dist.items(), key=lambda x: -x[1]):
        print(f"  {ct:<14}  {cnt}")
    print(f"\nOutputs saved to:          {output_dir.resolve()}")
    print("=" * 60)
    print("\nQuery the index with:")
    print('  from sentence_transformers import SentenceTransformer')
    print('  import chromadb')
    print(f'  client = chromadb.PersistentClient(path="{args.db_path}")')
    print(f'  col    = client.get_collection("{args.collection}")')
    print('  model  = SentenceTransformer("all-MiniLM-L6-v2")')
    print('  q = "Is windscreen damage covered?"')
    print('  r = col.query(query_embeddings=[model.encode(q).tolist()], n_results=3)')
    print('  print(r["documents"])')


if __name__ == "__main__":
    main()
