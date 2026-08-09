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

# Keywords used to auto-tag chunks with damage classes. Word-boundary
# regex (not bare substring matching) -- see the tag_damage_classes
# docstring below for the specific bugs this fixes.
DAMAGE_KEYWORDS = {
    "dent": [
        r"\bdents?\b", r"\bdented\b", r"\bdenting\b", r"panel damage",
        r"body damage", r"sheet metal", r"\bimpact\b", r"\bcollision\b",
        r"deformations? of body panels?",
    ],
    "scratch": [
        r"\bscratch(es|ed|ing)?\b", r"surface damage", r"\babrasion\b",
        r"paint damage", r"\bpaintwork\b", r"\bkeying\b", r"\bkeyed\b",
        r"paint abrasion", r"paint scratches?",
    ],
    "crack": [
        r"\bcracks?\b", r"\bcracked\b", r"\bcracking\b", r"\bfracture[sd]?\b",
        r"structural damage", r"\bchassis\b",
    ],
    "broken_lamp": [
        r"\blamps?\b", r"\bheadlamps?\b", r"\bheadlights?\b",
        r"\btaillights?\b", r"tail lamps?", r"\bindicators?\b",
        r"broken lamp", r"lighting unit", r"\breflectors?\b",
        r"fog lamps?", r"fog lights?",
    ],
    "shattered_glass": [
        r"\bwindscreens?\b", r"\bwindshields?\b", r"window glass",
        r"side window", r"door window", r"\bglazing\b", r"quarter glass",
        r"sunroof glass", r"glass component", r"glass part",
        r"glass damage", r"glass exclusion", r"glass coverage",
        r"glass claim", r"glass replacement", r"glass repair",
        r"\bshattered\b", r"broken glass", r"glass scratches",
        r"laminated (safety )?glass", r"\bsafety glass\b", r"glass surface",
        # bare "glass" as a fallback signal, excluding the confirmed
        # false-positive pattern ("fibre glass" / "fiberglass" body material)
        r"(?<!fibre )(?<!fiber )\bglass\b",
    ],
    "flat_tyre": [
        r"\btyres?\b", r"\btires?\b", r"flat tyre", r"\bpuncture[sd]?\b",
        r"\btubes?\b", r"burst tyre", r"tyre valve", r"tyre carcass",
        r"tyre and tube", r"zero-pressure",
    ],
}

# Clause type keywords for metadata tagging. "exclusions?" is included so
# that a heading like "3.6 Tyre Exclusions" or "A. Exclusions Relating to
# Panel and Body Damage" (see contextualize()) tags correctly even though
# it never says "excluded" or "shall not be liable" in that exact form.
CLAUSE_TYPE_KEYWORDS = {
    "coverage": [
        r"will indemnify", r"shall pay", r"\bcovers?\b", r"\bcovered\b",
        r"insured against", r"\bcompensation\b", r"\breimburse\b",
        r"\bpayable\b", r"we will pay", r"undertakes to indemnify",
    ],
    "exclusion": [
        r"shall not be liable", r"not covered", r"\bexcluded\b",
        r"does not cover", r"no claim", r"not payable", r"excluded from",
        r"we do not cover", r"not admissible", r"\bexclusions?\b",
    ],
    "sub_limit": [
        r"limited to", r"\bmaximum\b", r"not exceeding", r"\bupto\b",
        r"up to", r"subject to a maximum", r"capped at",
    ],
    "condition": [
        r"provided that", r"subject to", r"on condition", r"only if",
        r"condition precedent", r"provided always",
    ],
    "definition": [
        r"\bmeans\b", r"shall mean", r"defined as", r"refers to",
        r"herein referred",
    ],
}

# Two tiers of chunk-boundary markers.
#
# HEADING_PATTERNS: structural section/subsection headings (SECTION,
# PART, CHAPTER, lettered sub-sections like "A. Exclusions Relating to
# Panel and Body Damage", numbered sub-headings like "3.1 Panel and Dent
# Exclusions"). These become the "current heading" breadcrumb prepended
# to every chunk that falls under them (see chunk_document).
#
# ITEM_PATTERNS: numbered/lettered list items ("1. Consequential loss...",
# "a) ..."). These still force their own chunk boundary (so each item
# stays atomic) but do NOT overwrite the current heading -- a bare list
# item is not itself a new section, and treating it as one is what
# disconnected exclusion/coverage items from their governing heading in
# the original version of this script (see the section-context write-up
# in the Milestone 2 report, Section 6.2).
HEADING_PATTERNS = [
    r"^SECTION\s+[IVX]+",
    r"^PART\s+[A-Z]\.",
    r"^CHAPTER\s+\d+\.",
    r"^[A-Z][A-Z\s&/]+:$",
    r"^\d\.\d+\s+[A-Za-z]",
    r"^[A-Z]\.\s+[A-Z][a-z]",
    r"^ADD-ON\s+COVER",
    r"^GENERAL\s+EXCEPTION",
    r"^CONDITIONS?:",
    r"^EXCLUSIONS?:",
    r"^DEDUCTIBLE:",
    r"^CLAIM\s+PROCEDURE",
]
ITEM_PATTERNS = [
    r"^\d+\.\s+[A-Z]",
    r"^[a-z]\)\s",
]

HEADING_RE = re.compile("|".join(HEADING_PATTERNS), re.IGNORECASE)
ITEM_RE = re.compile("|".join(ITEM_PATTERNS), re.IGNORECASE)
SECTION_RE = re.compile(HEADING_RE.pattern + "|" + ITEM_RE.pattern, re.IGNORECASE)

MAX_HEADING_LEN = 90  # truncate long heading lines before using as a breadcrumb

# PDF Extraction

def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from a PDF using pdfplumber's default reading-order
    extraction.

    An earlier version of this function tried to auto-detect two-column
    layouts by bisecting each page at width/2 and comparing word counts
    against full-page extraction. None of the 5 synthetic policy PDFs are
    actually laid out in two columns; the bisection instead cut through
    single-column text and table cells at an arbitrary x-coordinate
    (verified on policy_4 page 1's coverage table and policy_2 page 2's
    tyre-exclusion paragraph, where it split "INSURANCE" into "I" /
    "SURANCE" and truncated words like "lightning" -> "lightnin"). The
    word-count heuristic used to trigger this ("left + right word count >
    full-page word count * 1.05") fires on 3 of 21 pages across the
    corpus purely from incidental word-splitting noise at the bisection
    line, not genuine multi-column content, and each time it fires it
    destroys otherwise-correct text. It has been removed.
    """
    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pages_text.append(page.extract_text() or "")
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


# Structure-aware chunking

def split_into_sections(text: str) -> list:
    """
    Split text on structural headings and list items (both act as hard
    chunk boundaries). Returns a list of {"text": str, "is_heading": bool}
    dicts. is_heading is True only when the boundary line that opened
    this section is a real structural heading (HEADING_RE), not a bare
    numbered/lettered list item (ITEM_RE) -- callers use this to update
    a "current heading" breadcrumb without a list item wrongly resetting
    it to itself.
    """
    sections = []
    current = []
    current_is_heading = False

    for line in text.splitlines():
        stripped = line.strip()
        starts_heading = bool(HEADING_RE.match(stripped))
        starts_item = (not starts_heading) and bool(ITEM_RE.match(stripped))

        if (starts_heading or starts_item) and current:
            sections.append({"text": "\n".join(current).strip(), "is_heading": current_is_heading})
            current = [line]
            current_is_heading = starts_heading
        else:
            current.append(line)

    if current:
        sections.append({"text": "\n".join(current).strip(), "is_heading": current_is_heading})

    return [s for s in sections if len(s["text"]) > 20]


def chunk_document(text: str, chunk_size: int = 300, chunk_overlap: int = 40) -> list:
    """
    Structure-aware chunking: split on headings and list items first (so
    each list item stays atomic rather than merging across an unrelated
    boundary), then apply RecursiveCharacterTextSplitter within each
    resulting section. Returns a list of (heading, chunk_text) tuples,
    where heading is the nearest preceding structural heading -- e.g. a
    numbered exclusion item ("1. Consequential loss...") is paired with
    the heading of the section it lives under ("EXCLUSIONS UNDER SECTION
    I" or "3.1 Panel and Dent Exclusions"), not just the item text
    itself. This is what lets contextualize() (below) prepend the
    governing heading before embedding, so an itemised exclusion clause
    still reads as an exclusion once split out of its parent section.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )

    sections = split_into_sections(text)
    current_heading = ""
    chunks = []  # (heading, chunk_text)

    for section in sections:
        if section["is_heading"]:
            first_line = section["text"].splitlines()[0].strip()
            current_heading = first_line[:MAX_HEADING_LEN]

        for sub_chunk in splitter.split_text(section["text"]):
            sub_chunk = sub_chunk.strip()
            if len(sub_chunk) > 40:
                chunks.append((current_heading, sub_chunk))

    return chunks


def contextualize(heading: str, chunk: str) -> str:
    """
    Prepend the governing heading as a short breadcrumb, unless the
    chunk already begins with that heading text (the first chunk under a
    heading does, since the heading line is the first line of its own
    section). This is the text that actually gets embedded and stored --
    the heading is not just tagging metadata, so retrieval itself (not
    just the clause_type/damage_classes tags) benefits from the context.
    """
    if not heading:
        return chunk
    if chunk.strip().lower().startswith(heading.strip().lower()[:30]):
        return chunk
    return f"[{heading}] {chunk}"


# Damage-class and clause-type tagging
#
# Both taggers score the *contextualized* chunk (heading + body), so a
# heading like "3.6 Tyre Exclusions" or "EXCLUSIONS UNDER SECTION I"
# contributes its own keywords even when the item text itself doesn't
# restate them. Keyword lists use word-boundary regex, refined after a
# full manual review of all 179 chunks against the source PDFs found
# three substring bugs in the original naive `in` matching: "light"
# matched inside "lightning" (a fire peril, unrelated to lamps), "wheel"
# matched inside "wheel arch" (a body panel, not a tyre), and bare
# "glass" matched both genuine window glass and "fibre glass" (a body
# material).

def tag_damage_classes(text: str) -> list:
    """Return list of damage classes mentioned or implied in a chunk."""
    lower = text.lower()
    tags = []
    for cls, patterns in DAMAGE_KEYWORDS.items():
        if any(re.search(p, lower) for p in patterns):
            tags.append(cls)
    return tags


def tag_clause_type(heading: str, body: str) -> str:
    """
    Return the most likely clause type for a chunk.

    A heading-level signal takes priority over a body-level one. Manual
    review of the corpus found this matters concretely: several exclusion
    items explain what is *not* covered by contrasting it with "...a
    specific identifiable accidental event covered under this policy" --
    the body text's own "covered" keyword then ties against the heading's
    "shall not be liable" / "Exclusions" signal, and a naive combined
    score resolves the tie to "coverage" by dict order, which is wrong
    every time it happens. The heading (e.g. "The Company shall not be
    liable to make any payment in respect of:", "3.2 Scratch Exclusions")
    is the more reliable indicator of the chunk's true governing
    category, so it wins outright whenever it carries any signal at all;
    body-only scoring is the fallback for chunks under a heading that
    doesn't itself indicate a clause type (e.g. "CLAIM PROCEDURE").

    The DEPRECIATION/DEDUCTIBLE/... check is deliberately case-sensitive
    and heading-scoped: a case-insensitive bare-word version (tried
    during manual review) hijacked chunks whose own body prose happened
    to mention "depreciation" mid-sentence in an unrelated exclusion item.
    """
    heading_lower = heading.lower()
    heading_scores = {
        ct: sum(1 for p in patterns if re.search(p, heading_lower))
        for ct, patterns in CLAUSE_TYPE_KEYWORDS.items()
    }
    if re.search(r"\b(DEPRECIATION|DEDUCTIBLE|INSURED DECLARED VALUE|IDV|NO CLAIM BONUS|YOUR EXCESS)\b", heading):
        heading_scores["sub_limit"] += 1

    if any(v > 0 for v in heading_scores.values()):
        return max(heading_scores, key=heading_scores.get)

    body_lower = body.lower()
    body_scores = {
        ct: sum(1 for p in patterns if re.search(p, body_lower))
        for ct, patterns in CLAUSE_TYPE_KEYWORDS.items()
    }
    best = max(body_scores, key=body_scores.get)
    return best if body_scores[best] > 0 else "general"


# Deduplication

def dedup_chunks(chunks: list, threshold: float = 0.90) -> list:
    """
    Remove near-identical chunks based on normalised character overlap.
    Two chunks are considered near-duplicates if their Jaccard similarity
    of word trigrams exceeds the threshold. chunks is a list of
    (heading, chunk_text) tuples; only chunk_text is compared, since the
    heading is shared context rather than distinguishing content.
    """
    def trigrams(text):
        words = text.lower().split()
        return set(zip(words, words[1:], words[2:])) if len(words) >= 3 else set(words)

    unique  = []
    seen_tg = []

    for heading, chunk in chunks:
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
            unique.append((heading, chunk))
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
                "heading":       m.get("heading", ""),
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
            "heading":        meta.get("heading", ""),
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
        writer.writerow(["chunk_id", "doc_id", "heading", "damage_classes",
                         "clause_type", "chunk_len", "text"])
        for i, (chunk, meta) in enumerate(zip(all_chunks, all_metadata)):
            writer.writerow([
                f"chunk_{i:05d}",
                meta.get("doc_id", ""),
                meta.get("heading", ""),
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

        # Tag each chunk. chunks_dedup is a list of (heading, chunk_text)
        # tuples; contextualize() prepends the heading as a breadcrumb to
        # form the text that actually gets embedded, stored, and tagged.
        doc_id = pdf_path.stem
        for idx, (heading, chunk) in enumerate(chunks_dedup):
            ctx_text    = contextualize(heading, chunk)
            damage_cls  = tag_damage_classes(ctx_text)
            clause_type = tag_clause_type(heading, chunk)
            all_chunks.append(ctx_text)
            all_metadata.append({
                "doc_id":         doc_id,
                "chunk_index":    len(all_chunks) - 1,
                "damage_classes": damage_cls,
                "clause_type":    clause_type,
                "text":           ctx_text,
                "heading":        heading,
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
