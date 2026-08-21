"""
Policy PDF -> tagged, contextualised chunks.

Ported from scripts/preprocess_policy_pdfs.py, which stays in place as the
Milestone 2/3 batch artifact. The keyword lists, chunk-boundary rules, and
tagging precedence below are NOT defaults -- each encodes a specific bug
found by manually reviewing all 179 chunks of the original corpus against
the source PDFs, and re-deriving them from scratch would reintroduce those
bugs. The comments record what each rule is defending against so a future
edit can tell a tuned value from an arbitrary one.
"""
import logging
import re

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)

# Damage-class keywords. Word-boundary regex, not bare substring `in`
# matching: the naive version matched "light" inside "lightning" (a fire
# peril, not a lamp), "wheel" inside "wheel arch" (a body panel, not a
# tyre), and bare "glass" inside "fibre glass" (a body material).
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
        r"(?<!fibre )(?<!fiber )\bglass\b",
    ],
    "flat_tyre": [
        r"\btyres?\b", r"\btires?\b", r"flat tyre", r"\bpuncture[sd]?\b",
        r"\btubes?\b", r"burst tyre", r"tyre valve", r"tyre carcass",
        r"tyre and tube", r"zero-pressure",
    ],
}

# Clause-type keywords. "exclusions?" is present so a heading like
# "3.6 Tyre Exclusions" tags correctly even though it never says "excluded"
# or "shall not be liable" in those exact words.
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

# Two tiers of chunk boundary. HEADING_PATTERNS are structural section
# headings, and become the breadcrumb prepended to every chunk beneath them.
# ITEM_PATTERNS are numbered/lettered list items: they still force a chunk
# boundary (each item stays atomic) but must NOT overwrite the current
# heading -- treating a bare list item as a new section is what disconnected
# exclusion items from their governing heading in the original script.
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
ITEM_PATTERNS = [r"^\d+\.\s+[A-Z]", r"^[a-z]\)\s"]

HEADING_RE = re.compile("|".join(HEADING_PATTERNS), re.IGNORECASE)
ITEM_RE = re.compile("|".join(ITEM_PATTERNS), re.IGNORECASE)

MAX_HEADING_LEN = 90


def extract_pdf_text(pdf_path) -> str:
    """pdfplumber's default reading-order extraction.

    Deliberately does NOT attempt two-column detection. An earlier version
    bisected each page at width/2 when a word-count heuristic fired; none of
    the corpus PDFs are two-column, and the bisection cut through
    single-column text and table cells mid-word ("INSURANCE" -> "I" /
    "SURANCE"). It fired on 3 of 21 pages purely from noise at the split
    line, destroying correct text each time.
    """
    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pages_text.append(page.extract_text() or "")
    except Exception as e:
        log.error("Failed to extract %s: %s", pdf_path, e)
        return ""
    return "\n\n".join(pages_text)


def clean_text(raw: str) -> str:
    """Strip page furniture and normalise PDF typographic artefacts."""
    cleaned = []
    for line in raw.splitlines():
        stripped = line.strip()

        if re.match(r"^Page\s+\d+\s+of\s+\d+", stripped, re.IGNORECASE):
            continue
        if re.match(r"^Policy Wordings", stripped, re.IGNORECASE):
            continue
        if re.match(r"^UIN:\s+IRDAN", stripped, re.IGNORECASE):
            continue
        if re.match(r"^\*+$", stripped):
            continue

        stripped = stripped.replace("ﬁ", "fi").replace("ﬂ", "fl")
        stripped = stripped.replace("–", "-").replace("—", " - ")
        stripped = stripped.replace("‘", "'").replace("’", "'")
        stripped = stripped.replace("“", '"').replace("”", '"')
        stripped = re.sub(r"  +", " ", stripped)

        if stripped:
            cleaned.append(stripped)
    return "\n".join(cleaned)


def split_into_sections(text: str) -> list:
    """Split on headings and list items (both hard boundaries). is_heading
    marks only real structural headings, so callers can track a breadcrumb
    without a list item resetting it to itself."""
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


def chunk_document(text: str, chunk_size: int, chunk_overlap: int) -> list:
    """Structure-aware chunking -> list of (heading, chunk_text). Splits on
    structure first so each list item stays atomic, then applies the
    recursive splitter within each section. The heading returned alongside
    each chunk is the nearest preceding *structural* heading, which is what
    lets an itemised exclusion still read as an exclusion once it has been
    split out of its parent section."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )

    current_heading = ""
    chunks = []
    for section in split_into_sections(text):
        if section["is_heading"]:
            current_heading = section["text"].splitlines()[0].strip()[:MAX_HEADING_LEN]
        for sub_chunk in splitter.split_text(section["text"]):
            sub_chunk = sub_chunk.strip()
            if len(sub_chunk) > 40:
                chunks.append((current_heading, sub_chunk))
    return chunks


def contextualize(heading: str, chunk: str) -> str:
    """Prepend the governing heading as a breadcrumb. This is the text that
    gets embedded and stored, so retrieval itself benefits from the section
    context -- not just the metadata tags."""
    if not heading:
        return chunk
    if chunk.strip().lower().startswith(heading.strip().lower()[:30]):
        return chunk
    return f"[{heading}] {chunk}"


def tag_damage_classes(text: str) -> list:
    """Damage classes mentioned or implied. Scores the *contextualized*
    chunk, so a heading like "3.6 Tyre Exclusions" contributes its keywords
    even when the item body doesn't restate them."""
    lower = text.lower()
    return [cls for cls, patterns in DAMAGE_KEYWORDS.items()
            if any(re.search(p, lower) for p in patterns)]


def tag_clause_type(heading: str, body: str) -> str:
    """Most likely clause type, with the heading signal taking priority.

    Heading-wins is load-bearing: several exclusion items describe what is
    NOT covered by contrasting it with "...a specific identifiable
    accidental event covered under this policy". The body's own "covered"
    keyword then ties against the heading's "shall not be liable" signal,
    and a naive combined score breaks the tie toward "coverage" by dict
    order -- wrong every time it happens.

    The DEPRECIATION/DEDUCTIBLE check is case-sensitive and heading-scoped
    on purpose: a case-insensitive bare-word version hijacked chunks whose
    body merely mentioned "depreciation" mid-sentence.
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


def dedup_chunks(chunks: list, threshold: float) -> list:
    """Drop near-identical chunks by word-trigram Jaccard similarity. Only
    chunk_text is compared -- the heading is shared context, so including it
    would make every chunk under one heading look mutually similar."""
    def trigrams(text):
        words = text.lower().split()
        return set(zip(words, words[1:], words[2:])) if len(words) >= 3 else set(words)

    unique, seen = [], []
    for heading, chunk in chunks:
        tg = trigrams(chunk)
        if any(tg and ref and len(tg & ref) / len(tg | ref) >= threshold for ref in seen):
            continue
        unique.append((heading, chunk))
        seen.append(tg)
    return unique


def parse_policy_pdf(pdf_path, chunk_size: int, chunk_overlap: int,
                     dedup_threshold: float) -> list:
    """Full pipeline for one PDF -> list of chunk dicts ready to embed."""
    text = clean_text(extract_pdf_text(pdf_path))
    chunks = dedup_chunks(chunk_document(text, chunk_size, chunk_overlap), dedup_threshold)

    out = []
    for idx, (heading, chunk) in enumerate(chunks):
        ctx_text = contextualize(heading, chunk)
        out.append({
            "chunk_index": idx,
            "heading": heading,
            "text": ctx_text,
            "damage_classes": tag_damage_classes(ctx_text),
            "clause_type": tag_clause_type(heading, chunk),
        })
    return out
