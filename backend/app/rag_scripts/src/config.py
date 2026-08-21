"""
Single source of truth for paths, thresholds, and model identifiers used by
the src/ claim-processing pipeline.

Why this exists: the scripts/ pipeline (kept as-is, see docs) scattered these
values across module-level constants in six different files -- the embedding
model name was hardcoded three times, the RRF weights lived in
scripts/hybrid_retrieval.py, the severity bins in scripts/yolo_schema.py, and
configs/pipeline_config.yaml held a partly-overlapping subset that nothing
actually read. Anything tuned by an experiment belongs in one place with the
experiment that justified it recorded next to it.

Values carried over unchanged (and why they are not free parameters):
  - DENSE_WEIGHT / SPARSE_WEIGHT: swept against all 50 eval incidents; 3:1
    was the lowest (most dense-weighted) ratio reaching peak Precision@3
    without regressing MRR. See scripts/hybrid_retrieval.py's module
    docstring for the full sweep table.
  - SEVERITY_BINS: the global area-ratio proxy from scripts/eda_vehide.py,
    fit against *bounding-box* area (polygon -> bbox), NOT segmentation-mask
    area. See src/inference/yolo_detector.py for why that distinction
    changes what the detector must compute.
  - ESCALATION_CONFIDENCE_THRESHOLD: still an uncalibrated placeholder --
    no calibrated value exists anywhere in this repo. Flagged rather than
    silently presented as tuned.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data locations.
#
# Deliberately NOT data/user_policies/ -- that belongs to the scripts/
# pipeline, which is kept as-is and owns it. Both pipelines name their
# ChromaDB collections "user_{user_id}", so pointing both at one directory
# means two independently-evolving codebases sharing a mutable store that
# each one deletes and recreates collections in. They stayed out of each
# other's way only because their user_ids happened to differ. src/ gets its
# own root so the separation is structural rather than coincidental.
DATA_DIR = ROOT / "data"
PIPELINE_DIR = DATA_DIR / "claim_pipeline"
USER_POLICY_DIR = PIPELINE_DIR / "policies"
USER_DB_PATH = USER_POLICY_DIR / "chroma_db"
USER_CHUNKS_DIR = USER_POLICY_DIR / "chunks"
USER_MANIFEST_DIR = USER_POLICY_DIR / "manifests"

# Claim run outputs + LangGraph checkpoint store
CLAIM_OUTPUT_DIR = PIPELINE_DIR / "claims"
CHECKPOINT_DB = PIPELINE_DIR / "claims" / "checkpoints.sqlite"

# Vision
YOLO_CHECKPOINT = ROOT / "models" / "YOLO-8s" / "last.pt"
YOLO_CONF_THRESHOLD = 0.25  # ultralytics inference floor; below this nothing is emitted
YOLO_IMG_SIZE = 640

CLASS_ID = {
    "dent": 0,
    "scratch": 1,
    "crack": 2,
    "broken_lamp": 3,
    "shattered_glass": 4,
    "flat_tyre": 5,
}
CLASS_NAMES = list(CLASS_ID.keys())

# Global area-ratio severity proxy (scripts/eda_vehide.py). Fit on bbox area.
SEVERITY_BINS = [0.0, 0.02, 0.08, 1.0]
SEVERITY_LABELS = ["minor", "moderate", "severe"]

# Uncalibrated placeholder -- see module docstring.
ESCALATION_CONFIDENCE_THRESHOLD = 0.50

# Retrieval
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40
DEDUP_THRESHOLD = 0.90

DENSE_WEIGHT = 3.0
SPARSE_WEIGHT = 1.0
RRF_K = 60
CANDIDATE_POOL = 20

RETRIEVAL_POOL = 15
CLAUSES_PER_TYPE = 5
MIN_CLAUSE_SCORE = 0.01

# Clause buckets. "definition" rides along with coverage for one specific,
# verified reason documented in src/retrieval/clause_retriever.py -- it is a
# narrow mitigation for a known auto-tagger mislabel, not a general relaxation.
COVERAGE_CLAUSE_TYPES = {"coverage", "definition"}
EXCLUSION_CLAUSE_TYPES = {"exclusion", "sub_limit", "condition"}

# Report Agent (Groq Cloud, OpenAI-compatible API)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
REPORT_MODEL = "llama-3.3-70b-versatile"
REPORT_TEMPERATURE = 0.2
REPORT_MAX_RETRIES = 3

VALID_VERDICTS = {"covered", "excluded", "conditional", "needs_review"}

# The decisions a human reviewer may return when releasing a paused claim.
# Validated at the resume boundary rather than only in the CLI's argparse,
# because resume_claim() is a public entry point any future caller (an app,
# an endpoint) reaches directly.
VALID_REVIEW_DECISIONS = {"approve", "reject", "override"}
