"""
Generate 10 contrastive synthetic claim payloads -- the actual JSON bundles
that would go to the Report Agent LLM, per the schema in
scripts/report_context.py.

"Contrastive" here means the 10 scenarios were deliberately chosen to stress
different parts of the pipeline rather than being 10 similar dent claims:
single vs. multi-damage, every severity bucket, a below-threshold-confidence
detection (escalation path), multi-instance-of-one-class, and two scenarios
tied to specific documented corpus quirks -- policy_5's tyre-only-covered-
with-concurrent-damage condition, and the vandalism/malicious-act wording
drift that motivated the hybrid retrieval fix in scripts/hybrid_retrieval.py
-- to check that fix actually holds up inside the full pipeline, not just
the isolated 50-incident eval.

Each scenario carries a `user_id` rather than a `selected_doc_id`: the
fixed 5-policy catalog and its claimant-selection step (formerly
scripts/policy_catalog.py + scripts/policy_selector.py) are gone -- each
user now uploads their own policy PDF (scripts/ingest_user_policy.py) and
the pipeline is scoped to it automatically. For this regeneration there is
no real upload flow to exercise, so the same 5 synthetic policy PDFs used
throughout the project (data/policy_pdfs/synthetic/) are re-used as 5 stand-
in "users" -- one ingested per doc, user_id == the old doc_id -- so the
scenarios still deliberately span all 5 policies and pin the two bug-
surfacing claims to the ones whose corpus quirks they exercise: CLAIM_02/
CLAIM_08 -> policy_4 (the table-garbling chunk_00122), CLAIM_09 -> policy_1
(the chunk_00004 definition-mistagging fix). See docs/rag_support_mile3.md
for the original corpus-quality writeups those fixes responded to.

Detections use yolo_schema.make_detection(class_name, area_ratio,
confidence) -- area_ratio is picked directly to land in a specific severity
bucket (see yolo_schema.SEVERITY_BINS) rather than deriving it from made-up
bbox coordinates.
"""
import json
from pathlib import Path

from ingest_user_policy import ingest
from report_context import ContextBundleBuilder
from yolo_schema import make_detection

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "rag_outputs" / "mile3" / "payloads"
SYNTHETIC_PDF_DIR = ROOT / "data" / "policy_pdfs" / "synthetic"

# Stand-in "users", one per synthetic policy PDF -- see module docstring.
DEMO_USERS = {
    "policy_1_bharat_suraksha": "policy_1_bharat_suraksha.pdf",
    "policy_2_safedrive_assurance": "policy_2_safedrive_assurance.pdf",
    "policy_3_quickclaim_general": "policy_3_quickclaim_general.pdf",
    "policy_4_autoguard_premium": "policy_4_autoguard_premium.pdf",
    "policy_5_valuemotor": "policy_5_valuemotor.pdf",
}

CLAIM_SCENARIOS = [
    {
        "claim_id": "CLAIM_01_dent_minor_clean",
        "user_id": "policy_1_bharat_suraksha",
        "incident_narrative": "While parking, the vehicle was struck by an adjacent car door, leaving a noticeable dent on the front left door panel.",
        "detections": [("dent", 0.01, 0.91)],
    },
    {
        "claim_id": "CLAIM_02_glass_severe_stone",
        "user_id": "policy_4_autoguard_premium",
        "incident_narrative": "The windscreen shattered after a large stone was projected by a passing truck on the expressway.",
        "detections": [("shattered_glass", 0.22, 0.95)],
    },
    {
        "claim_id": "CLAIM_03_flat_tyre_alone_no_body_damage",
        "user_id": "policy_5_valuemotor",
        "incident_narrative": "A nail embedded in the road punctured the rear left tyre, causing a sudden blowout on the highway. No other part of the vehicle was affected.",
        "detections": [("flat_tyre", 0.03, 0.88)],
    },
    {
        "claim_id": "CLAIM_04_multi_pileup_severe",
        "user_id": "policy_3_quickclaim_general",
        "incident_narrative": "A three-vehicle pileup caused front dents, cracked bumper panels, and broken headlight units.",
        "detections": [
            ("dent", 0.09, 0.90),
            ("crack", 0.06, 0.85),
            ("broken_lamp", 0.04, 0.83),
        ],
    },
    {
        "claim_id": "CLAIM_05_scratch_vandalism_keyed",
        "user_id": "policy_2_safedrive_assurance",
        "incident_narrative": "A vandal keyed the vehicle overnight in the apartment parking, leaving deep scratches on both side doors.",
        "detections": [("scratch", 0.015, 0.87)],
    },
    {
        "claim_id": "CLAIM_06_low_confidence_dent",
        "user_id": "policy_1_bharat_suraksha",
        "incident_narrative": "Possible minor contact damage noticed on the rear bumper; unclear from the photo whether this is fresh damage.",
        "detections": [("dent", 0.008, 0.35)],
    },
    {
        "claim_id": "CLAIM_07_glass_vandalism_window",
        "user_id": "policy_2_safedrive_assurance",
        "incident_narrative": "Vandals smashed the rear window of the vehicle during a public disturbance event.",
        "detections": [("shattered_glass", 0.10, 0.89)],
    },
    {
        "claim_id": "CLAIM_08_crack_lamp_rearend",
        "user_id": "policy_4_autoguard_premium",
        "incident_narrative": "A rear-end shunt in traffic cracked the tail-light housing and the bumper below it.",
        "detections": [
            ("crack", 0.05, 0.86),
            ("broken_lamp", 0.03, 0.84),
        ],
    },
    {
        "claim_id": "CLAIM_09_multi_instance_hail_dents",
        "user_id": "policy_1_bharat_suraksha",
        "incident_narrative": "Hailstorm left multiple small dents across the bonnet and roof of the vehicle.",
        "detections": [
            ("dent", 0.012, 0.80, (0.3, 0.4)),
            ("dent", 0.009, 0.78, (0.5, 0.4)),
            ("dent", 0.011, 0.81, (0.7, 0.5)),
        ],
    },
    {
        "claim_id": "CLAIM_10_tyre_scratch_flood_debris",
        "user_id": "policy_5_valuemotor",
        "incident_narrative": "Driving through debris on a flooded road caused a tyre puncture and road rash scratches on the lower sill.",
        "detections": [
            ("flat_tyre", 0.07, 0.87),
            ("scratch", 0.015, 0.82),
        ],
    },
]


def build_detections(spec: list):
    detections = []
    for item in spec:
        if len(item) == 3:
            cls, area_ratio, conf = item
            detections.append(make_detection(cls, area_ratio, conf))
        else:
            cls, area_ratio, conf, center = item
            detections.append(make_detection(cls, area_ratio, conf, center=center))
    return detections


def main():
    print(f"Ingesting {len(DEMO_USERS)} demo users' policies...")
    for user_id, pdf_name in DEMO_USERS.items():
        summary = ingest(SYNTHETIC_PDF_DIR / pdf_name, user_id)
        print(f"  {user_id:28s} -> {summary['n_chunks']} chunks")

    builders = {user_id: ContextBundleBuilder(user_id) for user_id in DEMO_USERS}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_payloads = []
    for scenario in CLAIM_SCENARIOS:
        detections = build_detections(scenario["detections"])
        builder = builders[scenario["user_id"]]
        payload = builder.build(
            claim_id=scenario["claim_id"],
            incident_narrative=scenario["incident_narrative"],
            detections=detections,
        )
        all_payloads.append(payload)
        with open(OUT_DIR / f"{scenario['claim_id']}.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"{scenario['claim_id']:38s} -> policy={payload['policy']['doc_id']:28s} "
              f"escalate={payload['escalation']['needs_human_review']}")

    with open(OUT_DIR.parent / "payloads_all.json", "w") as f:
        json.dump(all_payloads, f, indent=2)
    print(f"\nSaved {len(all_payloads)} payloads -> {OUT_DIR}")


if __name__ == "__main__":
    main()
