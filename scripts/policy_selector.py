"""
Stage 1 of the Report Agent pipeline: given a claim's detected damage
classes (and optionally their severities), auto-select which of the 5
synthetic policy documents governs this claim.

Why this exists: a real claims system already knows the policyholder's
insurer (looked up by policy number). This project has no such lookup, so
per the design discussion, the workflow instead picks whichever of the 5
policies' language best matches the claim's damage profile and treats that
as "the policy in effect" -- clearly labelled as an auto-match, not
identity verification (see docs/rag_support_mile3.md for the caveat this
implies).

Honesty check built into this module: the 5 synthetic policies were
deliberately authored (Milestone 2, Section 2.2) with *varied phrasing for
the same coverage concepts*, specifically to stress-test retrieval rather
than to be distinguishable by damage type. So damage-class-only matching is
not expected to reliably recover a "true" doc_id -- this module measures
that empirically (evaluate()) rather than assuming it works, the same way
scripts/hybrid_retrieval.py measured dense-vs-hybrid instead of assuming
hybrid would win.

Three aggregation heuristics turn a per-(doc, damage_class) score matrix
into a single doc_id pick:
  - mean_top_score:        average the best clause match per damage class
                            across the doc, pick the highest average.
  - max_score:              pick the doc holding the single strongest
                            match to any one damage class in the claim.
  - severity_weighted_mean: like mean_top_score, but each class's score is
                            weighted by that damage's severity in the
                            claim, so a claim's dominant/severe damage type
                            has more influence than a minor one also
                            present.

Run as a script to reproduce the comparison:
    python scripts/policy_selector.py --evaluate
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

from hybrid_retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "rag_outputs" / "mile3" / "policy_selection_eval.json"

DOC_IDS = [
    "policy_1_bharat_suraksha",
    "policy_2_safedrive_assurance",
    "policy_3_quickclaim_general",
    "policy_4_autoguard_premium",
    "policy_5_valuemotor",
]

# Reused verbatim from scripts/preprocess_policy_pdfs.py run_smoke_test --
# already validated there to reach a topically-correct top-1 for all 6
# classes, so re-derived here rather than writing new canned queries.
CLASS_QUERIES = {
    "dent": "Is dent damage covered under accidental external means?",
    "scratch": "Does the policy cover scratches and surface paint damage?",
    "crack": "Is cracking of bumper or body panels covered?",
    "broken_lamp": "Are broken headlamps and tail lights covered?",
    "shattered_glass": "Is windscreen and window glass damage covered?",
    "flat_tyre": "Is a flat tyre or tyre blowout covered under the policy?",
}

SEVERITY_WEIGHT = {"minor": 1.0, "moderate": 2.0, "severe": 3.0}


def build_score_matrix(retriever: HybridRetriever, pool_k: int = 1, agg: str = "sum") -> dict:
    """Precompute an aggregated fused score per (doc_id, damage_class) once
    -- 5 docs x 6 canonical class queries x pool_k lookups, reused across
    every claim in the eval rather than re-querying per claim.

    pool_k=1 (the original design) uses only each doc's single
    best-matching chunk per class -- a single number can't tell "one chunk
    mentions this topic" apart from "this document has a whole dedicated
    subsection on it." pool_k=5 (agg="sum") aggregates over the top-5
    chunks instead, to test whether a document having *multiple* relevant
    clauses on a topic (dedicated subsections, several exclusion items --
    documented as varying across the 5 policies in Milestone 2, e.g.
    policy_2's "dedicated subsections per damage type" vs. policy_3's
    "6 named exclusion schedules") is a real signal that a single top-1
    score was throwing away."""
    matrix = {doc_id: {} for doc_id in DOC_IDS}
    for doc_id in DOC_IDS:
        for cls, query in CLASS_QUERIES.items():
            scored = retriever.retrieve_scored(query, top_k=pool_k, doc_filter=doc_id)
            scores = [s for _, s in scored] or [0.0]
            matrix[doc_id][cls] = sum(scores) if agg == "sum" else sum(scores) / len(scores)
    return matrix


def score_mean_top_score(matrix: dict, damage_classes: list, **_) -> dict:
    return {d: sum(matrix[d][c] for c in damage_classes) / len(damage_classes) for d in DOC_IDS}


def score_max_score(matrix: dict, damage_classes: list, **_) -> dict:
    return {d: max(matrix[d][c] for c in damage_classes) for d in DOC_IDS}


def score_severity_weighted(matrix: dict, damage_classes: list, severities: dict = None, **_) -> dict:
    severities = severities or {}
    weights = [SEVERITY_WEIGHT.get(severities.get(c, "moderate"), 2.0) for c in damage_classes]
    total_w = sum(weights)
    return {
        d: sum(matrix[d][c] * w for c, w in zip(damage_classes, weights)) / total_w
        for d in DOC_IDS
    }


HEURISTICS = {
    "mean_top_score": score_mean_top_score,
    "max_score": score_max_score,
    "severity_weighted_mean": score_severity_weighted,
}


def _argmax_with_margin(score_dict: dict) -> tuple:
    ranked = sorted(score_dict.items(), key=lambda kv: -kv[1])
    top_doc, top_score = ranked[0]
    margin = top_score - ranked[1][1] if len(ranked) > 1 else top_score
    return top_doc, round(margin, 5)


class PolicySelector:
    """Public entry point used by scripts/report_context.py: pick a doc_id
    for a real claim using the winning heuristic (see evaluate() output)."""

    def __init__(self, retriever: HybridRetriever = None, heuristic: str = "mean_top_score"):
        self.retriever = retriever or HybridRetriever()
        self.matrix = build_score_matrix(self.retriever)
        self.heuristic_fn = HEURISTICS[heuristic]
        self.heuristic_name = heuristic

    def select(self, damage_classes: list, severities: dict = None) -> dict:
        doc_scores = self.heuristic_fn(self.matrix, damage_classes, severities=severities)
        doc_id, margin = _argmax_with_margin(doc_scores)
        per_class = {c: self.matrix[doc_id][c] for c in damage_classes}
        match_confidence = round(sum(per_class.values()) / len(per_class), 4)
        return {
            "doc_id": doc_id,
            "heuristic": self.heuristic_name,
            "match_confidence": match_confidence,
            "selection_margin": margin,
            "per_class_scores": {c: round(v, 4) for c, v in per_class.items()},
        }


# Exhaustive eval: does damage-class-only matching recover the "true" doc?
#
# Earlier version of this eval sampled 25 random (damage_classes, true_doc)
# claims and reported top-1 accuracy alone. That was the wrong instinct to
# fix by just raising a random sample size: mean_top_score / max_score are
# *deterministic* functions of a fixed 5-doc x 6-class score matrix (30
# numbers, already computed once by build_score_matrix). Every claim they
# could ever be asked about is fully determined by (a) which subset of the
# 6 classes is present and (b) which doc is asserted as "true" -- there are
# only 2^6 - 1 = 63 non-empty class subsets and 5 docs, i.e. 315 possible
# cases, total, ever. That is small enough to enumerate exactly rather than
# sample -- a full census has no sampling noise to report a confidence
# interval around, and costs nothing extra to compute since it reuses the
# same 30-value matrix (no new retrieval calls).
#
# severity_weighted_mean is the one heuristic that genuinely depends on an
# extra input (severity) with no natural "true" value, so it can't be
# folded into the same exhaustive census: with no severity given, its
# weights collapse to a uniform 2.0 and it is *mathematically identical* to
# mean_top_score (see score_severity_weighted's default). Testing it with
# *random* severity, as the old version did, only adds noise on top of
# noise. Instead, for every multi-class subset (57 of the 63), it is run
# under two deliberate, controlled severity patterns -- "first class severe,
# rest minor" and "last class severe, rest minor" -- so any difference from
# the mean_top_score baseline reflects the severity weighting actually
# doing something, not random chance.

ALL_CLASSES = list(CLASS_QUERIES.keys())


def enumerate_all_claims() -> list:
    """Every non-empty subset of the 6 damage classes x every doc as the
    asserted "true" doc -- 63 x 5 = 315 cases, the full relevant population."""
    claims = []
    for doc_id in DOC_IDS:
        for k in range(1, len(ALL_CLASSES) + 1):
            for combo in combinations(ALL_CLASSES, k):
                claims.append({
                    "claim_id": f"{doc_id}__{'-'.join(combo)}",
                    "true_doc_id": doc_id,
                    "damage_classes": list(combo),
                })
    return claims


def enumerate_severity_variant_claims() -> list:
    """Multi-class subsets (57 of the 63) x 5 docs x 2 deliberate severity
    patterns = 570 cases, for testing severity_weighted_mean specifically."""
    claims = []
    for doc_id in DOC_IDS:
        for k in range(2, len(ALL_CLASSES) + 1):
            for combo in combinations(ALL_CLASSES, k):
                classes = list(combo)
                for pattern, severe_cls in [("first_severe", classes[0]), ("last_severe", classes[-1])]:
                    severities = {c: ("severe" if c == severe_cls else "minor") for c in classes}
                    claims.append({
                        "claim_id": f"{doc_id}__{'-'.join(classes)}__{pattern}",
                        "true_doc_id": doc_id,
                        "damage_classes": classes,
                        "severities": severities,
                        "pattern": pattern,
                    })
    return claims


def _rank_of_true_doc(doc_scores: dict, true_doc_id: str) -> int:
    ranked = sorted(doc_scores.items(), key=lambda kv: -kv[1])
    return [d for d, _ in ranked].index(true_doc_id) + 1  # 1-indexed


def _score_claims(fn, matrix: dict, claims: list) -> dict:
    n = len(claims)
    ranks = []
    confusions = []
    for claim in claims:
        doc_scores = fn(matrix, claim["damage_classes"], severities=claim.get("severities"))
        rank = _rank_of_true_doc(doc_scores, claim["true_doc_id"])
        ranks.append(rank)
        if rank != 1:
            pred, _ = _argmax_with_margin(doc_scores)
            confusions.append({
                "claim_id": claim["claim_id"], "true": claim["true_doc_id"],
                "predicted": pred, "true_doc_rank": rank,
                "damage_classes": claim["damage_classes"],
            })
    return {
        "n_claims": n,
        "top1_accuracy": round(sum(1 for r in ranks if r == 1) / n, 3),
        "top2_accuracy": round(sum(1 for r in ranks if r <= 2) / n, 3),
        "mrr": round(sum(1 / r for r in ranks) / n, 3),
        "n_confusions": len(confusions),
        "sample_confusions": confusions[:5],
    }


def evaluate_pool_k_variants(retriever: HybridRetriever, exhaustive_claims: list, matrix_pool1: dict) -> dict:
    """Compare pool_k=1 (single best-matching chunk per doc/class -- reuses
    the matrix already computed by evaluate()) against pool_k=5 summed (does
    a document having *multiple* relevant clauses on a topic carry a signal
    pool_k=1 was throwing away?) on the identical 315-case exhaustive census."""
    matrices = {1: matrix_pool1, 5: build_score_matrix(retriever, pool_k=5, agg="sum")}
    results = {}
    for pool_k, m in matrices.items():
        results[f"pool_k={pool_k}"] = {
            "mean_top_score": _score_claims(score_mean_top_score, m, exhaustive_claims),
            "max_score": _score_claims(score_max_score, m, exhaustive_claims),
        }
    return results


def evaluate(retriever: HybridRetriever = None) -> dict:
    retriever = retriever or HybridRetriever()
    matrix = build_score_matrix(retriever)

    exhaustive_claims = enumerate_all_claims()
    primary = {
        "mean_top_score": _score_claims(score_mean_top_score, matrix, exhaustive_claims),
        "max_score": _score_claims(score_max_score, matrix, exhaustive_claims),
    }

    pool_k_comparison = evaluate_pool_k_variants(retriever, exhaustive_claims, matrix)

    sev_claims = enumerate_severity_variant_claims()
    first_severe = [c for c in sev_claims if c["pattern"] == "first_severe"]
    last_severe = [c for c in sev_claims if c["pattern"] == "last_severe"]
    multiclass_baseline_claims = [c for c in exhaustive_claims if len(c["damage_classes"]) >= 2]

    severity_analysis = {
        "n_multiclass_claims": len(multiclass_baseline_claims),
        "mean_top_score_baseline": _score_claims(score_mean_top_score, matrix, multiclass_baseline_claims),
        "severity_weighted_first_severe": _score_claims(score_severity_weighted, matrix, first_severe),
        "severity_weighted_last_severe": _score_claims(score_severity_weighted, matrix, last_severe),
    }

    # Flip analysis: on the identical (doc, class-subset) pairs, how often does
    # deliberately weighting one class as "severe" change the prediction versus
    # the unweighted mean_top_score baseline, and does that flip land on the
    # true doc or away from it?
    flips = {"first_severe": {"n_flips": 0, "flip_helped": 0, "flip_hurt": 0},
             "last_severe": {"n_flips": 0, "flip_helped": 0, "flip_hurt": 0}}
    for pattern, variant_claims in [("first_severe", first_severe), ("last_severe", last_severe)]:
        for claim in variant_claims:
            base_scores = score_mean_top_score(matrix, claim["damage_classes"])
            base_pred, _ = _argmax_with_margin(base_scores)
            sev_scores = score_severity_weighted(matrix, claim["damage_classes"], severities=claim["severities"])
            sev_pred, _ = _argmax_with_margin(sev_scores)
            if sev_pred != base_pred:
                flips[pattern]["n_flips"] += 1
                if sev_pred == claim["true_doc_id"] and base_pred != claim["true_doc_id"]:
                    flips[pattern]["flip_helped"] += 1
                elif base_pred == claim["true_doc_id"] and sev_pred != claim["true_doc_id"]:
                    flips[pattern]["flip_hurt"] += 1

    best = max(primary, key=lambda k: primary[k]["mrr"])
    return {
        "exhaustive_n": len(exhaustive_claims),
        "note": "Full census of all 63 non-empty damage-class subsets x 5 docs -- not a sample, no confidence interval needed.",
        "primary_heuristics": primary,
        "best_primary_heuristic": best,
        "pool_k_comparison": pool_k_comparison,
        "severity_weighted_analysis": severity_analysis,
        "severity_weighting_flip_analysis": flips,
        "score_matrix": {d: {c: round(v, 4) for c, v in m.items()} for d, m in matrix.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Policy-selection heuristic comparison")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.evaluate:
        report = evaluate()
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Exhaustive census: {report['exhaustive_n']} cases (63 class subsets x 5 docs)\n")
        for name, res in report["primary_heuristics"].items():
            print(f"{name:20s} top1={res['top1_accuracy']}  top2={res['top2_accuracy']}  mrr={res['mrr']}")
        print(f"\nBest primary heuristic (by MRR): {report['best_primary_heuristic']}")

        print("\npool_k comparison (single best chunk vs. sum of top-5 chunks per doc/class):")
        for pool_label, res in report["pool_k_comparison"].items():
            for name, m in res.items():
                print(f"  {pool_label:10s} {name:20s} top1={m['top1_accuracy']}  top2={m['top2_accuracy']}  mrr={m['mrr']}")

        sa = report["severity_weighted_analysis"]
        print(f"\nSeverity-weighted analysis ({sa['n_multiclass_claims']} multi-class claims):")
        print(f"  mean_top_score baseline      top1={sa['mean_top_score_baseline']['top1_accuracy']}  mrr={sa['mean_top_score_baseline']['mrr']}")
        print(f"  severity_weighted first_severe top1={sa['severity_weighted_first_severe']['top1_accuracy']}  mrr={sa['severity_weighted_first_severe']['mrr']}")
        print(f"  severity_weighted last_severe  top1={sa['severity_weighted_last_severe']['top1_accuracy']}  mrr={sa['severity_weighted_last_severe']['mrr']}")

        flips = report["severity_weighting_flip_analysis"]
        for pattern, f_res in flips.items():
            print(f"  {pattern}: {f_res['n_flips']} predictions flipped vs baseline "
                  f"({f_res['flip_helped']} helped, {f_res['flip_hurt']} hurt)")

        print(f"\nSaved -> {OUT_PATH}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
