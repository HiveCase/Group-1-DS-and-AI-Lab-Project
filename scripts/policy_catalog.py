"""
Policy catalog: the human-readable menu a claimant chooses from.

Design context: reading all 5 synthetic policies end-to-end (Milestone 3,
docs/rag_support_mile3.md Section 5) confirmed that *which* policy applies to
a claim cannot be inferred from the damage -- every policy covers all 6
damage classes, and everything that differs between them (insurer identity,
vehicle-age depreciation tier, which add-ons were bought, reporting
deadlines) is a fact about the customer's contract, not about the dented
panel in the photo. The 315-case census in policy_selector.py proved this
empirically; the source documents explain why structurally.

So the applicable policy is an explicit pipeline INPUT, not a guess. In a
real deployment it comes from the policy number -> schedule lookup. In this
project the claimant selects it, and because a claimant needs to recognise
their own cover, each option carries a short factual description of what that
policy does. These descriptions are drawn directly from the policy wordings
(data/policy_pdfs/synthetic/), not invented -- they are selection guidance,
so they must stay accurate to the documents.
"""

# doc_id -> catalog entry. `distinctive` is what sets this policy apart from
# the other four (all five share the same 6-class coverage and the same
# glass-nil-depreciation / tyre-only-with-concurrent-damage terms, so those
# are deliberately NOT listed as distinctive).
POLICY_CATALOG = {
    "policy_1_bharat_suraksha": {
        "insurer": "Bharat Suraksha Motor Insurance Co. Ltd",
        "product": "Motor Private Car Comprehensive (Own Damage + Third Party Liability)",
        "style": "Traditional IRDAI-style single-clause wording",
        "summary": (
            "Comprehensive cover with all damage types indemnified under one "
            "accidental-external-means clause. Standard age-based depreciation "
            "(Nil to 50%); glass at nil depreciation; fibre-glass parts at 30%. "
            "No add-on covers."
        ),
        "distinctive": "Single umbrella coverage clause; 11 numbered exclusions; no add-ons.",
    },
    "policy_2_safedrive_assurance": {
        "insurer": "SafeDrive Assurance Corporation",
        "product": "Standalone Own Damage cover with Nil-Depreciation add-on included",
        "style": "Modern plain-language, dedicated subsection per damage type",
        "summary": (
            "Standalone own-damage cover with Nil Depreciation built in as "
            "standard for vehicles under 3 years (75% depreciation waiver for "
            "3-5 years, none beyond 5 years). Separate detailed glass and tyre "
            "coverage sections."
        ),
        "distinctive": "Nil-depreciation included by default (age-tiered); sunroof glass excluded unless endorsed.",
    },
    "policy_3_quickclaim_general": {
        "insurer": "QuickClaim General Insurance Ltd",
        "product": "Motor Vehicle Package (Comprehensive Own Damage + compulsory Third Party)",
        "style": "Dense legal drafting with lettered exclusion schedules (A-F)",
        "summary": (
            "Comprehensive package with the most detailed exclusion schedules of "
            "the five (six named schedules, A-F). Glass at nil depreciation. "
            "Notable conditions: scratch/paint claims must be reported within 30 "
            "days; a pre-inspection report governs pre-existing damage."
        ),
        "distinctive": "Strictest procedural conditions (30-day scratch reporting, pre-inspection); most distractor exclusions.",
    },
    "policy_4_autoguard_premium": {
        "insurer": "AutoGuard Premium Insurance Services Ltd",
        "product": "Own Damage with Full-Cover Add-On Suite",
        "style": "Consumer-facing, coverage presented as a summary table",
        "summary": (
            "Premium-tier product presenting cover as a summary table, with a "
            "Full-Cover Add-On Suite (Nil Depreciation, Return-to-Invoice, "
            "Consumables). Glass nil depreciation standard. Compulsory deductible "
            "Rs. 2,500 per claim."
        ),
        "distinctive": "Most add-on options (nil-dep, return-to-invoice, consumables); Rs. 2,500 compulsory deductible.",
    },
    "policy_5_valuemotor": {
        "insurer": "ValueMotor Comprehensive Insurance Ltd",
        "product": "Affordable Comprehensive (Own Damage + Third Party)",
        "style": "Budget insurer, consumer-friendly 'you/your' language",
        "summary": (
            "Budget comprehensive cover with standard age-based depreciation and "
            "no nil-depreciation add-on (glass is still always nil depreciation; "
            "rubber/plastic/tyre always 50% deduction). Tyre cover explicitly "
            "conditional on concurrent body damage."
        ),
        "distinctive": "Cheapest tier, no depreciation add-on; strict driving-licence and deliberate-damage exclusions.",
    },
}

DOC_IDS = list(POLICY_CATALOG.keys())


class PolicyCatalog:
    """The selection menu presented to the claimant, and validation for the
    doc_id they pick. This is Stage 1 of the pipeline -- explicit selection,
    not inference (see module docstring)."""

    @staticmethod
    def list_options() -> list:
        """Return the menu of policies with descriptions, for the claimant to
        choose from."""
        return [
            {
                "doc_id": doc_id,
                "insurer": e["insurer"],
                "product": e["product"],
                "summary": e["summary"],
                "distinctive": e["distinctive"],
            }
            for doc_id, e in POLICY_CATALOG.items()
        ]

    @staticmethod
    def describe(doc_id: str) -> dict:
        if doc_id not in POLICY_CATALOG:
            raise ValueError(f"Unknown doc_id '{doc_id}'. Valid options: {DOC_IDS}")
        return POLICY_CATALOG[doc_id]

    @staticmethod
    def is_valid(doc_id: str) -> bool:
        return doc_id in POLICY_CATALOG


if __name__ == "__main__":
    import json
    print(json.dumps(PolicyCatalog.list_options(), indent=2))
