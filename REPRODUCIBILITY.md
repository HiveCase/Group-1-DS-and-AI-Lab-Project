# Reproducibility

This file records independent verification that this repository's key results can be reproduced
by someone who was **not** primarily responsible for assembling it, starting from a fresh
clone/download and following only the instructions in [`README.md`](README.md) and
[`backend/app/rag_scripts/README.md`](backend/app/rag_scripts/README.md).

For what "key results" means and exactly how to reproduce each one yourself, see `README.md` §7
("End-to-End Reproducibility") and §9 ("Evaluation & Results"), and the single entry-point script
[`scripts/reproduce.py`](scripts/reproduce.py).

---

## How to perform this verification

1. On a machine you have not used to develop this project (or at least a clean checkout
   directory), run:
   ```bash
   git clone <repo-url> verification-clone
   cd verification-clone
   python scripts/reproduce.py
   ```
2. Separately, follow `README.md` §5 to install and run the frontend, and submit at least one
   claim end-to-end through the UI (Claimant → Adjuster) as described in §7.
3. Fill in **one entry per verification pass** in the table below, using the template under it.
   Do not edit or delete a prior entry — append a new one, even if it re-verifies the same
   commit, so this file stays an honest running record.

---

## Verification log

<!--
TEMPLATE -- copy this block for each new verification entry and fill in every field.
Do not leave a field blank; write "N/A" explicitly if something genuinely does not apply.

### YYYY-MM-DD -- <Full Name> (<GitHub handle>)

- **Commit verified**: `<git rev-parse HEAD, short or full sha>`
- **Role on this project**: <e.g. "RAG evaluation" -- state that this is NOT the same area the
  verifier primarily worked on, per the assignment's independence requirement>
- **Machine / OS**: <e.g. "Windows 11, Python 3.12.4, Node 20.11" or "Ubuntu 22.04 in a fresh
  Docker container">
- **Steps followed**: <e.g. "README.md §7 end-to-end, plus `python scripts/reproduce.py`">
- **Key results reproduced?**
  - [ ] Live app: claim submitted, 5-agent pipeline completed, report visible in Adjuster portal
  - [ ] Backend test suite: `___ passed, ___ skipped` (compare to README.md's baseline: 67/2 without a live `GROQ_API_KEY`, 69/0 with one)
  - [ ] RAG retrieval evaluation: mean P@3 = `____`, MRR@5 = `____`
  - [ ] (optional) Report-generation / RAGAs evaluation, if API keys were available
  - [ ] (optional) YOLO retraining notebook, if a GPU/Colab/Kaggle account was available
- **Deviations or issues encountered**: <describe anything that didn't match the docs exactly,
  even minor -- a wrong path, a missing dependency, a version mismatch, a number that didn't
  match. "None" is a valid answer only if truly nothing came up.>
- **Fixes applied as a result** (if any): <link the commit/PR that fixed an issue this
  verification found, so the fix is traceable back to this entry>
- **Overall outcome**: <Reproduced / Reproduced with deviations / Not reproduced>
-->

*(No verification entries recorded yet. Add the first one using the template above before
submission.)*
