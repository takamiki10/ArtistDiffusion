# Box archive plan

Confirmed 5 October 2026: GitHub holds source, tests, setup instructions, documentation, and the archive index. Selected data and model artifacts will be stored on Box.

| Archive group | Retain |
| --- | --- |
| Frozen inference | Final checkpoint, matching normalization/schedule/metadata, MLP weights, robot assets |
| Numerical experiments | Evaluation inputs, split/path manifests, seed-level trajectories and metrics, benchmark settings and final outputs |
| Physical experiments | Essential Windows logs, original drawing scans, calibration/registration, trial metadata and final analyses |
| Windows restoration | Required handoff inputs, prepared batches/manifests, SDK/runtime provenance and required packages |
| Training reproduction | Training inputs/targets/splits, or verified regeneration inputs and instructions |

For each archive record filename, Box link or identifier, purpose, byte size, SHA-256, per-file manifest, matching code commit, restore location and verification date. Preserve relative paths and link the index from the README and final PDF.

Select exact files before compression. Distinguish physical from synthetic evidence. Review duplicate plots, caches, build products and diagnostics before omitting them; record omissions and retain originals until verification.

Source reorganization changes hashes. Validate historical artifacts with the commit recorded in their provenance in a separate checkout; do not rewrite historical hashes to match current source.

Test extraction, verify checksums, reproduce one offline result and confirm recipient access. Archive creation and Box upload are pending; no Box links are assigned yet.
