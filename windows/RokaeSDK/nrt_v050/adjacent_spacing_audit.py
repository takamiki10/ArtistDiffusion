"""Offline-only spacing audit. Reads frozen batches; never imports a vendor SDK.

Run from the project root with: python -m nrt_v050.adjacent_spacing_audit
Only the evidence JSON is written. No trajectory is regenerated or modified.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics

from neutral_v050_nrt_runner import generate_neutral, neutral_spacing_diagnostics

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "nrt_v050" / "prepared"
OUTPUT = ROOT / "nrt_v050" / "evidence" / "adjacent_spacing_audit.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_neutral(q0, joint, direction, amplitude_deg):
    """Previous neutral formula, ONLY for offline comparison; no command objects."""
    result = []
    for i in range(100):
        row = list(q0)
        if i not in (0, 99):
            row[joint - 1] += (1 if direction == "+" else -1) * math.radians(amplitude_deg) * math.sin(math.pi * (i / 99.0)) ** 4
        result.append(row)
    return result


def joint_space_spacing(rows):
    if len(rows) != 100 or any(len(row) != 6 for row in rows):
        raise ValueError("expected 100 six-joint vectors")
    if any(type(v) is not float or not math.isfinite(v) for row in rows for v in row):
        raise ValueError("expected finite binary64 joint values")
    distances = [math.dist(a, b) for a, b in zip(rows, rows[1:])]
    smallest = sorted(range(99), key=lambda i: (distances[i], i))[:5]
    duplicates = [[i, i + 1] for i, (a, b) in enumerate(zip(rows, rows[1:])) if a == b]
    return {"waypoint_count": 100, "adjacent_pair_count": 99,
            "metric": "six-dimensional Euclidean joint-space distance in radians",
            "pair_indexing": "zero-based [from, to]; ties sorted by lower index",
            "minimum_l2_rad": min(distances), "median_l2_rad": statistics.median(distances),
            "maximum_l2_rad": max(distances),
            "smallest_five": [{"pair": [i, i + 1], "l2_rad": distances[i]} for i in smallest],
            "exact_duplicate_consecutive_vectors": bool(duplicates), "duplicate_pairs": duplicates,
            "adjacent_l2_rad": distances}


def build_audit():
    manifest_path = PREPARED / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protected = {str(p.relative_to(ROOT)): sha256(p) for p in PREPARED.iterdir() if p.is_file()}
    batches = {}
    for condition, suffix in (("R", "A"), ("D", "B")):
        key = "path_0003_" + suffix
        path = PREPARED / (key + "_waypoints.json")
        metadata = manifest["batches"][key]
        actual_hash = sha256(path)
        if actual_hash != metadata["prepared_sha256"]:
            raise ValueError("frozen prepared hash mismatch: " + key)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["source_sha256"] != metadata["source_sha256"]:
            raise ValueError("frozen source hash mismatch: " + key)
        planned = [trial for trial in manifest["trials"] if trial["path"] == "path_0003" and trial["condition"] == condition]
        if len(planned) != 5 or any(trial["source"] != data["source"] for trial in planned):
            raise ValueError("unexpected trial mapping")
        batches[condition] = {"prepared_path": str(path.relative_to(ROOT)),
                              "prepared_sha256": actual_hash, "original_source": data["source"],
                              "original_source_sha256": data["source_sha256"],
                              "planned_trial_numbers": [trial["trial"] for trial in planned],
                              **joint_space_spacing(data["q_rad"])}
    q0 = [0.0] * 6
    comparison = {"amplitude_deg": 0.1, "q0_rad": q0, "joint": 1, "direction": "+",
                  "old_sin4": neutral_spacing_diagnostics(legacy_neutral(q0, 1, "+", .1), q0, 1),
                  "asymmetric_triangle": neutral_spacing_diagnostics(generate_neutral(q0, 1, "+", .1), q0, 1)}
    after = {str(p.relative_to(ROOT)): sha256(p) for p in PREPARED.iterdir() if p.is_file()}
    if protected != after:
        raise ValueError("frozen files changed during audit")
    return {"audit_kind": "OFFLINE_DIAGNOSTIC_ONLY", "physical_rd_outcomes_used": False,
            "rd_motion_authorized": False, "frozen_files_unchanged": True,
            "manifest_sha256": sha256(manifest_path), "protected_file_sha256": protected,
            "neutral_comparison": comparison, "path_0003": batches,
            "caveats": ["Distances are submitted joint-space spacing, not physical motion or controller acceptance.",
                        "Minimum nonzero neutral spacing excludes explicit zero gaps, which are reported separately.",
                        "The asymmetric triangular neutral profile has a unique peak and no consecutive duplicate targets.",
                        "No threshold or controller policy is changed by this diagnostic audit."]}


def main():
    audit = build_audit()
    OUTPUT.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Wrote", OUTPUT)
    print(json.dumps({k: {name: v[name] for name in ("minimum_l2_rad", "median_l2_rad", "maximum_l2_rad", "duplicate_pairs", "smallest_five")}
                      for k, v in audit["path_0003"].items()}, indent=2))
    for name in ("old_sin4", "asymmetric_triangle"):
        v = audit["neutral_comparison"][name]
        print(name, json.dumps({key: v[key] for key in ("minimum_nonzero_adjacent", "minimum_adjacent_including_zero", "first_interior_displacement", "minimum_nonzero_adjacent_pairs", "zero_adjacent_pairs")}))


if __name__ == "__main__":
    main()
