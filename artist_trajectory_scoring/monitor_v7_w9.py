from pathlib import Path

project = Path(
    "/mnt/ssd/artistDiffusion/ArtistDiffusion/"
    "artist_trajectory_scoring"
)

base = project / (
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v7_cost_improving_residual_targets_100paths_fast"
)

roots = [base] + [
    project / (
        "data/cartesian_expert_dataset_v3/"
        "diffusion_v7_cost_improving_residual_targets_100paths_fast_"
        f"w9_shard{i}"
    )
    for i in range(1, 10)
]

seen = set()

for root in roots:
    state_root = root / "window_state"

    if not state_root.is_dir():
        continue

    for path in state_root.glob("*/window_*.json"):
        seen.add((path.parent.name, path.name))

total = 100 * 18
done = len(seen)
percentage = 100.0 * done / total

print(
    f"Overall: {done}/{total} unique windows "
    f"({percentage:.2f}%)"
)

for index, root in enumerate(roots[1:], start=1):
    count = len(list(
        (root / "window_state").glob("*/window_*.json")
    ))
    print(f"Worker {index}: {count} saved windows")
