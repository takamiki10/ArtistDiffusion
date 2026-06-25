# Q-learning Cartesian Cost Debugger

This is a deliberately small, robot-free test bed for Cartesian path costs. It
does **not** load a URDF, run forward kinematics, or solve IK. A tabular
Q-learning agent moves a point on an 8-connected x-y grid around a CSV path so
that cutting, loops, cusps, and backward progress can be seen before reward
terms are copied into `optimize_cartesian_path.py`.

## Input and outputs

The input CSV must contain `t,x,y,z`. Only x and y affect the environment. The
z value at the closest desired-path index is retained in `learned_path.csv` for
future extension.

Training writes the following files under `--output_dir`:

- `q_table.npy`: dense float32 Q-table
- `learned_path.csv`: deterministic greedy rollout and per-step diagnostics
- `training_log.csv`: reward, success, error, backward motion, and cusps by episode
- `metrics.json`: rollout metrics, reward breakdown, grid settings, and weights
- `qlearn_debug_plot.png`: desired/learned x-y, tracking error, path index, and reward

## Generate the first target

From this directory (`artist_trajectory_scoring`):

```bash
python generate_cartesian_test_paths.py \
  --output_dir data/cartesian_test_paths \
  --num_steps 100 \
  --duration 1.0
```

The first target is then:

```text
data/cartesian_test_paths/arc_001/desired_path.csv
```

## Train

```bash
python train_qlearn_cartesian_debug.py \
  --path_csv data/cartesian_test_paths/arc_001/desired_path.csv \
  --output_dir data/cartesian_test_paths/arc_001/qlearn_debug \
  --episodes 3000 \
  --w_point 2.0 \
  --w_segment 5.0 \
  --w_alignment 0.25 \
  --w_backward 1.0 \
  --w_cusp 1.0 \
  --w_step 0.01 \
  --finish_bonus 25.0
```

All reward weights are configurable. Costs are subtracted from reward; the
finish bonus is added. Use `python train_qlearn_cartesian_debug.py --help` for
grid, episode, epsilon, and Q-learning settings.

The exact terms are:

- **point**: distance to the next desired point after the previously closest index
- **segment**: distance to the geometrically nearest path segment
- **alignment**: `1 - dot(action_direction, local_path_tangent)`
- **backward**: number of indices lost when the closest-path index decreases
- **cusp**: negative dot product of consecutive action directions, clipped at zero
- **step**: constant cost on every action
- **finish**: bonus near the final path point and one of the final two indices

The ordered next-point term supplies a local forward target, while the segment
term independently exposes geometric cutting. The closest index is intentionally
not forced to be monotonic: backward motion remains observable and punishable.

## Replot existing results

```bash
python plot_qlearn_cartesian_debug.py \
  --path_csv data/cartesian_test_paths/arc_001/desired_path.csv \
  --output_dir data/cartesian_test_paths/arc_001/qlearn_debug
```

## Reading the debugger

- Cutting: low point error but visibly shortened geometry or elevated segment error
- Looping: repeated x-y regions and a path-index trace that stalls or oscillates
- Cusps: nonzero `cusp_count`, especially with reversals visible in the learned path
- Backtracking: downward jumps in the closest-path-index plot and nonzero backward metrics
- Failure to finish: `finished: false` in `metrics.json`, often indicating weights or
  exploration make the terminal bonus unreachable

Compare runs in separate output directories and change one weight at a time.
The Q-table grows with `x_cells * y_cells * path_points * 9 * 8`; increase
`--grid_spacing` if a large path would make the table impractical.
