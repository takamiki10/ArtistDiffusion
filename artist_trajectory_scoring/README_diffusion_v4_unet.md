# ArtistDiffusion v4 Conditional U-Net

v4 keeps the project diffusion-based. It does not replace diffusion with IK or an MLP. IK remains useful for generating expert demonstrations and for baseline comparison, while the learned trajectory generator is still a conditional diffusion model.

The main architectural change is replacing the weaker temporal CNN with a Stanford Diffusion Policy-style 1D conditional U-Net. The model denoises a full joint trajectory sequence with Conv1d residual blocks, U-Net down/up paths, skip connections, sinusoidal diffusion timestep embeddings, and FiLM-style conditioning from the desired path features.

## Dataset

v4 reuses the existing v2 diffusion dataset:

```text
data/cartesian_expert_dataset_v3/diffusion_v2/
```

Condition shape is expected to be `(N, 100, 13)`:

```text
[x, y, z, dx, dy, dz, normalized_t, q_start_1, ..., q_start_6]
```

Target shape is expected to be `(N, 100, 6)`:

```text
delta_q(t) = expert_q(t) - q_start
```

Training prefers normalized condition and normalized target arrays when the `.npz` files provide them. The target for diffusion is normalized `delta_q`.

## Training

From `ArtistDiffusion/artist_trajectory_scoring`:

```bash
python train_conditional_diffusion_trajectory_v4_unet.py \
  --dataset_dir data/cartesian_expert_dataset_v3/diffusion_v2 \
  --epochs 2000 \
  --batch_size 32 \
  --hidden_dim 256 \
  --lr 1e-4 \
  --num_diffusion_steps 100 \
  --output_dir data/cartesian_expert_dataset_v3/diffusion_v4_unet \
  --device cuda
```

Training writes:

```text
best_model.pt
last_model.pt
train_log.csv
config.json
```

The checkpoint stores the model state, optimizer state, epoch, best validation loss, model config, diffusion schedule config, and normalization statistics when available.

## Sampling

```bash
python sample_conditional_diffusion_trajectory_v4_unet.py \
  --checkpoint data/cartesian_expert_dataset_v3/diffusion_v4_unet/best_model.pt \
  --dataset_dir data/cartesian_expert_dataset_v3/diffusion_v2 \
  --max_paths 83 \
  --num_samples 1 \
  --output_dir data/cartesian_expert_dataset_v3/diffusion_v4_unet/samples_single \
  --device cuda
```

For each path, sampling writes:

```text
diffusion_v4_pred_q.csv
metrics.json
plot.png
```

`diffusion_v4_pred_q.csv` uses the active joint convention:

```text
t,q1,q2,q3,q4,q5,q6
```

## Interpreting Results

The first success criterion is that v4 single-sample trajectories should follow the condition better than previous simple diffusion models. The expected early signal is improved Cartesian tracking while retaining smooth joint motion.

The sampler prints:

```text
evaluated count
accepted count
mean path_error
mean Cartesian error
mean max Cartesian error
worst max Cartesian error
mean joint velocity cost
mean joint acceleration cost
mean generation time
```

Accepted paths are intended to satisfy:

```text
mean Cartesian error <= 0.01
max Cartesian error <= 0.03
```

## Limitations

Cartesian FK-based cost is currently used for evaluation and ranking only, not as a differentiable training loss. Training remains standard DDPM epsilon prediction in normalized joint-delta space.

## Future Work

Good next steps after the v4 architecture is stable:

- Expand the expert dataset.
- Revisit best-of-K sampling from the stronger v4 model.
- Add differentiable FK or an FK surrogate for training-time Cartesian loss.
- Wire a stable xMateCR7 FK evaluator into the sampler using `robot.update_cfg(cfg)` and `robot.get_transform(frame_to=ee_link)`, not `robot.link_fk(...)`.
