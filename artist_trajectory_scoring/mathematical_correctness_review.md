The main mathematical issue is that the repository uses **two different meanings of velocity, acceleration, and jerk**: unscaled finite differences for trajectory selection, and timestamp-based derivatives for deployment reporting. It also contains materially different legacy and current pipelines, including a confirmed legacy MLP output-convention error.

I reviewed the source, configuration artifacts, dataset arrays, MLP checkpoint metadata, and installed FK implementation. **No files were modified.** The primary pipeline below is the frozen v8.1 deployment path; legacy differences are identified explicitly.

Let \(q_i\in\mathbb R^6\) denote joint positions, \(p_i^\star\in\mathbb R^3\) desired Cartesian positions, and \(p_i=f(q_i)\) FK positions. Joint positions are in radians; Cartesian positions are in metres. Use \(i\) for trajectory samples and \(k\) for diffusion timesteps.

**1. Pipeline traced**

The current path is:

\[
P^\star
\longrightarrow \text{path-conditioned MLP}
\longrightarrow \text{adaptive IK prior }Q^P
\longrightarrow \text{normalized residual diffusion}
\longrightarrow \text{FK-scored candidates}
\longrightarrow \text{receding-horizon execution}
\longrightarrow \text{deployment validation}.
\]

The MLP predicts normalized **absolute joint positions**:

\[
q_i^{\rm MLP}
=\sigma_y\odot
f_\theta\!\left((c_i^{\rm MLP}-\mu_x)/\sigma_x\right)+\mu_y.
\]

Adaptive position IK uses

\[
\min_{\ell\le q\le u}
\|f(q)-p_i^\star\|_2^2+
\lambda_{\rm smooth}\|q-q_{\rm ref}\|_2^2.
\]

- Sources: [predict_path_conditioned_mlp.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/predict_path_conditioned_mlp.py:76), `make_features`, `predict_q`, lines 76–105; [generate_ik_seed_path.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_ik_seed_path.py:317), `solve_ik_from_initial_guess`, lines 317–349.
- Adaptive settings: \(\lambda_{\rm smooth}=0.01\), then \(0.001\) for the second stage; the second stage is triggered by maximum position error exceeding \(0.03\) m. [generate_adaptive_mlp_ik_bootstrap_prior.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_adaptive_mlp_ik_bootstrap_prior.py:814), `adaptive_refine_path`, lines 814–875.
- Units: the IK objective combines \(\mathrm m^2\) and \(\mathrm{rad}^2\). Its coefficients therefore encode a unit-dependent tradeoff.
- Issue: this is regularized position fitting, not an equality-constrained Cartesian tracking solution.

Frozen deployment settings are \(N=100\), prediction horizon \(H=32\), execution horizon \(E=8\), anchoring horizon \(A=8\), \(K=8\), 50 DDIM steps, \(\eta=0\), target scale \(1\), output multiplier \(0.125\). [generate_joint_trajectory_diffusion_v8_1.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_joint_trajectory_diffusion_v8_1.py:44), module constants, lines 44–58.

**2. Cartesian tracking error, RMS, and maximum**

For a region containing \(n\) samples:

\[
r_i=p_i-p_i^\star,\qquad e_i=\|r_i\|_2,
\]

\[
E_{\rm mean}=\frac1n\sum_i e_i,\qquad
E_{\rm RMS}=\sqrt{\frac1n\sum_i e_i^2},\qquad
E_{\max}=\max_i e_i.
\]

The implementation also reports

\[
J_{\rm path}=\frac1n\sum_i\sum_{a=1}^{3}r_{ia}^2
=E_{\rm RMS}^2.
\]

- Sources: [generate_diffusion_v7_cost_improving_residual_targets.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_diffusion_v7_cost_improving_residual_targets.py:584), `trajectory_metrics`, lines 584–629; [trajectory_costs.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/trajectory_costs.py:44>), `compute_cartesian_error_metrics`, lines 44–56; [generate_joint_trajectory_diffusion_v8_1.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_joint_trajectory_diffusion_v8_1.py:1422), `main`, lines 1422–1433.
- Units: \(r_i,e_i,E_{\rm mean},E_{\rm RMS},E_{\max}\): metres; \(J_{\rm path}\): \(\mathrm m^2\).
- Interpretation: errors compare corresponding sample indices. They are not nearest-path distances, registration errors, or DTW errors.
- Issue: Cartesian RMS sums the three coordinate errors before averaging. It is \(\sqrt3\) times the coordinate-pooled RMS:
  \[
  \sqrt{\operatorname{mean}_{i,a}r_{ia}^2}=E_{\rm RMS}/\sqrt3.
  \]
- Issue: these are sample averages, not time-weighted averages. With nonuniform timestamps, they do not equal continuous-time RMS.

Joint-reference RMSE uses a different reduction:

\[
E_{q,\rm RMS}=
\sqrt{\frac1{6n}\sum_{i,j}(q_{ij}-q_{ij}^{\rm expert})^2},
\qquad
E_{q,\max}=\max_{i,j}|q_{ij}-q_{ij}^{\rm expert}|.
\]

Source: [evaluate_prior_refinement_fk_robot_costs.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/evaluate_prior_refinement_fk_robot_costs.py:317>), `q_error_metrics`, lines 317–319. Units: radians. This evaluates agreement with one IK solution, which need not be the only valid solution.

**3. Velocity, acceleration, jerk, and \(\Delta t\)**

The selection and training-difference convention is

\[
Dq_i=q_{i+1}-q_i,
\]

\[
D^2q_i=q_{i+2}-2q_{i+1}+q_i,
\]

\[
D^3q_i=q_{i+3}-3q_{i+2}+3q_{i+1}-q_i.
\]

No division by elapsed time occurs.

The v7/v8/v8.1 derivative costs are

\[
C_r(Q)=\frac1{n-r}\sum_{i=0}^{n-r-1}
\|D^rq_i\|_2^2,\qquad r=1,2,3.
\]

- Source: [generate_diffusion_v7_cost_improving_residual_targets.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_diffusion_v7_cost_improving_residual_targets.py:539), `derivative_cost`, lines 539–541; calls in `trajectory_metrics`, lines 627–629.
- Dimensions: \(D^rQ\in\mathbb R^{(n-r)\times6}\).
- Units: differences retain radians; squared costs retain \(\mathrm{rad}^2\). They may be described as per-sample differences, but not as SI time derivatives.
- \(\Delta t\): implicit index spacing \(1\) sample, **not a documented one-second interval**.

Legacy helpers instead average over joints too:

\[
\widetilde C_r(Q)=
\frac1{6(n-r)}\sum_{i,j}(D^rq_{ij})^2
=\frac{C_r(Q)}6.
\]

Sources: [trajectory_costs.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/trajectory_costs.py:59>), `compute_joint_velocity_cost`, `compute_joint_acceleration_cost`, lines 59–72; [evaluate_prior_refinement_fk_robot_costs.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/evaluate_prior_refinement_fk_robot_costs.py:322>), `smoothness_costs`, lines 322–330.

**Issue:** identically named costs can differ by a factor of six across repository versions.

Deployment reporting uses an actual differentiation operator \(G_t\):

\[
V=G_tQ,\qquad A=G_tV=G_t^2Q,\qquad J=G_tA=G_t^3Q,
\]

where each \(G_t\) is `np.gradient(..., timestamps, axis=0, edge_order=2)`.

Source: [generate_joint_trajectory_diffusion_v8_1.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_joint_trajectory_diffusion_v8_1.py:794), `dynamics`, lines 794–798.

For uniform spacing \(h\),

\[
(G_tf)_i=\frac{f_{i+1}-f_{i-1}}{2h},
\]

with endpoint rules

\[
(G_tf)_0=\frac{-3f_0+4f_1-f_2}{2h},
\quad
(G_tf)_{n-1}=\frac{3f_{n-1}-4f_{n-2}+f_{n-3}}{2h}.
\]

For nonuniform interior spacing \(h_-=t_i-t_{i-1}\), \(h_+=t_{i+1}-t_i\),

\[
(G_tf)_i=
-\frac{h_+f_{i-1}}{h_-(h_-+h_+)}
+\frac{(h_+-h_-)f_i}{h_-h_+}
+\frac{h_-f_{i+1}}{h_+(h_-+h_+)}.
\]

Endpoint derivatives come from the corresponding three-point one-sided quadratic interpolant. This operator is applied three times.

- Dimensions: all three exported arrays are \(100\times6\).
- Units: rad/s, rad/s², rad/s³.
- Default timestamps:
  \[
  t_i=\frac{10i}{99}\ {\rm s},\qquad
  h=\frac{10}{99}\approx0.1010101\ {\rm s}.
  \]
  Sources: `parse_args`, line 129; `load_input`, lines 576–586, in [generate_joint_trajectory_diffusion_v8_1.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_joint_trajectory_diffusion_v8_1.py:576).
- Supplied timestamps override this default; they need only be finite and strictly increasing.
- The Cartesian CSV input generator maps source time to
  \[
  t_i^{\rm physical}
  =D\frac{t_i^{\rm source}-t_0^{\rm source}}
  {t_{N-1}^{\rm source}-t_0^{\rm source}},
  \]
  preserving relative timing while assigning a requested duration. [generate_deployment_input_from_cartesian_csv.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_deployment_input_from_cartesian_csv.py:218), `read_cartesian_csv`, lines 218–264.

The smoothness benchmark defaults to a common 10-second duration, so its SI derivatives describe standardized timing rather than necessarily the original execution timing. Sources: [benchmark_ik_mlp_pipeline_smoothness.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/benchmark_ik_mlp_pipeline_smoothness.py:200), `parse_args`, lines 200–205; [compare_ik_mlp_pipeline_jerk_over_time.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/compare_ik_mlp_pipeline_jerk_over_time.py:810), `align_common_duration`, lines 810–886, and `compute_derivatives`, lines 1033–1043.

**4. Is jerk physical jerk?**

**In candidate selection and the v8.1 jerk guard: no. In timestamp-based deployment and benchmark reporting: yes, as a numerical estimate.**

To turn the forward third difference into a time-scaled approximation on a uniform grid:

\[
\widehat{\dddot q}_i=\frac{D^3q_i}{h^3},
\qquad
C_{\rm physical\ jerk}=\frac{C_3(Q)}{h^6}.
\]

However,

\[
G_t^3Q\ne D^3Q/h^3
\]

in general: they use different stencils and boundary treatment.

The history-aware guard prepends up to three already executed states and retains third-difference stencils whose newest sample belongs to the proposed prefix:

\[
C_{3,\rm hist}
=\frac1{|\mathcal I|}
\sum_{i\in\mathcal I}\|D^3q_i\|_2^2.
\]

It requires

\[
C_{3,\rm hist}(Q^{\rm candidate})
-C_{3,\rm hist}(Q^{\rm fallback})
\le10^{-12}.
\]

Source: [evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py:109), `executed_history`, `history_aware_incremental_jerk_cost`, lines 109–137; `evaluate_candidates_v8_1`, lines 340–353.

- Units: cost difference and its tolerance are in the squared discrete-difference convention, \(\mathrm{rad}^2\).
- Issue: this is not a maximum physical jerk constraint.
- Issue: it compares against the current anchored fallback, not directly against the original complete prior.
- Issue: changing physical timestamps changes reported physical jerk without changing the selector’s objective.

Physical reporting also uses

\[
J_{\rm RMS}=\sqrt{\frac1{6N}\sum_{i,j}J_{ij}^2},
\]

and trapezoidal integration for

\[
I_J\approx\int\sum_jJ_j(t)^2\,dt,
\]

with units rad/s³ and rad²/s⁵, respectively. Source: [compare_ik_mlp_pipeline_jerk_over_time.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/compare_ik_mlp_pipeline_jerk_over_time.py:1046), `integrate`, `per_joint_summary`, `method_summary`, lines 1046–1111.

**5. Trajectory, residual, and conditioning dimensions**

The current v8 contract is:

| Quantity | Shape | Units |
|---|---:|---|
| Desired/FK Cartesian path | \(100\times3\) | m |
| Joint path/prior | \(100\times6\) | rad |
| Cartesian prediction window | \(32\times3\) | m |
| Joint/residual prediction window | \(32\times6\) | rad |
| Normalized diffusion state/noise | \(B\times32\times6\) | Dimensionless |
| Raw/normalized condition | \(B\times32\times39\) | Mixed units / standardized |
| U-Net input residual layout | \(B\times6\times32\) | Dimensionless |
| U-Net condition layout | \(B\times39\times32\) | Standardized |
| Orientation trajectory | \(100\times3\times3\), or \(100\times4\) XYZW quaternions | Dimensionless |
| Orientation error | \(100\) | rad |

Sources: [train_conditional_diffusion_trajectory_v8.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/train_conditional_diffusion_trajectory_v8.py:43), constants, lines 43–47; [train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet.py:539), `predict_noise`, lines 539–557.

The 39 channels, in exact zero-based order, are:

| Channels | Formula | Units before normalization |
|---|---|---|
| 0–2 | \(p_i^\star\) | m |
| 3–5 | \(p_i^\star-p_{i-1}^\star\), zero at path start/padded future samples | m per sample |
| 6 | \(i/(N-1)\), clamped at the endpoint | Dimensionless |
| 7–12 | \(q_0^P\), repeated | rad |
| 13–18 | Current joint state, repeated | rad |
| 19–24 | Prior/anchored-prior \(q_i^P\) | rad |
| 25–30 | \(q_i^P-q_0^P\) | rad |
| 31–33 | \(f(q_i^P)\) | m |
| 34–36 | \(f(q_i^P)-p_i^\star\) | m |
| 37 | \(\|f(q_i^P)-p_i^\star\|_2\) | m |
| 38 | Target scale \(s\) | Dimensionless |

Source: [evaluate_diffusion_v8_teacher_forced_all_windows.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/evaluate_diffusion_v8_teacher_forced_all_windows.py:2308), `build_recursive_condition_norm`, lines 2308–2401. Training construction is in [build_diffusion_v6_strong_prior_residual_window_dataset.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/build_diffusion_v6_strong_prior_residual_window_dataset.py:318), `build_condition`, lines 318–353.

Issues:

- Desired “delta” channels are not Cartesian velocity.
- Progress is sample-index progress, not physical time or arc length.
- Orientation and timestamps are absent from diffusion conditioning.
- Training uses the prior’s current state; recursive inference uses the actually executed state and recomputed anchored FK. This creates a distribution change.

Version differences:

- v4: full trajectories \(100\times6\), conditions \(100\times13\); see [diagnose_diffusion_v4_pure_sampling_vs_noised_expert.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/diagnose_diffusion_v4_pure_sampling_vs_noised_expert.py:230>), `main`, lines 230–247.
- v5: \(32\times31\) conditions, the first 31 channels above; [build_diffusion_v5_residual_window_dataset.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/build_diffusion_v5_residual_window_dataset.py:174>), `build_condition_window`, lines 174–200.
- v5b/v6/v7: \(32\times38\), without target scale.
- Canonical MLP: \(3N+1+3=304\) inputs per sample by default: flattened whole desired path, time/progress, current desired XYZ. Output: six normalized absolute joint positions. `make_timestep_dataset`, [train_path_conditioned_mlp.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/train_path_conditioned_mlp.py:87), lines 87–111.

**6. Exact FK function and coordinate frame**

The current authoritative chain is:

`compute_fk_positions` → `fk_trajectory` → `fk_one` →

```python
robot.update_cfg({"joint1": q1, ..., "joint6": q6})
robot.get_transform(frame_to="xMateCR7_link6")
```

It returns the translation column:

\[
f(q)={}^{W}T_6(q)_{0:3,3}.
\]

Sources: [evaluate_diffusion_v8_teacher_forced_all_windows.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/evaluate_diffusion_v8_teacher_forced_all_windows.py:2514), `compute_fk_positions`, lines 2514–2522; [generate_diffusion_v7_cost_improving_residual_targets.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_diffusion_v7_cost_improving_residual_targets.py:456), `make_robot_context`, `fk_one`, `fk_trajectory`, lines 456–497.

For the supplied URDF,

\[
\begin{aligned}
{}^{W}T_6(q)=&
R_z(q_1)\,T_z(0.296)\,R_y(q_2)\,
T_z(0.49)\,R_y(-q_3)\\
&\cdot T_z(0.36)\,R_z(q_4)\,
T_y(-0.151)\,R_y(-q_5)\,
T_z(0.1265)\,R_z(q_6).
\end{aligned}
\]

Here rotations are homogeneous \(4\times4\) rotations and translations are in metres.

Source: [gazebo_xMateCR7.urdf](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/robot_model/rokae_ros_ws/rokae_ros_pkg/src/rokae_xMateCR7_moveit_config/config/gazebo_xMateCR7.urdf:38), fixed base and joint definitions, lines 38–42, 64–69, 102–107, 140–145, 178–185, 218–223, 256–261.

**Returned position is the origin of `xMateCR7_link6`, expressed in `world`.** The fixed world-to-base transform is identity in this URDF.

I verified locally that the default transform equals the explicitly world-referenced transform. The installed `yourdfpy` defaults to its scene base frame: [urdf.py](/home/s/.local/lib/python3.10/site-packages/yourdfpy/urdf.py:1141), `get_transform`, lines 1141–1171.

Issues:

- This is not automatically a brush-tip/TCP position. No tool-tip offset is applied in this FK path.
- With this model, link6-origin position is independent of \(q_6\). Its rotation changes, but position tracking cannot assess that joint’s orientation contribution.
- Legacy `score_trajectory.rokae_forward_kinematics` explicitly specifies `frame_from="world"`; its placeholder FK exists but is not the function called by the scorer. [score_trajectory.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/score_trajectory.py:24), lines 24–59, 185–186.

**7. Orientation evaluation**

The diffusion condition, target-selection score, and core v8/v8.1 metrics evaluate **position only**.

The deployment wrapper additionally computes

\[
\theta_i=
\cos^{-1}\!\left[
\operatorname{clip}\left(
\frac{\operatorname{tr}(R_\star^\top R_i)-1}{2},
-1,1\right)\right].
\]

- Source: [orientation_aware_adaptive_ik.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/orientation_aware_adaptive_ik.py:272), `orientation_geodesic_angle`, `orientation_error_trajectory`, lines 272–302.
- Units: radians; \(\theta_i\in[0,\pi]\).
- Target convention:
  \[
  R_\star=R_z(\mathrm{yaw})R_y(\mathrm{pitch})R_x(\mathrm{roll}),
  \]
  with XYZW quaternions. `quaternion_from_euler_rpy`, lines 107–133.
- Deployment requires:
  \[
  \max_i\theta_i\le g_R,\quad 0<g_R\le0.05\ {\rm rad},
  \]
  and
  \[
  \max_i|p_{i,z}-z_\star|\le g_z,\quad 0<g_z\le0.001\ {\rm m}.
  \]
  Both prior and final trajectory are checked.

Source: [generate_joint_trajectory_diffusion_v8_1.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_joint_trajectory_diffusion_v8_1.py:1265), `main`, lines 1265–1360; gate constants at 57–58.

The orientation-aware prior IK objective adds

\[
w_R\|\operatorname{Log}(R_\star^\top R(q))^\vee\|_2^2,
\qquad w_R=1.
\]

Source: [orientation_aware_adaptive_ik.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/orientation_aware_adaptive_ik.py:328), `solve_full_pose_ik_from_initial_guess.objective`, lines 328–349.

**Issue:** orientation/Z checks occur after the frozen rollout. They do not guide candidate ranking, and failure causes complete-trajectory rejection rather than an orientation-aware reselection.

**8. Residual definition, normalization, scaling, and reconstruction**

There are three different residual meanings.

For v4:

\[
\Delta q_i=q_i^{\rm expert}-q_{\rm start}.
\]

For v5/v6 supervised residual windows:

\[
r_i=q_i^{\rm expert}-q_i^P.
\]

Source: [build_diffusion_v6_strong_prior_residual_window_dataset.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/build_diffusion_v6_strong_prior_residual_window_dataset.py:417), `build_windows`, lines 417–421.

For v7/v8, targets are cost-improving candidate corrections:

\[
r_i^{\rm base}=q_i^{\rm candidate}-q_i^P,\qquad
r_i^{(s)}=s\,r_i^{\rm base},
\]

\[
q_i^{\rm target}=q_i^P+r_i^{(s)},
\qquad s\in\{0.125,0.25,0.5,0.75,1\}.
\]

Each scaled candidate is independently reevaluated before retention. Source: [generate_diffusion_v8_multitarget_scaled_residual_targets.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_diffusion_v8_multitarget_scaled_residual_targets.py:903), `evaluate_scaled_targets`, lines 903–949; reconstruction assertions in `validate` logic, lines 1827–1846.

For v8, over training rows \(b\) and window samples \(i\):

\[
\mu_{r,j}=\operatorname{mean}_{b,i}r_{bij},\qquad
\sigma_{r,j}=\operatorname{std}_{b,i}r_{bij}.
\]

If a standard deviation is \(\le10^{-8}\), the stored divisor becomes \(1\).

\[
x_{0,bij}=\frac{r_{bij}-\mu_{r,j}}{\sigma_{r,j}},
\qquad
\bar c_{bid}=\frac{c_{bid}-\mu_{c,d}}{\sigma_{c,d}}.
\]

The target-scale channel is explicitly exempted:

\[
\mu_{c,38}=0,\quad \sigma_{c,38}=1,\quad \bar c_{i,38}=s.
\]

Sources: [build_diffusion_v8_multitarget_scaled_training_dataset.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/build_diffusion_v8_multitarget_scaled_training_dataset.py:169), `training_normalization`, lines 169–192; `split_arrays`, lines 226–250; `main`, lines 356–378.

Inference denormalizes once:

\[
\widehat r=\sigma_r\odot\widehat x_0+\mu_r,
\]

then applies the separate output multiplier \(\gamma\):

\[
q_i^{\rm candidate}=q_i^{A}+\gamma\widehat r_i,
\qquad \gamma=0.125.
\]

Sources: [evaluate_diffusion_v8_teacher_forced_all_windows.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/evaluate_diffusion_v8_teacher_forced_all_windows.py:2455), `sample_ddim_candidates`, lines 2455–2464; [evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py:319), `evaluate_candidates_v8_1`, lines 319–323.

Issues:

- \(s\) is a learned condition; \(\gamma\) is an external multiplication. They are not interchangeable.
- No division by \(s\) occurs during denormalization.
- Joint subtraction is ordinary subtraction, not wrapped angular subtraction or a manifold logarithm.
- v8 statistics weight every retained target row equally; windows with more retained targets contribute more.
- v7 differs: its residual statistics use row weights \(1/\text{targets-per-window}\), and condition statistics use unique training windows. [build_diffusion_v7_cost_improving_training_dataset.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/build_diffusion_v7_cost_improving_training_dataset.py:416), `weighted_stats`, lines 416–424; `main`, lines 546–555.

Anchoring uses

\[
q_i^A=q_{s+i}^P+w_i(q_s^{\rm executed}-q_s^P),
\]

\[
w_i=1-3u_i^2+2u_i^3,\qquad
u_i=\operatorname{clip}(i/A,0,1),
\]

with \(w_i=0\) for \(i\ge A\). The candidate’s first point is overwritten with the current state; points \(1,\ldots,E\) are executed.

Sources: [evaluate_diffusion_v8_anchored_recursive_rollout.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/evaluate_diffusion_v8_anchored_recursive_rollout.py:146), `smoothstep`, `build_anchored_prior_window`, lines 146–206; v8.1 `run_rollout_v8_1`, lines 456–518.

The helper supports bound-adjusted anchoring weights, but the frozen v8.1 call at lines 459–465 does not pass bounds. Safety checks subsequently determine acceptability.

**9. Exact diffusion forward process and training loss**

The current schedule uses \(K_d=1000\):

\[
\beta_k=10^{-4}+\frac{k}{K_d-1}(0.02-10^{-4}),
\quad
\alpha_k=1-\beta_k,
\quad
\bar\alpha_k=\prod_{\ell=0}^{k}\alpha_\ell.
\]

Training samples \(k\) uniformly from \(0,\ldots,K_d-1\) and \(\epsilon\sim\mathcal N(0,I)\):

\[
x_k=\sqrt{\bar\alpha_k}\,x_0+
\sqrt{1-\bar\alpha_k}\,\epsilon.
\]

The model predicts noise, and reconstruction is

\[
\widehat x_0=
\frac{x_k-\sqrt{1-\bar\alpha_k}\,
\epsilon_\theta(x_k,k,\bar c)}
{\sqrt{\bar\alpha_k}}.
\]

Sources: [train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet.py:410), `build_schedule`, `add_noise`, `reconstruct_x0`, lines 410–440; [train_conditional_diffusion_trajectory_v8.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/train_conditional_diffusion_trajectory_v8.py:1286), `train_epoch`, lines 1286–1311.

All diffusion-state quantities are dimensionless. Diffusion \(k\) is unrelated to physical trajectory time.

The general v8 objective is

\[
L=
L_\epsilon+0.25L_0+0.05L_1+
0.10L_2+0.05L_3+0.10L_{\rm boundary}.
\]

Define physical residual prediction error

\[
e=\widehat r-r.
\]

For derivative order \(r=0,1,2,3\),

\[
L_r=
\frac1B\sum_b
\frac{\sum_iw_{r,i}\left[
\frac16\sum_j(D^re_{bij}/S_{r,j})^2\right]}
{\sum_iw_{r,i}},
\]

where \(w_{r,i}=2\) for the first \(E-r\) entries and \(1\) elsewhere, then normalized to mean one.

\[
L_\epsilon=\frac1{BH6}\sum_{b,i,j}
(\widehat\epsilon_{bij}-\epsilon_{bij})^2.
\]

The boundary loss is the weighted average, with weights \(2,1,2,1\), of joint-averaged squared normalized errors at:

\[
e_0/S_0,\quad e_{H-1}/S_0,\quad
De_0/S_1,\quad De_{H-2}/S_1.
\]

Sources: [train_conditional_diffusion_trajectory_v8.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/train_conditional_diffusion_trajectory_v8.py:1044), `temporal_weights`, `weighted_temporal_joint_mse`, `compute_loss_components`, lines 1044–1206; default coefficients, lines 203–210.

The scales \(S_r\) use training residual differences: ordinary standard deviation first, then \(1.4826\,\mathrm{MAD}\), then \(10^{-8}\). `robust_joint_scale`, `build_auxiliary_normalization`, lines 786–815.

**Crucially, the frozen deployed checkpoint is epsilon-only:**

\[
\boxed{L=L_\epsilon.}
\]

All five auxiliary coefficients are zero in [training_metadata.json](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/models/diffusion_v8_multitarget_scaled_residual_unet_100paths_epsilon_only_seed42/training_metadata.json:20), lines 20–25. A paper describing the deployed checkpoint as trained with jerk or acceleration losses would be incorrect.

Training rows are sampled with probabilities proportional to clipped, mean-one versions of

\[
n_{\rm window}(b)^{-1}
n_{\rm scale}(b)^{-1}
\mathrm{quality}_b^{0.25}.
\]

Weights affect sampling, not a second multiplication of the v8 loss. Source: `build_sampler_weights`, lines 839–864.

Legacy differences:

- v7 defaults to weighted epsilon MSE:
  \[
  L=\frac{\sum_bw_bL_{\epsilon,b}}{\sum_bw_b}.
  \]
  [train_conditional_diffusion_trajectory_v7.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/train_conditional_diffusion_trajectory_v7.py:316>), `weighted_average`, `train_epoch`, lines 316–352.
- v4 optionally penalizes differences of **predicted noise**:
  \[
  L=L_\epsilon+\lambda\operatorname{mean}(D\widehat\epsilon)^2.
  \]
  That is not a joint-trajectory smoothness loss. Its default coefficient is zero. [train_conditional_diffusion_trajectory_v4_unet.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/train_conditional_diffusion_trajectory_v4_unet.py:148>), `smoothness_loss`, `run_epoch`, lines 148–182; default at line 218.
- v4 accepts raw target keys without actually normalizing them; normalization statistics may only be recorded. Thus its mathematical state depends on the selected NPZ key. `DiffusionV2Dataset.__init__`, lines 77–121.

Current sampling starts from independent Gaussian noise. For selected reverse indices \(k\to k'\),

\[
\sigma_k=\eta\sqrt{
\frac{1-\bar\alpha_{k'}}{1-\bar\alpha_k}
\left(1-\frac{\bar\alpha_k}{\bar\alpha_{k'}}\right)},
\]

\[
x_{k'}=\sqrt{\bar\alpha_{k'}}\widehat x_0+
\sqrt{1-\bar\alpha_{k'}-\sigma_k^2}\widehat\epsilon+
\sigma_kz.
\]

The final virtual index has \(\bar\alpha_{-1}=1\). Frozen \(\eta=0\). Source: [evaluate_diffusion_v6_teacher_forced_validation.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/evaluate_diffusion_v6_teacher_forced_validation.py:631), `sample_batch`, lines 631–671.

**10. Exact current robot-aware score and candidate selection**

Define the relative metric change

\[
\delta_f(m)=
\frac{m(Q)-m(Q^P)}{\max(|m(Q^P)|,f)}.
\]

The score is exactly

\[
\boxed{
S=
4\delta(E_{\rm mean})
+2\delta(E_{95})
+\delta(E_{\max})
+0.5\delta(C_2)
+0.25\delta(C_3)
+\delta(B)
+0.5\delta(A_B)
+0.25\delta(P_{\rm sing}).
}
\]

| Term | Weight | Denominator floor | Metric units |
|---|---:|---:|---|
| Prefix mean position error | 4 | \(10^{-4}\) m | m |
| Prefix 95th-percentile position error | 2 | \(10^{-4}\) m | m |
| Prefix maximum position error | 1 | \(10^{-4}\) m | m |
| Prefix second-difference cost \(C_2\) | 0.5 | \(10^{-8}\) | rad² |
| Prefix third-difference cost \(C_3\) | 0.25 | \(10^{-8}\) | rad² |
| Maximum boundary component step \(B\) | 1 | \(10^{-4}\) rad | rad |
| Boundary second-difference norm \(A_B\) | 0.5 | \(10^{-4}\) | rad |
| Prefix singularity penalty | 0.25 | \(10^{-4}\) | See below |

Source: [generate_diffusion_v7_cost_improving_residual_targets.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_diffusion_v7_cost_improving_residual_targets.py:110), `ScoreWeights`, `MetricFloors`, lines 110–127; `relative_delta`, `delta_score`, lines 687–708.

There is **no velocity term, RMS Cartesian term, orientation term, or collision term** in this score.

Boundary definitions, for proposed execution \(q_0,\ldots,q_{E-1}\), history \(q_{-2},q_{-1}\), and prior tail \(q_E^P\):

\[
B=\max\left(
\|q_0-q_{-1}\|_\infty,\,
\|q_E^P-q_{E-1}\|_\infty
\right),
\]

\[
A_B=\max\left(
\|q_0-2q_{-1}+q_{-2}\|_2,\,
\|q_E^P-2q_{E-1}+q_{E-2}\|_2
\right),
\]

omitting unavailable history terms. Source: `boundary_metrics`, lines 544–580.

The positional Jacobian is finite-differenced using \(\epsilon=10^{-5}\) rad and joint-bound clipping:

\[
J_{:,j}=
\frac{f(q_j^+)-f(q_j^-)}{q_j^+-q_j^-}.
\]

It is \(3\times6\), in m/rad. Manipulability and penalty are

\[
m=\exp\!\left[
\operatorname{clip}\left(
\sum_{\ell=1}^{3}\log\max(\sigma_\ell(J),10^{-12}),
-60,60\right)\right],
\]

\[
P_{\rm sing}
=\operatorname{mean}_i
\min\left(\frac1{m_i+10^{-6}},10^{12}\right).
\]

Sources: `positional_jacobian`, lines 500–517; `manipulability_and_penalty`, lines 530–536.

Issues:

- This is translational manipulability, not full six-dimensional pose manipulability.
- Before numerical regularization, \(m\) has units \((\mathrm m/\mathrm{rad})^3\). The epsilon/floor constants implicitly assume that unit convention.
- The score is dimensionless after relative normalization, but remains sensitive to numerical floors and the baseline.
- Local negative scores do not mathematically guarantee a negative complete-path score.

Selection requires all gates below, \(S<0\), and the history-aware jerk guard. Among eligible candidates, v8.1 minimizes \(S\); scores within \(10^{-12}\) tie, then smaller history-jerk delta and lower candidate index win. No eligible candidate means anchored-prior fallback.

Sources: [evaluate_diffusion_v8_teacher_forced_all_windows.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/evaluate_diffusion_v8_teacher_forced_all_windows.py:1012), `sample_is_selectable`, lines 1012–1019; [evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py:140), `select_v8_1_candidate`, lines 140–170.

For complete-path reporting:

- The **legacy** score invents an exit back to the prior endpoint, because `tail_q=prior_q[-1]`.
- The **internal** score sets both boundary terms to zero:
  \[
  S_{\rm internal}=
  4\delta(E_{\rm mean})+2\delta(E_{95})+\delta(E_{\max})
  +0.5\delta(C_2)+0.25\delta(C_3)+0.25\delta(P_{\rm sing}).
  \]

Source: [evaluate_diffusion_v8_teacher_forced_all_windows.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/evaluate_diffusion_v8_teacher_forced_all_windows.py:2622), `compute_full_trajectory_metrics`, lines 2622–2708.

**Issue:** these two full-path scores are not interchangeable.

**11. Constraints actually checked**

Current candidate acceptance implements:

\[
\ell_j-10^{-7}\le q_{ij}\le u_j+10^{-7},
\]

\[
\max|\text{internal and boundary joint steps}|
\le0.20+10^{-7}\ {\rm rad},
\]

\[
E_{\rm mean,full}^{C}
\le\max(2E_{\rm mean,full}^{P},
E_{\rm mean,full}^{P}+0.01\ {\rm m}),
\]

\[
E_{\rm mean,prefix}^{P}-E_{\rm mean,prefix}^{C}
\ge\max(10^{-5}\ {\rm m},0.005E_{\rm mean,prefix}^{P}),
\]

\[
C_2^C\le1.1C_2^P+10^{-10},\qquad
C_3^C\le1.1C_3^P+10^{-10},
\]

\[
B^C\le B^P+0.01,\qquad
A_B^C\le A_B^P+0.01.
\]

It also checks finite values/boundaries and labels an internal step over \(0.20+10^{-7}\) rad as a branch jump.

Sources: [generate_diffusion_v7_cost_improving_residual_targets.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_diffusion_v7_cost_improving_residual_targets.py:711), `acceptance_reasons`, lines 711–749; [evaluate_diffusion_v7_teacher_forced_validation.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/evaluate_diffusion_v7_teacher_forced_validation.py:611), `target_acceptance_arguments`, lines 611–618.

Supplied URDF bounds, in radians:

\[
u=(3.1416,3.1416,4.9742,3.1416,3.1416,6.1082),
\qquad \ell=-u.
\]

The 0.01-rad joint safety margin is reporting-only, not a hard rejection criterion. Source: [generate_ik_seed_path.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_ik_seed_path.py:181), `check_joint_limits`, lines 181–289.

Fallback checks are narrower: executed-prefix joint limits, finite values, and internal/entry steps. Source: [evaluate_diffusion_v8_anchored_recursive_rollout.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/evaluate_diffusion_v8_anchored_recursive_rollout.py:348), `recursive_executed_prefix_hard_safety_reasons`, lines 348–374.

Deployment adds strict actual internal step \(\le0.20\) rad, timestamp checks, final orientation/Z gates, and independent FK/limit checks. Source: [generate_joint_trajectory_diffusion_v8_1.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_joint_trajectory_diffusion_v8_1.py:872), `final_safety_checks`, lines 872–951.

**Not checked by the current selector/deployment safety verdict:**

- Physical joint velocity limits.
- Physical acceleration or jerk limits.
- Torque, effort, inverse dynamics, or actuator bandwidth.
- Self-collision or environment collision.
- Continuous-time collision/limit satisfaction between samples.
- Brush contact force or contact dynamics.
- A hard minimum manipulability threshold.

The URDF declares velocity \(10\) and effort \(300\) for each revolute joint, but the reviewed gates consume its position bounds only. Robot loading uses `load_meshes=False`; no collision-query path is present in the current scoring chain.

Consequently, `full_path_safety_pass` means satisfaction of these implemented kinematic checks, not complete dynamic or collision feasibility.

An older benchmark does check

\[
\max_i\|D^2q_i\|_2\le0.25,\qquad
\max_i\|D^3q_i\|_2\le0.25,
\]

and \(\le1.1\) times the paired buffer values. These are post-evaluation decision criteria in discrete units, not SI acceleration/jerk limits. Sources: [evaluate_base_tail_final_benchmark.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/evaluate_base_tail_final_benchmark.py:3339>), decision logic, lines 3339–3366; [evaluate_global_anchored_receding_horizon_rollout.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/evaluate_global_anchored_receding_horizon_rollout.py:327>), `derivative_costs`, lines 327–369.

**12. Legacy scores and implementation/comment discrepancies**

The standalone score is

\[
J=J_{\rm path}+0.01C_2.
\]

Source: [score_trajectory.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/score_trajectory.py:134), `compute_path_error`, `compute_smoothness_cost`, `score_trajectory`, lines 134–191; default weights 208–209.

The “Stanford-style” default cost is

\[
J=J_{\rm path}+L_x+L_y+L_z+0.01\widetilde C_2
=2J_{\rm path}+0.01\widetilde C_2.
\]

Source: [trajectory_costs.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/trajectory_costs.py:75>), `compute_stanford_style_trajectory_cost`, lines 75–95.

**Issue:** default Cartesian error is counted twice.

The legacy robot/shape/drawing costs are

\[
J_R=E_{\rm mean}+0.25E_{\max}
+0.01\widetilde C_1+0.01\widetilde C_2
+0.001\widetilde C_3+10L_{\rm lim},
\]

\[
J_S=J_R+0.5E_{\rm start}+0.5E_{\rm end}
+F+0.5\,\mathrm{DTW},
\]

\[
J_D=J_S+0.5T+0.5P+0.25L+S_{\rm shape}.
\]

Here:

- \(L_{\rm lim}=\operatorname{mean}_{i,j}([\ell_j-q_{ij}]_++[q_{ij}-u_j]_+)^2\).
- \(F\): discrete Fréchet distance.
- DTW: minimum cumulative Euclidean alignment cost divided by the longer sample count.
- \(T\): desired-segment-length-weighted \(1-\) tangent cosine.
- \(P\): mean absolute difference of normalized cumulative arc length.
- \(L=|\log(\text{predicted length}/\text{desired length})|\).
- \(S_{\rm shape}\): mean Euclidean discrepancy after arc-length resampling, subtracting each path’s initial point, and dividing by desired path length.

Source: [evaluate_prior_refinement_fk_robot_costs.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/evaluate_prior_refinement_fk_robot_costs.py:348>), named metric functions, lines 348–507; `total_cost`, `shape_total_cost`, `drawing_total_cost`, lines 510–576; weights at 61–74.

Units mix metres, rad², and dimensionless quantities. Additionally, missing/nonfinite Cartesian metrics can be replaced by zero in these legacy totals. That can produce deceptively low scores when FK is unavailable.

The older base-tail selector uses a further distinct score:

\[
\begin{aligned}
J_{\rm BT}={}&J_{\rm prefix}+0.35J_{\rm tail}
+0.50J_{\rm ref,q}+J_{\rm ref,p}\\
&+0.75J_{\rm terminal}+0.50J_{\rm late}
+J_{\rm shift}+0.75J_{\rm extension}+10J_{\rm safety}.
\end{aligned}
\]

Its constituent terms are exactly:

\[
J_{\rm prefix}
=J_D+E_{\rm RMS}^2+0.01C_1+0.01C_2+0.001C_3
+\|\text{entry step}\|_2^2,
\]

\[
J_{\rm tail}
=J_D+E_{\rm RMS}^2+0.01C_1+0.01C_2+0.001C_3,
\]

\[
J_{\rm ref,q}=\operatorname{mean}_i\|q_i-q_i^{\rm global}\|^2,
\quad
J_{\rm ref,p}=\operatorname{mean}_i\|f(q_i)-f(q_i^{\rm global})\|^2,
\]

\[
J_{\rm terminal}
=\|p_{H-1}-p^\star_{H-1}\|^2
+0.25\|q_{H-1}-q_{H-1}^{\rm global}\|^2,
\]

\[
J_{\rm late}=J_D(\text{last eight samples})+E_{\rm RMS,late}^2,
\]

\[
J_{\rm shift}
=\|\text{candidate shift step}\|_2^2
+\|\text{retained-tail shift step}\|_2^2.
\]

\(J_{\rm extension}\) sums squared extension maximum step, velocity RMS, acceleration RMS, next-buffer mean Cartesian error, next-buffer joint/Cartesian reference RMS drifts, extension clipping count, and squared maximum extension clipping.

\(J_{\rm safety}\) sums raw/post-clip joint-limit counts and mean-square violations, candidate/retained-tail clipping counts, and squared maximum clipping magnitudes.

Sources: [evaluate_global_anchored_receding_horizon_rollout.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/evaluate_global_anchored_receding_horizon_rollout.py:810>), `practical_candidate_metrics`, lines 810–912; [evaluate_base_tail_final_benchmark.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/evaluate_base_tail_final_benchmark.py:76>), frozen weights, lines 76–98.

This score mixes units and includes derivative penalties both inside \(J_D\) and explicitly. Its fallback buffer is always eligible; the safety function uses relative non-worsening checks that can tolerate an already unsafe buffer. `safety_status`, lines 988–1096; `select_candidate`, lines 1122–1142.

The most consequential confirmed comment/implementation mismatch is the **legacy MLP export convention**:

- The checkpoint’s `y_mean` exactly matches the mean of training absolute joint positions.
- Canonical reconstruction is
  \[
  q=\sigma_q\odot y+\mu_q.
  \]
- Two legacy exporters instead compute
  \[
  q_{\rm exported}
  =q_{\rm start}+\sigma_{\Delta q}\odot y+\mu_{\Delta q}.
  \]

Sources: [generate_mlp_v3_train_test_predictions.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_mlp_v3_train_test_predictions.py:2), module description, lines 2–11, and `export_split_predictions`, lines 209–211; [generate_mlp_v3_test_predictions.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_mlp_v3_test_predictions.py:590), `main`, lines 590–595.

The current adaptive generator explicitly corrects this interpretation: [generate_adaptive_mlp_ik_bootstrap_prior.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_adaptive_mlp_ik_bootstrap_prior.py:323), `checkpoint_target_interpretation`, `canonical_mlp_full_q`, lines 323–400.

Other discrepancies/ambiguities:

- “Velocity/acceleration/jerk cost” frequently means discrete differences.
- “Robust” auxiliary scales normally use ordinary standard deviation, with MAD only as fallback.
- Generic v8 auxiliary-loss descriptions do not describe the deployed epsilon-only checkpoint.
- `score_trajectory.py:185` still says to replace FK later, although real FK is already called.
- The standalone scorer does not use timestamps in its cost, and does not verify equality of the desired and joint time grids.

**13. Train/test leakage and test information during generation**

The findings have different strengths and should not be conflated.

**Confirmed: older trainers use test data for model selection.**

- v4 treats filenames containing `test` as validation and saves the checkpoint minimizing that loss:
  \[
  e^\star=\arg\min_e L_{\rm test}(\theta_e).
  \]
  [train_conditional_diffusion_trajectory_v4_unet.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/train_conditional_diffusion_trajectory_v4_unet.py:124>), `find_split_files`, lines 124–132; `main`, lines 226–239, 286–296.
- v5 selects `best_checkpoint.pt` using `test_loss`. [train_conditional_diffusion_trajectory_v5_residual_unet.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/train_conditional_diffusion_trajectory_v5_residual_unet.py:777), `main`, lines 777–823.
- v5c selects using test candidate RMSE against expert joints. [train_residual_window_predictor_v5c.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/train_residual_window_predictor_v5c.py:570>), `main`, lines 570–612.

These files function as validation sets. Results on those same files are not untouched-test estimates.

**Confirmed: expert-derived initial states are available during generation.**

The stored train and test archives satisfy exactly

\[
q_{\rm start}=q^{\rm expert}_0
\]

for all 418 training and 83 test paths. The adaptive generator loads `q_start` and uses it in generation/retry seeds.

Sources: [generate_adaptive_mlp_ik_bootstrap_prior.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_adaptive_mlp_ik_bootstrap_prior.py:245), `load_generation_dataset`, lines 245–294; `retry_seeds`, lines 484–499. Binary evidence: `diffusion_v2/diffusion_train_v2.npz` and `diffusion_test_v2.npz`; binary arrays have no source line numbers.

This is legitimate if the problem explicitly provides the robot’s initial joint configuration. It is privileged information if the claimed task is Cartesian-path-only generation with no supplied initial state.

**Confirmed: noised-expert diagnostics use test trajectory information in generation.**

\[
x_{\rm init}
=\sqrt{\bar\alpha_k}\,x_0^{\rm test}
+\sqrt{1-\bar\alpha_k}\epsilon.
\]

Source: [diagnose_diffusion_v4_pure_sampling_vs_noised_expert.py](</mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/old scripts/diagnose_diffusion_v4_pure_sampling_vs_noised_expert.py:288>), `main`, lines 288–320.

These are reconstruction diagnostics, not independent trajectory-generation evaluations. The current v8.1 Gaussian sampler does not use this initialization.

**Potential leakage helper, currently unused:** `output_to_q` chooses an output interpretation by lower RMSE against `expert_q`:

\[
\arg\min_{a\in\{q_{\rm direct},q_{\rm start}+q_{\rm direct}\}}
\operatorname{RMSE}(a,q^{\rm expert}).
\]

Source: [generate_mlp_v3_test_predictions.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_mlp_v3_test_predictions.py:387), `output_to_q`, lines 387–426.

I found no repository call site for this helper. It is a leakage risk if reused, not evidence that current inference takes this branch.

**Current v8 diffusion train/validation arrays are path-disjoint.**

I verified:

- Training: 8,866 rows, 78 paths.
- Validation: 2,123 rows, 20 paths.
- Zero path-name overlap between these arrays.
- Training-only normalization is explicitly computed and checked.

Sources: [build_diffusion_v8_multitarget_scaled_training_dataset.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/build_diffusion_v8_multitarget_scaled_training_dataset.py:350), `main`, lines 350–378; [train_conditional_diffusion_trajectory_v8.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/train_conditional_diffusion_trajectory_v8.py:565), `validate_dataset`, lines 565–607.

**But that split is not an end-to-end unseen-path split.** Those v8 development paths originate from the original MLP training population. The saved MLP checkpoint names `train_episodes.npz`, containing 418 paths, as its training source. Thus “held out from diffusion training” does not mean “held out from the upstream learned prior.”

**Potential validation-preprocessing contamination:** v8 target diversity scales are computed across generated candidate populations before the later v8 train/validation split:

\[
S_j^{\rm diversity}
\leftarrow\text{all available valid base residuals},
\]

\[
d(r,r')=
\sqrt{\operatorname{mean}_{i,j}
((r_{ij}-r'_{ij})/S_j^{\rm diversity})^2}.
\]

Sources: [generate_diffusion_v8_multitarget_scaled_residual_targets.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/generate_diffusion_v8_multitarget_scaled_residual_targets.py:803), `robust_joint_residual_std`, `diversity_distance`, lines 803–831; `adaptive_diversity_scale`, lines 1139–1149; subsequent split in the dataset builder, lines 356–359.

This lets future validation-path statistics influence target retention. It is separate from—and does not contradict—the correctly training-only final normalization.

**Current path-disjoint confirmation has a cleaner inference boundary.**

`load_test_prior_records` supplies only desired paths, prior joints, prior FK, and metadata to the rollout. Expert trajectories do not enter its candidate score.

Source: [evaluate_diffusion_v8_1_path_disjoint_test_prior.py](/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring/evaluate_diffusion_v8_1_path_disjoint_test_prior.py:274), `load_test_prior_records`, lines 274–350.

I also found zero exact Cartesian-array duplicates between the 418-path train and 83-path test archives. This does not establish absence of near-duplicates.

Limitations:

- The formal population contains 30 paths with successful priors; it estimates performance conditional on prior success, not unconditional performance over all requested paths. `validate_manifest`, lines 136–155.
- Older experiments already use the repository’s test population for validation and diagnostics. New diffusion seeds do not make those physical paths an untouched test set.
- Using the requested desired Cartesian path to score candidates is legitimate task conditioning, not label leakage.
- File provenance and frozen manifests cannot prove that human hyperparameter choices were never informed by earlier test results.

| Paper concept | Code implementation | Correct mathematical expression | Potential issue |
|---|---|---|---|
| Cartesian tracking error | Same-index FK position discrepancy | \(e_i=\|f(q_i)-p_i^\star\|_2\), m | No nearest-path alignment or orientation |
| Mean tracking error | Mean Euclidean distance | \(\frac1N\sum_i e_i\), m | Not RMS |
| Cartesian RMS | Root mean squared Euclidean distance | \(\sqrt{\frac1N\sum_i e_i^2}\), m | Differs by \(\sqrt3\) from coordinate-pooled RMS |
| Maximum error | Maximum Euclidean position error | \(\max_i e_i\), m | Sampled maximum only |
| Path loss | Mean squared Euclidean error | \(\frac1N\sum_i e_i^2\), m² | “Stanford-style” defaults count it twice |
| Selection velocity/acceleration/jerk | `np.diff`, orders 1/2/3 | \(D^rq_i\), rad | No physical \(\Delta t\) |
| Derivative cost | Mean squared joint-vector norm | \(C_r=\operatorname{mean}_i\|D^rq_i\|^2\) | Legacy implementations divide by six |
| Physical jerk | Three timestamp-aware gradients | \(J=G_t^3Q\), rad/s³ | Different stencil from \(D^3Q/h^3\) |
| Default physical timestep | 100 samples over 10 s | \(h=10/99\) s | Selection ignores this timing |
| FK position | `yourdfpy`, link6 translation | \({}^{W}T_6(q)_{0:3,3}\) | Link origin, not a calibrated brush TCP |
| Orientation tracking | Final deployment geodesic gate | \(\theta_i=\acos((\operatorname{tr}(R_\star^\top R_i)-1)/2)\) | Absent from frozen candidate ranking |
| Residual target | Scaled correction to prior | \(r^{(s)}=s(q^{C}-q^P)\) | Distinct from \(q-q_{\rm start}\) |
| Residual normalization | Training-channel z-score | \(x_0=(r-\mu_r)/\sigma_r\) | Target multiplicity affects v8 statistics |
| Generated joints | Anchored prior plus scaled decoded residual | \(q^C=q^A+0.125(\sigma_r\widehat x_0+\mu_r)\) | Target scale and output multiplier differ |
| Diffusion forward process | Linear DDPM schedule | \(x_k=\sqrt{\bar\alpha_k}x_0+\sqrt{1-\bar\alpha_k}\epsilon\) | Diffusion time is not robot time |
| Deployed training loss | Epsilon-only checkpoint | \(L=\operatorname{mean}(\widehat\epsilon-\epsilon)^2\) | No active jerk/acceleration auxiliary loss |
| Robot-aware selection | Eight weighted relative metric changes | \(S=4\delta E_\mu+2\delta E_{95}+\delta E_{\max}+0.5\delta C_2+0.25\delta C_3+\delta B+0.5\delta A_B+0.25\delta P_{\rm sing}\) | Local baseline-relative surrogate |
| Jerk guard | Newly realized third-difference energy | \(\Delta C_{3,\rm hist}\le10^{-12}\) | Not an SI jerk bound |
| Internal full-path score | Boundary terms removed | \(S_{\rm internal}=S\) with \(B=A_B=0\) | Differs from legacy endpoint-return score |
| Joint feasibility | URDF position bounds plus step gates | \(\ell-10^{-7}\le q\le u+10^{-7}\), steps \(\le0.20\) rad | No physical velocity/torque/collision guarantee |
| Translational manipulability | SVD of \(3\times6\) positional Jacobian | \(m\approx\prod_{\ell=1}^3\sigma_\ell(J_p)\) | Not full-pose manipulability |
| Canonical MLP prediction | Absolute-joint output normalization | \(q=\sigma_qy+\mu_q\) | Legacy exporters incorrectly use delta normalization |
| Held-out diffusion validation | Disjoint v8 path sets | \(\mathcal P_{\rm train}\cap\mathcal P_{\rm val}=\varnothing\) | Upstream MLP already saw these paths |
| Independent test generation | Desired path plus supplied prior/start | \(\widehat Q=G(P^\star,Q^P,q_{\rm start},z)\) | Expert-derived start requires explicit task assumption |
| Test-set independence | Legacy checkpoint selection on test files | \(e^\star=\arg\min_eL_{\rm test}(\theta_e)\) | Test population used as validation |
| Noised-expert evaluation | Reverse diffusion from corrupted target | \(x_{\rm init}=\sqrt{\bar\alpha}x_0^{\rm expert}+\sqrt{1-\bar\alpha}\epsilon\) | Reconstruction diagnostic, not independent generation |
