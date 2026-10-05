> CURRENT PROTOCOL: [python_v050_nrt_experiment_plan.md](python_v050_nrt_experiment_plan.md) supersedes prior RT/30 s physical-execution assumptions. The experiment now uses exact waypoint sequences under a fixed Python v0.5.0 NRT policy, geometric metrics, and initially path_0003 with 5 R + 5 D trials. Earlier 10/30 s analyses remain numerical/historical references, not actual NRT timing. Windows SDK import is currently blocked; no hardware is authorized.

> SUPERSEDED FOR PHYSICAL EXECUTION: use [physical_timing_amendment_30s.md](physical_timing_amendment_30s.md). Physical duration is 30 s, SDK choice v0.5.0. The original formula/rationale below is retained; its 10 s command schedule and v0.5.1 API evidence are historical, not the selected physical backend.

# Execution timing design — prospective, offline only

## Decision recorded before interpolation implementation

Use a nominal **1 ms / 1000 Hz** future `RtControllerMode::jointPosition` stream from one C++ owner. This is the cycle documented by the local v0.5.1.rt_0 `setControlLoop` API, not inferred from the subscription rates. No hardware backend is implemented or enabled in this project.

**Reject piecewise-linear (PL) interpolation as the physical execution convention.** It is retained only as a labeled offline reference. PL passes through the knots but has discontinuous velocity at interior knots and at a stationary start/end. The local C++ manual §5.3.1 (PDF pp.29–30) requires a smooth joint-space trajectory, at least continuously differentiable velocity, and recommends zero starting/ending velocity and acceleration. Finite differences of PL do not remove its acceleration impulses. Letting an undocumented SDK/controller filter smooth PL would change the trajectory and its knot values.

The prospective host-reference candidate is **C2 piecewise quintic Hermite interpolation**, applied identically to all R/D files, fixed here for mathematical regularity and stationary endpoints, before examining condition-specific performance. It is an application-generated sequence of documented `JointPosition` commands, not a claimed built-in ROKAE quintic trajectory API. It preserves every original timestamp and knot. This candidate is not certified physically executable: audit acceleration/jerk, continuous bounds, filters and current settings first. No alternative is selected according to which favors D.

## Definition

Let h_i=t_(i+1)−t_i and s_i=(q_(i+1)−q_i)/h_i, componentwise. At an interior knot i:

- v_i=(h_i*s_(i−1)+h_(i−1)*s_i)/(h_(i−1)+h_i).
- a_i=2*(s_i−s_(i−1))/(h_(i−1)+h_i).
- v_0=a_0=v_99=a_99=0, so holding q constant outside [0,10] has matching position, velocity and acceleration.

On interval i set u=(t−t_i)/h_i and q(u)=sum(c_k*u^k,k=0..5). Define c0=q_i, c1=h_i*v_i, c2=h_i²*a_i/2, A=q_(i+1)−c0−c1−c2, B=h_i*v_(i+1)−c1−2*c2, C=h_i²*a_(i+1)−2*c2, then c3=10A−4B+C/2, c4=−15A+7B−C, c5=6A−3B+C/2.

At exact stored timestamps the evaluator returns the original stored binary64 knot directly. Endpoint derivatives are imposed analytically. No knot smoothing, rounding, clipping or wrapping occurs. Interior derivatives are deterministic interpolation choices, not measured velocities. This curve can overshoot the knot envelope; jerk is piecewise finite and may jump at knots. Those properties are audited, not concealed.

## Sampling and time origin

Evaluate the reference at tau_k=k/1000 seconds for k=0..10000: **10001 reference updates including both endpoints**, 10000 intervals over 10 seconds. Do not accumulate `time += 0.001`. The knot interval is 10/99 seconds, not 0.1 or 0.101 seconds. Only endpoints coincide exactly with the nominal millisecond grid; nearest-grid displacement for interior knots is at most 0.494949… ms. Do not snap or retime knots. The continuous host function satisfies q(t_i)=q_i; a fixed-rate stream does not transmit most knots at their exact times. Neither sampled values nor that mathematical identity proves exact physical passage through knots.

Future execution t=0 is the controller acceptance/application of the first experimental q0 update after separately authorized positioning has finished and readiness checks pass. Host callback entry/exit and sequence k are observable proxies, not proof of controller application time. The local callback has no controller timestamp/duration argument. Preserve the host start event and its clock/latency uncertainty; do not fit a favorable shift using R/D errors. The implementation must retain nominal schedule time separately from actual host times and never silently catch up or stretch timing after a deadline miss.

At nominal t=10 update 10000 requests exactly q99. The mathematical reference holds q99 before/after the interval and has zero endpoint v/a. Whether a final command flagged `setFinished()` is applied before the SDK stop takes effect is not established by the inspected declarations; this ordering must be resolved before motion. The documented `setFinished()`/`stopMove()` stops RT motion, and `stopMove()` is not an RL stop. No autonomous approach, retreat, mode/power command, endpoint stop routine, or motion CLI exists here.

## Fidelity limits and gates

The local `setControlLoop` documentation says the SDK filters returned commands. `setFilterLimit(bool,double)` exposes clipping/filter parameters (0–1000 Hz); `setFilterFrequency(joint,cart,torque,ec)` configures controller filters (1–1000 Hz). No verified identity/bypass setting or default transfer function is established. Do not assume 0 or 1000 Hz disables filtering. Both filter stages and valid `q_c` semantics must be resolved before claiming faithful execution. No filter setter is called by this offline project.

Audit separately: host-reference extrema, 1 ms samples/steps, SDK-returned-versus-transmitted reference if observable, controller `q_c`, measured `q_m`, host deadlines, queue lag, errors, and any available acquisition/application timestamps. Windows callback timeliness at 1 kHz is unproven. SDK `setRealtimeOptimization` applies only on Linux. A supported nominal interval is not a demonstrated real-time guarantee.

Reported PL interval slopes and finite-difference accelerations are descriptive quantities. Candidate quintic continuous extrema use derivative-root search, with numerical rather than formal interval certification. Full robot dynamics, torque, collision, tracking reserve, filter effects and operating-mode limits remain separate requirements. The existing Jetson analysis only supports linear/previous interpolation; a prospective analysis extension matching this quintic reference is needed before R/D collection. The package is left untouched.
