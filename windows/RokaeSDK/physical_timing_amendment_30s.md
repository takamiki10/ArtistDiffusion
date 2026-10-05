> CURRENT PROTOCOL: [python_v050_nrt_experiment_plan.md](python_v050_nrt_experiment_plan.md) supersedes prior RT/30 s physical-execution assumptions. The experiment now uses exact waypoint sequences under a fixed Python v0.5.0 NRT policy, geometric metrics, and initially path_0003 with 5 R + 5 D trials. Earlier 10/30 s analyses remain numerical/historical references, not actual NRT timing. Windows SDK import is currently blocked; no hardware is authorized.

# Prospective physical timing amendment: 30 seconds

This amendment is made **before any physical R/D data collection**, as confirmed by the user. It supersedes the old 10-second physical convention. The numerical benchmark remains **100 samples over standardized 10 s**, unchanged. Selected physical paths remain 0001, 0003 and 0006, with A=R and B=D.

## Reference definition and scaling proof

Set physical duration **30.0 s**, physical timestamps **T_i=3*t_i**, and preserve all 100 original binary64 joint knots in every command. Use exactly the prior C2 quintic Hermite construction with scaled timestamps. No IK/diffusion regeneration, smoothing of knots, clipping, wrapping or path reselection occurs. The original package remains immutable. New derived physical-knot CSVs are explicitly labeled 30 s and live outside the package.

Let q_10(t) be the prior continuous quintic and define q_30(T)=q_10(T/3). For uniform scaling, h'=3h, slopes'=slopes/3, interior v'=v/3 and a'=a/9. Normalized quintic coefficients c0...c5 are unchanged because h'v'=hv and h'^2*a'=h^2*a. Consequently the continuous curve has identical position extrema, speed maxima divided by **3**, acceleration by **9**, and jerk by **27**. Locations of extrema occur at physical times three times their benchmark times. C2 continuity, zero endpoint v/a and the finite piecewise jerk convention remain unchanged.

This is an exact mathematical statement; recomputing from binary64 scaled timestamps produces small roundoff differences. [physical_reference_30s.py](physical_reference_30s.py) directly reconstructs both curves, finds numerical derivative roots, compares against the previous C++ extrema, verifies exact knot bits and exports the derived physical files. It does not merely divide the old table. At exact stored physical timestamps it returns the original knot directly. It holds q0 before 0 and q99 after 30.

## Offline results

[Continuous demand CSV](artifacts/physical_30s_v050/continuous_demands_30s.csv) contains all 36 command/joint rows: position envelope, max speed/acceleration/jerk, jerk in degrees, minimum margin to configured jerk in both units, and speed/acceleration margins. [Verification](artifacts/physical_30s_v050/verification.json) records:

- All 56 immutable package hashes verified; 3600 original joint-value bit comparisons exact.
- Largest physical-versus-time-scaled phase discrepancy: 1.9984014443252818e-15 rad.
- Max position-min/max scaling discrepancies: 0 and 2.220446049250313e-16 rad.
- Max speed/acceleration/jerk scaling discrepancies: 4.3298697960381105e-15, 4.3298697960381105e-15, 8.171241461241152e-14 in their respective SI units.
- Worst 30 s jerk: **44.67963285109179 rad/s^3**, path_0003 D J1.
- Smallest jerk margin: **42.58682974862469 rad/s^3**, versus configured 87.26646259971648 rad/s^3. All 36 rows have positive speed, acceleration and jerk margins to the supplied live values.

Numerical extrema are derivative-root searches, not formally certified interval bounds. The analytic scaling proof establishes the uniform transformation; it does not prove dynamic/torque/collision feasibility. Applicable RT enforcement remains undocumented, but the host references no longer deliberately exceed the live configured jerk number. Filtering/tracking should be measured under the unchanged candidate convention, not used to tune a more favorable R/D trajectory.

## Future command schedule and timing

The prior nominal 1 ms proposal is retained **conditionally** for a future verified v0.5.0 RT interface, not claimed as a confirmed Python v0.5.0 command API. At that rate use T_k=k/1000 for k=0...30000: 30001 values including endpoints. The physical knot interval is 30/99=10/33 s. Only knots 0,33,66,99 mathematically coincide with the nominal millisecond grid (0,10,20,30 s); binary64 evaluation still uses the exact stored knot times. Maximum mathematical nearest-grid displacement is 16/33 ms, approximately 0.484848 ms. Do not snap or retime knots.

Future t=0 remains first experimental q0 application after separately authorized positioning. At t=30 request q99 with zero reference v/a and hold thereafter. Controller application time, final-command completion ordering, filtering and host deadlines remain acquisition/control evidence to measure or establish once a usable version-specific interface exists. No command loop is enabled by this amendment.

## Physical analysis and paper interpretation

Compare both R and D physical measurements with **q_30(T)** and a 30-second window. Use verified timestamps/origin; do not optimize alignment separately to favor either condition. Joint/Cartesian reference errors use the same link6/model/frame convention and the 30 s target. Do not feed 30 s measurements into an unchanged 10 s target evaluator. Keep the benchmark package unchanged and implement/review a separate physical analysis adapter before collection; this offline evaluator provides the required reference.

Physical jerk has physical units and duration. Do not compare its raw values with Section V's standardized 10 s jerk as the same metric. If a normalized supplementary comparison is used, label the transform explicitly: qdot_10=3*qdot_30, qddot_10=9*qddot_30, jerk_10=27*jerk_30. The integral of squared jerk scales by 1/243 under a threefold time stretch; other jerk metrics require their own definition. Do not silently apply a factor of 27 to every aggregate jerk metric.

Prospective wording: "For physical execution, both reference and diffusion-refined trajectories will be uniformly time-scaled to 30 s while preserving their joint-space knots, providing conservative margins with respect to the configured controller motion parameters." After execution actually occurs, change "will be" to "were". The sentence reports configured-parameter margins, not certified safety or proven RT enforcement.

The path_0006 C2 maximum remains 3.1425140906387883 rad, within live displayed +/-360-degree joint-position safety bounds and above the Jetson 3.1416 bound. Time scaling does not resolve or alter that model/controller discrepancy. Full effective limits, load, clearance and later neutral Stage B remain separate. No paper-plane/pen-tip/contact calibration is introduced for the free-space link6 experiment.
