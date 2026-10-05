> CURRENT PROTOCOL: [python_v050_nrt_experiment_plan.md](python_v050_nrt_experiment_plan.md) supersedes prior RT/30 s physical-execution assumptions. The experiment now uses exact waypoint sequences under a fixed Python v0.5.0 NRT policy, geometric metrics, and initially path_0003 with 5 R + 5 D trials. Earlier 10/30 s analyses remain numerical/historical references, not actual NRT timing. Windows SDK import is currently blocked; no hardware is authorized.

> Current prospective physical convention: 30 s, unchanged knots, SDK v0.5.0 only. Start with [time_constrained_workaround_report.md](time_constrained_workaround_report.md). The older 10 s C++ audit below remains benchmark/offline history, not a physical execution plan.

# ROKAE experiment: offline preparation

Start with [experimental_interface_offline_report.md](experimental_interface_offline_report.md). The project contains no hardware backend or vendor-library link. Loading a CSV cannot cause motion. `--stationary-capture` is a blocked placeholder and has not been run; an enabled stationary logger still requires implementation and reviewed lifecycle evidence.

Implemented components: strict package-pinned binary64 CSV loader; PL audit and prospective C2 quintic reference; per-joint demand/precision audits; decoded binary logger, queue, events and CSV conversion; synthetic tests and an independent Python verifier. No production sender, controller project, RobotAssist setting or handoff-package file is changed.

## Reproduce the offline checks

Tested toolchain: WSL Ubuntu, g++ 15.2.0, C++17, AddressSanitizer and UndefinedBehaviorSanitizer. g++/CMake were installed into WSL with approval for this task. No ROKAE SDK was installed or linked. This demonstrates offline component behavior only; the host/toolchain is not a verified SDK deployment or a Windows 1 kHz timing qualification.

From this directory in WSL:

```bash
bash build_and_test.sh
```

The script builds with warnings and sanitizers, runs synthetic tests, validates all 56 package hashes, audits all six trajectories, records 32 labeled synthetic states and independently checks numerical/binary results with Python's standard library. Every audit uses a new directory under `artifacts`; `artifacts/latest_run.txt` identifies the latest successful complete run. The tests also leave small manifest-tamper fixtures under `build_offline`. It does not call stationary capture.

Individual offline commands, from this directory, require a new output directory:

```bash
build_offline/experiment_offline --audit ../laptop_handoff/laptop_handoff artifacts/my_new_audit
build_offline/experiment_offline --synthetic-capture artifacts/my_new_synthetic_capture
```

There is no execute/motion CLI. The stationary placeholder returns a blocked status without constructing a robot or attempting a connection. Do not use the vendor read-state examples for passive testing; inspected examples also command motion.

`CMakeLists.txt` supplies an alternative C++17 build configuration. That path and a Windows/MSVC build have not been exercised here. The tested build is the shell script above. Libraries provide `csv_header` and `csv_record`; a general hardware-file conversion CLI and metadata validation backend remain future work.

## Documents and artifacts

- [Architecture](experiment_interface_architecture.md): local SDK versions, ownership, coexistence and lifecycle findings.
- [Execution timing](execution_timing_design.md): exact candidate formula, 1 ms sampling, endpoint behavior and fidelity gates.
- [Logger format](logger_format.md) and [metadata template](session_metadata.template.json): raw field offsets, validity and timing provenance.
- [Current-limit verification](current_limit_verification.md): operator screenshot/export checklist, with no setting changes.
- [Characterization plan](measurability_characterization_plan.md): stationary Stage A, separately authorized neutral Stage B, prospective threshold freeze.
- `evidence/sdk_source_manifest.json`: exact inspected local archive/member hashes; `evidence/protected_files_verification.json`: original controller/configuration baseline verification.
- Latest successful run: `artifacts/run_20260913T073830/`, including [demand tables](artifacts/run_20260913T073830/audit/demand_audit_summary.md), [R/D demand differences](artifacts/run_20260913T073830/audit/rd_demand_differences.csv), [precision audit](artifacts/run_20260913T073830/audit/representation_audit.csv) and [independent verification](artifacts/run_20260913T073830/independent_verification.json).

All synthetic files are explicitly labeled `SYNTHETIC_ONLY`. They are not robot measurements and must not enter the physical experiment dataset. Earlier artifact directories are retained as development history; use the latest successful run for the report.
