#!/bin/bash
# Offline-only build/test/audit. No vendor SDK is linked and no network is used.
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p build_offline artifacts
g++ --version > build_offline/compiler_version.txt
flags=(-std=c++17 -O1 -g -Wall -Wextra -Wpedantic -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer -pthread -Iinclude)
g++ "${flags[@]}" -c src/core.cpp -o build_offline/core.o
g++ "${flags[@]}" -c src/logger.cpp -o build_offline/logger.o
g++ "${flags[@]}" build_offline/core.o build_offline/logger.o tests/tests.cpp -o build_offline/offline_tests
g++ "${flags[@]}" build_offline/core.o build_offline/logger.o src/main.cpp -o build_offline/experiment_offline
build_offline/offline_tests | tee build_offline/test_results.txt
run_id="run_$(date -u +%Y%m%dT%H%M%S)"
mkdir "artifacts/$run_id"
build_offline/experiment_offline --audit ../laptop_handoff/laptop_handoff "artifacts/$run_id/audit"
build_offline/experiment_offline --synthetic-capture "artifacts/$run_id/synthetic"
python3 verify_and_summarize.py "artifacts/$run_id"
printf '%s\n' "$run_id" > artifacts/latest_run.txt
# The blocked stationary-capture mode is deliberately not invoked in this task.
