# Offline logger format and implementation boundary

This is a tested decoded-record logger and synthetic source, not an SDK acquisition backend. No constructor, socket, subscription or robot command is called. The binary format preserves the values supplied to it; it cannot establish the precision or freshness of a future SDK source.

## Source contract

Required measured joints map to SDK `q_m`; desired fields are `dq_m`, row-major 4×4 `pos_m`, and `q_c` only when valid in the selected mode. All joint values are radians and velocities rad/s. Pose translation is meters under the documented SDK convention; exact flange/TCP identity, base frame and active tool must be verified and recorded. Separately queried operation/power/operating-mode state has its own read bracket and must not be presented as atomically sampled with joints.

Every acquired record carries logger sequence, host monotonic read-start/end, bytes returned by the state update, overall SDK error, five field return codes, reported and validated field masks, native q/dq/pose/target values and optional separately timed state. Invalid doubles initially contain NaN, not substitute zeros. Masks are q=1, dq=2, pose=4, target=8, state=16. Unavailable optional fields are allowed; required q absence, inconsistent claimed validity, nonfinite values or an invalid rigid transform invalidate analysis. Transform checks use 1e-8 for the homogeneous last row and 1e-6 for rotation orthonormality/determinant.

No controller sample timestamp or source frame counter was found in the inspected public state API. The optional source-sequence slot is reserved for a genuinely exposed future source, and is absent in the SDK plan. Logger sequence counts acquired records, not controller frames. Host gaps cannot prove how many controller frames were lost. Alarm-log timestamps are not state timestamps.

## Binary v1

Each frame is **445 bytes**: ASCII `XCRAW001` (8), little-endian payload length 369 (4), payload (369), then lowercase ASCII SHA-256 of that payload (64). This is `sdk_decoded_raw`, not native TCP/UDP packet capture. No C++ struct padding is serialized. Integers are explicit little-endian widths; doubles preserve IEEE-754 binary64 bits, including negative zero and invalid native values.

| Payload offset | Field | Encoding |
|---:|---|---|
| 0 | logger sequence | uint64 |
| 8, 16 | read start, read end | uint64 nanoseconds |
| 24, 32 | separate state read start, end | uint64 nanoseconds |
| 40, 44, 48 | update bytes, reported mask, validated mask | uint32 each |
| 52 | overall SDK error | int32 |
| 56 | q, dq, pose, target, state return codes | 5 × int32 |
| 76 | measured q | 6 × binary64 |
| 124 | measured dq | 6 × binary64 |
| 172 | row-major measured end pose | 16 × binary64 |
| 300 | controller target q | 6 × binary64 |
| 348 | operation status, power state, operating mode | 3 × int32 |
| 360 | source-sequence-present | uint8, 0 or 1 |
| 361 | optional source sequence | uint64; ignored if absent |

The incremental parser accepts split/coalesced frames. Unknown magic/length, checksum corruption, inconsistent validity and truncated tails fail closed. Preserve the original file when conversion fails; do not discard bytes, resynchronize silently or repair it in place. Hashes detect accidental corruption, not authenticity.

## Events, queue and timing

`Logger::ingest` has one acquisition caller. `pump`/`flush` have one disk-writer caller. Owned records pass through a bounded mutex/deque queue; the synthetic executable exercises a separate writer thread. Overflow reports the lost logger sequence, returns failure and invalidates the session. Writer/flush failures latch failure and stop further ingestion; emergency events remain in memory if the event sink also fails. This does not guarantee persistence after process/power loss.

Events are JSON Lines with code, related logger sequence, `host_event_ns` from `std::chrono::steady_clock`, read-start/end when available, and detail. Zero event read-bracket fields mean unavailable. The clock epoch is unspecified; record its implementation and origin in metadata. Synthetic sample times are deliberately generated numbers and are not synchronized to real event times.

A timeout produces an event with its read bracket and no stale sample. Timeout alone is diagnostic and does not latch the library's invalid flag; future protocol acceptance must examine timeout/gap budgets as well as this flag. Duplicate/backward host ends, malformed required fields, claimed invalid optional data, actual source-counter discontinuities and queue/writer failures do latch invalidity. Every retained acquired sample stays in raw storage, including invalid/duplicate records. The future backend must distinguish a timed-out wait from an SDK error and use the actual field return codes.

The present mutex, dynamic deque/event allocations, SHA work, event copies and stream I/O have **not** been qualified for 1 kHz deadlines. Before an RT backend, use a reviewed preallocated acquisition queue and bounded emergency diagnostics, and prove synchronization/latency under load. No RT safety or bounded execution-time claim follows from the offline thread test.

## CSV conversion and metadata

`csv_header`/`csv_record` implement diagnostic conversion; the synthetic executable demonstrates raw-file parse → CSV. There is no arbitrary hardware-file conversion CLI yet. Conversion requires a verified joint permutation/sign/offset supplied through `Conversion`. For controller joint j, `q_jetson[map[j]] = sign[j] * (q_controller[j] - offset[j])`; velocity omits offsets. This is not automatic wrapping. The synthetic demo uses only a labeled synthetic identity mapping.

CSV includes `trial_id,timestamp,q1_measured…q6_measured`, host times, sequence, both masks, overall/field SDK codes, optional dq, target, pose and state. `timestamp` is host read-end minus a declared origin, in seconds, and is not controller acquisition/application time. `controller_timestamp` stays empty. Unavailable/invalid fields stay empty; invalid and duplicate rows remain visible. Values use 17 significant digits. Raw values and metadata remain the authority; CSV alone does not establish a valid experimental trial or alignment.

Use [session_metadata.template.json](session_metadata.template.json). Null values are unresolved gates, not defaults. A real backend must populate and validate it, record final file hashes/counters, link immutable commands/settings and fail closed when required provenance is missing. The synthetic run writes a smaller metadata file explicitly labeled synthetic. No hardware metadata was fabricated.

Tests cover valid values, exact bit preservation, every two-frame split point, coalescing, corrupt framing/checksums, truncated tails, timeouts, unavailable fields, NaN, malformed pose, queue overflow, duplicate host time, synthetic source-counter gaps, raw/event writer failures, mapping rejection and threaded queue order. Dropped-frame testing exercises a source that exposes a counter; it does not claim the ROKAE API does.
