# ArtistDiffusion system handover

This repository brings together the Linux/Jetson trajectory pipeline and the Windows ROKAE component.

| Location | Start here |
| --- | --- |
| Linux / Jetson | [Pipeline README](linux/artist_trajectory_scoring/README.md) |
| Windows | [Windows collection and restoration status](windows/README.md) |
| System handover | [Component map](docs/handover/COMPONENT_MAP.md) |
| Data and models on Box | [Archive plan](docs/handover/BOX_ARCHIVE_PLAN.md) |

## Linux

Run pipeline commands from `linux/artist_trajectory_scoring/`. Supporting folders and their internal paths are unchanged. The dependency list is `linux/requirements.txt`. The root dev-container configuration opens `/workspace/linux`.

The Linux Box package is named **Linux**. Extract its pipeline-relative payloads into `linux/artist_trajectory_scoring/` as specified by its restore guide. A Box URL has not yet been recorded here. For artifacts requiring exact historical source hashes, use their recorded commit in a separate checkout; do not alter provenance hashes.

## Windows

`windows/RokaeSDK/` preserves 66 byte-verified source/documentation files collected from `C:/RokaeSDK`. This is a source snapshot: large experiment artifacts, SDK/runtime files and external inputs still need a separate Box package and restoration instructions. Original source files were preserved unchanged; historical absolute paths and older status notes remain to be reconciled.

## Data storage

GitHub holds code, tests, setup instructions and documentation. Box holds selected datasets, checkpoints, physical recordings and scans. Archive links, checksums, matching code commits and restore paths should be recorded in the handover index. The data directories are intentionally excluded from Git.

Git history preceding this organization remains available. The move changes the outer directory layout; it does not alter Linux algorithm source or the imported Windows source.
