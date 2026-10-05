# Windows ROKAE component - source collected

Confirmed source: `C:/RokaeSDK`. `RokaeSDK/` contains 66 byte-verified source/documentation files (762,344 bytes), collected on 5 October 2026. The original installation was not changed. No hardware was connected and nothing was uploaded.

`provenance/windows_source_inventory.json` inventories all 1,371 files (893,999,641 bytes), with relative paths, sizes, SHA-256, and dispositions. Source copy includes root scripts/docs, C++ source/headers/tests, NRT preparation/audit source, and physical trace-analysis source. Larger files marked `archive_selection_pending` remain at the original location; this is an undecided retention status, not an exclusion recommendation. The copy is not yet a complete runnable package.

Historical records in `C:/RokaeSDK/evidence/protected_files_verification.json` also refer to:

`C:/Rokae/RobotAssist_5.0.11.0313/workspace/M82522101354/Artist_movement_finish/`

The user confirmed `C:/RokaeSDK` as the Windows component for this handover. Historical RobotAssist paths above remain contextual evidence. Original README/protocol notes describe earlier import blockers; later physical session logs and drawing scans exist, so the latest operating status must be reconciled with evidence.

## Further collection

Retain essential physical sessions, original drawing scans, prepared inputs, matching manifests, SDK/runtime dependencies and provenance. Some scripts reference `C:/laptop_handoff/laptop_handoff` or machine-specific paths. Collect and verify these dependencies before running a restored example. Separate synthetic output from physical evidence. Do not infer that a source copy alone reproduces the experiment.

## Suggested additional collection layout

- `robotassist_project/`: complete original project tree, preserving relative paths, tasks, frame definitions, variables, and configuration needed to reopen it.
- `host_tools/`: any custom Windows sender, converter, or launcher actually used with that project.
- `environment/`: RobotAssist/SDK/firmware versions, prerequisites, configuration notes, and installation instructions.
- `provenance/`: source paths, collection date, file sizes, SHA-256 hashes, and an inventory of omitted files.

Inspect source instructions before importing. Inventory and hash the source, copy it without modifying the working installation, then verify the copy. Preserve the initial snapshot before making documentation or portability changes. If this component has Git history, preserve it separately and document how it maps to the imported snapshot.

Do not copy the entire RobotAssist application installation into Git by default. Determine which project files and runtime dependencies are actually required. Preserve necessary installers or vendor bundles in the separate handover archive with version and retrieval information.

Record how the controller consumes `SmartJoint_Data_diffusion.csv`: actual filename/location, columns, angle units, timing convention, stroke markers, and handling of pen-up transitions. Compatibility remains unverified until the controller code is inspected.
