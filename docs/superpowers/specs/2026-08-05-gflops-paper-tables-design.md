# GFLOPs Paper Tables Design

## Goal

Fill the detector-efficiency tables in `paper/main.tex` with defensible GFLOPs values while keeping local measurements and published related-work values clearly separated.

## Measurement definition

- Unit: end-to-end GFLOPs per nuScenes timestamp.
- Input: six camera images, each cropped to `320 x 800`.
- Batch size: 1 timestamp.
- Scope: backbone, neck, temporal controller and policy, memory updater, detector decoder, and prediction head.
- Convention: one multiply-add is two FLOPs.
- Dynamic methods: report the arithmetic mean over frames, including scene-start refreshes, forced refreshes, and policy actions.
- Precision and hardware do not change operation counts; latency and memory remain separate measured quantities.

The repository base config fixes `final_dim=(320, 800)`. Both dimensions are divisible by the pipeline's padding divisor, so padding does not enlarge the input.

## Evidence tiers

Every value will have one of four labels in the provenance report:

1. `local-measured`: produced in this repository with the profiling command, config, checkpoint, and dataset sample recorded.
2. `local-weighted`: computed from locally measured fixed-path costs and action counts from the same complete sequential validation run.
3. `published`: copied from a primary paper that uses the same six-view `320 x 800` input and the same two-FLOPs-per-MAC convention.
4. `not-reported`: the primary source lacks a comparable absolute value and no compatible local reproduction has completed.

Estimated values derived only from latency, keep ratios, or layer depth will not appear as measured GFLOPs. A paper cell remains `n/r` until it reaches evidence tier 1 or 2.

## Local profiler

Add one command-line utility, `tools/profile_gflops.py`, which builds the configured detector and validation pipeline using the repository's existing MMDetection3D interfaces. It will load the requested checkpoint, run batch size 1, and record:

- total MACs and GFLOPs;
- operator-level counts;
- unsupported operators;
- input tensor shape;
- config and checkpoint paths;
- requested and observed temporal action;
- scene-start status; and
- per-frame results in JSONL plus a summary JSON.

The implementation will reuse the installed `fvcore` JIT flop analysis rather than add a dependency. Unsupported reshape, indexing, scalar, and metadata operations may be explicitly whitelisted because they do not perform multiply-accumulate work. Every other unsupported operator fails the measurement by default and must receive a reviewed counting rule before its result can enter the paper. The summary will never silently treat an unsupported compute-heavy operator as zero.

## Fixed and dynamic execution paths

The first profiling pass covers the distinct paths required by the current paper:

- dense L24 without temporal updating;
- L24 with temporal memory enabled;
- naive reuse with no backbone execution;
- partial L8 with and without memory updating;
- partial L12 with memory updating;
- partial L16 with memory updating; and
- each updater variant listed in the updater ablation.

Each forced path is checked against the action reported by the temporal-memory controller. Variable schedules and learned policies are profiled in their natural sequential validation order. Their headline GFLOPs are the direct mean across the profiled frames. If direct per-frame tracing is prohibitively slow, the accepted fallback is a weighted mean of locally measured path costs using action frequencies from the same complete validation run; the provenance report must identify this as `local-weighted` and include the counts and formula.

## Related work

Use primary sources only. The initial audit establishes:

- ToC3D publishes 4,601 GFLOPs for dense StreamPETR ViT-L and 3,131 GFLOPs for ToC3D-Faster at six-view `320 x 800`, using exactly two FLOPs per MAC.
- ToC3D does not publish an absolute ToC3D-Fast value.
- SToRe3D does not publish absolute ViT-L GFLOPs for the 1/3 and 1/10 rows. Its isolated 397-GFLOP reduction claim is for the ViT-B 1/2 setting and cannot be transplanted.

Missing related-work values will remain `n/r` in the first paper revision. A later targeted reproduction may use the authors' official code and checkpoint, but its result must be reported as a local reproduction rather than a published number.

## Paper changes

After measurements pass validation:

- fill GFLOPs in the main comparison and depth/policy ablation tables;
- add a GFLOPs column to updater and detector-efficiency policy tables where compute is part of the comparison;
- do not add GFLOPs to detection-error, regression, or signal-predictability tables;
- define the measurement convention in each relevant caption or a shared experimental-protocol paragraph;
- replace claims that FLOPs are unreported once the corresponding values exist; and
- preserve unrelated uncommitted edits already present in `paper/main.tex`.

The related-work block must not imply that published and locally measured latency were obtained on the same hardware. GFLOPs may be compared only when the input and counting convention match.

## Provenance report

Create `paper/gflops_provenance.md` with one row per paper entry. Required fields are method label, table, GFLOPs, evidence tier, resolution, camera count, counting convention, config, checkpoint or citation, aggregation rule, unsupported operators, and notes.

The local profiler's JSON outputs remain under `outputs/gflops/` and are not committed unless they are small summaries needed to reproduce the table.

## Validation

Automated tests will cover:

- two-FLOPs-per-MAC conversion;
- arithmetic and action-frequency weighted aggregation;
- rejection of empty measurements;
- rejection of material unsupported operators;
- preservation of per-frame provenance; and
- batch/resolution validation for `[1, 6, 3, 320, 800]`.

The profiler will be tested first with synthetic modules, then smoke-tested on one real validation timestamp. Before paper values are accepted, L24 must be numerically stable across repeated traces and operator coverage must be reviewed. The dense local number is compared with ToC3D's 4,601-GFLOP StreamPETR reference as a scale sanity check, not an equality requirement, because this repository uses RepDETR3D and a different detector integration.

## Non-goals

- No retraining is required to count FLOPs.
- No GFLOPs are inferred from latency.
- No cross-method checkpoint conversion is included in the first pass.
- No unrelated manuscript cleanup is included.
