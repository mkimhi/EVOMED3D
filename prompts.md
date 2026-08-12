# Codex prompts

### Use these prompts one at a time. Do not paste all prompts at once.

## Prompt 1: Repository scan

Read `AGENTS.md` and `docs/PROJECT_BRIEF.md`. Then inspect the repository to find the RepDETR3D EVA-02 config, the image backbone forward path, the detector head input, and where 3D position embeddings are computed or added. Do not modify code yet. Return a concise implementation map with file paths, classes/functions, feature tensor shapes if visible, and whether the detector feature interface is pre-PE or post-PE.

## Prompt 2: Minimal temporal-memory config plumbing

Read `AGENTS.md`, `docs/PROJECT_BRIEF.md`, `docs/DETECTOR_ADAPTERS.md`, and `docs/IMPLEMENTATION_NOTES.md`. Add config plumbing for `temporal_memory.enabled=false` that preserves baseline behavior exactly. Do not implement the updater yet. Add safe defaults and shape/debug logging hooks behind flags. Keep existing configs unchanged; create a new RepDETR3D EVA-02 temporal-memory config derived from the baseline.

## Prompt 3: Detector adapter skeleton

Read `AGENTS.md` and `docs/DETECTOR_ADAPTERS.md`. Implement a detector-agnostic adapter skeleton and a RepDETR3D adapter that can extract full features and inject features back into the detector head without changing behavior when temporal memory is disabled. Add identity tests or debug assertions where possible.

## Prompt 4: PE location and ego-motion note

Read `AGENTS.md` and `docs/PE_AND_EGOMOTION.md`. Locate the 3D PE code path. Refactor minimally so memory can be cached before PE if feasible. If not feasible, add explicit metadata marking memory as post-PE and create a documented fallback mode. Add identity ego-motion tests for the relative transform utility.

## Prompt 5: Naive reuse baseline

Read `AGENTS.md`, `docs/ARCHITECTURE.md`, and `docs/EXPERIMENTS.md`. Implement `updater="reuse"` with fixed `keyframe_interval`. Ensure `K=1` matches the full path. Reset memory on scene boundaries and discontinuities. Add debug logs for keyframe/non-keyframe decisions.

## Prompt 6: Intermediate EVA feature extraction

Read `AGENTS.md`, `docs/ARCHITECTURE.md`, and `docs/IMPLEMENTATION_NOTES.md`. Implement extraction of one requested intermediate EVA-02 layer/block feature for RepDETR3D. Avoid storing all layer outputs. Add config options for 25%, 50%, and 75% depth if possible. Return the exact tensor shapes.

## Prompt 7: Gated residual updater  - WE ARE HERE!!!

Read `AGENTS.md`, `docs/ARCHITECTURE.md`, and `docs/DISTILLATION.md`. Implement a lightweight gated residual memory updater. It should accept previous memory, projected partial current features, and optional ego-motion embedding. Initialize the residual scale conservatively so early training behaves close to reuse.

## Prompt 8: Feature distillation

Read `AGENTS.md` and `docs/DISTILLATION.md`. Add optional full-feature teacher distillation during training on non-keyframes. The teacher path must be stop-gradient. Add an inference-time assertion that the full teacher is not called on non-keyframes when `tm_inference_teacher_forbidden=true`.

## Prompt 9: Profiling -here

Read `AGENTS.md`, `docs/EXPERIMENTS.md`, and `docs/IMPLEMENTATION_NOTES.md`. Add profiling utilities to report backbone/full, backbone/partial, updater, head, total latency, and GPU memory. Use CUDA synchronization. Keep profiling optional and disabled by default.

## Prompt 10: Experiment configs

Read `AGENTS.md` and `docs/EXPERIMENTS.md`. Create configs for: baseline, naive reuse K=3, naive reuse K=5, gated residual K=3 with distillation, gated residual K=5 with distillation, and gated residual with ego conditioning. Do not delete or mutate original configs.

## Prompt 11: StreamPETR adapter stub - we can run baseline exp before that!!!!!

Read `AGENTS.md`, `docs/DETECTOR_ADAPTERS.md`, and `docs/PE_AND_EGOMOTION.md`. Add a StreamPETR adapter stub using the same interface as RepDETR3D. Do not complete the implementation unless the feature and PE locations are clear. Include TODOs with exact file paths and functions to modify.