# AGENTS.md

This repository implements temporal memory updating for efficient camera-only 3D detection.

The project should support both StreamPETR-style and RepDETR3D-style detectors. Do not hard-code the implementation to one detector unless the task explicitly asks for a detector-specific patch.

## Context policy

Do not read every markdown file by default. Start with this file and `docs/PROJECT_BRIEF.md`. Then read only the design note relevant to the requested change:

- Architecture or module design: `docs/ARCHITECTURE.md`
- 3D position embedding, ego-motion, or coordinate-frame issues: `docs/PE_AND_EGOMOTION.md`
- Training losses, teacher features, or freezing strategy: `docs/DISTILLATION.md`
- StreamPETR vs RepDETR3D compatibility: `docs/DETECTOR_ADAPTERS.md`
- Experiment configs and ablations: `docs/EXPERIMENTS.md`
- Repo integration and coding rules: `docs/IMPLEMENTATION_NOTES.md`
- Literature framing: `docs/RELATED_WORK_NOTES.md`
- Current implementation plan: `docs/IMPLEMENTATION.md`

When context is limited, prefer reading the most specific file rather than loading all docs.

## Non-negotiable goals

1. Preserve baseline behavior when temporal memory is disabled.
2. Implement the memory updater behind a detector-agnostic adapter interface.
3. Separate appearance memory from 3D positional encoding whenever possible.
4. Include a naive reuse baseline before implementing complicated update modules.
5. Include full-encoder teacher distillation for non-keyframe memory updates.
6. Measure actual wall-clock latency, GPU memory, mAP, and NDS. FLOPs alone are insufficient.

## Checkpoints

- RepDETR3D EVA-02 trained detector checkpoint: `ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth`.
  - Source run: `tools/train_repdetr3d_eva02_bs4.sh`.
  - Source artifact: `work_dirs/repdetr3d_eva02_800_bs4_seq_24e/iter_84384.pth`.
  - Validation from `work_dirs/stream_petr_eva.log`: mAP `0.5310`, NDS `0.6189`.
  - Use this as the detector initialization for temporal-memory project training.

## Definitions

- Full memory: final image-token/image-feature representation produced by the full backbone/neck path on a keyframe.
- Partial feature: intermediate ViT/block feature from the current frame, computed with less depth or cheaper processing.
- Updated memory: feature passed to the detector head/decoder on non-keyframes after combining previous memory with current partial features.
- Keyframe: frame where the full encoder path is run.
- Non-keyframe: frame where the full final memory is approximated by the updater.

## Expected implementation style

Prefer small, testable modules:

- `TemporalMemoryController`: decides keyframe vs non-keyframe and owns cached memory.
- `DetectorAdapter`: extracts/injects features for StreamPETR or RepDETR3D.
- `MemoryUpdater`: combines previous memory and current partial features.
- `PEHandler`: handles 3D PE recomputation or ego-motion-aware transformation.
- `DistillationLoss`: optional teacher loss between updated memory and full current memory.
- FramePolicy: decides per frame whether to use L24 full encoding or a partial layer such as L8, L12, or L16. Keep learned policies behind the controller instead of embedding policy logic directly in a detector.

Keep existing configs intact. Add new configs instead of mutating baseline configs.

For learned policy experiments, distinguish deployable online inputs from oracle/offline analysis inputs:

- Online metadata may use timestamp, frame index, ego pose/speed/yaw-rate, and other values available at inference.
- Offline metadata may additionally use annotation/object-count fields, but those rows are oracle ablations and should not be presented as deployable policy results.
- Cascade policies may probe a cheap partial feature such as L8 similarity before choosing L8, L12, or L24; include that probe cost in latency reporting.
- policy=dict(type="mlp", checkpoint=None, load_weights=False, feature_names=[...]) is allowed only for randomly initialized/train-from-scratch policy experiments. Evaluation policies should load an exported checkpoint unless the row is explicitly marked untrained.

## Minimum tests before training

- With memory disabled, outputs should match baseline within numerical tolerance.
- With `keyframe_interval=1`, outputs should match the full path.
- With naive reuse, no crash across a sequence.
- With memory update enabled, tensor shapes must match the detector input exactly.
- Ego-motion transformation code must have an identity-transform test.

## Naming

Use `tm_` or `temporal_memory_` prefixes for new modules, configs, and flags.
