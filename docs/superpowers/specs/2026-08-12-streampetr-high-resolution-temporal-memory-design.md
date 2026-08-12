# StreamPETR 640x1600 Temporal-Memory Design

## Goal

Establish a camera-only StreamPETR EVA/ViT-L detector at 640x1600 on
nuScenes, then use it as the full-encoder teacher for dense temporal-memory
experiments. The primary comparison is a high-resolution StreamPETR baseline;
SEPatch3D and ToC3D are external token-compression comparisons, not initial
dependencies.

## Scope and constraints

- Train on four A100 GPUs with 40 GB each for 24 epochs.
- Use only camera inputs. The dataset pipeline must set `use_lidar=False` and
  `use_radar=False` and must not include point-cloud loaders.
- Preserve the existing StreamPETR object-query temporal memory semantics.
- New files and flags use `tm_` or `temporal_memory_` prefixes. Existing
  baseline configurations remain unmodified.
- Measure synchronized end-to-end wall-clock latency and peak GPU memory, in
  addition to mAP and NDS.

## Model and initialization

Create a new StreamPETR EVA/ViT-L configuration derived from the project's
existing EVA-02 StreamPETR configuration. Set image augmentation final size to
640x1600 while retaining six camera views and the existing camera calibration,
timestamp, and ego-pose metadata flow.

Initialize the backbone from:

```text
ckpts/eva02_L_coco_det_sys_o365_remapped.pth
```

Do not initialize this teacher from the 800-resolution RepDETR3D detector
checkpoint. That checkpoint has resolution-specific detector/neck/head
adaptation and is useful only for RepDETR3D experiments. The EVA-02
pretraining checkpoint is the common transferable initialization.

The high-resolution configuration must explicitly support interpolation of
absolute positional embeddings when the ViT token grid differs from its
pretraining grid. A smoke test will verify the loaded tensor shapes and report
which parameters, if any, are skipped.

## Training recipe

Start with one sample per GPU. Select gradient accumulation after the first
forward/backward profiling run so the effective global batch is stable without
exceeding memory. Use the same optimizer family, learning-rate schedule, data
augmentation conventions, and 24-epoch schedule as the compatible StreamPETR
EVA baseline unless the smoke run proves that a batch-dependent adjustment is
necessary. Any learning-rate scaling must be written in the new config and
recorded in the run metadata.

Save only the required periodic/latest and best validation checkpoints; set a
small `max_keep_ckpts` value to prevent high-resolution experiments from
exhausting local storage.

## Staged execution

### Stage A: configuration and memory smoke test

1. Load the 640x1600 config and EVA checkpoint on one GPU.
2. Run one train forward/backward pass and record maximum allocated/reserved
   memory.
3. Run a 100--500 iteration distributed smoke training job.
4. Run a bounded validation subset and verify finite losses, predictions, and
   NuScenes evaluation formatting.

Promotion criteria: no positional-embedding load failure, no OOM, finite
losses, and a successful bounded validation pass.

### Stage B: high-resolution full-path teacher

Train the full StreamPETR detector for 24 epochs on four GPUs. Evaluate the
final/best checkpoint on the full validation set and record:

- mAP and NDS;
- synchronized total latency, backbone/neck latency, and head latency;
- peak GPU memory;
- configuration path, checkpoint, git commit, batch/accumulation settings,
  and image resolution.

This checkpoint is the full-memory source for later non-keyframe teacher
distillation.

### Stage C: temporal-memory transfer

Implement the StreamPETR `DetectorAdapter` only after Stage B is successful.
The adapter intercepts dense image appearance features before StreamPETR's 3D
position embedding and before its head consumes them. It does not replace or
modify StreamPETR's object-query history.

For every frame, compute timestamp-current 3D PE from current camera geometry
and ego pose. Cache/update appearance memory only:

```text
A_t = full image appearance features                 (keyframe)
A_hat_t = updater(A_(t-1), partial_feature_t, motion) (non-keyframe)
D_t = A_t or A_hat_t + PE_t
```

The first updater is a gated residual module conditioned on current partial
features and relative ego motion. On non-keyframe training frames, the Stage B
teacher full features are extracted under `no_grad()` and used for the existing
full-encoder feature-distillation loss. Teacher extraction is forbidden at
inference.

## Required ablations and gates

Run in this order:

1. `temporal_memory.enabled=False`: matches the new full-path teacher.
2. `enabled=True, keyframe_interval=1`: matches the full path within numerical
   tolerance.
3. Naive reuse at K=2, K=3, and K=5.
4. Partial-only current feature at the same partial depths.
5. Gated residual update with full-encoder distillation, then ego-motion
   conditioning.
6. Online policy experiments only after the fixed-K updater works.

Required tests before training the updater:

- baseline-disabled and K=1 equivalence;
- feature shape compatibility at the StreamPETR head boundary;
- current-frame 3D PE recomputation when reusing memory;
- identity relative-pose test;
- no teacher call during non-keyframe inference;
- short train/test sequence with no cache-state leakage across scenes.

## SEPatch3D and ToC3D relationship

SEPatch3D and ToC3D provide relevant 640x1600 StreamPETR baselines and
token-compression ideas. Their released full-detector checkpoints may be used
only with their matching upstream code/configuration unless parameter and
feature-interface compatibility is demonstrated. They are not assumed to be
drop-in checkpoints for this repository.

After Stage C is stable, token compression may be combined with temporal memory
in a separate ablation: (a) dense high-resolution teacher, (b) token
compression only, (c) temporal memory only, and (d) combined. This separation
keeps accuracy and latency attribution clear.

## Non-goals

- Do not delete or regenerate NuScenes metadata as part of this work.
- Do not use raw LiDAR, radar, or sweep files in the camera-only pipeline.
- Do not port SEPatch3D or ToC3D modules during the baseline-teacher phase.
- Do not alter RepDETR3D configs or temporal-memory behavior during this
  StreamPETR baseline phase.
