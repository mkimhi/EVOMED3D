# Related work notes

## Core positioning

The project is related to temporal modeling and efficient token processing, but the intended distinction is:

> Instead of recomputing, pruning, or compressing only current-frame tokens, update dense multi-view image-token memory across time using partial current-frame ViT computation.

## Closest 3D detection works

### StreamPETR

StreamPETR is the conceptual baseline. It propagates historical information through object queries in an online camera-only 3D detector. The proposed project should not claim that temporal memory is new; it should claim that dense image-feature memory updating is different from object-query temporal propagation.

### ToC3D / token compression

Token compression methods reduce the number of current-frame tokens processed by a ViT-based detector. They do not primarily update dense final encoder memory from previous-frame memory plus partial current-frame features.

### Adaptive tokenization / SEPatch3D-style methods

Adaptive tokenization changes the image patch/token granularity and allocates computation to informative regions. This is complementary to temporal memory updating.

### SToRe3D

SToRe3D is close because it connects 2D token sparsity and 3D query relevance and stores/reuses filtered features. Be careful when contrasting against it. The distinction should be that this project learns to reconstruct/update the detector-consumed feature memory from partial current-frame computation, with explicit keyframe/non-keyframe operation.

### BEV temporal methods

BEVDet4D, BEVFormer-style temporal fusion, RecurrentBEV/VideoBEV, and similar methods propagate BEV-space state. This project operates at the image-token/feature interface before sparse 3D decoding.

### PETR / DETR3D / CAPE

PETR-style methods use 3D position embeddings and camera geometry. This is important for the PE/ego-motion part of the project. CAPE is especially relevant because it discusses camera-view position embedding and temporal modeling with ego-motion.

## Older video efficiency works

Deep Feature Flow and related video recognition/detection works use keyframes and propagate features to nearby frames. This is important prior art. The project should acknowledge this family and explain why multi-view camera-only 3D detection with PETR-style geometry and ViT memory is a different setting.

## Suggested framing sentence

Use language like:

```text
Our method revisits keyframe-based feature reuse in the setting of modern sparse-query camera-only 3D detection. Unlike prior 3D token-pruning methods that reduce current-frame tokens, and unlike StreamPETR-style object-query memory, we maintain a dense multi-view image-feature memory and learn to update it from partial current-frame ViT features while recomputing or conditioning on the current 3D geometry.
```

## Claims to avoid

Avoid claiming:

- temporal memory is new
- keyframe feature reuse is new
- ego-motion-aware temporal fusion is new
- token pruning/compression is the same as this method

## Claims that may be defensible

Potential defensible claims after experiments:

1. Dense encoder memory has temporal redundancy in camera-only 3D detection.
2. Partial ViT features are sufficient to update final detector features between keyframes.
3. Distillation from full encoder features stabilizes temporal memory updates.
4. Separating appearance memory from current-frame 3D PE improves non-keyframe performance.
5. The method is complementary to token compression/adaptive tokenization.
