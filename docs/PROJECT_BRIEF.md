# Project brief: Temporal Memory Updating for Efficient Camera-only 3D Detection

## Core idea

Camera-only 3D detectors with ViT/EVA-style image backbones repeatedly recompute expensive image features for every timestamp. Existing efficient 3D detection work mostly compresses or prunes current-frame tokens, or propagates sparse object/BEV state. This project instead maintains a dense image-feature memory and updates it between keyframes using partial current-frame computation.

For timestamp `t`:

```text
if keyframe:
    M_t = FullEncoder(I_t)
else:
    Z_t_l = PartialEncoder(I_t, layer=l)
    M_t = MemoryUpdater(M_{t-1}, Z_t_l, ego_motion, optional_metadata)
```

The updated memory `M_t` is then passed to the normal detector neck/head/decoder.

## Intended claim

The intended claim is not simply temporal modeling. The intended claim is:

> Dense multi-view image-token memory can be temporally updated from partial ViT computation, reducing full image-encoder cost while preserving most of the camera-only 3D detection accuracy.

## Baselines

Always compare against:

1. Full detector at every frame.
2. Full detector every `K` frames with naive memory reuse between keyframes.
3. Partial feature only, without previous memory.
4. Proposed memory update.
5. Proposed memory update with feature distillation.

## Supported detectors

This project should work first with RepDETR3D. It should be written so the same logic can later be applied to StreamPETR.
The method should be detector-agnostic at the feature interface level:

```text
images -> backbone/neck features -> detector head/decoder
```

The temporal memory module should intercept the features between backbone/neck and detector head, not rewrite the whole detector.

## Main risks

1. 3D positional encoding may make cached memory geometrically stale after ego-motion.
2. Intermediate ViT features may be too low-level or distribution-shifted for the detector head.
3. The update module may cost enough to erase the speedup.
4. The naive reuse baseline may be surprisingly strong.
5. Freezing most pretrained weights may underfit because the detector receives out-of-distribution mixed features.

## First success criterion

For `K=3` or `K=5`, obtain a better accuracy/latency tradeoff than naive reuse, while retaining a meaningful speedup over full encoding every frame.

## Adaptive policy

The project now includes an adaptive policy direction in addition to fixed keyframe intervals. A learned policy may choose L8, L12, or L24 per frame based on estimated difficulty. Online policy inputs must be available at inference time. Offline annotation/object-count inputs are useful as oracle analysis only. The online + L8 cascade policy is deployable only if its extra block-8 probe cost is included in the latency comparison.
