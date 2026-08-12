# Summary: Adaptive Early-Feature Memory Updating for Camera-Only 3D Detection

Camera-only 3D detectors with large ViT/EVA backbones spend most of their runtime repeatedly re-encoding highly correlated multi-camera video frames. Our core observation is that the full encoded scene does not need to be recomputed at every timestamp. Early-exit or early-layer features are often sufficient to update a previous encoded scene representation, as long as the system can detect when the representation has drifted too far and must refresh with the full encoder.

The project therefore maintains a dense multi-view image-feature memory. On keyframes, the detector runs the full encoder and stores the final encoded feature map. On non-keyframes, it runs only a partial encoder, for example EVA block L8, L12, or L16, projects those early features into the detector feature interface, and updates the cached full representation:

```text
keyframe:     M_t = FullEncoder(I_t)                         # L24
non-keyframe: P_t = Project(PartialEncoder(I_t, layer=l))     # L8/L12/L16
              M_t = MemoryUpdater(M_{t-1}, P_t, metadata)
```

The updated memory is then passed to the normal detector head. This keeps the detection architecture mostly unchanged: the temporal module sits between the image backbone/neck and the DETR-style 3D detection head. The first implementation targets RepDETR3D with EVA-02, but the interface is detector-agnostic and intended to transfer to StreamPETR-style detectors.

## Key Problem: Representation Drift

The main failure mode is representation drift: the cached encoded scene becomes stale when the current frame changes too much. Drift is correlated with high ego speed, large translation between frames, high yaw rate, new objects entering the scene, objects disappearing, dense nearby objects, and other scene changes. Naive reuse confirms this risk: reusing full features without update is efficient but loses too much accuracy as the keyframe gap grows.

We studied drift by measuring adjacent-frame changes in encoded features. The main target was final-layer global cosine similarity, `similarity_L24_mean`, computed across the final EVA block feature maps. Its validation distribution is broad enough to support policy learning: approximately min 0.49, median 0.72, p75 0.77, p95 0.93, max 0.96. Final-feature similarity drops with ego translation and speed, yaw-rate, disappeared/new-object proxies, and near-object counts.

We also computed patch/grid CKA for L24 features, excluding CLS/register-token concerns raised by recent ViT representation-analysis work. This CKA signal is not redundant with cosine similarity: Spearman correlation between `cka_L24_mean` and `similarity_L24_mean` is about 0.50. Patch CKA varies substantially across frames and cameras, so it provides a complementary drift label. In practice, global mean similarity already works well as a policy target, while patch CKA is useful as a more local/token-structure-sensitive diagnostic.

## Predicting Drift

We tested whether drift can be predicted before paying for the full encoder. Three policy input regimes are important:

1. **Online metadata only:** timing and ego-motion features available at inference: frame index, scene-start flag, time delta, ego x/y/z, ego yaw, ego translation, speed, acceleration, yaw rate, absolute yaw rate, yaw acceleration, and absolute yaw acceleration. With HistGradientBoosting, online metadata predicts `similarity_L24_mean` with R2 about 0.706 and MAE about 0.039. The strongest five online features are ego translation, ego speed, absolute yaw rate, ego y, and ego x. Without absolute pose, the best five are ego translation, speed, absolute yaw rate, acceleration, and absolute yaw acceleration.

2. **Offline/oracle metadata:** online metadata plus annotation-derived features such as number of objects, object distance statistics, near-object counts at 10/20/30/50 m, box dimensions/area/volume, object velocity, camera annotation counts, 2D box area, and proxy counts for newly appeared, matched, and disappeared objects. These features are useful for analysis and upper bounds, but they are not deployable because they rely on ground truth or offline labels.

3. **Online metadata + L8 cascade:** first run a cheap L8 partial encoder probe, compare L8 features to the cached previous L8 state, then use L8 mean/per-camera similarity together with online metadata to decide whether L8 is enough, whether L12 is needed, or whether the frame is hard and should use L24. This signal is very predictive: metadata + L8 similarity reaches roughly R2 0.95 and MAE 0.015 for L24 cosine similarity. For patch CKA, online metadata gives R2 about 0.889, while online + L8 improves to about 0.933.

## Policy Network

The runtime policy is a small MLP classifier. Its input is a fixed vector of scalar policy features from one of the three regimes above. Its output is a difficulty class:

```text
easy   -> L8  partial update
medium -> L12 partial update
hard   -> L24 full encoder refresh
```

The MLP is trained offline from drift labels derived from `similarity_L24_mean` or CKA. Current exported policy checkpoints use cross-entropy over the easy/medium/hard classes. The policy can load pretrained weights from `outputs/policy_signal_stats/val/mlp_policy_*.pt`; it can also be configured with `load_weights=False` for random or train-from-scratch ablations. The online MLP policy is deployable, the offline metadata policy is an oracle upper bound, and the L8 cascade policy is deployable only if the L8 probe cost is counted in latency.

## Empirical Verdict

The preliminary results support the paper thesis. A full RepDETR3D EVA-02 baseline reaches about NDS 61.9 / mAP 53.1 at 2.68 FPS. Naive reuse improves speed but degrades sharply. Partial-only L8 and L12 already recover much of the accuracy while being faster than the full path. A gated residual memory updater with L8 at K=3 reaches about NDS 58.2 / mAP 49.0 at 6.12 FPS, showing that early features can update a previous encoded scene with only modest degradation. A deterministic cycle policy, L24-L8-L12-L16, further improves the tradeoff, reaching about NDS 59.3 / mAP 50.1 at 4.85 FPS.

The stronger claim is that a learned policy should push this closer to the Pareto frontier: use L8 when the scene is stable, L12 when drift is moderate, and L24 only when motion or scene change makes the cached memory unreliable. The metadata and similarity studies show that these hard cases are predictable. In particular, high-speed ego motion, large yaw-rate, close/dense objects, and object appearance/disappearance are exactly the cases where the policy should spend more compute.

Although the adaptive policy detector runs are still being trained, the direction is clear: even deterministic early-feature memory policies improve efficiency without significant performance loss, and smarter policies should give very efficient inference with almost no degradation. This is especially compelling because the base detector is a relatively simple DETR3D-style architecture. With a stronger EVA backbone and partial-encoder memory updates, a simple camera-only detector can approach strong modern accuracy while beating other efficient camera-only 3D detection candidates on the accuracy/latency tradeoff.
