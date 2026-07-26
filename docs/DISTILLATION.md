# Distillation and training strategy

## Why distillation is needed

Intermediate ViT/EVA features may be lower-level than final encoder features. Even if EVA-02 features are strong, a layer-12 or layer-18 representation is not guaranteed to be compatible with a detector head trained on final features. Without distillation, the memory updater may learn slowly or collapse to naive reuse.

## Teacher-student setup

During training on non-keyframes, optionally run both paths:

```text
Teacher: F_t = FullEncoder(I_t)              # stop-gradient target
Student: M_t_hat = Updater(M_{t-1}, Z_t_l)  # detector input feature
```

Loss:

```text
L_total = L_det(M_t_hat) + lambda_feat * L_feat(M_t_hat, stopgrad(F_t))
```

Optional additional losses:

```text
L_cos = 1 - cosine(M_t_hat, stopgrad(F_t))
L_token = SmoothL1(M_t_hat, stopgrad(F_t))
L_norm = abs(norm(M_t_hat) - norm(F_t))
```

Start simple:

```text
L_feat = SmoothL1(M_t_hat, stopgrad(F_t))
```

## Where to distill

Prefer distilling at the exact feature interface consumed by the detector head.

If the detector uses multi-scale features, distill each scale with separate weights:

```text
L_feat = sum_s lambda_s * SmoothL1(M_t_hat_s, F_t_s)
```

If memory is pre-PE, distill pre-PE appearance features. If only post-PE features are available, document that clearly.

## Freezing strategy

Use staged training.

### Stage 0: baseline verification

No new module. Verify original detector training/inference.

### Stage 1: train updater only

Freeze backbone. Freeze most detector components if possible. Train:

- partial feature projection
- memory updater
- optional small normalization/adapters

Use distillation strongly.

### Stage 2: train updater + neck/head

Train:

- memory updater
- neck
- detector head/decoder

Keep backbone frozen or LoRA-only.

### Stage 3: light end-to-end adaptation

Unfreeze later backbone blocks or add LoRA/adapters. Use only if Stage 2 is not enough.

## Recommended initial hyperparameters

Start with:

```text
lambda_feat = 1.0
feature_loss = SmoothL1
keyframe_interval = 3
partial_layer = middle EVA block
optimizer = same as detector baseline, smaller LR for pretrained parts
```

Then sweep:

```text
lambda_feat in [0.1, 0.5, 1.0, 2.0]
partial_layer in [25%, 50%, 75% depth]
keyframe_interval in [2, 3, 5]
```

## Avoiding teacher leakage at inference

The full teacher path may be run during training for distillation, but non-keyframe inference must not compute the full encoder output.

Add a config flag:

```text
tm_train_with_teacher = true/false
tm_inference_teacher_forbidden = true
```

If `tm_inference_teacher_forbidden=true`, assert that the full encoder is not called on non-keyframes.

## Metrics to log

For every experiment log:

- mAP
- NDS
- per-class AP if available
- latency per frame
- latency separated into backbone, updater, head
- GPU memory
- feature distillation loss
- cosine similarity to full teacher memory
- keyframe vs non-keyframe performance