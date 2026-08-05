# nuScenes Detection-Error Table Design

## Goal

Create `paper/nuscenes_detection_errors.md` as a paper-ready comparison of
three requested RepDETR3D experiments using NDS, mAP, mATE, mASE, mAOE, mAVE,
and mAAE.

## Table

Use one row per experiment and these columns:

```text
Method | NDS ↑ | mAP ↑ | mATE ↓ | mASE ↓ | mAOE ↓ | mAVE ↓ | mAAE ↓
```

Report all metrics as nuScenes fractions to four decimal places. Retain the
user-requested experiment names. Add a concise note defining the metric
direction and listing the local evaluation log used for each row.

## Provenance

- K=1 full-path baseline: `work_dirs/stream_petr_eva.log`.
- L16 row: `work_dirs/repdetr3d_eva02_800_bs4_seq_24e_tm_gated_k3_l16_no_distill/20260723_074835.log`.
- Reset4 feature-noise 0.10 policy: `work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset4_similarity_online_offline_teacher_forcing_feature_noise_no_distill/20260804_062800.log`.

The L16 row retains the requested RESULTS.md display name while using the
evaluation record that produced its reported 60.81 NDS and 51.96 mAP.

## Verification

Check every table value against its cited log and verify that the Markdown
table has three rows, seven metric columns, consistent four-decimal formatting,
and explicit higher/lower-is-better markers.
