# Similarity Results

Temporal final-feature similarity has been run for gaps 1-5 on the base RepDETR3D EVA02 checkpoint.

The command below runs the first four temporal-gap comparisons in one pass on a 5-video training subset. Gap 5 was run separately with a longer queue. Each run writes complete metrics to `summary.json` and `summary.csv`, with metric names prefixed by `gap_1/`, `gap_2/`, etc. Per-camera heatmap images are written under the same output directory.

```bash
python tools/feature_similarity.py --mode frame-gap-final --adapter repdetr3d --config projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_baseline.py --checkpoint ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth --data-split train --queue-length 5 --anchor-frame 0 --gaps 1 2 3 4 --num-videos 5 --num-samples 200 --out-dir work_dirs/feature_similarity/repdetr3d_eva02_final_frame_gaps
```

| similarity name | description | time gap | cosine/global | CKA/global | RSM Pearson | RSM Spearman | per-camera metrics | center/border metrics | foreground/background metrics | near/far foreground metrics | heatmaps | command for running |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|
| RepDETR3D EVA02 final feature temporal similarity gap 1 | Compares final detector features from adjacent frames, `(t_i, t_i+1)`, on a 5-video training subset. This measures how much dense final memory changes after one frame. | 1 | 0.7696 | 0.7615 | 0.7816 | 0.7450 | cosine camera range 0.7588-0.7892 | cosine center 0.7696, border 0.7696 | cosine foreground 0.6921, background 0.7958 | cosine near 0.7006, far 0.5944 | `work_dirs/feature_similarity/repdetr3d_eva02_final_frame_gaps_base/gap_1_*` | `python tools/feature_similarity.py --mode frame-gap-final --adapter repdetr3d --config projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_baseline.py --checkpoint ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth --data-split train --queue-length 5 --anchor-frame 0 --gaps 1 2 --num-videos 5 --num-samples 200 --out-dir work_dirs/feature_similarity/repdetr3d_eva02_final_frame_gaps_base` |
| RepDETR3D EVA02 final feature temporal similarity gap 2 | Compares final detector features two frames apart, `(t_i, t_i+2)`, on a 5-video training subset. This measures short-range drift relevant to K=3 reuse/update settings. | 2 | 0.7574 | 0.7522 | 0.7718 | 0.7341 | cosine camera range 0.7301-0.7713 | cosine center 0.7504, border 0.7597 | cosine foreground 0.6757, background 0.7857 | cosine near 0.6860, far 0.5714 | `work_dirs/feature_similarity/repdetr3d_eva02_final_frame_gaps_base/gap_2_*` | `python tools/feature_similarity.py --mode frame-gap-final --adapter repdetr3d --config projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_baseline.py --checkpoint ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth --data-split train --queue-length 5 --anchor-frame 0 --gaps 1 2 --num-videos 5 --num-samples 200 --out-dir work_dirs/feature_similarity/repdetr3d_eva02_final_frame_gaps_base` |
| RepDETR3D EVA02 final feature temporal similarity gap 3 | Compares final detector features three frames apart, `(t_i, t_i+3)`, on a 5-video training subset. This estimates feature drift across the longest non-keyframe span for K=4-style schedules. | 3 | 0.7476 | 0.7397 | 0.7605 | 0.7221 | cosine camera range 0.7072-0.7690 | cosine center 0.7401, border 0.7502 | cosine foreground 0.6600, background 0.7786 | cosine near 0.6699, far 0.5597 | `work_dirs/feature_similarity/repdetr3d_eva02_final_frame_gaps_base34/gap_3_*` | `python tools/feature_similarity.py --mode frame-gap-final --adapter repdetr3d --config projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_baseline.py --checkpoint ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth --data-split train --queue-length 5 --anchor-frame 0 --gaps 3 4 --num-videos 5 --num-samples 200 --out-dir work_dirs/feature_similarity/repdetr3d_eva02_final_frame_gaps_base34` |
| RepDETR3D EVA02 final feature temporal similarity gap 4 | Compares final detector features four frames apart, `(t_i, t_i+4)`, on a 5-video training subset. This measures stale-memory drift across the full 5-frame window. | 4 | 0.7063 | 0.7009 | 0.7603 | 0.6923 | cosine camera range 0.6657-0.7369 | cosine center 0.7031, border 0.7207 | cosine foreground 0.5909, background 0.7594 | cosine near 0.6056, far 0.4543 | `work_dirs/feature_similarity/repdetr3d_eva02_final_frame_gaps_base34/gap_4_*` | `python tools/feature_similarity.py --mode frame-gap-final --adapter repdetr3d --config projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_baseline.py --checkpoint ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth --data-split train --queue-length 5 --anchor-frame 0 --gaps 3 4 --num-videos 5 --num-samples 200 --out-dir work_dirs/feature_similarity/repdetr3d_eva02_final_frame_gaps_base34` |



• Foreground/background:

  - Uses 2D GT boxes from the train batch: gt_bboxes.
  - For each camera, each GT 2D box is scaled from image resolution to feature-map resolution.
  - Tokens inside any GT 2D box become foreground=True.
  - Everything else is background.

  Center/border:

  - Built directly on the feature grid.
  - center is the middle region using border_ratio=0.25.
  - So roughly the central 50% height by 50% width is center.
  - Everything outside that is border.

  Near/far:

  - Uses GT object depths from the train batch: depths.
  - It computes the median depth across available GT objects in the batch.
  - Foreground boxes with depth <= median are near_foreground.
  - Foreground boxes with depth > median are far_foreground.

  Important caveat: these masks are approximate feature-token masks from 2D projected boxes, not exact object silhouettes.

## Policy L24 Patch-CKA Signal

Motivation: the paper *Register and [CLS] tokens induce a decoupling of local and
global features in large ViTs* suggests that global special tokens can inflate
CKA-style similarity. The policy CKA table below is therefore computed from the
detector patch/grid feature maps shaped `[B, N_cam, C, H, W]`. In the current
RepDETR3D EVA path, CLS/register tokens are not present in these maps.

Command:

```bash
python tools/policy_feature_similarity.py --config projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_baseline.py --checkpoint ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth --data-split val --queue-length 2 --layers L24:final --out-csv outputs/policy_signal_stats/val/policy_feature_cka_l24.csv --include-cka --device cuda:0
```

Output:

```text
outputs/policy_signal_stats/val/policy_feature_cka_l24.csv
```

Rows: `6019` validation frames. Rows with previous-frame CKA: `5869`.

Distribution for adjacent-frame `cka_L24_mean`:

```text
min    0.0315
p01    0.0398
p05    0.0464
p25    0.0601
median 0.0760
p75    0.1087
p95    0.4648
p99    0.5415
max    0.6006
mean   0.1280
std    0.1283
```

Per-frame camera summaries also vary substantially:

| target | min | median | p75 | p95 | max | mean | std |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cka_L24_min` | 0.0150 | 0.0421 | 0.0589 | 0.3265 | 0.5324 | 0.0783 | 0.0940 |
| `cka_L24_max` | 0.0376 | 0.1243 | 0.1898 | 0.6098 | 0.8419 | 0.1921 | 0.1688 |
| `cka_L24_spread` | 0.0080 | 0.0823 | 0.1355 | 0.3180 | 0.5762 | 0.1138 | 0.0920 |

Spearman correlation between `cka_L24_mean` and cosine `similarity_L24_mean`:
`0.5010`. This means patch CKA is related to, but not redundant with, the
previous cosine target.

Grouped-CV predictability with `HistGradientBoostingRegressor` and
`GroupKFold(scene_token)`:

| target | feature set | R2 | MAE |
|---|---|---:|---:|
| `cka_L24_mean` | metadata online | 0.8891 +/- 0.0367 | 0.0231 +/- 0.0008 |
| `cka_L24_mean` | metadata offline | 0.8753 +/- 0.0622 | 0.0221 +/- 0.0016 |
| `cka_L24_mean` | metadata online + L8 cosine | 0.9330 +/- 0.0171 | 0.0197 +/- 0.0015 |
| `cka_L24_mean` | metadata offline + L8 cosine | 0.9431 +/- 0.0229 | 0.0173 +/- 0.0018 |
| `cka_L24_min` | metadata online | 0.7585 +/- 0.1005 | 0.0212 +/- 0.0030 |
| `cka_L24_min` | metadata offline | 0.7572 +/- 0.0723 | 0.0211 +/- 0.0040 |
| `cka_L24_min` | metadata online + L8 cosine | 0.8647 +/- 0.0385 | 0.0175 +/- 0.0029 |
| `cka_L24_min` | metadata offline + L8 cosine | 0.8787 +/- 0.0181 | 0.0166 +/- 0.0028 |

Notebook for exploration and policy modeling:

```text
notebooks/policy_metadata_explore_l24_cka.ipynb
```
