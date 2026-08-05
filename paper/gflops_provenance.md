# GFLOPs provenance

All local values are end-to-end inference GFLOPs per nuScenes timestamp at
batch size 1 with six `320 x 800` camera images. One multiply--accumulate is
counted as two floating-point operations. The profiler counts matrix
multiplication, convolution, FlashAttention, and multi-scale deformable
attention MACs; elementwise nonlinearities, indexing, reshaping, and metadata
bookkeeping are recorded as unsupported non-MAC operators and contribute zero
MACs. The exact operator lists are retained in each linked `summary.json` and
`frames.jsonl` output.

## Measured path costs

| Path | GFLOPs | Evidence | Config | Checkpoint | Profiler output |
|---|---:|---|---|---|---|
| Dense/full L24 | 5161.797648 | local-measured | `repdetr3d_eva02_800_bs2_seq_24e.py` | `ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth` | `outputs/gflops/dense_l24/summary.json` |
| Temporal full L24 | 5161.797648 | local-measured | `repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset5_similarity_online_offline_teacher_forcing_no_distill.py`, with `max_frames_between_full=4` | `...reset4_similarity_online_offline_teacher_forcing_feature_noise_no_distill/iter_42192.pth` | `outputs/gflops/l24_v2/summary.json` |
| Reuse, no backbone | 35.877299 | local-measured | `repdetr3d_eva02_800_bs2_seq_24e_tm_reuse_k3.py` | baseline checkpoint | `outputs/gflops/reuse_v2/summary.json` |
| Partial-only L8 | 2127.303190 | local-measured | `repdetr3d_eva02_800_bs2_seq_24e_tm_partial_only_l8.py` | baseline checkpoint | `outputs/gflops/partial_only_l8/summary.json` |
| Partial-only L12 | 2951.091599 | local-measured | `repdetr3d_eva02_800_bs2_seq_24e_tm_partial_only_l12.py` | baseline checkpoint | `outputs/gflops/partial_only_l12/summary.json` |
| Partial-only L16 | 3774.880007 | local-measured | `repdetr3d_eva02_800_bs2_seq_24e_tm_partial_only_l16.py` | baseline checkpoint | `outputs/gflops/partial_only_l16/summary.json` |
| Gated update L8 | 2194.149910 | local-measured | `repdetr3d_eva02_800_bs2_seq_24e_tm_gated_k3_l8_no_distill.py` | baseline checkpoint | `outputs/gflops/updater_gated_l8/summary.json` |
| Gated update L12 | 3017.938319 | local-measured | same four-level config with forced L12 | same reset4 feature-noise checkpoint | `outputs/gflops/l12_v2/summary.json` |
| Gated update L16 | 3841.726727 | local-measured | same four-level config with forced L16 | same reset4 feature-noise checkpoint | `outputs/gflops/l16_v2/summary.json` |
| Cross-attention update L8 | 2209.089568 | local-measured | `repdetr3d_eva02_800_bs2_seq_24e_tm_attention_k3_l8_no_distill.py` | baseline checkpoint | `outputs/gflops/updater_attention_l8/summary.json` |
| Gated cross-attention update L8 | 2242.512928 | local-measured | `repdetr3d_eva02_800_bs2_seq_24e_tm_gated_attention_k3_l8_no_distill.py` | baseline checkpoint | `outputs/gflops/updater_gated_attention_l8/summary.json` |
| Per-camera attention update L8 | 2182.977568 | local-measured | `repdetr3d_eva02_800_bs2_seq_24e_tm_per_camera_attention_k3_l8_no_distill.py` | baseline checkpoint | `outputs/gflops/updater_per_camera_attention_l8/summary.json` |
| Inverted attention update L8 | 2209.089568 | local-measured | `repdetr3d_eva02_800_bs2_seq_24e_tm_inverted_attention_k3_l8_no_distill.py` | baseline checkpoint | `outputs/gflops/updater_inverted_attention_l8/summary.json` |

The baseline checkpoint is valid for operation counting even where updater or
projector weights are absent: weights change tensor values, not the executed
module graph or tensor shapes.

The trainable policy has 42 inputs, a 32-unit hidden layer, and four outputs,
or 1,472 MACs per learned decision. Its maximum contribution is
0.000002944 GFLOPs per timestamp and therefore does not change any value at the
reported 0.1-GFLOP precision. Reset and age-forced actions bypass the MLP.

## Validation-set weighted values

Fixed schedules are aggregated over all 6,019 validation timestamps and 150
scene resets. K=3 has 2,082 L24 frames and 3,937 non-full frames; K=2 has
3,037 L24 and 2,982 reuse frames. The four-action cycle has counts
`L24=1537`, `L8=1500`, `L12=1500`, and `L16=1482`.

| Paper entry | Table(s) | GFLOPs | Evidence | Aggregation |
|---|---|---:|---|---|
| Full encoding every frame | main, policy, fixed | 5161.8 | local-measured | dense L24 path |
| Naive reuse K=2 | fixed | 2622.3 | local-weighted | K=2 counts × L24/reuse paths |
| Naive reuse K=3 | main, policy, fixed | 1809.0 | local-weighted | K=3 counts × L24/reuse paths |
| Partial-only L8, K=3 | policy, updater, fixed | 3176.9 | local-weighted | K=3 counts × L24/partial-only L8 paths |
| Partial-only L12, K=3 | fixed | 3715.8 | local-weighted | K=3 counts × L24/partial-only L12 paths |
| Partial-only L16, K=3 | fixed | 4254.6 | local-weighted | K=3 counts × L24/partial-only L16 paths |
| Gated residual L8, K=3 | policy, updater, fixed | 3220.7 | local-weighted | K=3 counts × L24/gated L8 paths |
| Gated residual L12, K=3 | policy | 3759.5 | local-weighted | K=3 counts × L24/gated L12 paths |
| Gated residual L16, K=3 | policy | 4298.3 | local-weighted | K=3 counts × L24/gated L16 paths |
| Cross-attention L8, K=3 | updater | 3230.4 | local-weighted | K=3 counts × L24/cross-attention paths |
| Gated cross-attention L8, K=3 | updater | 3252.3 | local-weighted | K=3 counts × L24/gated-attention paths |
| Per-camera attention L8, K=3 | updater | 3213.4 | local-weighted | K=3 counts × L24/per-camera paths |
| Inverted attention L8, K=3 | updater | 3230.4 | local-weighted | K=3 counts × L24/inverted-attention paths |
| Gated residual L8 + distillation, K=3 | updater | 3220.7 | local-weighted | teacher branch is disabled at inference; same graph as gated L8 |
| Variable L24/L8/L12/L16 cycle | main, policy, fixed | 3562.9 | local-weighted | complete cycle counts × gated action paths |
| Accuracy policy (reset4) | main, policy, extended | 4229.3 | local-weighted | `L8=27, L12=2090, L16=797, L24=3105` over 6,019 frames; L24 splits into 790 forced, 2,165 learned, and 150 scene-start actions |
| Fast policy (reset5, hard-aware) | main, extended | 4440.8 | local-weighted | `L8=0, L12=1541, L16=785, L24=3693` over 6,019 frames; L24 splits into 402 forced, 3,141 learned, and 150 scene-start actions |
| Other learned-policy rows | extended | n/r | not-reported | no complete action histogram; latency is not used as a proxy |

## Related work

| Paper entry | GFLOPs | Evidence | Source and comparability note |
|---|---:|---|---|
| StreamPETR ViT-L baseline | 4601 | published | [ToC3D Appendix Table 12](https://arxiv.org/html/2409.00633) reports 2300.5 GMACs / 4601 GFLOPs end to end. The paper also states six cameras and reports the `320 x 800` setting. |
| ToC3D-Fast ViT-L | n/r | not-reported | The primary paper reports latency but no absolute end-to-end MAC/FLOP total for Fast. |
| ToC3D-Faster ViT-L | 3131 | published | [ToC3D Appendix Table 12](https://arxiv.org/html/2409.00633) reports 1565.5 GMACs / 3131 GFLOPs end to end. |
| SToRe3D-1/3 ViT-L | n/r | not-reported | [SToRe3D](https://arxiv.org/html/2605.14110) reports FPS but no absolute ViT-L GFLOPs for this row. |
| SToRe3D-1/10 ViT-L | n/r | not-reported | The paper's 397-GFLOP reduction is for ViT-B SToRe3D-1/2, so it is not transferred to this row. |

Published and local GFLOPs share the six-view resolution and two-FLOPs-per-MAC
convention, but the local detector is RepDETR3D and the published detector is
StreamPETR. Latencies come from different hardware/profiling environments and
must not be interpreted as a controlled hardware comparison.

## Headline policy action-count commands

The reset4 and reset5 counts were produced by single-GPU, scene-ordered full
validation runs using `tools/test.py --policy-stats --eval bbox`. The reset4
run used the no-noise reset4 checkpoint and
`model.temporal_memory.max_frames_between_full=4`; the reset5 run used the
hard-weighted checkpoint and its configured maximum age of five. The policy
MLP contribution noted above is below the displayed precision.

The reset5 row has more measured GFLOPs than reset4 despite its lower ledger
latency. This is not reconciled by altering either measurement: the paper
reports the exact action-weighted operation count and the existing wall-clock
record independently. A controlled latency rerun with a fixed warm-up,
measurement boundary, and isolated GPU is still required to explain the
ordering.
