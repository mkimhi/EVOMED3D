# nuScenes Detection-Error Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a paper-ready Markdown table reporting NDS, mAP, mATE, mASE, mAOE, mAVE, and mAAE for three requested RepDETR3D experiments.

**Architecture:** Transcribe the final nuScenes validation metrics directly from the three identified evaluation logs into one Markdown file. Keep provenance beside the table and validate the exact values and row count without adding a reporting script.

**Tech Stack:** Markdown, shell verification, git.

---

### Task 1: Create and verify the metric table

**Files:**
- Create: `paper/nuscenes_detection_errors.md`

- [ ] **Step 1: Confirm the source values**

Run:

```bash
rg -n 'mATE|mASE|mAOE|mAVE|mAAE|NDS:|mAP:' \
  work_dirs/stream_petr_eva.log \
  work_dirs/repdetr3d_eva02_800_bs4_seq_24e_tm_gated_k3_l16_no_distill/20260723_074835.log \
  work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset4_similarity_online_offline_teacher_forcing_feature_noise_no_distill/20260804_062800.log
```

Expected final records:

```text
K=1:          0.6189 0.5310 0.5209 0.2514 0.2596 0.2353 0.1989
L16:          0.6081 0.5196 0.5420 0.2578 0.2809 0.2440 0.1922
Feature 0.10: 0.6057 0.5139 0.5439 0.2564 0.2771 0.2421 0.1932
```

- [ ] **Step 2: Create the Markdown report**

Create `paper/nuscenes_detection_errors.md` with this content:

```markdown
# nuScenes Detection Error Metrics

| Method | NDS ↑ | mAP ↑ | mATE ↓ | mASE ↓ | mAOE ↓ | mAVE ↓ | mAAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| RepDETR3D EVA02 temporal K=1 BASELINE | 0.6189 | 0.5310 | 0.5209 | 0.2514 | 0.2596 | 0.2353 | 0.1989 |
| RepDETR3D EVA02 partial-only L16 baseline | 0.6081 | 0.5196 | 0.5420 | 0.2578 | 0.2809 | 0.2440 | 0.1922 |
| RepDETR3D EVA02 gated residual trainable MLP policy 4-level reset4 similarity online + offline teacher forcing feature-noise 0.10 no distill | 0.6057 | 0.5139 | 0.5439 | 0.2564 | 0.2771 | 0.2421 | 0.1932 |

Higher is better for NDS and mAP; lower is better for all five true-positive error metrics.

## Provenance

- K=1 baseline: `work_dirs/stream_petr_eva.log`.
- L16 row: `work_dirs/repdetr3d_eva02_800_bs4_seq_24e_tm_gated_k3_l16_no_distill/20260723_074835.log`.
- Reset4 feature-noise 0.10 policy: `work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset4_similarity_online_offline_teacher_forcing_feature_noise_no_distill/20260804_062800.log`.
```

- [ ] **Step 3: Verify formatting and values**

Run:

```bash
test "$(rg -c '^\| RepDETR3D' paper/nuscenes_detection_errors.md)" -eq 3
rg -n '0\.6189.*0\.5310.*0\.5209.*0\.2514.*0\.2596.*0\.2353.*0\.1989|0\.6081.*0\.5196.*0\.5420.*0\.2578.*0\.2809.*0\.2440.*0\.1922|0\.6057.*0\.5139.*0\.5439.*0\.2564.*0\.2771.*0\.2421.*0\.1932' paper/nuscenes_detection_errors.md
git diff --check
```

Expected: three matching metric rows and no whitespace errors.

- [ ] **Step 4: Commit**

```bash
git add paper/nuscenes_detection_errors.md
git commit -m "docs: report nuScenes detection errors"
```
