# Temporal Policy Qualitative Figure Design

## Goal

Create a reproducible Figure 4 from the trained RepDETR3D temporal-policy
checkpoint. The figure must show how the deployed policy allocates encoder
depth across temporal regimes and include one clear early-exit failure.

## Scope

The new figure pipeline will not modify inference behavior or baseline
configs. It will consume a sequential validation rollout and produce a single
publication-ready PDF/PNG plus an auditable manifest of selected frames.

## Inputs

- Temporal policy config:
  `projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset5_similarity_online_offline_teacher_forcing_no_distill.py`.
- Temporal checkpoint:
  `work_dirs/repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset4_similarity_online_offline_teacher_forcing_no_distill/iter_42192.pth`.
- Full-depth baseline checkpoint:
  `ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth`.

The policy rollout is the source of selected depth, confidence, memory age,
and detections. The full-depth model is used to provide the oracle L24 action
and reference predictions for selected frames.

## Figure Layout

The figure has four labeled sequence rows and four chronological timestamps per
row. Each timestamp shows a front camera crop, a matched BEV detection render,
and a fixed-size telemetry strip containing selected depth, policy confidence,
memory age, and full-depth oracle depth. Optional gate maps are compact insets
only when captured without changing the model.

The rows are selected from recorded rollout values:

1. Stable low-speed cruising, dominated by L8 actions.
2. A sharp turn with a transition to L16 or L24.
3. A dense or newly-visible-object interval with deeper action.
4. A policy early-exit failure where the temporal result misses a nearby object
   visible in the L24 reference; the missed region is marked red.

## Components and Data Flow

`tools/export_temporal_policy_figure.py` will roll out the policy sequentially
and save per-frame records and visual artifacts. Its records include tokens,
scene/frame indices, action logits/probabilities, selected action, forced flag,
memory age, ego-motion metadata, detections, and full-depth comparison data.
Selection uses explicit predicates and stores why each sequence qualified.

`tools/make_temporal_policy_figure.py` will validate a complete manifest and
assemble the final static plate using fixed panel dimensions. It will fail
clearly when a requested regime or comparison artifact is absent.

The paper placeholder will be replaced with the generated asset and a caption
that identifies the temporal checkpoint and describes the failure case as a
qualitative observation, not an aggregate metric.

## Validation

- Unit-test trace/selection helpers using synthetic frame records.
- Verify all selected rows have exactly four ordered frames and contain required
  telemetry fields.
- Verify the generated PNG/PDF exists, is non-empty, and has fixed four-by-four
  layout dimensions.
- Run a CUDA smoke rollout on one sequence before the full figure rollout.

## Constraints

- Keep the pipeline detector-agnostic at the recorded feature interface where
  feasible; RepDETR3D-specific rendering belongs in the adapter layer.
- Do not present forced age refreshes as policy decisions.
- Preserve all user worktree changes outside the new figure files.
