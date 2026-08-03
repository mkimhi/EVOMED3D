# EvoMem3D Paper Revision Design

## Objective

Rewrite `paper/main.tex` as a coherent WACV 2027 algorithms-track paper whose central contribution is a policy network for adaptive-depth temporal feature-memory updating in camera-only 3D detection.

## Central Message

EvoMem3D avoids uniform full-backbone computation on streaming multi-camera input. A lightweight policy chooses the minimum useful EVA-02 encoding depth for each frame. Partial encodings at blocks 8, 12, or 16 provide current-frame evidence that updates a cached dense feature representation; block 24 performs a full refresh. The detector head remains unchanged.

Refresh caps such as reset-$K$ are implementation safeguards against unbounded memory age. They are not part of the method's primary identity and belong in experimental details or the appendix.

## Evidence Hierarchy

1. Lead with end-to-end nuScenes accuracy and measured throughput from completed `RESULTS.md` rows.
2. Use stale reuse and partial-only inference to establish why both temporal memory and current-frame computation are needed.
3. Use fixed-depth and cyclic schedules as controlled baselines for adaptive policy routing.
4. Use grouped scene-level notebook analyses to explain which signals predict representation drift.
5. Separate deployable online inference inputs from current-frame annotation features, which are oracle analyses only.
6. Mark unverified or missing evidence with visible red placeholders; never infer numerical results.

## Main-Paper Structure

- Introduction: repeated dense encoding, temporal redundancy, adaptive-depth memory update, headline trade-off.
- Related work: temporal 3D detection, efficient ViTs and token routing, video feature reuse, dynamic depth.
- Method: canonical feature memory, partial encoding, gated update, policy inputs and actions, training objectives.
- Experiments: setup summary, main accuracy-efficiency frontier, update mechanism ablation, policy-feature analysis, discussion.
- Conclusion: adaptive partial encoding as the principal result.

Implementation details, optimizer schedules, full latency protocol, refresh caps, negative updater variants, distillation details, and extended policy rows move to the appendix.

## Tables and Figures

- Main accuracy-efficiency table: full computation, reuse, representative fixed partial/update baselines, and strongest deployable policy variants.
- Updater ablation table: partial-only, gated residual, attention variants, and distillation.
- Policy-feature table: online metadata, shallow feature probe, offline/oracle features, cosine target, and CKA target.
- Extended appendix policy table: all completed policy experiments from `RESULTS.md`; incomplete runs become red placeholders only when scientifically useful.
- Reuse suitable notebook figures after checking labels and visual quality.
- Add red placeholders for a manually created overview diagram and qualitative prediction sequences when suitable prediction artifacts are unavailable.

## Writing Constraints

- Preserve the current WACV 2027 review template.
- Explain partial encoding conceptually before giving EVA block numbers.
- Do not make reset intervals prominent in the title, abstract, contributions, or core method narrative.
- Avoid implementation-heavy prose in the main paper.
- Remove obsolete author comments and speculative empty tables.
- Keep claims proportional to the recorded evidence, especially where latency excludes or assumes overlap for a feature probe.
