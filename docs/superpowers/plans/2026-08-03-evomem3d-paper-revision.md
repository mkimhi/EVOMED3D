# EvoMem3D Paper Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a coherent WACV 2027 manuscript centered on policy-driven partial encoding for adaptive temporal feature memory, using every completed experiment and clearly marking missing evidence.

**Architecture:** Rewrite `paper/main.tex` around a claim-evidence hierarchy rather than preserving its incomplete prose. Main-paper tables will contain the strongest controlled and deployable results; implementation details, negative ablations, and the complete experiment inventory will move to the appendix. Notebook outputs will support policy-feature analysis, while unsupported visual material will use explicit red placeholders.

**Tech Stack:** WACV LaTeX, BibTeX, existing nuScenes experiment records, saved Jupyter notebook outputs.

## Global Constraints

- Preserve the WACV 2027 algorithms-track review template.
- Make the policy network, adaptive encoder depth, and meaning of partial encoding the primary contribution.
- Treat reset intervals as appendix-level safety details.
- Separate deployable inference features from oracle annotation inputs.
- Do not invent missing numerical results or citations.
- Move training and latency protocol details to the appendix.

---

### Task 1: Evidence and Claim Matrix

**Files:**
- Read: `RESULTS.md`
- Read: `notebooks/policy_metadata_explore.ipynb`
- Read: `notebooks/policy_metadata_explore_l24_cka.ipynb`
- Read: `docs/PROJECT_BRIEF.md`
- Read: `docs/ARCHITECTURE.md`
- Read: `docs/EXPERIMENTS.md`
- Read: current temporal-memory source through CodeGraph

**Interfaces:**
- Consumes: repository experiment records and implemented behavior.
- Produces: verified values and evidence classifications used by all manuscript sections.

- [ ] **Step 1: Classify completed detector experiments**

Record the 30 `RESULTS.md` rows as completed or missing based on the four metric fields; classify each policy input as deployable, training-only teacher information, or oracle inference information.

- [ ] **Step 2: Extract notebook evidence**

Use the saved grouped five-fold scene-CV results for 5,869 frames across 150 nuScenes validation scenes. Preserve cosine and CKA as separate targets and report their feature-set definitions.

- [ ] **Step 3: Define claim-safe headline comparisons**

Use full recomputation (61.90 NDS, 53.10 mAP, 2.68 FPS), fixed L8 update (58.23, 48.98, 6.12), and the strongest deployable adaptive row (60.43, 54.41, 5.11) without implying that reset frequency is the innovation.

### Task 2: Rewrite the Main Narrative

**Files:**
- Modify: `paper/main.tex`
- Read/verify: `paper/main.bib`

**Interfaces:**
- Consumes: Task 1 evidence matrix.
- Produces: title, abstract, introduction, related work, method, experiments, discussion, limitations, and conclusion.

- [ ] **Step 1: Rewrite title and abstract**

State the adaptive-depth memory problem, L8/L12/L16/L24 action space, unchanged detector head, and the measured headline comparison. Avoid reset-$K$ terminology.

- [ ] **Step 2: Rewrite introduction and contributions**

Explain why full dense encoding is wasteful in streaming input; why stale reuse fails; and why partial features are current-frame corrections rather than independent detector inputs.

- [ ] **Step 3: Simplify the method section**

Retain only the canonical memory equation, partial encoding/projecting/updating equations, lightweight policy definition, online feature timing, and training objectives. Move module dimensions and schedules to the appendix.

- [ ] **Step 4: Rewrite the experiment interpretation**

Organize results around three questions: value of partial encoding, value of temporal memory, and value of learned adaptive depth. Explicitly discuss negative attention and distillation results.

- [ ] **Step 5: Rewrite discussion, limitations, and conclusion**

Bound claims to one detector/backbone/dataset and acknowledge single-run reporting, latency-protocol gaps, feature-probe accounting caveats, and teacher-forcing distribution shift.

### Task 3: Build Evidence-Complete Tables

**Files:**
- Modify: `paper/main.tex`

**Interfaces:**
- Consumes: all completed metrics and notebook summaries.
- Produces: compact main-paper and comprehensive appendix tables.

- [ ] **Step 1: Main accuracy-efficiency table**

Include full, reuse, L8/L12/L16 partial-only, fixed L8 memory update, cyclic depth, and representative learned policy rows. Compute speedups directly from recorded FPS.

- [ ] **Step 2: Updater ablation table**

Compare partial-only L8, gated residual, four attention variants, and gated residual with feature distillation at the same L8/K=3 setting.

- [ ] **Step 3: Policy ablation table**

Include completed online-only, memory-age safeguard, prior-feature, teacher-forced, hard-weighted, and oracle rows, with an explicit inference-input column.

- [ ] **Step 4: Policy-signal table**

Report grouped-CV regression for cosine and CKA using online metadata, offline metadata, online plus current L8 probe, and offline plus current L8 probe.

- [ ] **Step 5: Extended appendix inventory**

Include every completed detector result not already represented in the main paper. Add red commented placeholders only for scientifically important missing experiments.

### Task 4: Integrate Figures and Placeholders

**Files:**
- Modify: `paper/main.tex`
- Reuse: `outputs/policy_signal_stats/val/explore/*.png`
- Reuse: `outputs/policy_signal_stats/val/explore_cka_l24/*.png`

**Interfaces:**
- Consumes: saved notebook plots and available prediction artifacts.
- Produces: publication-useful policy evidence or explicit manual-work placeholders.

- [ ] **Step 1: Inspect candidate notebook figures**

Check the target distribution, grouped-CV metric, feature importance, and confusion-matrix plots for readable labels and defensible captions.

- [ ] **Step 2: Select minimal supporting figures**

Use at most one compact policy-analysis figure in the main paper; move secondary figures to the appendix.

- [ ] **Step 3: Add manual figure placeholders**

Insert red boxed placeholders for a paradigm diagram and qualitative prediction sequences if no suitable prediction images exist. Each placeholder must specify the panels and intended message.

### Task 5: Appendix and Reproducibility Details

**Files:**
- Modify: `paper/main.tex`

**Interfaces:**
- Consumes: configs, implementation source, and experiment records.
- Produces: implementation, training, latency, extended results, and qualitative-analysis appendices.

- [ ] **Step 1: Describe implementation succinctly**

Document the detector feature boundary, EVA exits, 1x1 gated residual updater, scene/discontinuity resets, full-teacher training path, and inference teacher prohibition.

- [ ] **Step 2: Move incomplete protocols to red appendix placeholders**

Request exact optimizer/freeze details, hardware/software versions, warm-up/timing protocol, peak memory, action frequencies, and random-seed reporting only where repository evidence is absent.

- [ ] **Step 3: State reproducibility and ethics items**

Add data availability, limitations, conflict-of-interest, funding, author contributions, and AI-assistance placeholders appropriate for an anonymous WACV draft.

### Task 6: Validate the Manuscript

**Files:**
- Verify: `paper/main.tex`
- Verify: `paper/main.bib`

**Interfaces:**
- Consumes: revised manuscript.
- Produces: compiling, internally consistent paper or an exact blocker report.

- [ ] **Step 1: Run static checks**

Run `rg -n "\\\\mk|\\\\yi|TODO|todo|preliminary|--" paper/main.tex`, check unmatched citations/labels, and run `git diff --check`.

- [ ] **Step 2: Compile LaTeX**

Run `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from `paper/`. If `latexmk` is unavailable, run the available `pdflatex`/`bibtex` sequence and report the limitation.

- [ ] **Step 3: Inspect the generated PDF**

Check page count, table overflow, red placeholders, figure readability, broken references, and accidental author comments.

- [ ] **Step 4: Audit numerical claims**

Reconcile every NDS, mAP, FPS, latency, regression, and classification value against `RESULTS.md` or saved notebook output before completion.
