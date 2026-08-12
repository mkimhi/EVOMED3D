# RepDETR3D Token Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the useful SEPatch3D tri-state token routing pieces into StreamPETR and expose an opt-in RepDETR3D token-budget path.

**Architecture:** Keep fixed patch size and avoid SEPatch3D FlexiViT/adaptive patch code. Add reusable tensor routing utilities, then call a dense geometry-preserving budget mask from `SparseHead` after RepDETR3D flattens multi-level image features. True variable-length routing remains isolated in utilities because RepDETR3D's deformable CUDA attention indexes dense feature maps by `spatial_shapes`.

**Tech Stack:** PyTorch, MMDetection/MMDetection3D config registry, pytest.

---

### Task 1: Router Utilities

**Files:**
- Create: `projects/mmdet3d_plugin/models/routing/tri_state_router.py`
- Create: `projects/mmdet3d_plugin/models/routing/__init__.py`
- Test: `mmdetection3d/tests/test_models/test_tri_state_router.py`

- [x] **Step 1: Write failing tests for tensor routing and dense budget masking**

Add tests that import `combine_keep_and_bypass`, `pool_ambient_tokens`, `route_tokens`, `make_budget_mask`, and `apply_dense_budget_mask`.

- [ ] **Step 2: Run tests to verify import failure**

Run: `python -m pytest mmdetection3d/tests/test_models/test_tri_state_router.py -q`
Expected: fails because `projects.mmdet3d_plugin.models.routing` does not exist yet.

- [ ] **Step 3: Implement minimal router utilities**

Copy the SEPatch3D tensor router helpers, then add `make_budget_mask` and `apply_dense_budget_mask` for RepDETR3D's dense feature-map constraint.

- [ ] **Step 4: Run tests to verify router utilities pass**

Run: `python -m pytest mmdetection3d/tests/test_models/test_tri_state_router.py -q`
Expected: all tests pass.

### Task 2: RepDETR3D Head Hook

**Files:**
- Modify: `projects/mmdet3d_plugin/models/dense_heads/sparse_head.py`
- Modify: `projects/mmdet3d_plugin/datasets/nuscenes_dataset.py`
- Modify: `projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e.py`
- Create: `projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_router.py`

- [ ] **Step 1: Add routing kwargs to `SparseHead`**

Keep defaults disabled. Store latest routing metadata for inspection.

- [ ] **Step 2: Apply dense budget masking after feature flattening**

When enabled, rank tokens by feature norm, choose keep count from ego speed if present, and zero-fill or ambient-fill dropped token positions without changing `spatial_flatten`.

- [ ] **Step 3: Add ego-speed plumbing**

Add `ego_speed` to dataset info and RepDETR3D collect keys so routing configs can condition budgets on speed.

- [ ] **Step 4: Add opt-in router config**

Create a config inheriting the EVA02 RepDETR3D baseline and enabling `routing_mode='ambient_pool'` with speed-conditioned ratios.

### Task 3: Docs

**Files:**
- Create: `docs/sepatch3d/tri_state_token_routing_roadmap.md`
- Create: `docs/sepatch3d/token_routing_for_repdetr3d.md`

- [ ] **Step 1: Copy and sanitize relevant SEPatch3D routing docs**

Keep the fixed-token tri-state idea. Remove SEPatch3D-specific paths and archived variable patch-size instructions.

- [ ] **Step 2: Document the RepDETR3D limitation**

State that the first hook preserves dense geometry for deformable attention and does not reduce CUDA attention sequence length until the attention op gains sparse token indexing.

### Task 4: Verification

- [ ] **Step 1: Run focused tests**

Run: `python -m pytest mmdetection3d/tests/test_models/test_tri_state_router.py -q`
Expected: pass.

- [ ] **Step 2: Compile touched Python files**

Run: `python -m py_compile projects/mmdet3d_plugin/models/routing/tri_state_router.py projects/mmdet3d_plugin/models/dense_heads/sparse_head.py projects/mmdet3d_plugin/datasets/nuscenes_dataset.py`
Expected: exit code 0.

- [ ] **Step 3: Inspect diff**

Run: `git diff -- projects/mmdet3d_plugin/models/routing projects/mmdet3d_plugin/models/dense_heads/sparse_head.py projects/mmdet3d_plugin/datasets/nuscenes_dataset.py projects/configs/RepDETR3D docs/sepatch3d mmdetection3d/tests/test_models/test_tri_state_router.py`
Expected: only router migration changes.
