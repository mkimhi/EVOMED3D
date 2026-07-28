# Temporal Depth and Mixture-of-Depths Routers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-sample L8/L12/L16/L24 detection-plus-budget router and a 50%-capacity global-block MoD router to temporal-memory RepDETR3D EVA-02.

**Architecture:** Each routable sample first computes L8, then a hard router uses pooled L8, cached pre-PE memory, and online metadata to choose exactly one depth. MoD keeps L1--L8/L24 and all window blocks dense, routing half of every camera's tokens at global L9/L12/L15/L18/L21. Cache state is per sample, not per GPU batch.

**Tech Stack:** Python 3.8, PyTorch 1.13, MMCV 1.7, EVA-02 ViT, RepDETR3D, pytest.

## Global Constraints

- Disabled temporal memory and forced L24 exactly preserve the full detector feature path.
- Training batch size four emits four independent sample actions; actions are shared only across cameras belonging to the same sample.
- Router evaluation/test sets `data.samples_per_gpu=1`; training batch size remains inherited from the base config.
- Router inputs are online-only L8, cached memory, timestamp/frame age, and ego metadata. No GT, annotations, offline data, or drift target is allowed.
- Depth costs are `[8,12,16,24]/24`, target cost is `0.50`, and reset-5 forces L24. Router loss is detection plus cost/balance loss only.
- L8/L12/L16 retain the existing layer-specific projectors/updaters and L24 retains the existing full path; the depth router adds no exit adapters.
- MoD capacity is exactly 50% per camera at global blocks `(9,12,15,18,21)`; top-k, pack/scatter, adapters, and router overhead are part of latency.

---

### Task 1: Create depth-router and batch-cache primitives

**Files:**
- Create: `projects/mmdet3d_plugin/models/temporal_memory/routers.py`
- Modify: `projects/mmdet3d_plugin/models/temporal_memory/controller.py`
- Modify: `projects/mmdet3d_plugin/models/temporal_memory/__init__.py`
- Create: `tests/test_temporal_memory_routers.py`
- Modify: `tests/test_temporal_memory_controller.py`

**Interfaces:**
- `DepthExitRouter(feature_dim, metadata_dim, hidden_dim)` returns `DepthRouteOutput(logits, probabilities, hard_actions[B], hard_one_hot[B,4], expected_cost)`.
- `depth_router_losses(route, target_cost, balance_prior, balance_weight)` returns `loss_tm_budget`, `loss_tm_balance`, and expected cost.
- `TemporalMemoryController.get_or_update_samples(requests, training=False)` owns `states[sample_key]` with independent `scene_token`, `frame_index`, `frames_since_full`, and memory.

- [ ] **Step 1: Write failing tests**

```python
def test_router_has_one_action_and_budget_grad_per_sample():
    r = DepthExitRouter(4, 3, 8)
    out = r(torch.randn(4, 6, 4, 2, 2), torch.randn(4, 6, 4, 2, 2), torch.randn(4, 3), training=True)
    assert out.hard_actions.shape == (4,)
    assert out.hard_one_hot.sum(1).tolist() == [1., 1., 1., 1.]
    loss = depth_router_losses(out, .5, (.25, .25, .25, .25), .1)
    (loss['loss_tm_budget'] + loss['loss_tm_balance']).backward()
    assert any(p.grad is not None for p in r.parameters())

def test_controller_keeps_memory_separate_per_sample_key():
    c = TemporalMemoryController(enabled=True, updater='partial_only', max_frames_between_full=5)
    c.get_or_update_samples([request('a', 'L24', 1.), request('b', 'L24', 2.)])
    out = c.get_or_update_samples([request('a', 'L8', 3.), request('b', 'L8', 4.)])
    assert [x.item() for x in out] == [3., 4.]
```

- [ ] **Step 2: Verify failure**

Run `conda run --no-capture-output -n token pytest tests/test_temporal_memory_routers.py tests/test_temporal_memory_controller.py -k 'router_has or separate_per_sample' -v`. Expected: FAIL because the new interfaces are missing.

- [ ] **Step 3: Implement**

Pool cameras/spatial tokens, project current and previous vectors, concatenate metadata, and use hard Gumbel-Softmax in training/argmax in evaluation. Expected cost uses soft probabilities; budget is squared target error; balance is batch-mean probability error and decays to zero. Do not calculate similarity, drift, GT, or classification-target loss. Refactor controller state so single-sample APIs wrap per-sample state; enforce reset-5 before honoring a partial request.

- [ ] **Step 4: Verify and commit**

Run `conda run --no-capture-output -n token pytest tests/test_temporal_memory_routers.py tests/test_temporal_memory_controller.py -v`. Expected: PASS. Commit with `git add projects/mmdet3d_plugin/models/temporal_memory/routers.py projects/mmdet3d_plugin/models/temporal_memory/controller.py projects/mmdet3d_plugin/models/temporal_memory/__init__.py tests/test_temporal_memory_routers.py tests/test_temporal_memory_controller.py && git commit -m "feat: add temporal router primitives and sample cache"`.

### Task 2: Continue EVA from L8 and integrate per-sample exits

**Files:**
- Modify: `projects/mmdet3d_plugin/models/backbones/eva_vit.py`
- Modify: `projects/mmdet3d_plugin/models/temporal_memory/adapters/repdetr3d_adapter.py`
- Modify: `projects/mmdet3d_plugin/models/detectors/repdetr3d.py`
- Modify: `projects/mmdet3d_plugin/models/temporal_memory/projectors.py`
- Modify: `tests/test_temporal_memory_partial_eva.py`
- Modify: `tests/test_temporal_memory_repdetr3d_routing.py`

**Interfaces:**
- `EVAViT.prepare_tokens(images)` and `EVAViT.forward_blocks(tokens, start_index=0, stop_at=None)` avoid recomputing patch/L1--L8 work.
- `_tm_route_depth_batch(frame_data, img_metas, training)` returns feature tensors in original batch order plus route and router losses.

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize('stop_at', [7, 11, 15, 23])
def test_eva_continuation_matches_direct_prefix(stop_at):
    b = make_small_eva(depth=24); image = torch.randn(2, 3, 64, 64)
    tokens = b.prepare_tokens(image); l8 = b.forward_blocks(tokens, 0, 7)
    assert torch.allclose(b.forward_blocks(l8, 8, stop_at), b.forward_blocks(tokens, 0, stop_at), atol=1e-5, rtol=1e-5)

def test_depth_groups_restore_original_batch_order_and_reject_gt():
    model = make_router_stub(actions=['L8', 'L12', 'L24', 'L16'])
    output, route, losses = model._tm_route_depth_batch(batch_of_four(), metas_of_four(), training=True)
    assert route.hard_actions.tolist() == [0, 1, 3, 2]
    assert [item['id'] for item in output] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match='online-only'):
        model._tm_depth_router_inputs(l8(), memory(), {'gt_bboxes_3d': torch.ones(1)})
```

- [ ] **Step 2: Verify failure**

Run `conda run --no-capture-output -n token pytest tests/test_temporal_memory_partial_eva.py tests/test_temporal_memory_repdetr3d_routing.py -k 'continuation or depth_groups' -v`. Expected: FAIL because continuation and grouped routing are missing.

- [ ] **Step 3: Implement**

Move patch/absolute-position setup into `prepare_tokens`; retain `_forward_blocks` as a backward-compatible wrapper. Map L8/L12/L16 to stops 7/11/15. In RepDETR3D, extract L8 for all B samples, build metadata exclusively from online scalar keys, look up cache states, route, execute equal-depth groups, select the existing layer-specific partial projector by action, and scatter results. L24 bypasses projectors and uses the original full path. Add `freeze_backbone_only=True` that freezes only EVA while head/router/projector/updater stay trainable; preserve existing `freeze_pretrained_detector` behavior. Return budget every iteration and a graph-connected zero balance loss after warm-up.

- [ ] **Step 4: Verify and commit**

Run `conda run --no-capture-output -n token pytest tests/test_temporal_memory_partial_eva.py tests/test_temporal_memory_adapters.py tests/test_temporal_memory_repdetr3d_routing.py tests/test_temporal_memory_projector.py -v`. Expected: PASS. Commit with `git add projects/mmdet3d_plugin/models/backbones/eva_vit.py projects/mmdet3d_plugin/models/temporal_memory/adapters/repdetr3d_adapter.py projects/mmdet3d_plugin/models/detectors/repdetr3d.py projects/mmdet3d_plugin/models/temporal_memory/projectors.py tests/test_temporal_memory_partial_eva.py tests/test_temporal_memory_repdetr3d_routing.py && git commit -m "feat: route temporal depth per sample"`.

### Task 3: Implement packed token routing at global EVA blocks

**Files:**
- Modify: `projects/mmdet3d_plugin/models/temporal_memory/routers.py`
- Modify: `projects/mmdet3d_plugin/models/backbones/eva_vit.py`
- Create: `tests/test_temporal_memory_mod_router.py`

**Interfaces:**
- `TokenDepthRouter(channels, hidden_channels)` scores dense `[BN,H,W,C]` tokens.
- `select_topk_tokens(scores, capacity_ratio=.5)` returns `[BN,K]` indices and `[BN,N]` mask where `K=ceil(.5*N)`.
- `VisionRotaryEmbeddingFast.forward_indices(tokens, indices)` applies RoPE at original flattened positions.
- `Block.forward_mod(x, router, adapter, capacity_ratio)` supports only global blocks.

- [ ] **Step 1: Write failing tests**

```python
def test_mod_capacity_is_exactly_half_per_camera():
    indices, mask = select_topk_tokens(torch.tensor([[0., 3., 1., 2.], [4., 1., 3., 2.]]), .5)
    assert indices.shape == (2, 2)
    assert mask.sum(1).tolist() == [2, 2]

def test_mod_bypasses_unselected_tokens_and_rejects_window_blocks():
    x = torch.randn(1, 2, 2, 8)
    out, _ = make_global_block().forward_mod(x, FixedTokenRouter([[1, 3]]), TokenDepthAdapter(8, 4), .5)
    assert torch.equal(out.reshape(1, 4, 8)[:, [0, 2]], x.reshape(1, 4, 8)[:, [0, 2]])
    with pytest.raises(ValueError, match='global-attention'):
        make_window_block().forward_mod(x, TokenDepthRouter(8, 4), TokenDepthAdapter(8, 4), .5)
```

- [ ] **Step 2: Verify failure**

Run `conda run --no-capture-output -n token pytest tests/test_temporal_memory_mod_router.py -v`. Expected: FAIL because MoD primitives do not exist.

- [ ] **Step 3: Implement**

Flatten each camera grid, select fixed top-k, gather selected tokens plus original positions, apply indexed RoPE in a factored global attention sequence helper, run selected attention/MLP, multiply selected updates by router scores for detection-loss gradient, apply zero-initialized token adapter, and scatter into a clone of the input residual state. Reject `window_size > 0`; skipped tokens equal their input residual state. Register `EVAViT.configure_temporal_mod(routed_block_indices=(8,11,14,17,20), capacity_ratio=.5, router_hidden_channels=256, adapter_hidden_channels=256)`, `clear_temporal_mod`, and stats. Use `forward_mod` only on these global blocks; dense path is unchanged when disabled.

- [ ] **Step 4: Verify and commit**

Run `conda run --no-capture-output -n token pytest tests/test_temporal_memory_mod_router.py tests/test_temporal_memory_partial_eva.py -v`. Expected: PASS. Commit with `git add projects/mmdet3d_plugin/models/temporal_memory/routers.py projects/mmdet3d_plugin/models/backbones/eva_vit.py tests/test_temporal_memory_mod_router.py && git commit -m "feat: add global-block temporal MoD routing"`.

### Task 4: Add matched configs and result rows

**Files:**
- Create: `projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_depth_router_budget50_reset5.py`
- Create: `projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_mod_global50_frozen.py`
- Create: `projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_mod_global50_frozen_adapter_warmup.py`
- Create: `projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_tm_mod_global50_joint.py`
- Modify: `RESULTS.md`
- Modify: `tests/test_temporal_memory_config.py`

- [ ] **Step 1: Write failing config tests**

```python
def test_router_configs_are_online_only_reset5_and_single_sample_eval():
    cfg = Config.fromfile(DEPTH_ROUTER_CONFIG)
    assert cfg.model.temporal_memory.max_frames_between_full == 5
    assert cfg.router_eval_cfg.data.samples_per_gpu == 1
    assert 'gt_bboxes_3d' not in cfg.model.temporal_memory.router.feature_names

def test_mod_configs_differ_only_by_backbone_training_regime():
    frozen, joint = Config.fromfile(MOD_FROZEN), Config.fromfile(MOD_JOINT)
    assert frozen.model.temporal_memory.mod_router.routed_blocks == (9, 12, 15, 18, 21)
    assert frozen.model.temporal_memory.mod_router.capacity_ratio == .5
    assert frozen.model.temporal_memory.mod_router.train_backbone is False
    assert joint.model.temporal_memory.mod_router.train_backbone is True
```

- [ ] **Step 2: Verify failure**

Run `conda run --no-capture-output -n token pytest tests/test_temporal_memory_config.py -k 'router_configs or mod_configs' -v`. Expected: FAIL because configs do not exist.

- [ ] **Step 3: Implement configs and docs**

All configs load `./ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth`, set reset-5, online-only input names, and `router_eval_cfg.data.samples_per_gpu=1`. The depth router uses target .50 with the existing L8/L12/L16 projectors and no exit-adapter warm-up. MoD configs use global blocks `(9,12,15,18,21)`, capacity .50, frozen/no-warm-up, frozen/warm-up, and joint N-vium-style warm-up variants. Append five blank-metric `RESULTS.md` rows stating no GT/no drift target, unbatched evaluation, and exact commands ending `--cfg-options data.samples_per_gpu=1 model.tm_debug_latency=True`.

- [ ] **Step 4: Verify and commit**

Run `conda run --no-capture-output -n token pytest tests/test_temporal_memory_config.py -v`. Expected: PASS. Commit all new configs, `RESULTS.md`, and config tests with message `feat: add temporal router experiment configs`.

### Task 5: Verify full-path compatibility and smoke-run both families

**Files:**
- Modify: `tests/test_temporal_memory_repdetr3d_routing.py`

- [ ] **Step 1: Write failing L24-equivalence test**

```python
def test_forced_l24_depth_router_matches_full_features():
    model = make_small_repdetr_with_depth_router(force_action='L24')
    routed = model._tm_route_depth_batch(batch_of_one(), metas_of_one(), training=False)[0]
    full = model._profile_extract_full_features(batch_of_one(), len_queue=1, training_mode=False)
    assert_features_close(routed, full, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Verify, smoke test, and commit**

Run all focused tests: `conda run --no-capture-output -n token pytest tests/test_temporal_memory_routers.py tests/test_temporal_memory_mod_router.py tests/test_temporal_memory_controller.py tests/test_temporal_memory_partial_eva.py tests/test_temporal_memory_repdetr3d_routing.py tests/test_temporal_memory_config.py -v`. Expected: PASS.

Run one GPU training iteration of the primary depth and frozen-MoD configs with `CUDA_VISIBLE_DEVICES=0`, `tools/dist_train.sh`, `runner.max_iters=1`, and separate `work_dirs/tm_depth_router_smoke` / `work_dirs/tm_mod_router_smoke`. Evaluate resulting checkpoints with `data.samples_per_gpu=1` and `model.tm_debug_latency=True`; logs must show action counts plus expected/realized depth cost, or fixed MoD capacity, and timing components. Commit the final compatibility test with message `test: verify temporal router compatibility`.

## Plan Self-Review

- Tasks 1--2 cover per-sample cache, depth routes, online-only inputs, reset-5, the existing L8/L12/L16 projectors, and detection-plus-budget training.
- Task 3 covers exact 50% token MoD only where EVA uses global attention; external config indices `(9,12,15,18,21)` map to internal `(8,11,14,17,20)`.
- Tasks 4--5 cover frozen versus joint experiments, requested result rows, single-sample evaluation, baseline checks, and bounded GPU verification.
