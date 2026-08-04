# Inference Example Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one CLI that exports three short nuScenes scene excerpts as six-camera inputs, GT, predictions, and EVA feature heatmaps for either a baseline or temporal checkpoint.

**Architecture:** A single script reuses the existing MMCV model/dataset construction and nuScenes formatter/renderer. Small pure helpers select chronological frames and construct 2-by-3 mosaics; a forward hook captures one requested EVA block during the same inference pass.

**Tech Stack:** Python, PyTorch, MMCV/MMDetection3D, Pillow, NumPy, Matplotlib, existing nuScenes renderer.

## Global Constraints

- Default output directory is exactly `inferece_example`.
- Default selection is the first five frames from each of the first three distinct validation scenes; ten frames is configurable.
- Every input and feature artifact contains all six synchronized cameras.
- Support both baseline and temporal-memory configs without modifying detector code.
- Add no dependency.
- Keep all generated images in one flat output directory.

---

### Task 1: Selection and six-camera visualization helpers

**Files:**
- Create: `tests/test_export_inference_examples.py`
- Create: `tools/export_inference_examples.py`

**Interfaces:**
- Produces: `select_scene_indices(data_infos, num_scenes, frames_per_scene) -> list[tuple[int, int, int]]`, where tuples are `(dataset_index, scene_ordinal, frame_ordinal)`.
- Produces: `activation_to_uint8_maps(feature, num_cameras=6) -> list[np.ndarray]`.
- Produces: `save_six_panel(images, labels, output_path, heatmap=False) -> None`.

- [ ] **Step 1: Write failing tests for scene selection and feature conversion**

```python
def test_select_scene_indices_takes_first_frames_from_distinct_scenes():
    infos = [{'scene_token': s} for s in ['a', 'a', 'a', 'b', 'b', 'c', 'c']]
    assert select_scene_indices(infos, 3, 2) == [
        (0, 0, 0), (1, 0, 1), (3, 1, 0),
        (4, 1, 1), (5, 2, 0), (6, 2, 1)]

def test_activation_to_uint8_maps_keeps_all_six_cameras():
    feature = torch.arange(6 * 2 * 3 * 4).reshape(6, 2, 3, 4).float()
    maps = activation_to_uint8_maps(feature)
    assert len(maps) == 6
    assert all(x.shape == (2, 3) and x.dtype == np.uint8 for x in maps)
```

- [ ] **Step 2: Run tests and confirm the imports fail because the script does not exist**

Run: `python -m pytest tests/test_export_inference_examples.py -q`
Expected: collection error naming `tools.export_inference_examples`.

- [ ] **Step 3: Implement the smallest pure helpers and a six-panel Pillow renderer**

```python
def select_scene_indices(data_infos, num_scenes, frames_per_scene):
    selected, scenes, counts = [], {}, {}
    for index, info in enumerate(data_infos):
        token = info['scene_token']
        if token not in scenes:
            if len(scenes) == num_scenes:
                continue
            scenes[token] = len(scenes)
            counts[token] = 0
        if counts[token] < frames_per_scene:
            selected.append((index, scenes[token], counts[token]))
            counts[token] += 1
    return selected
```

Normalize channel-mean absolute activation independently per camera with percentile clipping, convert it to `uint8`, colorize heatmaps with Pillow, and build a labeled 2-by-3 canvas.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_export_inference_examples.py -q`
Expected: all tests pass.

### Task 2: Integrated model inference and nuScenes rendering CLI

**Files:**
- Modify: `tools/export_inference_examples.py`
- Modify: `tests/test_export_inference_examples.py`

**Interfaces:**
- Consumes: Task 1 helpers.
- Produces: `main()` and CLI arguments `config`, `checkpoint`, `--output-dir`, `--num-scenes`, `--frames-per-scene`, `--feature-layer`, `--score-thr`, and `--cfg-options`.
- Produces: four PNGs per selected frame plus `manifest.json`.

- [ ] **Step 1: Write failing validation and artifact-naming tests**

```python
def test_validate_args_rejects_nonpositive_counts():
    with pytest.raises(ValueError, match='num-scenes'):
        validate_export_counts(0, 5)

def test_artifact_stem_is_stable():
    assert artifact_stem(2, 4) == 'scene02_frame04'
```

- [ ] **Step 2: Run tests and confirm failure from missing behavior**

Run: `python -m pytest tests/test_export_inference_examples.py -q`
Expected: failure naming `validate_export_counts` or `artifact_stem`.

- [ ] **Step 3: Implement the CLI and one-pass exporter**

Load config/plugin/model/checkpoint using the same calls as `tools/test.py`; make a shallow dataset copy containing selected `data_infos`; force sequential batch-size-one inference; register a hook on `img_backbone.blocks[feature_layer - 1]`; collect predictions, raw-image mosaics, feature mosaics, and controller `last_action`; format predictions with `dataset.format_results`; render paired GT/pred views with `tools.visual_nuscenes.NuScenes`; write `manifest.json`; remove the hook in `finally`.

Validation must fail before inference if paths, CUDA, six cameras, scene count, or feature layer are invalid. Existing unrelated files in the output directory remain untouched.

- [ ] **Step 4: Run tests, compilation, and CLI checks**

Run: `python -m pytest tests/test_export_inference_examples.py -q`
Expected: all tests pass.

Run: `python -m py_compile tools/export_inference_examples.py tests/test_export_inference_examples.py`
Expected: exit 0.

Run: `python tools/export_inference_examples.py --help`
Expected: exit 0 and all documented options present.

- [ ] **Step 5: Run a one-frame smoke export when CUDA, data, and checkpoint are available**

Run:

```bash
python tools/export_inference_examples.py \
  projects/configs/RepDETR3D/repdetr3d_eva02_800_bs4_seq_24e.py \
  ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth \
  --num-scenes 1 --frames-per-scene 1 \
  --output-dir inferece_example
```

Expected: four non-empty PNGs and one valid manifest for `scene00_frame00`.

- [ ] **Step 6: Inspect scope and diff**

Run: `git diff --check -- tools/export_inference_examples.py tests/test_export_inference_examples.py`
Expected: exit 0. Confirm no baseline or temporal detector source was modified.
