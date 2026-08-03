# Inference Example Export Design

## Goal

Add one minimal command-line script that runs either the full baseline or a temporal-memory experiment on short nuScenes excerpts and exports paper-ready input/output examples. The default export contains the first five frames from each of the first three distinct validation scenes. Ten frames per scene are supported by a command-line option.

## Interface

The script will be invoked as:

```bash
python tools/export_inference_examples.py CONFIG CHECKPOINT
```

Optional arguments will include:

- `--output-dir`, defaulting to `inferece_example` as requested;
- `--num-scenes`, defaulting to `3`;
- `--frames-per-scene`, defaulting to `5` and accepting `10`;
- `--feature-layer`, defaulting to EVA block `8`;
- `--score-thr`, controlling displayed prediction confidence;
- `--cfg-options`, matching the existing test script's configuration overrides.

Any compatible baseline or temporal-memory config/checkpoint pair may be supplied. The exporter will not contain hard-coded model weights or detector-specific temporal policy logic.

## Sample Selection and Temporal Semantics

The exporter will inspect the validation dataset's ordered `data_infos`, identify the first three distinct scene tokens, and select the first requested number of frames from each scene. Inference remains chronological within each scene so a temporal model's memory and policy state evolve normally. Scene boundaries already present in the model metadata trigger the normal memory reset behavior.

The selected records will be exposed through a shallow dataset copy rather than by rewriting the source dataset. This retains the repository's existing preprocessing and result-formatting behavior while avoiding inference over complete scenes.

## Exported Artifacts

All images will be placed directly in one flat directory, `inferece_example/`. Each frame produces four PNG files with a shared stem such as `scene00_frame00`:

1. `scene00_frame00_input.png`: a labeled 2-by-3 mosaic of all six synchronized nuScenes camera images in canonical order: front-left, front, front-right, back-left, back, and back-right.
2. `scene00_frame00_gt.png`: the existing nuScenes multi-camera and bird's-eye-view ground-truth rendering.
3. `scene00_frame00_pred.png`: the same rendering using model predictions above the configured score threshold.
4. `scene00_frame00_feature_L8.png`: a labeled 2-by-3 heatmap mosaic, one panel per camera, made from the mean absolute channel activation at the requested EVA block.

A compact `manifest.json` will record the config, checkpoint, feature layer, sample and scene tokens, source image paths, and policy action when the model exposes one. The manifest is supporting metadata; all visual artifacts remain in the single requested directory.

## Model and Feature Capture

The script will reuse the model and dataset construction used by `tools/test.py`: load the MMCV config, import the project plugin, build the validation dataset and sequential batch-size-one loader, build the detector, load the checkpoint, and run evaluation under `torch.no_grad()`.

An ordinary PyTorch forward hook on `model.img_backbone.blocks[feature_layer - 1]` will capture the requested intermediate EVA activation without changing baseline or temporal detector code and without a second backbone pass. EVA block outputs have shape `[B * N_camera, H, W, C]`. The visualization will average absolute activation over channels, normalize each camera map robustly, and apply a standard perceptually ordered colormap. The hook will be removed in a `finally` block.

If the supplied backbone does not expose EVA-style `blocks`, the script will stop with an actionable message instead of silently exporting the wrong representation. If the hook fires more than once for one sample, the first captured activation will be used and the event will be noted in the manifest.

## Prediction and Ground-Truth Rendering

Predictions will be collected in dataset order and converted with the dataset's existing nuScenes result formatter. The existing `tools/visual_nuscenes.py` renderer will then generate matched GT and prediction views for every selected sample token. Input mosaics will read the six original camera files listed in the selected `data_infos`, preserving raw visual content rather than inverse-normalizing model tensors.

Temporary formatted-result data may be created inside the output directory while rendering, then retained only when needed by the renderer or manifest. Existing user files will not be deleted. The script will create missing output directories and overwrite only files with the exact frame stems it is generating.

## Error Handling

The command will validate the config, checkpoint, nuScenes data files, CUDA availability, requested scene/frame counts, six-camera availability, and feature-layer range before expensive inference where possible. It will report when fewer than the requested number of frames exist in a scene and export the available frames. A failed render will identify the sample token and artifact type.

## Verification

Minimal verification will include:

- Python byte-compilation;
- CLI `--help` execution;
- a small test for selecting the first `N` frames from three scene tokens;
- a small test for converting six synthetic EVA feature tensors into a six-panel heatmap mosaic;
- when local data, checkpoint, and CUDA are usable, one smoke export with one scene and one frame, followed by checks that all four images and the manifest exist and are non-empty.

No new package will be introduced. The implementation will reuse PyTorch, MMCV, Pillow, Matplotlib, and the nuScenes utilities already required by this repository.
