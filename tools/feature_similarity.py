#!/usr/bin/env python
import argparse
import csv
import json
import os
import os.path as osp
import sys
from collections import defaultdict
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from scipy.stats import rankdata

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

import projects.mmdet3d_plugin  # noqa: F401,E402
from projects.mmdet3d_plugin.datasets.builder import build_dataloader  # noqa: E402
from projects.mmdet3d_plugin.models.temporal_memory.partial_features import resolve_eva_layer_index  # noqa: E402
from tools.eval_utils import limit_dataset  # noqa: E402


DEFAULT_CAMERA_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


def flatten_feature_list(features):
    tokens = []
    for feature in features:
        if feature.dim() != 5:
            raise ValueError(
                'feature tensors must have shape [B, N_cam, C, H, W], '
                f'got {tuple(feature.shape)}')
        b, n_cam, c, h, w = feature.shape
        tokens.append(feature.permute(0, 1, 3, 4, 2).reshape(b * n_cam * h * w, c))
    return torch.cat(tokens, dim=0)


def cosine_similarity_maps(student_features, teacher_features):
    maps = {}
    for level, (student, teacher) in enumerate(zip(student_features, teacher_features)):
        if student.shape != teacher.shape:
            raise ValueError(
                f'feature level {level} shape mismatch: '
                f'{tuple(student.shape)} vs {tuple(teacher.shape)}')
        cosine = F.cosine_similarity(student.float(), teacher.float(), dim=2)
        maps[f'level{level}'] = cosine.detach().cpu()
    return maps


def linear_cka(x, y, eps=1e-12):
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = torch.linalg.matrix_norm(x.T @ y, ord='fro') ** 2
    norm_x = torch.linalg.matrix_norm(x.T @ x, ord='fro')
    norm_y = torch.linalg.matrix_norm(y.T @ y, ord='fro')
    return hsic / (norm_x * norm_y + eps)


def pearsonr(x, y):
    x = torch.as_tensor(x, dtype=torch.float32).flatten()
    y = torch.as_tensor(y, dtype=torch.float32).flatten()
    if x.numel() < 2 or y.numel() < 2:
        return float('nan')
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) == 0.0:
        return float('nan')
    return float((x @ y / denom).clamp(-1, 1))


def spearmanr(x, y):
    x_rank = torch.from_numpy(rankdata(torch.as_tensor(x).detach().cpu().numpy()))
    y_rank = torch.from_numpy(rankdata(torch.as_tensor(y).detach().cpu().numpy()))
    return pearsonr(x_rank, y_rank)


def rsm_upper_triangle(features, max_tokens=1024):
    features = features.float()
    if features.size(0) > max_tokens:
        indices = torch.linspace(0, features.size(0) - 1, steps=max_tokens).long()
        features = features[indices]
    features = F.normalize(features, dim=1)
    rsm = features @ features.T
    rows, cols = torch.triu_indices(rsm.size(0), rsm.size(1), offset=1)
    return rsm[rows, cols]


def temporal_feature_pair(features, gap, anchor=0):
    anchor_features = []
    future_features = []
    for feature in features:
        if feature.dim() != 6:
            raise ValueError(
                'temporal features must have shape [B, T, N_cam, C, H, W], '
                f'got {tuple(feature.shape)}')
        if anchor + gap >= feature.size(1):
            raise ValueError(
                f'gap {gap} from anchor {anchor} exceeds temporal length {feature.size(1)}')
        anchor_features.append(feature[:, anchor])
        future_features.append(feature[:, anchor + gap])
    return anchor_features, future_features


def limit_dataset_by_scenes(dataset, num_scenes):
    if num_scenes is None:
        return dataset
    if num_scenes <= 0:
        raise ValueError('--num-videos must be a positive integer')
    if not hasattr(dataset, 'data_infos'):
        return dataset

    scene_tokens = []
    keep_indices = []
    for index, info in enumerate(dataset.data_infos):
        scene_token = info.get('scene_token') if isinstance(info, dict) else None
        if scene_token is None:
            continue
        if scene_token not in scene_tokens:
            if len(scene_tokens) >= num_scenes:
                break
            scene_tokens.append(scene_token)
        if scene_token in scene_tokens:
            keep_indices.append(index)

    dataset.data_infos = [dataset.data_infos[index] for index in keep_indices]
    if hasattr(dataset, 'flag'):
        dataset.flag = dataset.flag[keep_indices]
    return dataset


def resize_mask(mask, height, width):
    mask = mask.float()
    if mask.shape[-2:] == (height, width):
        return mask.bool()
    flat = mask.reshape(-1, 1, mask.size(-2), mask.size(-1))
    resized = F.interpolate(flat, size=(height, width), mode='nearest')
    return resized.reshape(*mask.shape[:-2], height, width).bool()


def center_mask(batch, n_cam, height, width, border_ratio=0.25):
    mask = torch.zeros(batch, n_cam, height, width, dtype=torch.bool)
    y0 = int(height * border_ratio)
    y1 = max(y0 + 1, int(height * (1.0 - border_ratio)))
    x0 = int(width * border_ratio)
    x1 = max(x0 + 1, int(width * (1.0 - border_ratio)))
    mask[:, :, y0:y1, x0:x1] = True
    return mask


def _masked_mean(values, mask):
    mask = mask.to(values.device)
    if mask.sum().item() == 0:
        return float('nan')
    return float(values[mask].float().mean())


def _feature_tokens_for_mask(features, mask):
    chunks = []
    for feature in features:
        b, n_cam, c, h, w = feature.shape
        level_mask = resize_mask(mask, h, w).to(feature.device)
        tokens = feature.permute(0, 1, 3, 4, 2)[level_mask]
        if tokens.numel() > 0:
            chunks.append(tokens.reshape(-1, c))
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


def compare_feature_sets(
    student_features,
    teacher_features,
    camera_names=None,
    masks=None,
    max_rsm_tokens=1024,
):
    camera_names = camera_names or DEFAULT_CAMERA_NAMES
    masks = masks or {}
    cosine_maps = cosine_similarity_maps(student_features, teacher_features)
    metrics = {}

    all_cosine = torch.cat([value.reshape(-1) for value in cosine_maps.values()])
    metrics['cosine/global_mean'] = float(all_cosine.float().mean())

    heatmaps = {}
    first_feature = student_features[0]
    batch, n_cam, _, first_h, first_w = first_feature.shape
    masks = dict(masks)
    masks.setdefault('center', center_mask(batch, n_cam, first_h, first_w))
    masks.setdefault('border', ~masks['center'])

    for level_name, values in cosine_maps.items():
        # [B, N, H, W] -> [N, H, W]
        heatmaps[level_name] = values.float().mean(dim=0).numpy()
        _, n_cam, h, w = values.shape
        for cam_idx in range(n_cam):
            cam_name = camera_names[cam_idx] if cam_idx < len(camera_names) else f'CAM_{cam_idx}'
            metrics[f'cosine/{cam_name}_mean'] = float(values[:, cam_idx].float().mean())
        spatial = values.float().mean(dim=(0, 1))
        for y in range(h):
            for x in range(w):
                metrics[f'cosine/spatial_y{y}_x{x}'] = float(spatial[y, x])

        center = center_mask(values.size(0), n_cam, h, w)
        metrics[f'cosine/{level_name}_center_mean'] = _masked_mean(values, center)
        metrics[f'cosine/{level_name}_border_mean'] = _masked_mean(values, ~center)

    for mask_name, mask in masks.items():
        mask = mask.detach().cpu().bool()
        masked_values = []
        inverse_values = []
        for values in cosine_maps.values():
            level_mask = resize_mask(mask, values.size(-2), values.size(-1))
            masked_values.append(values[level_mask])
            inverse_values.append(values[~level_mask])
        masked_values = torch.cat([x.reshape(-1) for x in masked_values if x.numel() > 0])
        inverse_values = torch.cat([x.reshape(-1) for x in inverse_values if x.numel() > 0])
        metrics[f'cosine/{mask_name}_mean'] = (
            float(masked_values.float().mean()) if masked_values.numel() else float('nan'))
        metrics[f'cosine/not_{mask_name}_mean'] = (
            float(inverse_values.float().mean()) if inverse_values.numel() else float('nan'))
        if mask_name == 'foreground':
            metrics['cosine/background_mean'] = metrics[f'cosine/not_{mask_name}_mean']

    student_tokens = flatten_feature_list(student_features).detach().cpu()
    teacher_tokens = flatten_feature_list(teacher_features).detach().cpu()
    metrics['cka/global'] = float(linear_cka(student_tokens, teacher_tokens))
    student_rsm = rsm_upper_triangle(student_tokens, max_tokens=max_rsm_tokens)
    teacher_rsm = rsm_upper_triangle(teacher_tokens, max_tokens=max_rsm_tokens)
    metrics['rsm/pearson'] = pearsonr(student_rsm, teacher_rsm)
    metrics['rsm/spearman'] = spearmanr(student_rsm, teacher_rsm)

    for cam_idx in range(n_cam):
        cam_name = camera_names[cam_idx] if cam_idx < len(camera_names) else f'CAM_{cam_idx}'
        student_cam = [feature[:, cam_idx:cam_idx + 1] for feature in student_features]
        teacher_cam = [feature[:, cam_idx:cam_idx + 1] for feature in teacher_features]
        st = flatten_feature_list(student_cam).detach().cpu()
        tt = flatten_feature_list(teacher_cam).detach().cpu()
        metrics[f'cka/{cam_name}'] = float(linear_cka(st, tt))
        metrics[f'rsm/{cam_name}_pearson'] = pearsonr(
            rsm_upper_triangle(st, max_tokens=max_rsm_tokens),
            rsm_upper_triangle(tt, max_tokens=max_rsm_tokens))
        metrics[f'rsm/{cam_name}_spearman'] = spearmanr(
            rsm_upper_triangle(st, max_tokens=max_rsm_tokens),
            rsm_upper_triangle(tt, max_tokens=max_rsm_tokens))

    for mask_name, mask in masks.items():
        st = _feature_tokens_for_mask(student_features, mask)
        tt = _feature_tokens_for_mask(teacher_features, mask)
        if st is not None and tt is not None and st.size(0) >= 2 and tt.size(0) >= 2:
            metrics[f'cka/{mask_name}'] = float(linear_cka(st.detach().cpu(), tt.detach().cpu()))
        inv = ~mask.bool()
        st_bg = _feature_tokens_for_mask(student_features, inv)
        tt_bg = _feature_tokens_for_mask(teacher_features, inv)
        if st_bg is not None and tt_bg is not None and st_bg.size(0) >= 2 and tt_bg.size(0) >= 2:
            metrics[f'cka/not_{mask_name}'] = float(linear_cka(st_bg.detach().cpu(), tt_bg.detach().cpu()))
            if mask_name == 'foreground':
                metrics['cka/background'] = metrics[f'cka/not_{mask_name}']

    return metrics, heatmaps


def save_camera_heatmaps(
    heatmaps,
    out_dir,
    camera_names=None,
    summary=None,
    heatmap_vmin=0.45,
    heatmap_vmax=0.85,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    camera_names = camera_names or DEFAULT_CAMERA_NAMES

    for level_name, level_maps in heatmaps.items():
        np.save(out_dir / f'{level_name}.npy', level_maps)
        for cam_idx, heatmap in enumerate(level_maps):
            cam_name = camera_names[cam_idx] if cam_idx < len(camera_names) else f'CAM_{cam_idx}'
            path = out_dir / f'{level_name}_{cam_name}.png'
            plt.figure(figsize=(5, 4))
            plt.imshow(heatmap, cmap='viridis', vmin=heatmap_vmin, vmax=heatmap_vmax)
            plt.colorbar(label='cosine similarity')
            plt.title(f'{level_name} {cam_name}')
            plt.tight_layout()
            plt.savefig(path, dpi=140)
            plt.close()

    if summary is not None:
        serializable = {
            key: (None if isinstance(value, float) and np.isnan(value) else value)
            for key, value in summary.items()
        }
        (out_dir / 'summary.json').write_text(json.dumps(serializable, indent=2, sort_keys=True))
        with (out_dir / 'summary.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for key in sorted(serializable):
                writer.writerow([key, serializable[key]])


class FeatureAnalysisAdapter:
    def extract_teacher_features(self, model, data):
        raise NotImplementedError

    def extract_student_features(self, model, data, layer=None):
        raise NotImplementedError

    def extract_final_sequence_features(self, model, data, queue_length):
        raise NotImplementedError

    def camera_names(self, data):
        return DEFAULT_CAMERA_NAMES

    def masks(self, data, feature_shape):
        return {}


class RepDETR3DFeatureAnalysisAdapter(FeatureAnalysisAdapter):
    def _img(self, data):
        img = data['img']
        if isinstance(img, (list, tuple)):
            return img[0]
        return img

    def _sequence_img(self, data, queue_length):
        img = self._img(data)
        if torch.is_tensor(img) and img.dim() == 5 and img.size(0) == queue_length:
            img = img.unsqueeze(0)
        return img

    def _img_metas(self, data):
        metas = data.get('img_metas')
        if isinstance(metas, (list, tuple)):
            return metas[0]
        return metas

    def extract_teacher_features(self, model, data):
        return model._squeeze_temporal_feature_dim(
            model.extract_img_feat(self._img(data), len_queue=1, training_mode=True))

    def extract_final_sequence_features(self, model, data, queue_length):
        features = model.extract_img_feat(
            self._sequence_img(data, queue_length),
            len_queue=queue_length,
            training_mode=True)
        for feature in features:
            if feature.dim() != 6:
                raise ValueError(
                    'expected final sequence features with shape [B, T, N_cam, C, H, W], '
                    f'got {tuple(feature.shape)}')
        return features

    def extract_student_features(self, model, data, layer=None):
        if layer is not None:
            if not getattr(model, 'temporal_memory_enabled', False):
                raise ValueError('layer extraction requires a temporal-memory config with a projector')
            frame_data = dict(data)
            frame_data['img'] = self._img(data)
            partial = model._profile_extract_partial_features(frame_data, f'block{layer}')
            return model.temporal_memory_projector(partial)
        if getattr(model, 'temporal_memory_enabled', False):
            frame_data = dict(data)
            frame_data['img'] = self._img(data)
            return model._extract_img_feat_temporal_memory(self._img_metas(data), **frame_data)
        return self.extract_teacher_features(model, data)

    def extract_backbone_layer_features(self, model, data, layer, depth=24):
        img = self._img(data)
        original_img = img
        if img.dim() == 6:
            img = img.flatten(1, 2)
        if img.dim() == 5 and img.size(0) == 1:
            img = img.squeeze(0)
        elif img.dim() == 5 and img.size(0) > 1:
            b, n_cam, c, h, w = img.size()
            img = img.reshape(b * n_cam, c, h, w)
        if getattr(model, 'use_grid_mask', False):
            img = model.grid_mask(img)

        layer_index = depth - 1 if layer == 'final' else resolve_eva_layer_index(layer, depth=depth)
        feature = model.img_backbone.forward_intermediate(img, layer_index)
        if feature.dim() != 4:
            raise ValueError(
                'raw backbone features must have shape [B*N_cam, C, H, W], '
                f'got {tuple(feature.shape)}')
        b = original_img.size(0)
        c, h, w = feature.shape[1:]
        if original_img.dim() == 6:
            t, n_cam = original_img.size(1), original_img.size(2)
            feature = feature.view(b * t, n_cam, c, h, w)
        else:
            n_cam = int(feature.size(0) / b)
            feature = feature.view(b, n_cam, c, h, w)
        return [feature]

    def _image_hw(self, data):
        img = self._img(data)
        if torch.is_tensor(img):
            return int(img.shape[-2]), int(img.shape[-1])
        return None

    def frame_data(self, data, frame_index):
        sliced = dict(data)
        img = self._img(data)
        if torch.is_tensor(img) and img.dim() == 6:
            sliced['img'] = img[:, frame_index]
        for key in ['gt_bboxes', 'depths', 'gt_labels', 'centers2d']:
            if key in data:
                sliced[key] = self._frame_value(data[key], frame_index)
        return sliced

    def _frame_value(self, value, frame_index):
        if torch.is_tensor(value):
            if value.dim() >= 2 and value.size(1) > frame_index:
                return value[:, frame_index]
            return value
        if isinstance(value, (list, tuple)):
            if len(value) > frame_index:
                return value[frame_index]
            return value
        return value

    def _select_current_frame(self, value, n_cam):
        if not isinstance(value, (list, tuple)) or not value:
            return value
        if len(value) == n_cam:
            return value
        return value[-1]

    def _iter_camera_values(self, value, batch, n_cam):
        if value is None:
            return [[None for _ in range(n_cam)] for _ in range(batch)]
        if torch.is_tensor(value):
            if value.dim() == 4 and value.size(0) == batch:
                return [[value[b, cam] if cam < value.size(1) else None for cam in range(n_cam)]
                        for b in range(batch)]
            if value.dim() == 3 and batch == 1:
                return [[value[cam] if cam < value.size(0) else None for cam in range(n_cam)]]
        if isinstance(value, (list, tuple)):
            if len(value) == batch:
                output = []
                for item in value:
                    item = self._select_current_frame(item, n_cam)
                    if isinstance(item, (list, tuple)):
                        output.append([item[cam] if cam < len(item) else None for cam in range(n_cam)])
                    else:
                        output.append([item] + [None for _ in range(n_cam - 1)])
                return output
            if batch == 1:
                value = self._select_current_frame(value, n_cam)
                return [[value[cam] if cam < len(value) else None for cam in range(n_cam)]]
        return [[None for _ in range(n_cam)] for _ in range(batch)]

    def _to_tensor(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        try:
            return torch.as_tensor(value)
        except (TypeError, ValueError):
            return None

    def masks(self, data, feature_shape):
        batch, n_cam, _, feature_h, feature_w = feature_shape
        image_hw = self._image_hw(data)
        if image_hw is None or 'gt_bboxes' not in data:
            return {}

        image_h, image_w = image_hw
        foreground = torch.zeros(batch, n_cam, feature_h, feature_w, dtype=torch.bool)
        near = torch.zeros_like(foreground)
        far = torch.zeros_like(foreground)
        boxes_by_cam = self._iter_camera_values(data.get('gt_bboxes'), batch, n_cam)
        depths_by_cam = self._iter_camera_values(data.get('depths'), batch, n_cam)

        all_depths = []
        for sample_depths in depths_by_cam:
            for depth_values in sample_depths:
                depth_tensor = self._to_tensor(depth_values)
                if depth_tensor is not None and depth_tensor.numel() > 0:
                    all_depths.append(depth_tensor.reshape(-1).float())
        depth_threshold = torch.cat(all_depths).median() if all_depths else None

        for b, sample_boxes in enumerate(boxes_by_cam):
            for cam, box_values in enumerate(sample_boxes[:n_cam]):
                boxes = self._to_tensor(box_values)
                if boxes is None or boxes.numel() == 0:
                    continue
                boxes = boxes.reshape(-1, boxes.shape[-1]).float()
                depth_values = depths_by_cam[b][cam] if b < len(depths_by_cam) and cam < len(depths_by_cam[b]) else None
                depths = self._to_tensor(depth_values)
                if depths is not None:
                    depths = depths.reshape(-1).float()
                for box_idx, box in enumerate(boxes):
                    if box.numel() < 4:
                        continue
                    x1, y1, x2, y2 = box[:4]
                    gx1 = int(torch.floor(x1 / max(image_w, 1) * feature_w).clamp(0, feature_w - 1))
                    gx2 = int(torch.ceil(x2 / max(image_w, 1) * feature_w).clamp(0, feature_w))
                    gy1 = int(torch.floor(y1 / max(image_h, 1) * feature_h).clamp(0, feature_h - 1))
                    gy2 = int(torch.ceil(y2 / max(image_h, 1) * feature_h).clamp(0, feature_h))
                    if gx2 <= gx1 or gy2 <= gy1:
                        continue
                    foreground[b, cam, gy1:gy2, gx1:gx2] = True
                    if depths is not None and depth_threshold is not None and box_idx < depths.numel():
                        target = near if depths[box_idx] <= depth_threshold else far
                        target[b, cam, gy1:gy2, gx1:gx2] = True

        masks = {}
        if foreground.any():
            masks['foreground'] = foreground
        if near.any():
            masks['near_foreground'] = near
        if far.any():
            masks['far_foreground'] = far
        return masks


class StreamPETRFeatureAnalysisAdapter(RepDETR3DFeatureAnalysisAdapter):
    def _as_feature_list(self, features):
        if torch.is_tensor(features):
            if features.dim() == 6 and features.size(1) == 1:
                features = features[:, 0]
            if features.dim() != 5:
                raise ValueError(f'expected StreamPETR features with 5 dims, got {tuple(features.shape)}')
            return [features]
        return features

    def extract_teacher_features(self, model, data):
        return self._as_feature_list(
            model.extract_img_feat(self._img(data), len_queue=1, training_mode=True))

    def extract_final_sequence_features(self, model, data, queue_length):
        features = self._as_feature_list(
            model.extract_img_feat(self._img(data), len_queue=queue_length, training_mode=True))
        for feature in features:
            if feature.dim() != 6:
                raise ValueError(
                    'expected final sequence features with shape [B, T, N_cam, C, H, W], '
                    f'got {tuple(feature.shape)}')
        return features

    def extract_student_features(self, model, data, layer=None):
        if layer is not None:
            raise NotImplementedError(
                'StreamPETR partial-layer feature analysis needs a detector partial-feature hook.')
        return self.extract_teacher_features(model, data)


def build_analysis_adapter(name):
    if name == 'repdetr3d':
        return RepDETR3DFeatureAnalysisAdapter()
    if name == 'streampetr':
        return StreamPETRFeatureAnalysisAdapter()
    raise KeyError(f'Unsupported analysis adapter: {name}')


def parse_args():
    parser = argparse.ArgumentParser(description='Feature similarity analysis')
    parser.add_argument('--mode', default='two-model', choices=['two-model', 'layer-vs-final', 'frame-gap-final'])
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--student-config')
    parser.add_argument('--student-checkpoint')
    parser.add_argument('--teacher-config')
    parser.add_argument('--teacher-checkpoint')
    parser.add_argument('--adapter', default='repdetr3d', choices=['repdetr3d', 'streampetr'])
    parser.add_argument('--layers', nargs='*', type=int, default=None)
    parser.add_argument('--layer', default='block8')
    parser.add_argument('--final-layer', default='final')
    parser.add_argument('--backbone-depth', type=int, default=24)
    parser.add_argument('--queue-length', type=int, default=5)
    parser.add_argument('--anchor-frame', type=int, default=0)
    parser.add_argument('--gaps', nargs='*', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--num-videos', type=int, default=None)
    parser.add_argument('--num-samples', type=int, default=200)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data-split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--max-rsm-tokens', type=int, default=1024)
    parser.add_argument('--heatmap-vmin', type=float, default=0.45)
    parser.add_argument('--heatmap-vmax', type=float, default=0.85)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def load_cfg(path, cfg_options=None):
    cfg = Config.fromfile(path)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        module_dir = getattr(cfg, 'plugin_dir', osp.dirname(path))
        module_path = module_dir.replace('/', '.').strip('.')
        importlib.import_module(module_path)
    return cfg


def prepare_dataset_cfg(cfg, split):
    data_cfg = getattr(cfg.data, split)
    if split != 'train':
        data_cfg.test_mode = True
    samples_per_gpu = data_cfg.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        data_cfg.pipeline = replace_ImageToTensor(data_cfg.pipeline)
    return data_cfg, samples_per_gpu


def configure_frame_gap_dataset_cfg(data_cfg, queue_length):
    data_cfg['queue_length'] = queue_length
    data_cfg['random_length'] = 0
    data_cfg['seq_mode'] = False
    return data_cfg


def build_model_from_cfg(cfg, checkpoint, device):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.to(device)
    model.eval()
    return model


def cuda_target_gpus(device):
    if device == 'cuda':
        return [0]
    if device.startswith('cuda:'):
        return [int(device.split(':', 1)[1])]
    raise ValueError(f'expected cuda device string, got {device}')


def move_data_to_device(data, device):
    if device.startswith('cuda'):
        return scatter(data, cuda_target_gpus(device))[0]
    return data


def accumulate_metrics(accumulator, metrics):
    for key, value in metrics.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        accumulator[key].append(float(value))


def averaged_metrics(accumulator):
    return {
        key: float(np.mean(values))
        for key, values in accumulator.items()
        if values
    }


def single_model_mode(mode):
    return mode in {'layer-vs-final', 'frame-gap-final'}


def validate_args(args):
    if single_model_mode(args.mode):
        missing = [name for name in ['config', 'checkpoint'] if getattr(args, name) is None]
    else:
        missing = [
            name for name in [
                'student_config',
                'student_checkpoint',
                'teacher_config',
                'teacher_checkpoint',
            ]
            if getattr(args, name) is None
        ]
    if missing:
        raise ValueError(f'missing required arguments for {args.mode}: {", ".join("--" + x.replace("_", "-") for x in missing)}')
    if args.mode == 'frame-gap-final':
        if args.queue_length < 2:
            raise ValueError('--queue-length must be at least 2 for frame-gap-final')
        max_gap = max(args.gaps)
        if args.anchor_frame + max_gap >= args.queue_length:
            raise ValueError(
                f'--queue-length {args.queue_length} is too short for anchor '
                f'{args.anchor_frame} and max gap {max_gap}')


def main():
    args = parse_args()
    validate_args(args)
    device = args.device if torch.cuda.is_available() or not args.device.startswith('cuda') else 'cpu'
    adapter = build_analysis_adapter(args.adapter)
    teacher_cfg = load_cfg(args.teacher_config or args.config, args.cfg_options)
    student_cfg = load_cfg(args.student_config or args.config, args.cfg_options)

    data_cfg, samples_per_gpu = prepare_dataset_cfg(teacher_cfg, args.data_split)
    if args.mode == 'frame-gap-final':
        configure_frame_gap_dataset_cfg(data_cfg, args.queue_length)
    dataset = build_dataset(data_cfg)
    dataset = limit_dataset_by_scenes(dataset, args.num_videos)
    dataset = limit_dataset(dataset, args.num_samples)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=teacher_cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=teacher_cfg.data.nonshuffler_sampler,
    )

    teacher = build_model_from_cfg(teacher_cfg, args.teacher_checkpoint or args.checkpoint, device)
    student = teacher if single_model_mode(args.mode) else build_model_from_cfg(
        student_cfg, args.student_checkpoint, device)

    metric_accumulators = defaultdict(list)
    heatmap_accumulators = defaultdict(list)

    layers = args.layers or [None]
    with torch.no_grad():
        for data in mmcv.track_iter_progress(data_loader):
            data = move_data_to_device(data, device)
            camera_names = adapter.camera_names(data)
            if args.mode == 'frame-gap-final':
                sequence_features = adapter.extract_final_sequence_features(
                    teacher, data, queue_length=args.queue_length)
                for gap in args.gaps:
                    anchor_features, future_features = temporal_feature_pair(
                        sequence_features, gap=gap, anchor=args.anchor_frame)
                    future_data = adapter.frame_data(data, args.anchor_frame + gap)
                    masks = adapter.masks(future_data, future_features[0].shape)
                    metrics, heatmaps = compare_feature_sets(
                        anchor_features,
                        future_features,
                        camera_names=camera_names,
                        masks=masks,
                        max_rsm_tokens=args.max_rsm_tokens)
                    prefix = f'gap_{gap}'
                    accumulate_metrics(metric_accumulators, {
                        f'{prefix}/{key}': value for key, value in metrics.items()
                    })
                    for key, value in heatmaps.items():
                        heatmap_accumulators[f'{prefix}_{key}'].append(value)
            elif args.mode == 'layer-vs-final':
                teacher_features = adapter.extract_backbone_layer_features(
                    teacher, data, args.final_layer, depth=args.backbone_depth)
                student_features = adapter.extract_backbone_layer_features(
                    student, data, args.layer, depth=args.backbone_depth)
                masks = adapter.masks(data, teacher_features[0].shape)
                metrics, heatmaps = compare_feature_sets(
                    student_features,
                    teacher_features,
                    camera_names=camera_names,
                    masks=masks,
                    max_rsm_tokens=args.max_rsm_tokens)
                prefix = f'{args.layer}_vs_{args.final_layer}'
                accumulate_metrics(metric_accumulators, {
                    f'{prefix}/{key}': value for key, value in metrics.items()
                })
                for key, value in heatmaps.items():
                    heatmap_accumulators[f'{prefix}_{key}'].append(value)
            else:
                teacher_features = adapter.extract_teacher_features(teacher, data)
                for layer in layers:
                    student_features = adapter.extract_student_features(student, data, layer=layer)
                    masks = adapter.masks(data, teacher_features[0].shape)
                    metrics, heatmaps = compare_feature_sets(
                        student_features,
                        teacher_features,
                        camera_names=camera_names,
                        masks=masks,
                        max_rsm_tokens=args.max_rsm_tokens)
                    prefix = f'layer{layer}' if layer is not None else 'student'
                    accumulate_metrics(metric_accumulators, {
                        f'{prefix}/{key}': value for key, value in metrics.items()
                    })
                    for key, value in heatmaps.items():
                        heatmap_accumulators[f'{prefix}_{key}'].append(value)

    summary = averaged_metrics(metric_accumulators)
    averaged_heatmaps = {
        key: np.mean(values, axis=0)
        for key, values in heatmap_accumulators.items()
    }
    save_camera_heatmaps(
        averaged_heatmaps,
        args.out_dir,
        camera_names=DEFAULT_CAMERA_NAMES,
        summary=summary,
        heatmap_vmin=args.heatmap_vmin,
        heatmap_vmax=args.heatmap_vmax)
    print(f'Wrote feature similarity summary and heatmaps to {args.out_dir}')


if __name__ == '__main__':
    main()
