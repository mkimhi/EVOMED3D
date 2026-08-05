from collections import Counter


EXPECTED_IMAGE_SHAPE = (1, 6, 3, 320, 800)
IGNORED_UNSUPPORTED_OPS = {
    'aten::__iand__',
    'aten::add',
    'aten::add_',
    'aten::all',
    'aten::atan2',
    'aten::clone',
    'aten::cos',
    'aten::cumsum',
    'aten::div',
    'aten::exp',
    'aten::eye',
    'aten::gelu',
    'aten::le',
    'aten::linspace',
    'aten::log',
    'aten::mean',
    'aten::mul',
    'aten::mul_',
    'aten::nan_to_num',
    'aten::neg',
    'aten::ones_like',
    'aten::pad',
    'aten::pow',
    'aten::prod',
    'aten::repeat',
    'aten::reshape',
    'aten::rsub',
    'aten::sigmoid',
    'aten::silu',
    'aten::sin',
    'aten::softmax',
    'aten::sqrt',
    'aten::squeeze_',
    'aten::sub',
    'aten::sum',
    'aten::topk',
    'aten::upsample_bicubic2d',
    'aten::view',
    'aten::permute',
    'aten::transpose',
    'aten::contiguous',
    'aten::slice',
    'aten::select',
    'aten::index',
    'aten::unsqueeze',
    'aten::squeeze',
    'aten::detach',
}


def flash_attention_macs(q_shape, kv_shape, cu_q_shape):
    total_q, heads, head_dim = (int(value) for value in q_shape)
    total_k = int(kv_shape[0])
    batch = int(cu_q_shape[0]) - 1
    if batch < 1 or total_q % batch or total_k % batch:
        raise ValueError('invalid packed flash-attention shapes')
    query_length = total_q // batch
    key_length = total_k // batch
    return 2 * batch * heads * query_length * key_length * head_dim


def deformable_attention_macs(value_shape, sampling_shape):
    batch, queries, groups, levels, points, coordinates = (
        int(value) for value in sampling_shape)
    if coordinates != 2 or int(value_shape[0]) != batch:
        raise ValueError('invalid deformable-attention shapes')
    group_dim = int(value_shape[3])
    return 5 * batch * queries * groups * levels * points * group_dim


def macs_to_gflops(macs):
    macs = float(macs)
    if macs < 0:
        raise ValueError('MAC count must be non-negative')
    return macs * 2 / 1e9


def validate_input_shape(shape, expected=EXPECTED_IMAGE_SHAPE):
    shape = tuple(int(value) for value in shape)
    if shape != tuple(expected):
        raise ValueError(
            f'expected image shape {tuple(expected)}, got {shape}')
    return shape


def validate_unsupported_ops(unsupported, ignored=IGNORED_UNSUPPORTED_OPS):
    unsupported = {
        name: int(count)
        for name, count in unsupported.items()
        if int(count) > 0
    }
    material = sorted(set(unsupported) - set(ignored))
    if material:
        raise ValueError(f'unsupported compute operators: {material}')
    return unsupported


def weighted_path_gflops(path_gflops, action_counts):
    counts = {
        action: int(count)
        for action, count in action_counts.items()
        if int(count) > 0
    }
    if not counts:
        raise ValueError('action counts are empty')
    missing = sorted(set(counts) - set(path_gflops))
    if missing:
        raise ValueError(f'missing path GFLOPs for actions: {missing}')
    return sum(
        float(path_gflops[action]) * count
        for action, count in counts.items()
    ) / sum(counts.values())


def summarize_frames(rows):
    rows = list(rows)
    if not rows:
        raise ValueError('no frame measurements')
    mean_macs = sum(float(row['macs']) for row in rows) / len(rows)
    return {
        'sample_count': len(rows),
        'mean_macs': mean_macs,
        'mean_gflops': macs_to_gflops(mean_macs),
        'action_counts': dict(Counter(row['action'] for row in rows)),
    }
