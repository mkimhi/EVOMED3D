from collections import Counter


EXPECTED_IMAGE_SHAPE = (1, 6, 3, 320, 800)
IGNORED_UNSUPPORTED_OPS = {
    'aten::reshape',
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
