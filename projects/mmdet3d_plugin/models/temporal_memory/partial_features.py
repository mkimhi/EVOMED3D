def resolve_eva_layer_index(layer_id, depth=24):
    presets = {
        '25%': 0.25,
        'quarter': 0.25,
        '50%': 0.50,
        'middle': 0.50,
        '75%': 0.75,
        'three_quarter': 0.75,
    }
    if isinstance(layer_id, str) and layer_id in presets:
        return max(0, min(depth - 1, int(depth * presets[layer_id]) - 1))
    if isinstance(layer_id, str) and layer_id.startswith('block'):
        return int(layer_id.replace('block', ''))
    return int(layer_id)
