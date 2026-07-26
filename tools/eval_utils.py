def limit_dataset(dataset, num_test):
    if num_test is None:
        return dataset
    if num_test <= 0:
        raise ValueError('--num_test must be a positive integer')

    dataset_size = len(dataset)
    limit = min(num_test, dataset_size)
    if limit == dataset_size:
        return dataset

    if hasattr(dataset, 'data_infos'):
        dataset.data_infos = dataset.data_infos[:limit]
        dataset._eval_sample_tokens = {
            info['token'] for info in dataset.data_infos
            if isinstance(info, dict) and 'token' in info
        }
    if hasattr(dataset, 'flag'):
        dataset.flag = dataset.flag[:limit]
    return dataset


def format_timing_summary(total_seconds, sample_count):
    if sample_count <= 0:
        return 'Inference timing: no samples measured'
    ms_per_sample = total_seconds * 1000.0 / sample_count
    fps = sample_count / total_seconds if total_seconds > 0 else float('inf')
    return (
        f'Inference timing: {ms_per_sample:.2f} ms/sample, '
        f'{fps:.2f} FPS over {sample_count} samples')
