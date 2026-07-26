# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _format_timing_line(name, seconds, samples):
    if samples <= 0:
        return f'{name}: no samples measured'
    return (
        f'{name}: {seconds * 1000.0 / samples:.2f} ms/sample, '
        f'{samples / seconds if seconds > 0 else float("inf"):.2f} FPS, '
        f'total {seconds:.2f}s')


def _print_timing_summary(timing, rank):
    dist.reduce(timing, dst=0)
    if rank != 0:
        return

    data_wait, forward, result_post, collect, samples = timing.tolist()
    samples = int(samples)
    print('\nTiming summary')
    print(_format_timing_line('  data_wait', data_wait, samples))
    print(_format_timing_line('  model_forward', forward, samples))
    print(_format_timing_line('  result_postprocess', result_post, samples))
    print(_format_timing_line('  result_collect', collect, samples))
    print(_format_timing_line(
        '  eval_loop_total', data_wait + forward + result_post + collect,
        samples))

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False,
                          timing_warmup=5):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    timing = dict(data_wait=0.0, forward=0.0, result_post=0.0, collect=0.0)
    measured_samples = 0
    data_start = time.perf_counter()
    for i, data in enumerate(data_loader):
        data_elapsed = time.perf_counter() - data_start

        _cuda_sync()
        forward_start = time.perf_counter()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        _cuda_sync()
        forward_elapsed = time.perf_counter() - forward_start

        post_start = time.perf_counter()
        # encode mask results
        if isinstance(result, dict):
            if 'bbox_results' in result.keys():
                bbox_result = result['bbox_results']
                batch_size = len(result['bbox_results'])
                bbox_results.extend(bbox_result)
            if 'mask_results' in result.keys() and result['mask_results'] is not None:
                mask_result = custom_encode_mask_results(result['mask_results'])
                mask_results.extend(mask_result)
                have_mask = True
        else:
            batch_size = len(result)
            bbox_results.extend(result)
        post_elapsed = time.perf_counter() - post_start

        if i >= timing_warmup:
            timing['data_wait'] += data_elapsed
            timing['forward'] += forward_elapsed
            timing['result_post'] += post_elapsed
            measured_samples += batch_size

        #if isinstance(result[0], tuple):
        #    assert False, 'this code is for instance segmentation, which our code will not utilize.'
        #    result = [(bbox_results, encode_mask_results(mask_results))
        #              for bbox_results, mask_results in result]
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
        data_start = time.perf_counter()

    # collect results from all ranks
    collect_start = time.perf_counter()
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None
    timing['collect'] = time.perf_counter() - collect_start

    timing_tensor = torch.tensor([
        timing['data_wait'],
        timing['forward'],
        timing['result_post'],
        timing['collect'],
        measured_samples,
    ], dtype=torch.float64, device='cuda')
    _print_timing_summary(timing_tensor, rank)

    if mask_results is None:
        return bbox_results
    return {'bbox_results': bbox_results, 'mask_results': mask_results}


def custom_single_gpu_test(model, data_loader, show=False, out_dir=None,
                           timing_warmup=5):
    model.eval()
    bbox_results = []
    mask_results = []
    have_mask = False
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    timing = dict(data_wait=0.0, forward=0.0, result_post=0.0)
    measured_samples = 0
    data_start = time.perf_counter()

    for i, data in enumerate(data_loader):
        data_elapsed = time.perf_counter() - data_start

        _cuda_sync()
        forward_start = time.perf_counter()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        _cuda_sync()
        forward_elapsed = time.perf_counter() - forward_start

        post_start = time.perf_counter()
        if isinstance(result, dict):
            if 'bbox_results' in result:
                bbox_result = result['bbox_results']
                batch_size = len(bbox_result)
                bbox_results.extend(bbox_result)
            if 'mask_results' in result and result['mask_results'] is not None:
                mask_result = custom_encode_mask_results(result['mask_results'])
                mask_results.extend(mask_result)
                have_mask = True
        else:
            batch_size = len(result)
            bbox_results.extend(result)
        post_elapsed = time.perf_counter() - post_start

        if i >= timing_warmup:
            timing['data_wait'] += data_elapsed
            timing['forward'] += forward_elapsed
            timing['result_post'] += post_elapsed
            measured_samples += batch_size

        for _ in range(batch_size):
            prog_bar.update()
        data_start = time.perf_counter()

    print('\nTiming summary')
    print(_format_timing_line('  data_wait', timing['data_wait'], measured_samples))
    print(_format_timing_line('  model_forward', timing['forward'], measured_samples))
    print(_format_timing_line(
        '  result_postprocess', timing['result_post'], measured_samples))
    print(_format_timing_line(
        '  eval_loop_total',
        timing['data_wait'] + timing['forward'] + timing['result_post'],
        measured_samples))

    if mask_results:
        return {'bbox_results': bbox_results, 'mask_results': mask_results}
    return bbox_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)
