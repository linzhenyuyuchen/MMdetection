"""
CommandLine:
    pytest tests/test_soft_nms.py
    python tests/test_soft_nms.py
"""
import numpy as np
import torch
import mmcv
import os
import argparse

from mmdet.ops.nms.nms_wrapper import soft_nms


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval a model with soft-nms')
    parser.add_argument('pkl', default='/home/mmdetection/work_dirs/ensemble_1.pkl', help='result pkl file path')
    parser.add_argument('num_classes', default=5, help='num of classes')
    return args

def test_soft_nms(pkl_file,num_classes):

    if os.path.exists(pkl_file):
        print(f'\nreading results from {pkl_file}')
        dt = mmcv.load(pkl_file)
    dt3 = []
    for i in range(len(dt)):
        tmp = []
        for j in range(num_classes):
            a = single_soft_nms(dt[i][j],0.7)
            tmp.append(a)
        dt3.append(tmp)
    mmcv.dump(dt3, pkl_file.replace(".pkl","_soft_nms.pkl"))


def single_soft_nms(base_dets,iou_thr = 0.7):
    """
    CommandLine:
        xdoctest -m tests/test_soft_nms.py test_soft_nms_device_and_dtypes_cpu
    """
    if len(base_dets)>0:
        dets = base_dets.astype(np.float32)
        new_dets, inds = soft_nms(dets, iou_thr)
        return new_dets
    return base_dets

def test_soft_nms_device_and_dtypes_cpu():
    """
    CommandLine:
        xdoctest -m tests/test_soft_nms.py test_soft_nms_device_and_dtypes_cpu
    """
    iou_thr = 0.7
    base_dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
                          [49.3, 32.9, 51.0, 35.3, 0.9],
                          [35.3, 11.5, 39.9, 14.5, 0.4],
                          [35.2, 11.7, 39.7, 15.7, 0.3]])

    # CPU can handle float32 and float64
    dets = base_dets.astype(np.float32)
    new_dets, inds = soft_nms(dets, iou_thr)
    assert dets.dtype == new_dets.dtype
    assert len(inds) == len(new_dets) == 4

    dets = torch.FloatTensor(base_dets)
    new_dets, inds = soft_nms(dets, iou_thr)
    assert dets.dtype == new_dets.dtype
    assert len(inds) == len(new_dets) == 4

    dets = base_dets.astype(np.float64)
    new_dets, inds = soft_nms(dets, iou_thr)
    assert dets.dtype == new_dets.dtype
    assert len(inds) == len(new_dets) == 4

    dets = torch.DoubleTensor(base_dets)
    new_dets, inds = soft_nms(dets, iou_thr)
    assert dets.dtype == new_dets.dtype
    assert len(inds) == len(new_dets) == 4

if __name__ == '__main__' :
    args = parse_args()
    test_soft_nms(args.pkl,args.num_classes)
