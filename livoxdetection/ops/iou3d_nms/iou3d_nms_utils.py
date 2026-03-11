

"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import torch
import numpy as np
try:
    from . import iou3d_nms_cuda
    _HAS_IOU3D_NMS_EXT = True
except Exception:
    iou3d_nms_cuda = None
    _HAS_IOU3D_NMS_EXT = False

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    boxes_a, is_numpy = check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    if _HAS_IOU3D_NMS_EXT:
        ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
        iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)
    else:
        # Axis-aligned approximation (ignores heading). Good enough to run CPU-only inference.
        xa_min = boxes_a[:, 0] - boxes_a[:, 3] * 0.5
        xa_max = boxes_a[:, 0] + boxes_a[:, 3] * 0.5
        ya_min = boxes_a[:, 1] - boxes_a[:, 4] * 0.5
        ya_max = boxes_a[:, 1] + boxes_a[:, 4] * 0.5

        xb_min = boxes_b[:, 0] - boxes_b[:, 3] * 0.5
        xb_max = boxes_b[:, 0] + boxes_b[:, 3] * 0.5
        yb_min = boxes_b[:, 1] - boxes_b[:, 4] * 0.5
        yb_max = boxes_b[:, 1] + boxes_b[:, 4] * 0.5

        inter_xmin = torch.maximum(xa_min[:, None], xb_min[None, :])
        inter_ymin = torch.maximum(ya_min[:, None], yb_min[None, :])
        inter_xmax = torch.minimum(xa_max[:, None], xb_max[None, :])
        inter_ymax = torch.minimum(ya_max[:, None], yb_max[None, :])

        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
        inter = inter_w * inter_h

        area_a = (xa_max - xa_min) * (ya_max - ya_min)
        area_b = (xb_max - xb_min) * (yb_max - yb_min)
        union = torch.clamp(area_a[:, None] + area_b[None, :] - inter, min=1e-6)
        ans_iou = inter / union

    return ans_iou.numpy() if is_numpy else ans_iou


def boxes_iou_bev(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    if not _HAS_IOU3D_NMS_EXT:
        raise ImportError('iou3d_nms_cuda is not built; cannot run GPU bev IoU')
    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    if not _HAS_IOU3D_NMS_EXT:
        raise ImportError('iou3d_nms_cuda is not built; cannot run GPU 3D IoU')

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms_cpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    CPU fallback: greedy 3D NMS using BEV IoU.
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    orig_device = boxes.device
    boxes = boxes.cpu().contiguous()
    scores = scores.cpu()
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes_ordered = boxes[order].contiguous()
    keep = []
    for i in range(len(order)):
        if len(keep) == 0:
            keep.append(i)
            continue
        iou = boxes_bev_iou_cpu(boxes_ordered[i : i + 1], boxes_ordered[keep])
        if iou.max() <= thresh:
            keep.append(i)
    keep_tensor = order[torch.tensor(keep, dtype=torch.long)]
    return keep_tensor.to(orig_device).contiguous(), None


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    if not _HAS_IOU3D_NMS_EXT:
        raise ImportError('iou3d_nms_cuda is not built; cannot run nms_gpu')
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    # Extension writes kept indices into CPU buffer
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].to(order.device)].contiguous(), None


def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    if not _HAS_IOU3D_NMS_EXT:
        raise ImportError('iou3d_nms_cuda is not built; cannot run nms_normal_gpu')
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].to(order.device)].contiguous(), None
