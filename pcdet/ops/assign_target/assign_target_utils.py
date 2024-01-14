from copy import deepcopy
from matplotlib.pyplot import box
import torch
import torch.nn as nn
from torch.autograd import Function

from ...utils import common_utils
from . import assign_target_cuda


def points_in_boxes_cpu(points, boxes, box_range=1):
    """
    Args:
        points: (num_points, 2)
        boxes: [x, y, dx, dy, heading], (x, y) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    assert boxes.shape[1] == 5
    assert points.shape[1] == 2
    if box_range < 1:
        boxes = deepcopy(boxes)
        boxes[:, 2:4] = boxes[:, 2:4] * box_range

    points, is_numpy = common_utils.check_numpy_to_torch(points)
    boxes, is_numpy = common_utils.check_numpy_to_torch(boxes)

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    assign_target_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices.numpy() if is_numpy else point_indices

def points_in_boxes_gpu(points, boxes, box_range=1):
    """
    Args:
        points: (num_points, 2)
        boxes: [x, y, dx, dy, heading], (x, y) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    assert boxes.shape[1] == 8
    assert points.shape[1] == 2
    
    boxes_2d = deepcopy(boxes[:, [0, 1, 3, 4, 6]])
    #boxes_2d[:, -1] = torch.arctan2(boxes[:, -1], boxes[:, -2])
    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    if box_range != 0:
        assign_target_cuda.points_in_boxes_gpu(boxes_2d.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices

def points_in_cross_area_gpu(centerx, centery, points, boxes, r=1):
    """
    Args:
        points: (num_points, 2)
        boxes: [x, y, dx, dy, heading], (x, y) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0], points.shape[1]), dtype=torch.int)

    if boxes.shape[0] > 0:
        assign_target_cuda.points_in_cross_area_gpu(centerx, centery, point_indices, r)
    
    return point_indices.flatten(1)

def positive_points_in_boxes_gpu(points, boxes, box_range=1):
    """
    Args:
        points: (num_points, 2)
        boxes: [x, y, dx, dy, heading], (x, y) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    assert boxes.shape[1] == 5
    assert points.shape[1] == 2
    if box_range < 1:
        boxes = deepcopy(boxes)
        boxes[:, 2:4] = boxes[:, 2:4] * box_range

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    assign_target_cuda.positive_points_in_boxes_gpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices

def heatmap_in_boxes_gpu(points, boxes, pc_range, voxel_size, feature_map_stride, box_size_scale=1, min_radius=3):
    """
    Args:
        points: (num_points, 2)
        boxes: [x, y, dx, dy, heading], (x, y) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    assert boxes.shape[1] == 5
    assert points.shape[1] == 2
    
    boxes = deepcopy(boxes)
    boxes[:, :2] = (boxes[:, :2] - pc_range[:2]) / (voxel_size[:2] * feature_map_stride)
    boxes[:, 2:4] = boxes[:, 2:4] * box_size_scale / (voxel_size[:2] * feature_map_stride)
    boxes[:, 2:4] = torch.clamp(boxes[:, 2:4], min=min_radius)
    points = (points - pc_range[:2]) / (voxel_size[:2] * feature_map_stride)

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]))
    assign_target_cuda.heatmap_in_boxes_gpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices

def heatmap_in_boxes_center_gpu(points, boxes, pc_range, voxel_size, feature_map_stride, box_size_scale=1, min_radius=3):
    """
    Args:
        points: (num_points, 2)
        boxes: [x, y, dx, dy, heading], (x, y) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    assert boxes.shape[1] == 5
    assert points.shape[1] == 2
    
    boxes = deepcopy(boxes)
    boxes[:, :2] = torch.floor((boxes[:, :2] - pc_range[:2]) / (voxel_size[:2] * feature_map_stride))
    boxes[:, 2:4] = torch.floor(boxes[:, 2:4] * box_size_scale / (voxel_size[:2] * feature_map_stride)) 
    boxes[:, 2:4] = torch.clamp(boxes[:, 2:4], min=min_radius)
    points = torch.floor((points - pc_range[:2]) / (voxel_size[:2] * feature_map_stride))

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]))
    assign_target_cuda.heatmap_in_boxes_center_gpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices

if __name__ == '__main__':
    pass
