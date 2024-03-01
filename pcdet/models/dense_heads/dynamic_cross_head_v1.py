import copy
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch.nn.init import kaiming_normal_
from ..model_utils import dcdet_utils, model_nms_utils
from ...utils import loss_utils
from ...ops.assign_target import assign_target_utils
from ...ops.iou3d_nms import iou3d_nms_utils

class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, class_num, stride=1, init_bias=-2.19, use_bias=False, norm_func=None):
        super().__init__()
        self.sep_head_dict = sep_head_dict
        if stride > 1:
            self.deblock = nn.Sequential(
                nn.ConvTranspose2d(input_channels, input_channels, kernel_size=stride, stride=stride, padding=0),
                nn.BatchNorm2d(input_channels),
                nn.ReLU()
                )
        else:
            self.deblock = nn.Identity()
        

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels'] * class_num
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels) if norm_func is None else norm_func(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=1))
            fc = nn.Sequential(*fc_list)

            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight, mean=0, std=0.001)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        x = self.deblock(x)
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class DynamicCrossHeadv1(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = np.array(voxel_size)

        self.blur_parameter = [torch.from_numpy(np.array(blur)) for blur in self.model_cfg.get('BLUR_PARAMETER', [[1, 0.3, 0.5]])]
        self.up_strides = self.model_cfg.get('UP_STRIDES', [1, 1, 1])
        self.cross_area_r = self.model_cfg.get('CROSS_AREA_R', 1)
        self.cross_area_samples = 1 + self.cross_area_r * (self.cross_area_r + 1) * 2
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.spatial_indices = self.generate_spatial_indices(self.point_cloud_range, self.voxel_size, self.feature_map_stride)
        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        norm_func = partial(nn.BatchNorm2d, eps=self.model_cfg.get('BN_EPS', 1e-5), momentum=self.model_cfg.get('BN_MOM', 0.1))
        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            norm_func(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=1, num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    stride=self.up_strides[idx],
                    class_num=len(cur_class_names),
                    init_bias=-np.log((1 - 0.01) / 0.01),
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                    norm_func=norm_func
                )
            )
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.get_dynamic_masks = loss_utils.DynamicPositiveMask(1, self.model_cfg.get('DCLA_REG_WEIGHT', 3), \
                                                                self.voxel_size * self.feature_map_stride)
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossSparse())
        self.add_module('reg_loss_func', loss_utils.RWIoULoss(self.voxel_size * self.feature_map_stride))
        if 'iou' in self.separate_head_cfg.HEAD_DICT:
            self.add_module('crit_iou', loss_utils.DCDetIoULoss())

    @staticmethod
    def generate_spatial_indices(point_cloud_range, voxel_size, stride):
        grid_size_x = int((point_cloud_range[3] - point_cloud_range[0]) / (voxel_size[0] * stride) + 0.1)
        grid_size_y = int((point_cloud_range[4] - point_cloud_range[1]) / (voxel_size[1] * stride) + 0.1)
        x_shifts = torch.arange(
                0, grid_size_x, dtype=torch.int,
            ).cuda()
        y_shifts = torch.arange(
                0, grid_size_y, dtype=torch.int,
            ).cuda()
        y_shifts, x_shifts = torch.meshgrid([
                y_shifts, x_shifts
            ])
        spatial_indices = torch.stack([x_shifts, y_shifts], dim=2)
        return spatial_indices

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, spatial_shape, feature_map_stride, num_max_objs=500,
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(spatial_shape[1] * spatial_shape[0], num_classes)

        ret_boxes = gt_boxes.new_zeros((num_max_objs, self.cross_area_samples, gt_boxes.shape[-1] - 1 + 1))
        reg_inds = gt_boxes.new_zeros(num_max_objs, self.cross_area_samples).long()
        cls_inds = gt_boxes.new_zeros(num_max_objs, self.cross_area_samples).long()
        mask = gt_boxes.new_zeros(num_max_objs, self.cross_area_samples).long()

        box_masks = ((gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0) & (gt_boxes[:, 5] > 0) & (gt_boxes[:, 0] >= self.point_cloud_range[0]) \
                     & (gt_boxes[:, 1] >= self.point_cloud_range[1]) & (gt_boxes[:, 0] < self.point_cloud_range[3]) \
                     & (gt_boxes[:, 1] < self.point_cloud_range[4]))
        
        box_num = box_masks.sum()
        gt_boxes = gt_boxes[box_masks]
        box_cls = gt_boxes[:, -1:] - 1

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=spatial_shape[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=spatial_shape[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        box_point_mask = assign_target_utils.points_in_cross_area_gpu(center_int[:, 0].contiguous(), center_int[:, 1].contiguous(), self.spatial_indices, gt_boxes, r=self.cross_area_r)

        collision_inds = torch.nonzero(box_point_mask.sum(0) > 1)
        
        box_point_mask[:, collision_inds] = 0

        #indices = torch.nonzero(box_point_mask)[:, 1].view(box_num, -1)
        sort_res = torch.sort(box_point_mask, descending=True, dim=-1, stable=True)
        
        sort_mask, sort_inds = sort_res[0][:, :self.cross_area_samples], sort_res[1][:, :self.cross_area_samples]
        
        reg_inds[:box_num] = sort_inds + spatial_shape[0] * spatial_shape[1] * box_cls
        cls_inds[:box_num] = sort_inds
        mask[:box_num] = sort_mask

        ret_boxes[:box_num, :, 0:2] = center[:, None, :] - self.spatial_indices.view(-1, 2)[sort_inds]
        ret_boxes[:box_num, :, 2:3] = z[:, None, None]
        ret_boxes[:box_num, :, 3:6] = gt_boxes[:, None, 3:6].log()
        ret_boxes[:box_num, :, 6:7] = torch.cos(gt_boxes[:, None, 6:7])
        ret_boxes[:box_num, :, 7:] = torch.sin(gt_boxes[:, None, 6:7])
        if gt_boxes.shape[1] > 8:
                ret_boxes[:box_num, :, 8:] = gt_boxes[:, None, 7:-1]

        cur_class_id = (gt_boxes[:, -1] - 1).long()

        heatmap[sort_inds, cur_class_id[:, None]] = 1

        return heatmap, ret_boxes, reg_inds, cls_inds, mask

    def assign_targets(self, gt_boxes, spatial_shape, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:
        """
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'reg_inds': [],
            'cls_inds': [],
            'masks': [],
            'gt_boxes': []
        }

        all_names = np.array(['bg', *self.class_names])
        for head_idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, reg_inds_list, cls_inds_list, masks_list, gt_boxes_list = [], [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)
                cur_spatial_shape = [cur_size * self.up_strides[head_idx] for cur_size in spatial_shape]
                heatmap, ret_boxes, reg_inds, cls_inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), 
                    gt_boxes=gt_boxes_single_head, 
                    spatial_shape=cur_spatial_shape, 
                    feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                )

                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                reg_inds_list.append(reg_inds.to(gt_boxes_single_head.device))
                cls_inds_list.append(cls_inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                gt_boxes_list.append(gt_boxes_single_head[:, :-1])

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['reg_inds'].append(torch.stack(reg_inds_list, dim=0))
            ret_dict['cls_inds'].append(torch.stack(cls_inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['gt_boxes'].append(gt_boxes_list)

        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        batch_size = self.forward_ret_dict['batch_size']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_hm = self.sigmoid(pred_dict['hm'])
            pred_hm = pred_hm.view(*pred_hm.shape[:2], -1).transpose(2, 1)
            pred_reg = torch.cat([pred_dict[head_name].view(batch_size, -1, len(self.class_names_each_head[idx]), pred_hm.shape[1]) for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)
            pred_reg = pred_reg.view(pred_reg.shape[0], pred_reg.shape[1], -1).transpose(2, 1)

            target_boxes = target_dicts['target_boxes'][idx]
            target_hm = target_dicts['heatmaps'][idx]

            masks = target_dicts['masks'][idx]
            reg_inds = target_dicts['reg_inds'][idx]
            cls_inds = target_dicts['cls_inds'][idx]

            pred_boxes = []
            pred_cls = []
            target_cls = []
            batch_spatial_indices = []
            for bs_idx in range(batch_size):
                pred_boxes.append(pred_reg[bs_idx][reg_inds[bs_idx]])
                pred_cls.append(pred_hm[bs_idx][cls_inds[bs_idx]])
                target_cls.append(target_hm[bs_idx][cls_inds[bs_idx]])
                batch_spatial_indices.append(self.spatial_indices.view(-1, 2)[cls_inds[bs_idx]])

            batch_spatial_indices = torch.stack(batch_spatial_indices)
            target_cls = torch.stack(target_cls)
            pred_boxes = torch.stack(pred_boxes).float()
            pred_cls = torch.stack(pred_cls)

            pred_boxes[..., -2:] = pred_boxes[..., -2:].sigmoid() * 2 - 1
            target_labels = target_cls.argmax(dim=-1)
            #  calculate IoU    
            iou_targets, pred_boxes_for_iou, gt_boxes_for_iou = self.get_iou_targets(pred_boxes, target_boxes, masks, batch_spatial_indices)

            # get dynamic positive masks 
            target_masks = self.get_dynamic_masks(pred_cls, target_cls, pred_boxes, target_boxes, \
                                                        masks, iou_targets, self.blur_parameter[idx].to(target_cls)[target_labels])
            pos_masks = target_masks.eq(1)

            # calculate cls loss
            for bs_idx in range(batch_size):
                target_hm[bs_idx][cls_inds[bs_idx]] *= target_masks[bs_idx][:, :, None]
            
            hm_loss = self.hm_loss_func(pred_hm, target_hm)
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            # calculate reg loss
            loc_loss = self.reg_loss_func(
                pred_boxes, target_boxes, pos_masks, r_factor=self.blur_parameter[idx].to(target_cls)[target_labels]
            )
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

            if 'iou' in pred_dict or self.model_cfg.get('IOU_REG_LOSS', False):
                if 'iou' in pred_dict:
                    pred_ious = pred_dict['iou']
                    pred_ious = pred_ious.view(batch_size, -1)
                    batch_pred_ious = []
                    for bs_idx in range(batch_size):
                        batch_pred_ious.append(pred_ious[bs_idx][reg_inds[bs_idx]])
                    batch_pred_ious = torch.stack(batch_pred_ious)
                    iou_loss = self.crit_iou(batch_pred_ious.view(-1), iou_targets.view(-1) * 2 - 1, pos_masks.view(-1))
                    loss += iou_loss
                    tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()

                if self.model_cfg.get('IOU_REG_LOSS', False):
                    iou_reg_loss = loss_utils.UpFormerIoUregLoss(
                        pred_boxes=pred_boxes_for_iou.view(-1, pred_boxes_for_iou.shape[-1]), 
                        gt_boxes=gt_boxes_for_iou.view(-1, gt_boxes_for_iou.shape[-1]),
                        mask=masks.view(-1),
                    )
                    iou_reg_loss = iou_reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                    loss += iou_reg_loss
                    tb_dict['iou_reg_loss_head_%d' % idx] = iou_reg_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def get_iou_targets(self, box_preds, box_targets, masks, spatial_indices):
        iou_targets = torch.zeros_like(masks).float()
        input_shape = box_preds.shape
        box_preds = box_preds.reshape(-1, input_shape[-1])
        box_targets = box_targets.reshape(-1, input_shape[-1])
        spatial_indices = spatial_indices.reshape(-1, spatial_indices.shape[-1])
        box_inds = masks.view(-1).nonzero().squeeze(-1)

        qboxes = self._get_predicted_boxes(box_preds.float(), spatial_indices)
        gboxes = self._get_predicted_boxes(box_targets, spatial_indices)

        iou_pos_targets = iou3d_nms_utils.boxes_aligned_iou3d_gpu(qboxes[box_inds], gboxes[box_inds]).detach()

        iou_targets.view(-1)[box_inds] = iou_pos_targets.squeeze(-1)
        iou_targets = torch.clamp(iou_targets, 0, 1)

        return iou_targets, qboxes, gboxes
    
    def _get_predicted_boxes(self, pred_boxes, spatial_indices):
        center, center_z, dim, rot_cos, rot_sin = pred_boxes[..., :2], pred_boxes[..., 2:3], pred_boxes[..., 3:6], \
                                                  pred_boxes[..., 6:7], pred_boxes[..., 7:8]
        dim = torch.exp(torch.clamp(dim, min=-5, max=5))
        angle = torch.atan2(rot_sin, rot_cos)
        xs = (spatial_indices[:, 0:1] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = (spatial_indices[:, 1:2] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        box_part_list = [xs, ys, center_z, dim, angle]
        pred_box = torch.cat((box_part_list), dim=-1)
        return pred_box


    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_bbox = pred_dict['bbox'].view(batch_size, 8, -1)
            batch_center = batch_bbox[:, :2]
            batch_center_z = batch_bbox[:, 2:3]
            batch_dim = batch_bbox[:, 3:6].exp()
            batch_rot_cos = batch_bbox[:, 6:7].sigmoid() * 2 - 1
            batch_rot_sin = batch_bbox[:, 7:8].sigmoid() * 2 - 1

            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            batch_iou = (pred_dict['iou'].view(batch_size, 1, -1) + 1) * 0.5 if 'iou' in pred_dict else None

            final_pred_dicts = dcdet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]

                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_iou' in final_dict:
                    pred_iou = torch.clamp(final_dict['pred_iou'], min=0, max=1.0)
                    IOU_RECTIFIER = final_dict['pred_scores'].new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(final_dict['pred_scores'], 1 - IOU_RECTIFIER[final_dict['pred_labels']]) * torch.pow(pred_iou, IOU_RECTIFIER[final_dict['pred_labels']])

                if post_process_cfg.NMS_CONFIG.NMS_TYPE not in  ['circle_nms', 'class_specific_nms']:
                    selected, selected_scores = model_nms_utils.class_agnostic_niv_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                    )
                elif post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms':
                    raise NotImplementedError

                final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                final_dict['pred_scores'] = selected_scores
                final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict
        
    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], spatial_shape=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict
            self.forward_ret_dict['batch_size'] = data_dict['batch_size']

        self.forward_ret_dict['pred_dicts'] = pred_dicts
        
        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict
