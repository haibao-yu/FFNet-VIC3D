# Copyright (c) DAIR-V2X (AIR). All rights reserved.
import torch
from torch import nn as nn
from mmcv.runner import force_fp32
from torch.nn import functional as F
import os
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from .single_stage import SingleStage3DDetector

class ReduceInfTC(nn.Module):
    def __init__(self, channel):
        super(ReduceInfTC, self).__init__()
        self.conv1_2 = nn.Conv2d(channel//2, channel//4, kernel_size=3, stride=2, padding=0)
        self.bn1_2 = nn.BatchNorm2d(channel//4, track_running_stats=True)
        self.conv1_3 = nn.Conv2d(channel//4, channel//8, kernel_size=3, stride=2, padding=0)
        self.bn1_3 = nn.BatchNorm2d(channel//8, track_running_stats=True)
        self.conv1_4 = nn.Conv2d(channel//8, channel//64, kernel_size=3, stride=2, padding=1)
        self.bn1_4 = nn.BatchNorm2d(channel//64, track_running_stats=True)

        self.deconv2_1 = nn.ConvTranspose2d(channel//64, channel//8, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm2d(channel//8, track_running_stats=True)
        self.deconv2_2 = nn.ConvTranspose2d(channel//8, channel//4, kernel_size=3, stride=2, padding=0)
        self.bn2_2 = nn.BatchNorm2d(channel//4, track_running_stats=True)
        self.deconv2_3 = nn.ConvTranspose2d(channel//4, channel//2, kernel_size=3, stride=2, padding=0, output_padding=1)
        self.bn2_3 = nn.BatchNorm2d(channel//2, track_running_stats=True)


    def forward(self, x):
        outputsize = x.shape
        #out = F.relu(self.bn1_1(self.conv1_1(x)))
        out = F.relu(self.bn1_2(self.conv1_2(x)))
        out = F.relu(self.bn1_3(self.conv1_3(out)))
        out = F.relu(self.bn1_4(self.conv1_4(out)))
         
        out = F.relu(self.bn2_1(self.deconv2_1(out)))
        out = F.relu(self.bn2_2(self.deconv2_2(out)))
        x_1 = F.relu(self.bn2_3(self.deconv2_3(out)))
        
        #x_1 = F.relu(self.bn2_4(self.deconv2_4(out)))
        return x_1
         
class PixelWeightedFusion(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusion, self).__init__()
        self.conv1_1 = nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        return x_1

@DETECTORS.register_module()
class V2XVoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(V2XVoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)

        self.inf_voxel_layer = Voxelization(**voxel_layer)
        self.inf_voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.inf_middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.inf_backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.inf_neck = builder.build_neck(neck)

        # TODO: channel configuration
        self.fusion_weighted = PixelWeightedFusion(384)
        self.encoder = ReduceInfTC(768)
        
    def generate_matrix(self,theta,x0,y0):
        import numpy as np
        c = theta[0][0]
        s = theta[1][0]
        matrix = np.zeros((3,3))
        matrix[0,0] = c
        matrix[0,1] = -s
        matrix[1,0] = s
        matrix[1,1] = c
        matrix[0,2] = -c * x0 + s * y0 + x0
        matrix[1,2] =  -c * y0 - s * x0 + y0
        matrix[2,2] = 1
        return matrix

    def extract_feat(self, points, img_metas=None, points_view='vehicle'):
        """Extract features from points."""
        if points_view == 'vehicle':
            voxels, num_points, coors = self.voxelize(points)
            voxel_features = self.voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0].item() + 1
            veh_x = self.middle_encoder(voxel_features, coors, batch_size)
            veh_x = self.backbone(veh_x)
            if self.with_neck:
                veh_x = self.neck(veh_x)
            return veh_x

        elif points_view == 'infrastructure':
            inf_voxels, inf_num_points, inf_coors = self.inf_voxelize(points)
            inf_voxel_features = self.inf_voxel_encoder(inf_voxels, inf_num_points, inf_coors)
            inf_batch_size = inf_coors[-1, 0].item() + 1
            inf_x = self.inf_middle_encoder(inf_voxel_features, inf_coors, inf_batch_size)
            inf_x = self.inf_backbone(inf_x)
            if self.with_neck:
                inf_x = self.inf_neck(inf_x)
            
            inf_x[0] = self.encoder(inf_x[0])
            return inf_x
        else:
            raise Exception('Points View is Error: {}'.format(points_view))

    
    def feature_fusion(self, veh_x, inf_x, img_metas, mode='fusion'):

        """ Method II: Based on affine transformation."""
        wrap_feats_ii = []
        
        '''
        for ii in range(len(veh_x[0])):
            inf_feature = inf_x[0][ii:ii+1]
            veh_feature = veh_x[0][ii:ii+1]

            calib_inf2veh_rotation = img_metas[ii]['inf2veh']['rotation']
            calib_inf2veh_translation = img_metas[ii]['inf2veh']['translation']
            inf_pointcloud_range = self.inf_voxel_layer.point_cloud_range
            # theta_rot = [[cos(-theta), sin(-theta), 0.0], [cos(-theta), sin(-theta), 0.0]], theta is in the lidar coordinate.
            # according to the relationship between lidar coordinate system and input coordinate system.
            theta_rot = torch.tensor([[calib_inf2veh_rotation[0][0], -calib_inf2veh_rotation[0][1], 0.0],
                                      [-calib_inf2veh_rotation[1][0], calib_inf2veh_rotation[1][1], 0.0]]).type(dtype=torch.float).cuda(next(self.parameters()).device)
            theta_rot = torch.unsqueeze(theta_rot, 0)
            grid_rot = F.affine_grid(theta_rot, size=torch.Size(veh_feature.shape), align_corners=False)
            # range: [-1, 1].
            # Moving right and down is negative.
            x_trans = -2 * calib_inf2veh_translation[0][0] / (inf_pointcloud_range[3] - inf_pointcloud_range[0])
            y_trans = -2 * calib_inf2veh_translation[1][0] / (inf_pointcloud_range[4] - inf_pointcloud_range[1])
            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).cuda(next(self.parameters()).device)
            theta_trans = torch.unsqueeze(theta_trans, 0)
            grid_trans = F.affine_grid(theta_trans, size=torch.Size(veh_feature.shape), align_corners=False)

            warp_feat_rot = F.grid_sample(inf_feature, grid_rot, mode='bilinear', align_corners=False)
            warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='bilinear', align_corners=False)

            wrap_feats_ii.append(warp_feat_trans)
        '''
        for ii in range(len(veh_x[0])):
            inf_feature = inf_x[0][ii:ii+1]
            veh_feature = veh_x[0][ii:ii+1]

            calib_inf2veh_rotation = img_metas[ii]['inf2veh']['rotation']
            calib_inf2veh_translation = img_metas[ii]['inf2veh']['translation']
            inf_pointcloud_range = self.inf_voxel_layer.point_cloud_range

            theta_rot = torch.tensor([[calib_inf2veh_rotation[0][0], -calib_inf2veh_rotation[0][1], 0.0],
                                      [-calib_inf2veh_rotation[1][0], calib_inf2veh_rotation[1][1], 0.0],
                                      [0,0,1]]).type(dtype=torch.float).cuda(next(self.parameters()).device)
            theta_rot= torch.FloatTensor(self.generate_matrix(theta_rot,-1,0)).type(dtype=torch.float).cuda(next(self.parameters()).device)
            # Moving right and down is negative.
            x_trans = -2 * calib_inf2veh_translation[0][0] / (inf_pointcloud_range[3] - inf_pointcloud_range[0])
            y_trans = -2 * calib_inf2veh_translation[1][0] / (inf_pointcloud_range[4] - inf_pointcloud_range[1])
            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans],[0.0, 0.0 , 1]]).type(dtype=torch.float).cuda(next(self.parameters()).device)
            theta_r_t=torch.mm(theta_rot,theta_trans, out=None)

            grid_r_t = F.affine_grid(theta_r_t[0:2].unsqueeze(0), size=torch.Size(veh_feature.shape), align_corners=False)
            warp_feat_trans = F.grid_sample(inf_feature, grid_r_t, mode='bilinear', align_corners=False)
            wrap_feats_ii.append(warp_feat_trans)
            
        wrap_feats = [torch.cat(wrap_feats_ii, dim=0)]
            
        if mode not in ['fusion', 'inf_only', 'veh_only']:
            raise Exception("Mode is Error: {}".format(mode))
        if mode == 'inf_only':
            return wrap_feats
        elif mode == 'veh_only':
            return veh_x
        veh_cat_feats = [torch.cat([veh_x[0], wrap_feats[0]], dim=1)]
        veh_cat_feats[0] = self.fusion_weighted(veh_cat_feats[0])

        return veh_cat_feats

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @torch.no_grad()
    @force_fp32()
    def inf_voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.inf_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def points_grid(self, points, point_cloud_range, voxel_size, feature_shape):
        """Assign each points with id.
        Args:
            feature_shape: N * C * H * W
        Returns:
            points_grids: N * 5, 0 - idx in H * W, [1, 2] - h, w idx in [H, W], [3, 4] - [0, 1] range for [H, W]
        Comment:
            LiDAR Coordinate Systems: x - forward, y - left, z - upward
        """
        origin_size_x = (point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]
        feat_scale = origin_size_x / feature_shape[3]

        points_grids = torch.zeros(len(points), 3).cuda()
        range_start_x = point_cloud_range[0]
        range_lenth_x = point_cloud_range[3] - point_cloud_range[0]
        voxel_size_x = voxel_size[0] * feat_scale
        W_L = range_lenth_x / voxel_size_x

        range_start_y = point_cloud_range[1]
        range_lenth_y = point_cloud_range[4] - point_cloud_range[1]
        voxel_size_y = voxel_size[1] * feat_scale
        H_L = range_lenth_y / voxel_size_y

        points_grids[:, 1] = torch.round((points[:, 1] - range_start_y) / voxel_size_y)
        points_grids[:, 2] = torch.round((points[:, 0] - range_start_x) / voxel_size_x)
        points_grids[:, 0] = points_grids[:, 1] * W_L + points_grids[:, 2]

        return points_grids, int(H_L), int(W_L)

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      infrastructure_points=None,
                      gt_bboxes_ignore=None,
                      ):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        for ii in range(len(infrastructure_points)):
            infrastructure_points[ii][:, 3] = 255 * infrastructure_points[ii][:, 3]
        feat_veh = self.extract_feat(points, img_metas, points_view='vehicle')
        feat_inf = self.extract_feat(infrastructure_points, img_metas, points_view='infrastructure')
        feat_fused = self.feature_fusion(feat_veh, feat_inf, img_metas, mode='fusion')
        outs = self.bbox_head(feat_fused)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, infrastructure_points=None, rescale=False):
        """Test function without augmentaiton."""
        for ii in range(len(infrastructure_points[0])):
            infrastructure_points[0][ii][:, 3] = 255 * infrastructure_points[0][ii][:, 3]
        
        feat_veh = self.extract_feat(points, img_metas, points_view='vehicle')
        feat_inf = self.extract_feat(infrastructure_points[0], img_metas, points_view='infrastructure')
        feat_fused = self.feature_fusion(feat_veh, feat_inf, img_metas, mode='fusion')
        outs = self.bbox_head(feat_fused)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        if len(bbox_list[0][0].tensor) == 0: # return a zero list when pre is empty
            bbox_list[0] = list(bbox_list[0])
            bbox_list[0][0].tensor = torch.zeros(1, 7).cuda(points[0].device)
            bbox_list[0][1] = torch.zeros(1).cuda(points[0].device)
            bbox_list[0][2] = torch.zeros(1).cuda(points[0].device)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""

        return None
    
    def tensor_to_pcd(self, tensor_points):
        size_float = 4
        list_pcd = []
        for ii in range(tensor_points.shape[0]):
            if tensor_points.shape[1] == 4:
                x, y, z, intensity = tensor_points[ii, 0], tensor_points[ii, 1], tensor_points[ii, 2], tensor_points[ii, 3]
            else:
                x, y, z = tensor_points[ii, 0], tensor_points[ii, 1], tensor_points[ii, 2]
                intensity = 1.0
            list_pcd.append((x, y, z, intensity))

        dt = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')])
        np_pcd = np.array(list_pcd, dtype=dt)

        new_metadata = {}
        new_metadata['version'] = '0.7'
        new_metadata['fields'] = ['x', 'y', 'z', 'intensity']
        new_metadata['size'] = [4, 4, 4, 4]
        new_metadata['type'] = ['F', 'F', 'F', 'F']
        new_metadata['count'] = [1, 1, 1, 1]
        new_metadata['width'] = len(np_pcd)
        new_metadata['height'] = 1
        new_metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        new_metadata['points'] = len(np_pcd)
        new_metadata['data'] = 'binary'
        pc_save = pypcd.PointCloud(new_metadata, np_pcd)
        
        return pc_save