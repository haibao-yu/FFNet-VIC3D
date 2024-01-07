# Copyright (c) DAIR-V2X (AIR). All rights reserved.
import torch
from torch import nn as nn
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS, build_backbone, build_neck
from .. import builder
from .single_stage import SingleStage3DDetector
import copy
import random
import json
import os
import numpy as np
from pypcd import pypcd

def QuantFunc(input, b_n=4):
    alpha = torch.abs(input).max()
    s_alpha = alpha / (2 ** (b_n - 1) - 1)

    input = input.clamp(min=-alpha,max=alpha)
    input = torch.round(input / s_alpha)
    input = input * s_alpha

    return input

def AttentionMask(image_1, image_2, img_shape=(576, 576), mask_shape=(36, 36), threshold=0.0):
    mask = torch.zeros((image_1.shape[0], mask_shape[0], 
                                            mask_shape[1])).cuda(image_1.device)

    feat_diff = torch.sum(torch.abs(image_1 - image_2), dim=1)
    stride = int(img_shape[0] / mask_shape[0])
    for bs in range(image_1.shape[0]):
        for kk in range(mask_shape[0]):
            for ll in range(mask_shape[1]):
                patch = feat_diff[bs, kk*stride:(kk+1)*stride, ll*stride:(ll+1)*stride]
                if patch.sum() > threshold:
                    mask[bs, kk, ll] = 1
    # sparse_ratio = mask.sum() / mask.numel()
    # print("Sparse Ratio: ", sparse_ratio)

    return mask

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

        self.with_quant = False

    def forward(self, x, attention_mask=None):
        outputsize = x.shape
        #out = F.relu(self.bn1_1(self.conv1_1(x)))
        out = F.relu(self.bn1_2(self.conv1_2(x)))
        out = F.relu(self.bn1_3(self.conv1_3(out)))
        out = F.relu(self.bn1_4(self.conv1_4(out)))

        if attention_mask is not None:
            for bs in range(out.shape[0]):
                out[bs] = attention_mask[bs] * out[bs]

        if self.with_quant:
            for ii in range(out.shape[0]):
                out[ii] = QuantFunc(out[ii])
         
        out = F.relu(self.bn2_1(self.deconv2_1(out)))
        out = F.relu(self.bn2_2(self.deconv2_2(out)))
        x_1 = F.relu(self.bn2_3(self.deconv2_3(out)))
        #x_1 = F.relu(self.bn2_4(self.deconv2_4(out)))
        return x_1

def read_json(path):
    with open(path, "r") as f:
        my_json = json.load(f)
        return my_json

class PixelWeightedFusion(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusion, self).__init__()
        self.conv1_1 = nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        return x_1

class FlowGenerator(nn.Module):
    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 with_neck=False,
                 neck=None):
        super(FlowGenerator, self).__init__()
        backbone_flow = copy.deepcopy(backbone)
        backbone_flow['in_channels'] = backbone['in_channels'] * 2
        self.inf_backbone = build_backbone(backbone_flow)
        self.inf_with_neck = with_neck
        if neck is not None:
            self.inf_neck = build_neck(neck)
        self.inf_voxel_layer = Voxelization(**voxel_layer)
        self.inf_voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.inf_middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.pre_encoder = ReduceInfTC(768)
        self.with_attention_mask = False
        
    def forward(self, points_t_0, points_t_1):
        voxels_t_0, num_points_t_0, coors_t_0 = self.inf_voxelize(points_t_0)
        voxel_features_t_0 = self.inf_voxel_encoder(voxels_t_0, num_points_t_0, coors_t_0)
        batch_size_t_0 = coors_t_0[-1, 0].item() + 1
        feat_t_0 = self.inf_middle_encoder(voxel_features_t_0, coors_t_0, batch_size_t_0)
        
        voxels_t_1, num_points_t_1, coors_t_1 = self.inf_voxelize(points_t_1)
        voxel_features_t_1 = self.inf_voxel_encoder(voxels_t_1, num_points_t_1, coors_t_1)
        batch_size_t_1 = coors_t_1[-1, 0].item() + 1
        feat_t_1 = self.inf_middle_encoder(voxel_features_t_1, coors_t_1, batch_size_t_1)

        flow_pred = torch.cat([feat_t_0, feat_t_1], dim=1)
        flow_pred = self.inf_backbone(flow_pred)
        if self.inf_with_neck:
            flow_pred = self.inf_neck(flow_pred)
        
        if self.with_attention_mask:
            attention_mask = AttentionMask(feat_t_0, feat_t_1)
            flow_pred[0] = self.pre_encoder(flow_pred[0], attention_mask)
        else:
            flow_pred[0] = self.pre_encoder(flow_pred[0])
        return flow_pred

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

@DETECTORS.register_module()
class FeatureFlowNet(SingleStage3DDetector):
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
        super(FeatureFlowNet, self).__init__(
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
        self.fusion_training = False
        self.flow_training = True
        self.mse_loss = nn.MSELoss()
        self.flownet = FlowGenerator(
            voxel_layer,
            voxel_encoder,
            middle_encoder,
            backbone,
            with_neck=self.with_neck,
            neck=neck)
        self.encoder = ReduceInfTC(768)
        
        try:
            self.data_root = train_cfg['data_root']
            self.pretraind_checkpoint_path = train_cfg['pretrained_model']
        except:
            pass
            self.data_root = test_cfg['data_root']
            self.pretraind_checkpoint_path = test_cfg['pretrained_model']
        self.flownet_pretrained = False
        if 'test_mode' in test_cfg.keys():
            self.test_mode = test_cfg['test_mode']
        else:
            self.test_mode = "FlowPred"
        self.count = 0
        
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

    def generate_matrix(self,theta,x0,y0):
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
    
    def points_shuffle(self, points, max_points_num=20000):
        import random
        for ii in range(len(points)):
            if len(points[ii]) > max_points_num:
                points_idxs = [jj for jj in range(len(points[ii]))]
                random.shuffle(points_idxs)
                points[ii] = points[ii][points_idxs[:max_points_num]]
        return points
            
    def feature_fusion(self, veh_x, inf_x, img_metas, mode='fusion'):
        
        wrap_feats_ii = []
        for ii in range(len(veh_x[0])):
            inf_feature = inf_x[0][ii:ii+1]
            veh_feature = veh_x[0][ii:ii+1]

            calib_inf2veh_rotation = img_metas[ii]['inf2veh']['rotation']
            calib_inf2veh_translation = img_metas[ii]['inf2veh']['translation']
            inf_pointcloud_range = self.inf_voxel_layer.point_cloud_range

            theta_rot = torch.tensor([[calib_inf2veh_rotation[0][0], -calib_inf2veh_rotation[0][1], 0.0],
                                      [-calib_inf2veh_rotation[1][0], calib_inf2veh_rotation[1][1], 0.0],
                                      [0,0,1]]).type(dtype=torch.float).cuda(next(self.parameters()).device)
            theta_rot= torch.FloatTensor(self.generate_matrix(theta_rot,-1,0)). \
                type(dtype=torch.float).cuda(next(self.parameters()).device)

            x_trans = -2 * calib_inf2veh_translation[0][0] / (inf_pointcloud_range[3] - inf_pointcloud_range[0])
            y_trans = -2 * calib_inf2veh_translation[1][0] / (inf_pointcloud_range[4] - inf_pointcloud_range[1])
            theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans], \
                [0.0, 0.0 , 1]]).type(dtype=torch.float).cuda(next(self.parameters()).device)
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

    def flownet_init(self):
        pretraind_checkpoint_path = self.pretraind_checkpoint_path
        flownet_pretrained = self.flownet_pretrained
        pretraind_checkpoint = torch.load(pretraind_checkpoint_path)['state_dict']
        pretraind_checkpoint_modify = {}

        checkpoint_source = 'single_infrastructure_side'
        for k, v in pretraind_checkpoint.items():
            if 'inf_' in k:
                checkpoint_source = 'v2x_voxelnet'
                break

        if checkpoint_source == 'single_infrastructure_side':
            for k, v in pretraind_checkpoint.items():
                pretraind_checkpoint_modify['inf_' + k] = v
                if flownet_pretrained:
                    pretraind_checkpoint_modify['flownet.inf_' + k] = v
        elif checkpoint_source == 'v2x_voxelnet':
            for k, v in pretraind_checkpoint.items():
                if 'inf_' in k and flownet_pretrained:
                    pretraind_checkpoint_modify['flownet.' + k] = v
                pretraind_checkpoint_modify[k] = v

        self.load_state_dict(pretraind_checkpoint_modify, strict=False)

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      infrastructure_points=None,
                      gt_bboxes_ignore=None):
        """Training forward function."""
        for ii in range(len(infrastructure_points)):
            infrastructure_points[ii][:, 3] = 255 * infrastructure_points[ii][:, 3]
        if self.fusion_training:
           
            feat_veh = self.extract_feat(points, img_metas, points_view='vehicle')
            feat_inf = self.extract_feat(infrastructure_points, img_metas, points_view='infrastructure')
            feat_fused = self.feature_fusion(feat_veh, feat_inf, img_metas)
            outs = self.bbox_head(feat_fused)
            loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
            losses = self.bbox_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if self.flow_training:
            inf_points_t_0 = []
            inf_points_t_1 = []
            inf_points_t_2 = []
            points_t_0_1 = []
            points_t_1_2 = []
            
            for ii in range(len(points)):
                inf_points_path_t_0 = os.path.join(self.data_root,
                                                   img_metas[ii]['infrastructure_pointcloud_bin_path_t_0'])
                tem_inf_points = torch.from_numpy(np.fromfile(inf_points_path_t_0, dtype=np.float32))
                tem_inf_points = torch.reshape(tem_inf_points, (-1, 4)).cuda(device=points[0].device)
                inf_points_t_0.append(tem_inf_points)

                inf_points_path_t_1 = os.path.join(self.data_root,
                                                   img_metas[ii]['infrastructure_pointcloud_bin_path_t_1'])
                tem_inf_points = torch.from_numpy(np.fromfile(inf_points_path_t_1, dtype=np.float32))
                tem_inf_points = torch.reshape(tem_inf_points, (-1, 4)).cuda(device=points[0].device)
                inf_points_t_1.append(tem_inf_points)

                inf_points_path_t_2 = os.path.join(self.data_root,
                                                   img_metas[ii]['infrastructure_pointcloud_bin_path_t_2'])
                tem_inf_points = torch.from_numpy(np.fromfile(inf_points_path_t_2, dtype=np.float32))
                tem_inf_points = torch.reshape(tem_inf_points, (-1, 4)).cuda(device=points[0].device)
                inf_points_t_2.append(tem_inf_points)

                points_t_0_1.append(
                    img_metas[ii]['infrastructure_t_0_1'])
                points_t_1_2.append(
                    img_metas[ii]['infrastructure_t_1_2'])
            
            for ii in range(len(inf_points_t_0)):
                inf_points_t_0[ii][:,3] = 255 * inf_points_t_0[ii][:,3]
                inf_points_t_1[ii][:,3] = 255 * inf_points_t_1[ii][:,3]
                inf_points_t_2[ii][:,3] = 255 * inf_points_t_2[ii][:,3]
            feat_inf_t_1 = self.extract_feat(inf_points_t_1, img_metas, points_view='infrastructure')
            feat_inf_t_2 = self.extract_feat(inf_points_t_2, img_metas, points_view='infrastructure')
            for ii in range(len(feat_inf_t_1)):
                feat_inf_t_1[ii] = feat_inf_t_1[ii].detach()
                feat_inf_t_2[ii] = feat_inf_t_2[ii].detach()
            
            loss_type = 'similarity_loss'         
            if loss_type == 'mse_loss':
                flow_pred = self.flownet(inf_points_t_0, inf_points_t_1)
                feat_inf_apprs = []
                for ii in range(len(flow_pred)):
                    for bs in range(len(points)):
                        feat_inf_t_1[ii][bs] = feat_inf_t_1[ii][bs] + flow_pred[ii][bs] / points_t_0_1[bs] * points_t_1_2[bs]
                    feat_inf_apprs.append(feat_inf_t_1[ii])
                
                similarity = torch.cosine_similarity(torch.flatten(feat_inf_t_2[0], start_dim=1, end_dim=3),
                                                 torch.flatten(feat_inf_apprs[0], start_dim=1, end_dim=3), dim=1)
                # print("The similarity is: ", similarity, points_t_1_2)

                for ii in range(len(feat_inf_apprs)):
                    if ii == 0:
                        if not self.fusion_training:
                            losses = {}
                        losses['mse_loss'] = self.mse_loss(feat_inf_apprs[ii], feat_inf_t_2[ii]) 
                    else:
                        losses['mse_loss'] = losses['mse_loss'] + self.mse_loss(feat_inf_apprs[ii], feat_inf_t_2[ii])
            
            if loss_type == 'similarity_loss':
                flow_pred = self.flownet(inf_points_t_0, inf_points_t_1)
                feat_inf_apprs = []
                for ii in range(len(flow_pred)):
                    for bs in range(len(points)):
                        tem_feat_inf_t_1_before_max = feat_inf_t_1[ii][bs].mean()
                        feat_inf_t_1[ii][bs] = feat_inf_t_1[ii][bs] + flow_pred[ii][bs] / points_t_0_1[bs] * points_t_1_2[bs]
                        tem_feat_inf_t_1_after_max = feat_inf_t_1[ii][bs].mean().detach()
                        feat_inf_t_1[ii][bs] = feat_inf_t_1[ii][bs] / tem_feat_inf_t_1_after_max * tem_feat_inf_t_1_before_max
                    feat_inf_apprs.append(feat_inf_t_1[ii])
                
                similarity = torch.cosine_similarity(torch.flatten(feat_inf_t_2[0], start_dim=1, end_dim=3),
                                                 torch.flatten(feat_inf_apprs[0], start_dim=1, end_dim=3), dim=1)
                # print("The similarity is: ", similarity, points_t_1_2)
                
                label = torch.ones(len(points), requires_grad=False).cuda(device=points[0].device)
                if not self.fusion_training:
                    losses = {}
                losses['similarity_loss'] = self.mse_loss(similarity, label)

                
            '''visualize the features
            import cv2 as cv
            for bs, data_info_idx in enumerate(data_info_idxs):
                inf_points_t_2_idx = self.flow_data_infos[data_info_idx]['infrastructure_idx_t_2']

                channel_idx = 1
                tensor_save = torch.sigmoid(feat_inf_t_2[0][bs, channel_idx]) * 255
                img_save_path = os.path.join('vis_feature', 'feat_t_2_' + inf_points_t_2_idx + '_' + str(channel_idx) + '.png')
                cv.imwrite(img_save_path, tensor_save.cpu().numpy())
                tensor_save = torch.sigmoid(feat_inf_apprs[0][bs, channel_idx]) * 255
                img_save_path = os.path.join('vis_feature', 'feat_t_2_' + inf_points_t_2_idx + '_appr_' + str(channel_idx) + '.png')
                cv.imwrite(img_save_path, tensor_save.detach().cpu().numpy())
            '''

        return losses

    def simple_test(self, points, img_metas, imgs=None, infrastructure_points=None, rescale=False):
        """Test function without augmentaiton."""
        for ii in range(len(infrastructure_points[0])):
            infrastructure_points[0][ii][:, 3] = 255 * infrastructure_points[0][ii][:, 3]
        feat_veh = self.extract_feat(points, img_metas, points_view='vehicle')
        feat_inf = self.extract_feat(infrastructure_points[0], img_metas, points_view='infrastructure')
        
        if self.test_mode not in ['FlowPred', 'OriginFeat', 'Async']:
            raise Exception('FlowNet Test Mode is Error: {}'.format(self.test_mode))
        
        if self.test_mode == "OriginFeat":
            feat_inf = self.extract_feat(infrastructure_points[0], img_metas, points_view='infrastructure')
            
        if self.test_mode == "Async":
            inf_points_t_1 = []
            
            for ii in range(len(points)):
                inf_points_path_t_1 = os.path.join(self.data_root,
                                                   img_metas[ii]['infrastructure_pointcloud_bin_path_t_1'])
                tem_inf_points = torch.from_numpy(np.fromfile(inf_points_path_t_1, dtype=np.float32))
                tem_inf_points = torch.reshape(tem_inf_points, (-1, 4)).cuda(device=points[0].device)
                inf_points_t_1.append(tem_inf_points)
            for ii in range(len(inf_points_t_1)):
                inf_points_t_1[ii][:,3] = 255 * inf_points_t_1[ii][:,3]
            feat_inf_t_1 = self.extract_feat(inf_points_t_1, img_metas, points_view='infrastructure')
            similarity = torch.cosine_similarity(torch.flatten(feat_inf_t_1[0], start_dim=1, end_dim=3),
                                                 torch.flatten(feat_inf[0], start_dim=1, end_dim=3), dim=1)
            # print("The similarity is: ", similarity)
            
            feat_inf = feat_inf_t_1

        if self.test_mode == "FlowPred":
            inf_points_t_0 = []
            inf_points_t_1 = []
            inf_points_t_2 = []
            points_t_0_1 = []
            points_t_1_2 = []
            if not os.path.exists('./result'):
                os.mkdir('./result')
            for ii in range(len(points)):
                inf_points_path_t_0 = os.path.join(self.data_root,
                                                   img_metas[ii]['infrastructure_pointcloud_bin_path_t_0'])
                tem_inf_points = torch.from_numpy(np.fromfile(inf_points_path_t_0, dtype=np.float32))
                tem_inf_points = torch.reshape(tem_inf_points, (-1, 4)).cuda(device=points[0].device)
                inf_points_t_0.append(tem_inf_points)

                inf_points_path_t_1 = os.path.join(self.data_root,
                                                   img_metas[ii]['infrastructure_pointcloud_bin_path_t_1'])
                tem_inf_points = torch.from_numpy(np.fromfile(inf_points_path_t_1, dtype=np.float32))
                tem_inf_points = torch.reshape(tem_inf_points, (-1, 4)).cuda(device=points[0].device)
                inf_points_t_1.append(tem_inf_points)

                inf_points_path_t_2 = os.path.join(self.data_root,
                                                   img_metas[ii]['infrastructure_pointcloud_bin_path_t_2'])
                tem_inf_points = torch.from_numpy(np.fromfile(inf_points_path_t_2, dtype=np.float32))
                tem_inf_points = torch.reshape(tem_inf_points, (-1, 4)).cuda(device=points[0].device)
                inf_points_t_2.append(tem_inf_points)

                points_t_0_1.append(
                    img_metas[ii]['infrastructure_t_0_1'])
                points_t_1_2.append(
                    img_metas[ii]['infrastructure_t_1_2'])
            
            for ii in range(len(inf_points_t_0)):
                inf_points_t_0[ii][:,3] = 255 * inf_points_t_0[ii][:,3]
                inf_points_t_1[ii][:,3] = 255 * inf_points_t_1[ii][:,3]
                inf_points_t_2[ii][:,3] = 255 * inf_points_t_2[ii][:,3]
            
            feat_inf_t_1 = self.extract_feat(inf_points_t_1, img_metas, points_view='infrastructure')
            feat_inf_t_2 = self.extract_feat(inf_points_t_2, img_metas, points_view='infrastructure')
            feat_inf_temp = self.extract_feat(inf_points_t_2, img_metas, points_view='infrastructure')
            flow_pred = self.flownet(inf_points_t_0, inf_points_t_1)
            feat_inf_apprs = []
            for ii in range(len(flow_pred)):
                for bs in range(len(points)):
                    tem_feat_inf_t_1_before_max = feat_inf_t_1[ii][bs].mean()
                    feat_inf_temp[ii][bs] = feat_inf_t_1[ii][bs] + flow_pred[ii][bs] / points_t_0_1[bs] * points_t_1_2[bs]
                    tem_feat_inf_t_1_after_max = feat_inf_temp[ii][bs].mean().detach()
                    feat_inf_temp[ii][bs] = feat_inf_temp[ii][bs] / tem_feat_inf_t_1_after_max * tem_feat_inf_t_1_before_max
                feat_inf_apprs.append(feat_inf_temp[ii])
            
            similarity = torch.cosine_similarity(torch.flatten(feat_inf_t_2[0], start_dim=1, end_dim=3),
                                                 torch.flatten(feat_inf_apprs[0], start_dim=1, end_dim=3), dim=1)
            # print("The similarity is: ", similarity, points_t_1_2)
            
            feat_inf = feat_inf_apprs
        feat_fused = self.feature_fusion(feat_veh, feat_inf, img_metas)
        
        '''
        import cv2
        hot_map_feat_cat = np.zeros((288, 288))+255
        #flow_pred feat_inf_t_1  feat_inf_t_2
        self.count+=1
        hot_map_feat_flow=np.zeros((288,288))
        for ii in range(flow_pred[0].shape[1]):
                hot_map_feat_flow = hot_map_feat_flow + torch.abs(flow_pred[0][0, ii]).cpu().detach().numpy()
        hot_map_feat_pred=np.zeros((288,288))
        for ii in range(feat_inf_apprs[0].shape[1]):
                hot_map_feat_pred = hot_map_feat_pred + torch.abs(feat_inf_apprs[0][0, ii]).cpu().detach().numpy()
        hot_map_feat_t1=np.zeros((288,288))
        for ii in range(feat_inf_t_1[0].shape[1]):
                hot_map_feat_t1 = hot_map_feat_t1 + torch.abs(feat_inf_t_1[0][0, ii]).cpu().detach().numpy()
        hot_map_feat_t2=np.zeros((288,288))
        for ii in range(feat_inf_t_2[0].shape[1]):
                hot_map_feat_t2 = hot_map_feat_t2 + torch.abs(feat_inf_t_2[0][0, ii]).cpu().detach().numpy()
        hot_map_feat_veh=np.zeros((288,288))
        for ii in range(feat_veh[0].shape[1]):
                hot_map_feat_veh = hot_map_feat_veh + torch.abs(feat_veh[0][0, ii]).cpu().detach().numpy()
        
        cv2.imwrite('./result/flow_'+str(self.count)+'.png', hot_map_feat_cat-hot_map_feat_flow*10)
        cv2.imwrite('./result/pred_'+str(self.count)+'.png', hot_map_feat_cat-hot_map_feat_pred*10)
        cv2.imwrite('./result/t2_'+str(self.count)+'.png', hot_map_feat_cat-hot_map_feat_t2*10)
        cv2.imwrite('./result/t1_'+str(self.count)+'.png', hot_map_feat_cat-hot_map_feat_t1*10)
        '''
                
        '''
        hot_map_feat_fused=np.zeros((288,288))
        for ii in range(feat_fused[0].shape[1]):
                hot_map_feat_fused = hot_map_feat_fused + torch.abs(feat_fused[0][0, ii]).cpu().detach().numpy()
        cv2.imwrite('./result/veh_'+str(self.count)+'.png', hot_map_feat_cat-hot_map_feat_veh*10)
        cv2.imwrite('./result/fusion_'+str(self.count)+'.png', hot_map_feat_cat-hot_map_feat_fused*10)
        # cv2.imwrite('./result/veh_'+str(self.count)+'.png', hot_map_feat_cat-feat_veh[0][0,79].cpu().detach().numpy()*7000)
        # cv2.imwrite('./result/fusion_'+str(self.count)+'.png', hot_map_feat_cat-feat_fused[0][0,79].cpu().detach().numpy()*7000)

        # cv2.imwrite('./result/flow_'+str(self.count)+'.png', hot_map_feat_cat-flow_pred[0][0,79].cpu().detach().numpy()*7000)
        # cv2.imwrite('./result/pred_'+str(self.count)+'.png', hot_map_feat_cat-feat_inf_apprs[0][0,79].cpu().detach().numpy()*7000)
        # cv2.imwrite('./result/t2_'+str(self.count)+'.png', hot_map_feat_cat-feat_inf_t_2[0][0,79].cpu().detach().numpy()*10000)
        # cv2.imwrite('./result/t1_'+str(self.count)+'.png', hot_map_feat_cat-feat_inf_t_1[0][0,79].cpu().detach().numpy()*7000)

        # feat_fused = self.feature_fusion(feat_veh, feat_inf, img_metas)
        # cv2.imwrite('./result/veh_'+str(self.count)+'.png', hot_map_feat_cat-feat_veh[0][0,79].cpu().detach().numpy()*7000)
        # cv2.imwrite('./result/fusion_'+str(self.count)+'.png', hot_map_feat_cat-feat_fused[0][0,79].cpu().detach().numpy()*7000)
        '''

        outs = self.bbox_head(feat_fused)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self):
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