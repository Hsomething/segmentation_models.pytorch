# -*- coding: UTF-8 -*-
# @Time        :   1:57
# @Author      :  Huangxiao
# @application :  
# @File        :  test.py
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn), Jingyi Xie (hsfzxjy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from ._base import EncoderMixin
# BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d
relu_inplace = True
# from .bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace

ALIGN_CORNERS = True
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats) \
            .permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, num_classes,align_corners,block = ["BOTTLENECK",'BASIC','BASIC','BASIC'],
                 fuse_method = "SUM",
                 num_modules = [1,1,4,3],
                 num_branches = [1,2,3,4],
                 num_channels = [[64],[48,96],[48,96,192],[48,96,192,384]],
                 num_blocks = [[4],[4,4],[4,4,4],[4,4,4,4]],
                 ocr_mid_channels = 512,ocr_key_channels = 256,bn_momentum = 0.1,
                 relu_inplace = True, **kwargs):
        global ALIGN_CORNERS

        super(HighResolutionNet, self).__init__()
        ALIGN_CORNERS = align_corners
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.num_branches = num_branches
        self.blocks= block

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=relu_inplace)

        num_channels_ = self.num_channels[0][0]
        block_ = blocks_dict[self.blocks[0]]
        num_blocks_ = self.num_blocks[0][0]
        self.layer1 = self._make_layer(block_, 64, num_channels_, num_blocks_)
        stage1_out_channel = block_.expansion * num_channels_


        num_channels_ = self.num_channels[1]
        block_ = blocks_dict[self.blocks[1]]
        num_channels_ = [
            num_channels_[i] * block_.expansion for i in range(len(num_channels_))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels_)
        self.stage2, pre_stage_channels = self._make_stage(
            {
                "NUM_MODULES":num_modules[1],
                "NUM_BRANCHES":num_branches[1],
                "NUM_BLOCKS":self.num_blocks[1],
                "NUM_CHANNELS":num_channels_,
                "BLOCK":self.blocks[1],
                "FUSE_METHOD":fuse_method,
            }, num_channels_)

        num_channels_ = self.num_channels[2]
        block_ = blocks_dict[self.blocks[2]]
        num_channels_ = [
            num_channels_[i] * block_.expansion for i in range(len(num_channels_))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels_)
        self.stage3, pre_stage_channels = self._make_stage(
            {
                "NUM_MODULES": num_modules[2],
                "NUM_BRANCHES": num_branches[2],
                "NUM_BLOCKS": self.num_blocks[2],
                "NUM_CHANNELS": num_channels_,
                "BLOCK": self.blocks[2],
                "FUSE_METHOD": fuse_method,
            }, num_channels_)

        num_channels_ = self.num_channels[3]
        block_ = blocks_dict[self.blocks[3]]
        num_channels_ = [
            num_channels_[i] * block_.expansion for i in range(len(num_channels_))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels_)
        self.stage4, pre_stage_channels = self._make_stage(
            {
                "NUM_MODULES": num_modules[3],
                "NUM_BRANCHES": num_branches[3],
                "NUM_BLOCKS": self.num_blocks[3],
                "NUM_CHANNELS": num_channels_,
                "BLOCK": self.blocks[3],
                "FUSE_METHOD": fuse_method,
            }, num_channels_, multi_scale_output=True)

        # last_inp_channels = np.int(np.sum(pre_stage_channels))
        # ocr_mid_channels = ocr_mid_channels
        # ocr_key_channels = ocr_key_channels
        #
        # self.conv3x3_ocr = nn.Sequential(
        #     nn.Conv2d(last_inp_channels, ocr_mid_channels,
        #               kernel_size=3, stride=1, padding=1),
        #     BatchNorm2d(ocr_mid_channels),
        #     nn.ReLU(inplace=relu_inplace),
        # )
        # self.ocr_gather_head = SpatialGather_Module(num_classes)
        #
        # self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
        #                                          key_channels=ocr_key_channels,
        #                                          out_channels=ocr_mid_channels,
        #                                          scale=1,
        #                                          dropout=0.05,
        #                                          )
        # self.cls_head = nn.Conv2d(
        #     ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        #
        # self.aux_head = nn.Sequential(
        #     nn.Conv2d(last_inp_channels, last_inp_channels,
        #               kernel_size=1, stride=1, padding=0),
        #     BatchNorm2d(last_inp_channels),
        #     nn.ReLU(inplace=relu_inplace),
        #     nn.Conv2d(last_inp_channels, num_classes,
        #               kernel_size=1, stride=1, padding=0, bias=True)
        # )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        # x0_h, x0_w = x[0].size(2), x[0].size(3)
        # x1 = F.interpolate(x[1], size=(x0_h, x0_w),
        #                    mode='bilinear', align_corners=ALIGN_CORNERS)
        # x2 = F.interpolate(x[2], size=(x0_h, x0_w),
        #                    mode='bilinear', align_corners=ALIGN_CORNERS)
        # x3 = F.interpolate(x[3], size=(x0_h, x0_w),
        #                    mode='bilinear', align_corners=ALIGN_CORNERS)
        #
        # feats = torch.cat([x[0], x1, x2, x3], 1)
        #
        # out_aux_seg = []
        #
        # # ocr
        # out_aux = self.aux_head(feats)
        # # compute contrast feature
        # feats = self.conv3x3_ocr(feats)
        #
        # context = self.ocr_gather_head(feats, out_aux)
        # feats = self.ocr_distri_head(feats, context)
        #
        # out = self.cls_head(feats)
        #
        # out_aux_seg.append(out_aux)
        # out_aux_seg.append(out)

        # return out_aux_seg
        return x

    def init_weights(self, pretrained='', ):
        logger.info('=> init weights from normal distribution')
        for name, m in self.named_modules():
            if any(part in name for part in {'cls', 'aux', 'ocr'}):
                # print('skipped', name)
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location={'cuda:0': 'cpu'})
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in
                               pretrained_dict.items()}
            print(set(model_dict) - set(pretrained_dict))
            print(set(pretrained_dict) - set(model_dict))
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            # for k, _ in pretrained_dict.items():
            # logger.info(
            #     '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        elif pretrained:
            raise RuntimeError('No such file {}'.format(pretrained))


class HRNetEncoder(HighResolutionNet,EncoderMixin):
    def __init__(self,num_classes,out_channels,align_corners,
                 block = ["BOTTLENECK",'BASIC','BASIC','BASIC'],
                 fuse_method = "SUM",
                 num_modules = [1,1,4,3],
                 num_branches = [1,2,3,4],
                 num_channels = [[64],[48,96],[48,96,192],[48,96,192,384]],
                 num_blocks = [[4],[4,4],[4,4,4],[4,4,4,4]],bn_momentum = 0.1,
                 relu_inplace = True,**kwargs):
        super().__init__(num_classes= num_classes,align_corners = align_corners,block = block,fuse_method =fuse_method,
                         num_modules = num_modules,num_branches = num_branches,num_channels=num_channels,
                         num_blocks = num_blocks,bn_momentum = bn_momentum,relu_inplace = relu_inplace,**kwargs
                         )
        self._depth = 5
        self._out_channels = out_channels
        self._in_channels = 3

        # del self.aux_head,self.conv3x3_ocr,self.ocr_gather_head,self.ocr_distri_head,self.cls_head

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1,self.bn1,self.relu,self.conv2,self.bn2,self.relu),
            self.layer1,
            self.stage2,
            self.stage3,
            self.stage4
        ]

    def forward(self,x):
        stages = self.get_stages()
        features = []
        x = stages[0](x)
        features.append(x)
        x = stages[1](x)
        features.append(x)
        x = stages[2](x)
        x_list = []
        for i in range(self.num_branches[1]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.num_branches[2]):
            if self.transition2[i] is not None:
                if i < self.num_branches[1]:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.num_branches[3]):
            if self.transition3[i] is not None:
                if i < self.num_branches[2]:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        features.append(x[0])
        features.append(x[1])
        features.append(x[2])
        features.append(x[3])
        return features

    def load_state_dict(self, state_dict, **kwargs):
        model_state_dict = self.state_dict()
        pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in
                           state_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_state_dict.keys()}
        state_dict.update(pretrained_dict)
        super().load_state_dict(state_dict, **kwargs)


url_map = {
    "hrnet18-s1":"https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnet_w18_small_v1-f460c6bc.pth",
    "hrnet18-s2":"https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnet_w18_small_v2-4c50a8cb.pth",
    "hrnet18":"https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w18-8cb57bb9.pth",
    "hrnet30":"https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w30-8d7f8dab.pth",
    "hrnet32":"https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w32-90d8c5fb.pth",
    "hrnet40":"https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w40-7cd397a4.pth",
    "hrnet44":"https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w44-c9ac8c18.pth",
    "hrnet48":"https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w48-abd2e6ab.pth",
    "hrnet64":"https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-hrnet/hrnetv2_w64-b47cc881.pth",
}

def _get_pretrained_settings(encoder):
    pretrained_settings = {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": url_map[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    }
    return pretrained_settings


HRNetEncoders = {
    "hrnet18-s1":{
        "encoder": HRNetEncoder,
        "pretrained_settings": _get_pretrained_settings("hrnet18-s1"),
        "params": {
            "out_channels": (3, 64, 16, 32, 64, 128),
            "align_corners":True,
            "num_classes":3,
            "block": ["BOTTLENECK",'BASIC','BASIC','BASIC'],
            "fuse_method": "SUM",
            "num_modules": [1,1,1,1],
            "num_branches": [1,2,3,4],
            "num_channels": [[32],[16,32],[16,32,64],[16,32,64,128]],
            "num_blocks": [[1],[2,2],[2,2,2],[2,2,2,2]],
            "bn_momentum": 0.1,
            "relu_inplace": True,
            "model_name": "HRNet-18-s1",
        },
    },
    "hrnet18-s2":{
        "encoder": HRNetEncoder,
        "pretrained_settings": _get_pretrained_settings("hrnet18-s2"),
        "params": {
            "out_channels": (3, 64, 18, 36, 72, 144),
            "align_corners":True,
            "num_classes":3,
            "block": ["BOTTLENECK",'BASIC','BASIC','BASIC'],
            "fuse_method": "SUM",
            "num_modules": [1,1,3,2],
            "num_branches": [1,2,3,4],
            "num_channels": [[64],[18,36],[18,36,72],[18,36,72,144]],
            "num_blocks": [[2],[2,2],[2,2,2],[2,2,2,2]],
            "bn_momentum": 0.1,
            "relu_inplace": True,
            "model_name": "HRNet-18-s2",
        },
    },
    "hrnet18":{
        "encoder": HRNetEncoder,
        "pretrained_settings": _get_pretrained_settings("hrnet18"),
        "params": {
            "out_channels": (3, 64, 18, 36, 72, 144),
            "align_corners":True,
            "num_classes":3,
            "block": ["BOTTLENECK",'BASIC','BASIC','BASIC'],
            "fuse_method": "SUM",
            "num_modules": [1,1,4,3],
            "num_branches": [1,2,3,4],
            "num_channels": [[64],[18,36],[18,36,72],[18,36,72,144]],
            "num_blocks": [[4],[4,4],[4,4,4],[4,4,4,4]],
            "bn_momentum": 0.1,
            "relu_inplace": True,
            "model_name": "HRNet-18",
        },
    },
    "hrnet30":{
            "encoder": HRNetEncoder,
            "pretrained_settings": _get_pretrained_settings("hrnet30"),
            "params": {
                "out_channels": (3, 64, 30,60,120,240),
                "align_corners":True,
                "num_classes":3,
                "block": ["BOTTLENECK",'BASIC','BASIC','BASIC'],
                "fuse_method": "SUM",
                "num_modules": [1,1,4,3],
                "num_branches": [1,2,3,4],
                "num_channels": [[64],[30,60],[30,60,120],[30,60,120,240]],
                "num_blocks": [[4],[4,4],[4,4,4],[4,4,4,4]],
                "bn_momentum": 0.1,
                "relu_inplace": True,
                "model_name": "HRNet-30",
            },
    },
    "hrnet32":{
            "encoder": HRNetEncoder,
            "pretrained_settings": _get_pretrained_settings("hrnet32"),
            "params": {
                "out_channels": (3, 64, 32, 64, 128, 256),
                "align_corners":True,
                "num_classes":3,
                "block": ["BOTTLENECK",'BASIC','BASIC','BASIC'],
                "fuse_method": "SUM",
                "num_modules": [1,1,4,3],
                "num_branches": [1,2,3,4],
                "num_channels": [[64],[32,64],[32,64,128],[32,64,128,256]],
                "num_blocks": [[4],[4,4],[4,4,4],[4,4,4,4]],
                "bn_momentum": 0.1,
                "relu_inplace": True,
                "model_name": "HRNet-32",
            },
    },
    "hrnet40":{
            "encoder": HRNetEncoder,
            "pretrained_settings": _get_pretrained_settings("hrnet40"),
            "params": {
                "out_channels": (3, 64, 40,80,160,320),
                "align_corners":True,
                "num_classes":3,
                "block": ["BOTTLENECK",'BASIC','BASIC','BASIC'],
                "fuse_method": "SUM",
                "num_modules": [1,1,4,3],
                "num_branches": [1,2,3,4],
                "num_channels": [[64],[40,80],[40,80,160],[40,80,160,320]],
                "num_blocks": [[4],[4,4],[4,4,4],[4,4,4,4]],
                "bn_momentum": 0.1,
                "relu_inplace": True,
                "model_name": "HRNet-40",
            }
    },
    "hrnet44":{
            "encoder": HRNetEncoder,
            "pretrained_settings": _get_pretrained_settings("hrnet44"),
            "params": {
                "out_channels": (3, 64, 44,88,176,352),
                "align_corners":True,
                "num_classes":3,
                "block": ["BOTTLENECK",'BASIC','BASIC','BASIC'],
                "fuse_method": "SUM",
                "num_modules": [1,1,4,3],
                "num_branches": [1,2,3,4],
                "num_channels": [[64],[44,88],[44,88,176],[44,88,176,352]],
                "num_blocks": [[4],[4,4],[4,4,4],[4,4,4,4]],
                "bn_momentum": 0.1,
                "relu_inplace": True,
                "model_name": "HRNet-44",
            },
        },
    "hrnet48":{
            "encoder": HRNetEncoder,
            "pretrained_settings": _get_pretrained_settings("hrnet48"),
            "params": {
                "out_channels": (3, 64, 48, 96, 192, 384),
                "align_corners":True,
                "num_classes":3,
                "block": ["BOTTLENECK",'BASIC','BASIC','BASIC'],
                "fuse_method": "SUM",
                "num_modules": [1,1,4,3],
                "num_branches": [1,2,3,4],
                "num_channels": [[64],[48,96],[48,96,192],[48,96,192,384]],
                "num_blocks": [[4],[4,4],[4,4,4],[4,4,4,4]],
                "bn_momentum": 0.1,
                "relu_inplace": True,
                "model_name": "HRNet-48",
            },
        },
    "hrnet64":{
            "encoder": HRNetEncoder,
            "pretrained_settings": _get_pretrained_settings("hrnet64"),
            "params": {
                "out_channels": (3, 64, 64,128,256,512),
                "align_corners":True,
                "num_classes":3,
                "block": ["BOTTLENECK",'BASIC','BASIC','BASIC'],
                "fuse_method": "SUM",
                "num_modules": [1,1,4,3],
                "num_branches": [1,2,3,4],
                "num_channels": [[64],[64,128],[64,128,256],[64,128,256,512]],
                "num_blocks": [[4],[4,4],[4,4,4],[4,4,4,4]],
                "bn_momentum": 0.1,
                "relu_inplace": True,
                "model_name": "HRNet-64",
            },
        },
}

def get_seg_model(cfg, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model
