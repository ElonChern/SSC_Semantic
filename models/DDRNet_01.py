#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
DDRNet
jieli_cn@163.com
"""

import torch
import torch.nn as nn
import sys
sys.path.append('/home/elon/SSC-master')
from models.my_projection_layer import Project2Dto3D
from models.DDR import DDR_ASPP3d
from models.DDR import BottleneckDDR2d, BottleneckDDR3d, DownsampleBlock3d
from models.modules import Process, Upsample, Downsample
from models.modules import SegmentationHead

# DDRNet
# ----------------------------------------------------------------------
class SSC_RGBD_DDRNet(nn.Module):
    def __init__(self, num_classes=20,feature=32):
        super(SSC_RGBD_DDRNet, self).__init__()
        print('SSC_RGBD_DDRNet: RGB and Depth streams with DDR blocks for Semantic Scene Completion')

        # w, h, d = 240, 144, 240
        w, h, d = 256, 32, 256
        full_scene_size = (256,256,32)
        project_scale = 1
        norm_layer=nn.BatchNorm3d
        bn_momentum=0.1
        dilations = [1, 2, 3]
        self.feature = feature
        # --- depth
        c_in, c, c_out, dilation, residual = 1, 4, 8, 1, True
        self.dep_feature2d = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, 0),  # reduction
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
        )
        self.project_layer_dep = Project2Dto3D(full_scene_size,project_scale)  # w=240, h=144, d=240
        self.dep_feature3d_downsample_1 = nn.Sequential(
            DownsampleBlock3d(8, 16),
            BottleneckDDR3d(c_in=16, c=4, c_out=16, dilation=1, residual=True),
        )
        
        self.dep_feature3d_downsample_2 = nn.Sequential(
            DownsampleBlock3d(16, 64),  # nn.MaxPool3d(kernel_size=2, stride=2)
            BottleneckDDR3d(c_in=64, c=16, c_out=64, dilation=1, residual=True),
        )
        

        # --- RGB
        c_in, c, c_out, dilation, residual = 3, 4, 8, 1, True
        self.rgb_feature2d = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, 0),  # reduction
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
        )
        self.project_layer_rgb = Project2Dto3D(full_scene_size,project_scale)  # w=240, h=144, d=240

        self.feature3d_downsample_1 = nn.Sequential(
            DownsampleBlock3d(8, 16),
            BottleneckDDR3d(c_in=16, c=4, c_out=16, dilation=1, residual=True),
        )
        
        self.feature3d_downsample_2 = nn.Sequential(
            DownsampleBlock3d(16, 32),  # nn.MaxPool3d(kernel_size=2, stride=2)
            BottleneckDDR3d(c_in=32, c=16, c_out=32, dilation=1, residual=True),
        )        
        
        # --- Voxel
        self.voxel_feature3d = nn.Sequential(nn.Conv3d(1, 8, kernel_size=3, padding=1, stride=1),
                                             nn.BatchNorm3d(8),
                                             nn.ReLU(inplace=True))
        
        self.process_l0 = nn.Sequential(
            Process(self.feature//2, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature//2, norm_layer, bn_momentum),
        )
        self.process_l1 = nn.Sequential(
            Process(self.feature, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature, norm_layer, bn_momentum),
        )
        self.process_l2 = nn.Sequential(
            Process(self.feature * 2, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature * 2, norm_layer, bn_momentum,expansion=8),
        )

        self.up_13_l2 = Upsample(
            self.feature * 4, self.feature * 2, norm_layer, bn_momentum
        )
        self.up_12_l1 = Upsample(
            self.feature * 2, self.feature, norm_layer, bn_momentum
        )
        self.up_l1_lfull = Upsample(
            self.feature, self.feature // 2, norm_layer, bn_momentum
        )

        self.ssc_head_1_4 = SegmentationHead(
            self.feature *2, self.feature *2, num_classes, dilations
        )
        self.ssc_head_1_2 = SegmentationHead(
            self.feature , self.feature , num_classes, dilations
        )        
        self.ssc_head_1_1 = SegmentationHead(
            self.feature // 2, self.feature // 2, num_classes, dilations
        )


        # -------------1/4

        # ck = 256
        # self.ds = DownsamplerBlock_3d(64, ck)
        ck = 32
        c = 16
        # c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True
        self.res3d_1d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=2, residual=True)
        self.res3d_2d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=3, residual=True)
        self.res3d_3d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=5, residual=True)

        self.res3d_1r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=2, residual=True)
        self.res3d_2r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=3, residual=True)
        self.res3d_3r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=5, residual=True)
        
        self.res3d_1v = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=2, residual=True)
        self.res3d_2v = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=3, residual=True)
        self.res3d_3v = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=5, residual=True)        

        self.aspp = DDR_ASPP3d(c_in=int(ck * 4), c=16, c_out=64)
        # self.aspp = DDR_ASPP3d(c_in=int(ck * 4), c=64, c_out=int(ck * 4))

        # 64 * 5 = 320
        self.conv_out = nn.Sequential(
            nn.Conv3d(320, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, num_classes, 1, 1, 0)
        )

        # ----  weights init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)  # gain=1
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)

    def forward(self, x_depth=None, x_rgb=None, data=None):
        # input: x (BS, 3L, 240L, 144L, 240L)
        # print('SSC: x.shape', x.shape)

        x0_rgb = self.rgb_feature2d(x_rgb)
        x0_rgb_pro = self.project_layer_rgb(x0_rgb, data) # [b, 8, 256,32,256] 
        x0_depth = self.dep_feature2d(x_depth)
        x0_depth_pro = self.project_layer_dep(x0_depth, data) # [b, 8, 256,32,256]
        x0_voxel = self.voxel_feature3d(data['occupancy']) # [b, 8, 256,32,256]
        
        # 原始数据融合
        f0_img_depth_0 = torch.add(x0_rgb_pro,x0_depth_pro) # [b, 8, 256,32,256]
        f0_pro2voxel_0 = torch.cat(( f0_img_depth_0, x0_voxel), dim=1) # [b, 16, 256,32,256]
        f0_voxel2pro_0 = torch.add(f0_img_depth_0, x0_voxel) # [b, 8, 256,32,256]
        
        # 第一次降采样 + 融合 
        f0_img_depth_1 = self.feature3d_downsample_1(f0_voxel2pro_0) # [b, 16, 128,16,128]  
        voxel_1 = self.process_l0(f0_pro2voxel_0) # [b, 16, 128,16,128]
        f0_voxel2pro_1 = torch.add(f0_img_depth_1,voxel_1) # [b, 16, 128,16,128]
        f0_pro2voxel_1 = torch.cat(( f0_img_depth_1, voxel_1), dim=1) # [b, 32, 128,16,128]
        
        # 第二次降采样 + 融合   
        f0_img_depth_2 = self.feature3d_downsample_2(f0_voxel2pro_1) # [b, 32, 64,8,64]  #
        voxel_2 = self.process_l1(f0_pro2voxel_1) # [b, 32, 64,8,64]          
        f0_voxel2pro_2 = torch.add(f0_img_depth_2,voxel_2) # [b, 32, 64,8,64] 
        f0_pro2voxel_2 = torch.cat(( f0_img_depth_2, voxel_2), dim=1) # [b, 64, 64,8,64] 
                   

        # voxel 分支
        x3d_l3 = self.process_l2(f0_pro2voxel_2)  # [b, 128, 32,4,32]   
        x3d_up_l2 = self.up_13_l2(x3d_l3) + f0_pro2voxel_2 #[b, 64, 64,8,64]
        # print(x3d_up_l2.shape)
        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + f0_pro2voxel_1 #[b, 32, 128,16,128]
        # print(x3d_up_l1.shape)
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)+ f0_pro2voxel_0 #[b, 16, 1256,32,256]

        out_scale_1_1__3D = self.ssc_head_1_1(x3d_up_lfull)
        out_scale_1_2__3D = self.ssc_head_1_2(x3d_up_l1)
        out_scale_1_4__3D = self.ssc_head_1_4(x3d_up_l2)        
        scores = {'pred_semantic_1_1': out_scale_1_1__3D, 'pred_semantic_1_2': out_scale_1_2__3D,
                  'pred_semantic_1_4': out_scale_1_4__3D}
        
        # img和depth 分支
        f0 = torch.add(f0_voxel2pro_2,voxel_2)

        x_4_d = self.res3d_1d(f0_voxel2pro_2)
        x_4_r = self.res3d_1r(voxel_2)
       

        f1 = torch.add(x_4_d, x_4_r)
        # print("x_4_r ={}".format(x_4_r.shape))
        
        x_5_d = self.res3d_2d(x_4_d)
        x_5_r = self.res3d_2r(x_4_r)

        f2 = torch.add(x_5_d, x_5_r)
        # print("x_5_r ={}".format(x_5_r.shape))


        x_6_d = self.res3d_3d(x_5_d)
        x_6_r = self.res3d_3r(x_5_r)
       
        f3 = torch.add(x_6_d, x_6_r)
        
        # print("f3 ={}".format(f3.shape))

        x = torch.cat((f0, f1, f2, f3), dim=1)  # channels concatenate
        # print('SSC: channels concatenate x', x.size())  # (BS, 256L, 60L, 36L, 60L)

        x = self.aspp(x)

        y = self.conv_out(x)  # (BS, 12L, 60L, 36L, 60L)

        return y,scores


if __name__ == '__main__':
    model = SSC_RGBD_DDRNet()
    x = torch.rand(1, 1, 480,640)
    y = torch.rand(1, 3, 480,640)
    out = model(x,y)