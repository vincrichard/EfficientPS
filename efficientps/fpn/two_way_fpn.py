import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
from inplace_abn import InPlaceABN


class TwoWayFpn(nn.Module):

    def __init__(self, in_feature_shape):
        super().__init__()
        # Channel information are the one given in the EfficientPS paper
        # Depending on the EfficientNet model chosen the number of channel will 
        # change
        # x4 size [B, 40, H, W] (input 40 channels)
        # Bottom up path layers
        self.conv_b_up_x4 = Conv2d(in_feature_shape[1], 256, 1)
        self.iabn_b_up_x4 = InPlaceABN(256)

        # Top down path layers
        self.conv_t_dn_x4 = Conv2d(in_feature_shape[1], 256, 1)
        self.iabn_t_dn_x4 = InPlaceABN(256)

        # x8 size [B, 64, H, W] (input 64 channels)
        # Bottom up path layers
        self.conv_b_up_x8 = Conv2d(in_feature_shape[2], 256, 1)
        self.iabn_b_up_x8 = InPlaceABN(256)

        # Top down path layers
        self.conv_t_dn_x8 = Conv2d(in_feature_shape[2], 256, 1)
        self.iabn_t_dn_x8 = InPlaceABN(256)

        # x16 size [B, 176, H, W] (input 176 channels)
        # In the paper they took the 5 block of efficient net ie 128 channels
        # But taking last block seem more pertinent and was already implemented
        # Skipping to id 3 since block 4 does not interest us
        # Bottom up path layers
        self.conv_b_up_x16 = Conv2d(in_feature_shape[3], 256, 1)
        self.iabn_b_up_x16 = InPlaceABN(256)

        # Top down path layers
        self.conv_t_dn_x16 = Conv2d(in_feature_shape[3], 256, 1)
        self.iabn_t_dn_x16 = InPlaceABN(256)

        # x32 size [B, 2048, H, W] (input 2048 channels)
        # Bottom up path layers
        self.conv_b_up_x32 = Conv2d(in_feature_shape[4], 256, 1)
        self.iabn_b_up_x32 = InPlaceABN(256)

        # Top down path layers
        self.conv_t_dn_x32 = Conv2d(in_feature_shape[4], 256, 1)
        self.iabn_t_dn_x32 = InPlaceABN(256)

        # Separable Conv and Inplace BN at the output of the FPN
        # x4
        self.depth_wise_conv_x4 = Conv2d(
            in_channels=256, out_channels=256, groups=256,  
            kernel_size=3, padding=1)
        self.iabn_out_x4 = InPlaceABN(256)
        # x8
        self.depth_wise_conv_x8 = Conv2d(
            in_channels=256, out_channels=256, groups=256,  
            kernel_size=3, padding=1)
        self.iabn_out_x8 = InPlaceABN(256)
        # x16
        self.depth_wise_conv_x16 = Conv2d(
            in_channels=256, out_channels=256, groups=256,  
            kernel_size=3, padding=1)
        self.iabn_out_x16 = InPlaceABN(256)
        # x32
        self.depth_wise_conv_x32 = Conv2d(
            in_channels=256, out_channels=256, groups=256,  
            kernel_size=3, padding=1)
        self.iabn_out_x32 = InPlaceABN(256)
        
    def forward(self, inputs):
        outputs = dict()

        #################################
        # Bottom up part of the network #
        #################################
        # x4 size
        # [B, C, x4W, x4H]
        b_up_x4 = inputs['reduction_2']
        # [B, C, x4W, x4H] -> [B, 256, x4W, x4H]
        b_up_x4 = self.conv_b_up_x4(b_up_x4)
        b_up_x4 = self.iabn_b_up_x4(b_up_x4)
        # [B, 256, x4W, x4H] -> [B, 256, x8W, x8H]
        b_up_x4_to_merge = F.interpolate(
            b_up_x4,
            size=(inputs['reduction_3'].shape[2], 
                  inputs['reduction_3'].shape[3]),
            mode='bilinear'
        )

        # x8 size
        # [B, C, x8W, x8H]
        b_up_x8 = inputs['reduction_3']
        # [B, C, x8W, x8H] -> [B, 256, x8W, x8H]
        b_up_x8 = self.conv_b_up_x8(b_up_x8)
        b_up_x8 = self.iabn_b_up_x8(b_up_x8)
        b_up_x8 = torch.add(b_up_x4_to_merge, b_up_x8)
        # [B, 256, x8W, x8H] -> [B, 256, x16W, x16H]
        b_up_x8_to_merge = F.interpolate(
            b_up_x8,
            size=(inputs['reduction_4'].shape[2], 
                  inputs['reduction_4'].shape[3]),
            mode='bilinear'
        )

        #x16 size (reduction_4 since we don't need block 4)
        # [B, C, x16W, x16H]
        b_up_x16 = inputs['reduction_4']
        # [B, C, x16W, x16H] -> [B, 256, x16W, x16H]
        b_up_x16 = self.conv_b_up_x16(b_up_x16)
        b_up_x16 = self.iabn_b_up_x16(b_up_x16)
        b_up_x16 = torch.add(b_up_x8_to_merge, b_up_x16)
        # [B, 256, x16W, x16H] -> [B, 256, x32W, x32H]
        b_up_x16_to_merge = F.interpolate(
            b_up_x16,
            size=(inputs['reduction_5'].shape[2],
                  inputs['reduction_5'].shape[3]),
            mode='bilinear'
        )

        #x32 size
        # [B, C, x32W, x32H]
        b_up_x32 = inputs['reduction_5']
        # [B, C, x32W, x32H] -> [B, 256, x32W, x32H]
        b_up_x32 = self.conv_b_up_x32(b_up_x32)
        b_up_x32 = self.iabn_b_up_x32(b_up_x32)
        b_up_x32 = torch.add(b_up_x16_to_merge, b_up_x32)

        ################################
        # Top down part of the network #
        ################################

        # x32 size
        # [B, C, x32W, x32H]
        t_dn_x32 = inputs['reduction_5']
        # [B, C, x32W, x32H] -> [B, 256, x32W, x32H]
        t_dn_x32 = self.conv_t_dn_x32(t_dn_x32)
        t_dn_x32 = self.iabn_t_dn_x32(t_dn_x32)
        # [B, 256, x32W, x32H] -> [B, 256, x16W, x16H]
        t_dn_x32_to_merge = F.interpolate(
            t_dn_x32,
            size=(inputs['reduction_4'].shape[2],
                  inputs['reduction_4'].shape[3]),
            mode='bilinear'
        )
        # Create output
        p_32 = torch.add(t_dn_x32, b_up_x32)
        p_32 = self.depth_wise_conv_x32(p_32)
        p_32 = self.iabn_out_x32(p_32)
        # outputs['P_32'] = p_32

        # x16 size
        # [B, C, x16W, x16H]
        t_dn_x16 = inputs['reduction_4']
        # [B, C, x16W, x16H] -> [B, 256, x16W, x16H]
        t_dn_x16 = self.conv_t_dn_x16(t_dn_x16)
        t_dn_x16 = self.iabn_t_dn_x16(t_dn_x16)
        t_dn_x16 = torch.add(t_dn_x32_to_merge, t_dn_x16)
        # [B, 256, x16W, x16H] -> [B, 256, x32W, x32H]
        t_dn_x16_to_merge =  F.interpolate(
            t_dn_x16,
            size=(inputs['reduction_3'].shape[2],
                  inputs['reduction_3'].shape[3]),
            mode='bilinear'
        )
        # Create output
        p_16 = torch.add(t_dn_x16, b_up_x16)
        p_16 = self.depth_wise_conv_x16(p_16)
        p_16 = self.iabn_out_x16(p_16)
        # outputs['P_16'] = p_16

        # x8 size
        # [B, C, x8W, x8H]
        t_dn_x8 = inputs['reduction_3']
        # [B, C, x8W, x8H] -> [B, 256, x8W, x8H]
        t_dn_x8 = self.conv_t_dn_x8(t_dn_x8)
        t_dn_x8 = self.iabn_t_dn_x8(t_dn_x8)
        t_dn_x8 = torch.add(t_dn_x16_to_merge, t_dn_x8)
        # [B, 256, x8W, x8H] -> [B, 256, x4W, x4H]
        t_dn_x8_to_merge = F.interpolate(
            t_dn_x8,
            size=(inputs['reduction_2'].shape[2],
                  inputs['reduction_2'].shape[3]),
            mode='bilinear'
        )
        # Create output
        p_8 = torch.add(t_dn_x8, b_up_x8)
        p_8 = self.depth_wise_conv_x8(p_8)
        p_8 = self.iabn_out_x8(p_8)
        # outputs['P_8'] = p_8

        # x4 size
        # [B, C, x4W, x4H]
        t_dn_x4 = inputs['reduction_2']
        # [B, C, x4W, x4H] -> [B, 256, x4W, x4H]
        t_dn_x4 = self.conv_t_dn_x4(t_dn_x4)
        t_dn_x4 = self.iabn_t_dn_x4(t_dn_x4)
        t_dn_x4 = torch.add(t_dn_x8_to_merge, t_dn_x4)
        
        # Create outputs
        p_4 = torch.add(t_dn_x4, b_up_x4)
        p_4 = self.depth_wise_conv_x4(p_4)
        p_4 = self.iabn_out_x4(p_4)
        # outputs['P_4'] = p_4

        return {
            'P_4': p_4,
            'P_8': p_8,
            'P_16': p_16,
            'P_32': p_32
        }
