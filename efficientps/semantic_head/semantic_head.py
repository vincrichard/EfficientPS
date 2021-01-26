import torch
import torch.nn as nn
from torch.nn import functional as F
from inplace_abn import InPlaceABN


class SemanticHead(nn.Module):

    def __init__(self, nb_class):
        super().__init__()

        self.dpc_x32 = DPC()
        self.dpc_x16 = DPC()

        self.lsfe_x8 = LSFE()
        self.lsfe_x4 = LSFE()

        self.mc_16_to_8 = MC()
        self.mc_8_to_4 = MC()

        self.last_conv = nn.Conv2d(512, nb_class, 1)

        self.softmax = nn.Softmax2d()


    def forward(self, inputs):
        # The forward is apply in a bottom up manner
        # x32 size
        p_32 = inputs['P_32']
        p_32 = self.dpc_x32(p_32)
        # [B, C, x32H, x32W] -> [B, C, x16H, x16W]
        p_32_to_merge = F.interpolate(
            p_32,
            scale_factor=(2, 2),
            mode='bilinear')
        # [B, C, x16H, x16W] -> [B, C, x4H, x4W]
        p_32 = F.interpolate(
            p_32_to_merge,
            scale_factor=(4, 4),
            mode='bilinear')

        # x16 size
        p_16 = inputs['P_16']
        p_16 = self.dpc_x16(p_16)
        p_16_to_merge = torch.add(p_32_to_merge, p_16)
        # [B, C, x16H, x16W] -> [B, C, x4H, x4W]
        p_16 = F.interpolate(
            p_16,
            scale_factor=(4, 4),
            mode='bilinear')
        # [B, C, x16H, x16W] -> [B, C, x8H, x8W]
        p_16_to_merge = self.mc_16_to_8(p_16_to_merge)

        # x8 size
        p_8 = inputs['P_8']
        p_8 = self.lsfe_x8(p_8)
        p_8 = torch.add(p_16_to_merge, p_8)
        # [B, C, x8H, x8W] -> [B, C, x4H, x4W]
        p_8_to_merge = self.mc_8_to_4(p_8)
        # [B, C, x8H, x8W] -> [B, C, x4H, x4W]
        p_8 = F.interpolate(
            p_8,
            scale_factor=(2, 2),
            mode='bilinear')

        # x4 size
        p_4 = inputs['P_4']
        p_4 = self.lsfe_x4(p_4)
        p_4 = torch.add(p_8_to_merge, p_4)

        # Create output
        # [B, 128, x4H, x4W] -> [B, 512, x4H, x4W]
        outputs = torch.cat((p_32, p_16, p_8, p_4), dim=1)       
        outputs = self.last_conv(outputs)
        outputs = F.interpolate(
            outputs,
            scale_factor=(4, 4),
            mode='bilinear')

        logits = self.softmax(outputs)
        return logits



class LSFE(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Separable Conv
        self.conv_1 = nn.Conv2d(256, 128, 3, groups=128, padding=1)
        self.conv_2 = nn.Conv2d(128, 128, 3, groups=128, padding=1)
        # Inplace BN + Leaky Relu
        self.abn_1 = InPlaceABN(128)
        self.abn_2 = InPlaceABN(128)

    def forward(self, inputs):
        # Apply first conv
        outputs = self.conv_1(inputs)
        outputs = self.abn_1(outputs)

        # Apply second conv
        outputs = self.conv_2(outputs)
        return self.abn_2(outputs)


class MC(nn.Module):

    def __init__(self):
        super().__init__()
        # Conv are similar to LSFE module
        # self.lfse = LSFE()
        # Separable Conv
        self.conv_1 = nn.Conv2d(128, 128, 3, groups=128, padding=1)
        self.conv_2 = nn.Conv2d(128, 128, 3, groups=128, padding=1)
        # Inplace BN + Leaky Relu
        self.abn_1 = InPlaceABN(128)
        self.abn_2 = InPlaceABN(128)

    def forward(self, inputs):
        # Apply first conv
        outputs = self.conv_1(inputs)
        outputs = self.abn_1(outputs)

        # Apply second conv
        outputs = self.conv_2(outputs)
        outputs = self.abn_2(outputs)

        # Apply conv
        # outputs = self.lfse(inputs)

        # Return upsample features
        return F.interpolate(
            outputs, 
            scale_factor=(2, 2),
            mode='bilinear')

class DPC(nn.Module):

    def __init__(self):
        super().__init__()
        options = {
            'in_channels'   : 256,
            'out_channels'  : 256,
            'kernel_size'   : 3,
            'groups'        : 256,
        }
        self.conv_first = nn.Conv2d(dilation=(1, 6), padding=(1, 6), **options)
        self.iabn_first = InPlaceABN(256)
        # Branch 1
        self.conv_branch_1 = nn.Conv2d(**options, padding=1)
        self.iabn_branch_1 = InPlaceABN(256)
        # Branch 2
        self.conv_branch_2 = nn.Conv2d(dilation=(6, 21), padding=(6, 21), **options)
        self.iabn_branch_2 = InPlaceABN(256)
        #Branch 3
        self.conv_branch_3 = nn.Conv2d(dilation=(18, 15), padding=(18, 15), **options)
        self.iabn_branch_3 = InPlaceABN(256)
        # Branch 4
        self.conv_branch_4 = nn.Conv2d(dilation=(6, 3), padding=(6, 3), **options)
        self.iabn_branch_4 = InPlaceABN(256)
        # Last conv
        # There is some mismatch in the paper about the dimension of this conv
        # In the paper it says "This tensor is then finally passed through a 
        # 1×1 convolution with 256 output channels and forms the output of the 
        # DPC module." But the overall schema shows an output of 128
        # The MC module schema also show an input of 256.
        # In order to have 512 channel at the concatenation of all layers, 
        # I choose 128 output channels
        self.conv_last = nn.Conv2d(1280, 128, 1)
        self.iabn_last = InPlaceABN(128)

    def forward(self, inputs):
        # First conv
        inputs = self.conv_first(inputs)
        inputs = self.iabn_first(inputs)
        # Branch 1
        branch_1 = self.conv_branch_1(inputs)
        branch_1 = self.iabn_branch_1(branch_1)
        # Branch 2
        branch_2 = self.conv_branch_2(inputs)
        branch_2 = self.iabn_branch_2(branch_2)
        # Branch 3
        branch_3 = self.conv_branch_3(inputs)
        branch_3 = self.iabn_branch_3(branch_3)
        # Branch 4 (take branch 3 as input)
        branch_4 = self.conv_branch_4(branch_3)
        branch_4 = self.iabn_branch_4(branch_4)
        # Concatenate
        # [B, 256, H, W] -> [B, 1280, H, W]
        concat = torch.cat(
            (inputs, branch_1, branch_2, branch_3, branch_4), 
            dim=1)
        # Last conv
        outputs = self.conv_last(concat)
        return self.iabn_last(outputs)


