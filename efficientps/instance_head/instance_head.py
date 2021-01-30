import torch.nn as nn

# from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import build_roi_heads
#   ANCHOR_GENERATOR:
#     SIZES: [[32], [64], [128], [256], [512]]  # One size for each in feature map
#     ASPECT_RATIOS: [[0.5, 1.0, 2.0]]  # Three aspect ratios (same for all in feature maps)

class InstanceHead(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        # self.rpn = RPNCustom(in_features=[32, 16, 8, 4])
        # Detectron 2 expect a dict of ShapeSpec object as input_shape
        input_shape = dict()
        for name, shape in zip(cfg.MODEL.RPN.IN_FEATURES, [4, 16, 8, 32]):
            input_shape[name] = ShapeSpec(channels=256, stride=shape)

        self.rpn = build_proposal_generator(cfg, input_shape=input_shape)

        self.roi_heads = build_roi_heads(cfg, input_shape)


    def forward(self, inputs, targets={}):
        losses = {}
        proposals, losses_rpn = self.rpn(inputs, targets['instance'])
        if self.training:
            _, losses_head = self.roi_heads(inputs, proposals, targets['instance'])
            losses.update(losses_rpn)
            losses.update(losses_head)
            return {}, losses
        else:
            pred_instances , _ = self.roi_heads(inputs, proposals)
            return pred_instances, {}
