import torch.nn as nn

# from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.layers import ShapeSpec


from typing import Dict, List, Optional, Tuple
from detectron2.config import configurable
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, ROIHeads, BaseMaskRCNNHead, build_roi_heads, select_foreground_proposals
from inplace_abn import InPlaceABN

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


    def forward(self, inputs, gt_instances):
        losses = {}
        proposals, losses_rpn = self.rpn(inputs, gt_instances)
        pred_instances, losses_head = self.roi_heads(inputs, proposals, gt_instances)
        losses.update(losses_rpn)
        losses.update(losses_head)
        return losses


@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(ROIHeads):

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        box_network: nn.Module,
        box_predictor: nn.Module,
        mask_head: nn.Module,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.pooler = pooler
        self.box_network = box_network
        self.box_predictor = box_predictor
        self.mask_head = mask_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = [1.0 / input_shape[lvl].stride for lvl in in_features]
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["box_network"] = BboxNetwork()
        ret["box_predictor"] = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=1024, height=1, width=1)
        )
        # Mask Network + Head
        ret["mask_head"] = MaskNetwork(cfg.MODEL.ROI_HEADS.NUM_CLASSES)
        return ret

    def forward(self, features, proposals, targets=None):
        """
        Args:
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        pooled_feature = self.pooler(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(self.box_network(pooled_feature)) #.mean(dim=[2, 3])

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            # Mask
            proposals, fg_selection_masks = select_foreground_proposals(
                proposals, self.num_classes
            )
            # Since the ROI feature transform is shared between boxes and masks,
            # we don't need to recompute features. The mask loss is only defined
            # on foreground proposals, so we need to select out the foreground
            # features.
            mask_features = pooled_feature[torch.cat(fg_selection_masks, dim=0)]
            del pooled_feature
            losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


class BboxNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # self.first_fc = nn.Linear()

# @ROI_MASK_HEAD_REGISTRY.register()
class MaskNetwork(BaseMaskRCNNHead):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_iabn_layers = []
        for i in range(4):
            separable_conv = nn.Conv2d(256, 256, kernel_size=3, groups=256)
            self.conv_iabn_layers.append(separable_conv)
            iabn = InPlaceABN(256)
            self.conv_iabn_layers.append(iabn)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.last_iabn = InPlaceABN(256)
        self.last_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def layers(self, x):
        for layer in self.conv_iabn_layers:
            x = layer(x)
        x = self.last_iabn(self.deconv(x))
        return self.last_conv(x)