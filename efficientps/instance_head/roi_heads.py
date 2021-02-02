import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from detectron2.config import configurable
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, ROIHeads, BaseMaskRCNNHead, select_foreground_proposals
from detectron2.layers import ShapeSpec
from detectron2.structures import Instances

from inplace_abn import InPlaceABN

@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(ROIHeads):

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_head: nn.Module,
        mask_pooler: ROIPooler,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.box_in_features = in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.mask_in_features = in_features
        self.mask_pooler = mask_pooler
        self.mask_head = mask_head
        self.mask_on = True

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        in_features = ret["in_features"] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_scales     = [1.0 / input_shape[lvl].stride for lvl in in_features]
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO

        ret["box_pooler"] = ROIPooler(
            output_size=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE,
        )
        ret["mask_pooler"] = ROIPooler(
            output_size=cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE,
        )

        ret["box_head"] = BboxNetwork(cfg)
        ret["box_predictor"] = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=1024, height=1, width=1)
        )
        # Mask Network + Head
        ret["mask_head"] = MaskNetwork(cfg.MODEL.ROI_HEADS.NUM_CLASSES)
        return ret

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            return proposals, losses
        else:
            # Check that there is proposal given by the rpn
            if sum([len(p) for p in proposals]) == 0:
                return None, {}
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            # Check if all proposal has been removed by the IoU thresh and score thresh
            if pred_instances is not None:
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
            if features.shape[0] == 0:
                return None
        else:
            features = {f: features[f] for f in self.mask_in_features}
            if features.shape[0] == 0:
                return None
        return self.mask_head(features, instances)

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        return instances


class BboxNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        in_channel = 256 * int((cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION))**2
        self.first_fc = nn.Linear(in_channel, 1024)
        self.first_iabn = InPlaceABN(1024)
        self.second_fc = nn.Linear(1024, 1024)
        self.second_iabn = InPlaceABN(1024)

    def forward(self, x):
        # x = self.max_pool(x)
        x = self.flatten(x)
        x = self.first_fc(x)
        x = self.first_iabn(x)
        x = self.second_fc(x)
        return self.second_iabn(x)

# @ROI_MASK_HEAD_REGISTRY.register()
class MaskNetwork(BaseMaskRCNNHead):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_iabn_layers = nn.ModuleList([])
        for i in range(4):
            separable_conv = nn.Conv2d(256, 256, kernel_size=3, groups=256, padding=1)
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