import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from .fpn import TwoWayFpn
import pytorch_lightning as pl
from .backbone import generate_backbone_EfficientPS, output_feature_size
from .semantic_head import SemanticHead
from .instance_head import InstanceHead
from .panoptic_segmentation_module import panoptic_segmentation_module
from .panoptic_metrics import generate_pred_panoptic
from panopticapi.evaluation import pq_compute


class EffificientPS(pl.LightningModule):
    """
    EfficientPS model see http://panoptic.cs.uni-freiburg.de/
    Here pytorch lightningis used https://pytorch-lightning.readthedocs.io/en/latest/
    """

    def __init__(self, cfg):
        """
        Args:
        - cfg (Config) : Config object from detectron2
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = generate_backbone_EfficientPS(cfg)
        self.fpn = TwoWayFpn(
            output_feature_size[cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID])
        self.semantic_head = SemanticHead(cfg.NUM_CLASS)
        self.instance_head = InstanceHead(cfg)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        predictions, _ = self.shared_step(x)
        return predictions

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        _, loss = self.shared_step(batch)
        # Add losses to logs
        [self.log(k, v) for k,v in loss.items()]
        return {'loss': sum(loss.values())}

    def shared_step(self, inputs):
        loss = dict()
        predictions = dict()
        # Feature extraction
        features = self.backbone.extract_endpoints(inputs['image'])
        pyramid_features = self.fpn(features)
        # Heads Predictions
        semantic_logits, semantic_loss = self.semantic_head(pyramid_features, inputs)
        pred_instance, instance_losses = self.instance_head(pyramid_features, inputs)
        # Output set up
        loss.update(semantic_loss)
        loss.update(instance_losses)
        predictions.update({'semantic': semantic_logits})
        predictions.update({'instance': pred_instance})
        return predictions, loss

    def validation_step(self, batch, batch_idx):
        predictions, loss = self.shared_step(batch)
        panoptic_result = panoptic_segmentation_module(self.cfg,
            predictions,
            self.device)
        return {
            'val_loss': sum(loss.values()),
            'panoptic': panoptic_result,
            'image_id': batch['image_id']
        }

    def validation_epoch_end(self, outputs):
        # Create and save all predictions files
        generate_pred_panoptic(self.cfg, outputs)

        # Compute PQ metric with panpticapi
        pq_res = pq_compute(
            gt_json_file= os.path.join(self.cfg.DATASET_PATH,
                                       self.cfg.VALID_JSON),
            pred_json_file= os.path.join(self.cfg.DATASET_PATH,
                                         self.cfg.PRED_JSON),
            gt_folder= os.path.join(self.cfg.DATASET_PATH,
                                    "gtFine/cityscapes_panoptic_val/"),
            pred_folder=os.path.join(self.cfg.DATASET_PATH, self.cfg.PRED_DIR)
        )
        self.log("PQ", 100 * pq_res["All"]["pq"])
        self.log("SQ", 100 * pq_res["All"]["sq"])
        self.log("RQ", 100 * pq_res["All"]["rq"])
        self.log("PQ_th", 100 * pq_res["Things"]["pq"])
        self.log("SQ_th", 100 * pq_res["Things"]["sq"])
        self.log("RQ_th", 100 * pq_res["Things"]["rq"])
        self.log("PQ_st", 100 * pq_res["Stuff"]["pq"])
        self.log("SQ_st", 100 * pq_res["Stuff"]["sq"])
        self.log("RQ_st", 100 * pq_res["Stuff"]["rq"])

    def configure_optimizers(self):
        if self.cfg.SOLVER.NAME == "Adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.cfg.SOLVER.BASE_LR)
        elif self.cfg.SOLVER.NAME == "SGD":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.cfg.SOLVER.BASE_LR,
                                        momentum=0.9,
                                        weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        else:
            raise ValueError("Solver name is not supported, \
                Adam or SGD : {}".format(self.cfg.SOLVER.NAME))
        return {
            'optimizer': optimizer,
            # 'scheduler': StepLR(optimizer, [120, 144], gamma=0.1),
            'scheduler': ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True),
            'monitor': 'PQ'
        }

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < self.cfg.SOLVER.WARMUP_ITERS:
            lr_scale = min(1., float(self.trainer.global_step + 1) /
                                    float(self.cfg.SOLVER.WARMUP_ITERS))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.cfg.SOLVER.BASE_LR

        # update params
        optimizer.step(closure=closure)


"""

# class EffificientPS(nn.Module):

#     def __init__(self):


#     def forward(self):

        # 1 - Apply backbone efficient det
        # input original image
        # output x4 x8 x16 x32 feature map

        # 2 - Apply 2-way fpn
        # inpout bacbone outputs
        # output P4, P8, P16, P32

        # 3 - Instance Segmentation Branch
        # For each FPN Layer

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_in_feature_maps:
            layer_outputs.append(self.rpn(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

            # Apply RPN
            # 3.1 - Apply Network
            # input One PX feature map
            # output [anchors_prob, anchors_bbox, anchors_logit(for the loss)]
            # 3.2 - Proposal Layer
            # input anchors_prob, anchors_bbox
            # output  bs x roi x (256x14x14)
            # Generate proposals

        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposal_layer([rpn_class, rpn_bbox],
                                proposal_count=proposal_count,
                                nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                anchors=self.anchors,
                                config=self.config)

                # 3.2.1 - Sort anchor by score
                # 3.2.2 - Retieve top k anchors
                # 3.2.3 - Apply delta to get refine anchors [batch, N, (y1, x1, y2, x2)]
                # 3.2.4 - Apply non max suppression
            # 3.3 - Compute targets for ROI
            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]

            # Normalize coordinates
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            gt_boxes = gt_boxes / scale

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_deltas, target_mask = \
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

            # 3.4 - Network classification / regression head
            # 3.5 - Network Mask head
            if not rois.size():
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
            else:
                # Network Heads
                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rois)

                # Create masks for detections
                mrcnn_mask = self.mask(mrcnn_feature_maps, rois)

            return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask]

            # 5 loss to compute here

        # 4 - Semantic Segmentation Branch
        # Follow the straight forward implementation
        # 1 loss

        # if training :
            # 5 Compute loss
            # 6 optimize

        # if inference:
            # 5 Panoptic Fusion



        # MASK RCNN
        # if mode == 'inference':
        #     # Network Heads
        #     # Proposal classifier and BBox regressor heads
        #     mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)

        #     # Detections
        #     # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
        #     detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)

        #     # Convert boxes to normalized coordinates
        #     # TODO: let DetectionLayer return normalized coordinates to avoid
        #     #       unnecessary conversions
        #     h, w = self.config.IMAGE_SHAPE[:2]
        #     scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
        #     if self.config.GPU_COUNT:
        #         scale = scale.cuda()
        #     detection_boxes = detections[:, :4] / scale

        #     # Add back batch dimension
        #     detection_boxes = detection_boxes.unsqueeze(0)

        #     # Create masks for detections
        #     mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)

        #     # Add back batch dimension
        #     detections = detections.unsqueeze(0)
        #     mrcnn_mask = mrcnn_mask.unsqueeze(0)

        #     return [detections, mrcnn_mask]
"""