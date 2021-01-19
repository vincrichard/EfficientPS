import torch
import torch.nn as nn

class EffificientPS(nn.Module):

    def __init__(self):

    
    def forward(self):

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