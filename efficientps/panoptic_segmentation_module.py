import torch
import torch.nn.functional as F
from detectron2.structures import Instances

def panoptic_segmentation_module(cfg, outputs, device):
    """
    Take output of both semantic and instance head and combine them to create
    panoptic predictions.

    Note there is no need to check for threshold score, compute overlap and
    sorted scores. Since Detectron2 inference function already has the
    `SCORE_THRESH_TEST` and `NMS_THRESH_TEST` that does those action. Furthermore
    all prediction are sorted reated to their scores

    Args:
    - cfg (Config) : Config object
    - outputs (dict) : Inference output of our model
    - device : Device used by the lightning module

    Returns:
    - canvas (tensor) : [B, H, W] Panoptic predictions
    """
    # If no instance prediction pass the threshold score > 0.5 IoU > 0.5
    # Returns the argmax of semantic logits
    if outputs['instance'] is None:
        return compute_output_only_semantic(outputs['semantic'])
    panoptic_result = []
    # Loop on Batch images / Instances
    for i, instance in enumerate(outputs['instance']):
        instance = check_bbox_size(instance)
        # If there is no proposal after the check compute panoptic output with
        # semantic information only
        if len(instance.pred_boxes.tensor) == 0:
            panoptic_result.append(
                        compute_output_only_semantic(outputs['semantic'][i]))
            continue
        semantic = outputs['semantic'][i]
        # Preprocessing
        Mla = scale_resize_pad(instance).to(device)
        # Compute instances
        Mlb = create_mlb(semantic, instance).to(device)
        Fl = compute_fusion(Mla, Mlb)
        # First merge instances with stuff predictions
        semantic_stuff_logits = semantic[:11,:,:]
        inter_logits = torch.cat([semantic_stuff_logits, Fl], dim=0)
        inter_preds = torch.argmax(inter_logits, dim=0)
        # Create canvas and merge everything
        canvas = create_canvas_thing(inter_preds, instance)
        canvas = add_stuff_from_semantic(cfg, canvas, semantic)
        panoptic_result.append(canvas)

    return torch.stack(panoptic_result)

def check_bbox_size(instance):
    """
    In some cases the width or height of a predicted bbox is 0. This function
    check all dimension and remove instances having this issue.
    Args:
    - instance (Instance) : detectron2 Instance object with prediction
    Returns:
    - new_instance (Instance) : dectron2 Instance with filtered prediction
    """
    new_instance = Instances(instance.image_size)
    boxes = instance.pred_boxes.tensor
    masks = instance.pred_masks
    inds = []
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        # Retrieve bbox dimension
        box = torch.round(box).cpu().numpy()
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        if h == 0 or w == 0:
            continue
        inds.append(i)
    new_instance.pred_masks = instance.pred_masks[inds]
    new_instance.pred_boxes = instance.pred_boxes[inds]
    new_instance.pred_classes = instance.pred_classes[inds]
    new_instance.scores = instance.scores[inds]
    return new_instance

def scale_resize_pad(instance):
    """
    In order to use both semantic and instances, mask must be rescale and fit
    the dimension of the bboxes predictions.
    Args:
    - instance (Instance) : an Instance object from detectron containg all
                            proposal bbox, masks, classes and scores
    """
    Mla = []
    boxes = instance.pred_boxes.tensor
    masks = instance.pred_masks
    # Loop on proposal
    for box, mask in zip(boxes, masks):
        # Retrieve bbox dimension
        box = torch.round(box).cpu().numpy()
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        # Resize mask to bbox dimension
        mask = F.interpolate(mask.unsqueeze(0), size=(h, w), mode='bilinear')
        mask = mask[0,0,...]
        # Start from an empty canvas to have padding
        canva = torch.zeros(instance.image_size)
        # Fit the upsample mask in the bbox prediction position
        canva[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = mask
        Mla.append(canva)
    return torch.stack(Mla)

def create_mlb(semantic, instance):
    """
    Create the semantic logit corresponding to each class prediction.
    Args:
    - semantic (tensor) : Semantic logits of one image
    - instance (Instance) : Instance object with all instance prediction for one
                            image
    Returns:
    - Mlb (tensor) : dim[Nb of prediction, H, W]
    """
    Mlb = []
    boxes = instance.pred_boxes.tensor
    classes = instance.pred_classes
    for bbox, cls in zip(boxes, classes):
        # Start from a black image
        canva = torch.zeros(instance.image_size)
        # Add the semantic value from the predicted class at the predicted bbox location
        canva[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = \
            semantic[cls,int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        Mlb.append(canva)
    return torch.stack(Mlb)

def compute_fusion(Mla, Mlb):
    """
    Compute the Hadamard product of the sum of sigmoid and the sum of Mla and
    Mlb. The Hadamard product is a fancy name for element-wise product.
    Args:
    - Mla (tensor) : Instance logit preprocess see `scale_resize_pad`
    - Mlb (tensor) : Semantic logits preprocess see `create_mlb`
    Returns:
    - Fl (tensor) : Fused mask logits
    """
    return (torch.sigmoid(Mla) + torch.sigmoid(Mlb)) * (Mla + Mlb)

def create_canvas_thing(inter_preds, instance):
    """
    From the intermediate prediction retrieve only the logits corresponding to
    thing classes.
    Args:
    -inter_preds (tensor): intermediate prediction
    -instance (Instance) : Instance object used to retrieve each classes of
                           instance prediction
    Returns:
    -canvas (tensor) : panoptic prediction containing only thing classes
    """
    # init to a number not in class category id
    canvas = torch.zeros_like(inter_preds)
    # Retrieve classes of all instance prediction (sorted by detectron2)
    classes = instance.pred_classes
    instance_train_id_to_eval_id = [24, 25, 26, 27, 28, 31, 32, 33, 0]
    # Used to label each instance incrementally
    track_of_instance = {}
    # Loop on instance prediction
    for id_instance, cls in enumerate(classes):
        # The stuff channel are the 11 first channel so we add an offset
        id_instance += 11
        # Compute mask for each instance and verify that no prediction has been
        # made
        mask = torch.where((inter_preds == id_instance) & (canvas==0))
        # If the instance is present on interpreds add its panoptic label to
        # the canvas and increment the id of instance
        if len(mask) > 0:
            nb_instance = track_of_instance.get(int(cls), 0)
            canvas[mask] = instance_train_id_to_eval_id[cls] * 1000 + nb_instance
            track_of_instance.update({int(cls):nb_instance+1})
    return canvas

def compute_output_only_semantic(semantic):
    """
    In case where no instance are suitable, we are returning the panoptic
    prediction base only on the semantic outputs.
    This is usefull mainly at the beginning of the training.
    Args:
    - semantic (tensor) : Output of the semantic head (either for the full
                          batch or for one image)
    """
    semantic_train_id_to_eval_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                23, 24, 25, 26, 27, 28, 31, 32, 33, 0]
    if len(semantic.shape) == 3:
        semantic_output = torch.argmax(semantic, dim=0)
    else:
        semantic_output = torch.argmax(semantic, dim=1)
    # apply reversed to avoid issue with reindexing the value
    for train_id in reversed(torch.unique(semantic_output)):
        mask = torch.where(semantic_output == train_id)
        # Create panoptic ids for instance thing or stuff
        if train_id > 11:
            semantic_output[mask] = semantic_train_id_to_eval_id[train_id] * 1000
        else:
            semantic_output[mask] = semantic_train_id_to_eval_id[train_id]
    return semantic_output

def add_stuff_from_semantic(cfg, canvas, semantic):
    """
    Compute the semantic output. If the output is not overlap with an existing
    prediction on the canvas (ie with a instance prediction) and the are is
    above the defined treshold, add the panoptic label of the stuff class
    on the canvas
    Args:
    - canvas (torch): canvas containing the thing class predictions
    - semantic (torch): logit output from the semantic head
    Return:
    - canvas (torch): Final panoptic prediction for an image
    """
    # Link between semantic and stuff classes in semantic prediction instance
    # classes have higher class training values
    stuff_train_id_to_eval_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]
    semantic_output = torch.argmax(semantic, dim=0)
    # Reverse to avoid overwrite classes information
    for train_id in reversed(torch.unique(semantic_output)):
        # If the detected section is a thing
        if train_id >= len(stuff_train_id_to_eval_id):
            continue
        # Compute mask where semantic is present and no things has been predicted
        mask = torch.where((semantic_output == train_id) & (canvas == 0))
        # Check the area is large enough
        if len(mask[0]) > cfg.INFERENCE.AREA_TRESH:
            # Compute mask where there is no thing classes
            canvas[mask] = stuff_train_id_to_eval_id[train_id]
    return canvas