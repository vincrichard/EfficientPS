import torch
import torch.nn.functional as F
from torchvision.ops import nms

def panoptic_segmentation_module(outputs):
    # If no instance prediction pass the threshold score > 0.5 IoU > 0.5
    # Returns the argmax of semantic logits
    if outputs['instance'] is None:
        return compute_output_only_semantic(outputs['semantic'])
    panoptic_result = []
    for i, instance in enumerate(outputs['instance']):
        semantic = outputs['semantic'][i]
        # nms(instance.pred_boxes.tensor, instance.scores, 0.5)
        # Preprocessing
        Mla = scale_resize_pad(instance)
        # Compute instances
        Mlb = create_mlb(semantic, instance)
        Fl = compute_fusion(Mla, Mlb)
        # First merge instances with stuff predictions
        semantic_stuff_logits = semantic[:11,:,:]
        inter_logits = torch.cat([semantic_stuff_logits, Fl], dim=0)
        inter_preds = torch.argmax(inter_logits, dim=0)
        # Create canvas and merge everything
        canvas = create_canvas_thing(inter_preds, instance)
        canvas = add_stuff_from_semantic(canvas, semantic)
        panoptic_result.append(canvas)

    return torch.stack(panoptic_result)

def scale_resize_pad(instance):
    Mla = []
    boxes = instance.pred_boxes.tensor
    masks = instance.pred_masks
    for box, mask in zip(boxes, masks):
        # Retrieve bbox dimension
        box = torch.round(box).numpy()
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        # Resize mask to bbox dimension
        mask = F.interpolate(mask.unsqueeze(0), size=(h, w), mode='bilinear')
        mask = mask[0,0,...]
        # Start from an empty canvas to have padding
        canva = torch.zeros(instance.image_size)
        canva[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = mask
        Mla.append(canva)
    return torch.stack(Mla)

# def check_for_overlap(instance):
#     pass

def create_mlb(semantic, instance):
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
    return (F.sigmoid(Mla) + F.sigmoid(Mlb)) * (Mla + Mlb)

def create_canvas_thing(inter_preds, instance):
    """
    Retrieve only the pixel in the bbox of the instance that predicted an instance
    all 11 first argument are stuff ids
    """
    # init to a number not in class category id
    canvas = torch.zeros_like(inter_preds)
    mask_id = inter_preds < 11
    # stuff_train_id_to_eval_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]
    classes = instance.pred_classes
    instance_train_id_to_eval_id = [24, 25, 26, 27, 28, 31, 32, 33, 0]
    track_of_instance = {}
    for id_instance, cls in enumerate(classes):
        # Add the id of stuff class already used
        id_instance += 11
        mask = torch.where(inter_preds == id_instance)
        if len(mask) > 0:
            nb_instance = track_of_instance.get(int(cls), 0)
            canvas[mask] = instance_train_id_to_eval_id[cls] * 1000 + nb_instance
            track_of_instance.update({int(cls):nb_instance+1})
    canvas[mask_id] = 0
    return canvas

def compute_output_only_semantic(semantic):
    """
    Only used if there is no instance predictions
    """
    semantic_train_id_to_eval_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                23, 24, 25, 26, 27, 28, 31, 32, 33, 0]
    semantic_output = torch.argmax(semantic, dim=1)
    # apply reversed to avoid issue with reindexing the value
    for train_id in reversed(torch.unique(semantic_output)):
        mask = torch.where(semantic_output==train_id)
        # Create instance panoptic id but only one instance will exists
        if train_id > 11:
            semantic_output[mask] = semantic_train_id_to_eval_id[train_id] * 1000
        else:
            semantic_output[mask] = semantic_train_id_to_eval_id[train_id]
    return semantic_output

def add_stuff_from_semantic(canvas, semantic):
    """
    Compute the area of stuff classes and check it is above the area treshold
    Add the panoptic label of those stuff classes to the canvas
    """
    area_thresh = 1024
    stuff_train_id_to_eval_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]
    semantic_output = torch.argmax(semantic, dim=0)
    for train_id in reversed(torch.unique(semantic_output)):
        # If the detected section is a thing
        if train_id >= len(stuff_train_id_to_eval_id):
            continue
        # Compute mask and verify area
        mask = torch.where(semantic_output==train_id)
        if len(mask[0]) > area_thresh:
            canvas[mask] = stuff_train_id_to_eval_id[train_id]
    return canvas