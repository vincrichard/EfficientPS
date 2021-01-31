import os
import json
import numpy as np
from PIL import Image
import torch.nn.functional as F
from panopticapi.utils import id2rgb

# GT_JSON = "/media/vincent/C0FC3B20FC3B0FE0/Elix/detectron2/datasets/cityscapes/gtFine/cityscapes_panoptic_val.json"
GT_JSON = "/home/ubuntu/Elix/cityscapes/gtFine/cityscapes_panoptic_val.json"
# PRED_JSON = "/media/vincent/C0FC3B20FC3B0FE0/Elix/detectron2/datasets/cityscapes/cityscapes_panoptic_preds.json"
PRED_JSON = "/home/ubuntu/Elix/cityscapes/cityscapes_panoptic_preds.json"

def create_output_file(annotations, panoptic_tensor, targets):
    # pred_dir = "/media/vincent/C0FC3B20FC3B0FE0/Elix/detectron2/datasets/cityscapes/preds"
    pred_dir = "/home/ubuntu/Elix/cityscapes/preds"
    if not os.path.exists(pred_dir): os.makedirs(pred_dir)
    for img_panoptic, image_id in zip(panoptic_tensor, targets['image_id']):
        img_data = dict()
        img_data['image_id'] = image_id
        img_data['segments_info'] = []
        img_panoptic = F.interpolate(
            img_panoptic.unsqueeze(0).unsqueeze(0).float(),
            size=(1024, 2048),
            mode='nearest'
        )[0,0,...]
        img_panoptic = img_panoptic.cpu().numpy()
        for ind, instance in enumerate(np.unique(img_panoptic)):
            img_data['segments_info'].append(
                {
                    'id': int(instance),
                    'category_id': int(instance) if instance < 1000 else int(instance % 1000)
                }
            )
        img_data['file_name'] = "{}_preds_panoptic.png".format(image_id)
        img = id2rgb(img_panoptic)
        img_to_save = Image.fromarray(img)
        img_to_save.save(os.path.join(pred_dir, img_data['file_name']))
        annotations.append(img_data)

def save_json_file(annotations):
    # Save prediction file
    with open(GT_JSON, "r") as f:
        json_data = json.load(f)
    json_data['annotations'] = annotations
    with open(PRED_JSON, "w") as f:
        f.write(json.dumps(json_data))
