import os
import json
import numpy as np
from PIL import Image
from panopticapi.utils import id2rgb

GT_JSON = "/media/vincent/C0FC3B20FC3B0FE0/Elix/detectron2/datasets/cityscapes/gtFine/cityscapes_panoptic_val.json"
PRED_JSON = "/media/vincent/C0FC3B20FC3B0FE0/Elix/detectron2/datasets/cityscapes/cityscapes_panoptic_preds.json"

def create_output_file(annotations, panoptic_tensor, targets):
    pred_dir = "/media/vincent/C0FC3B20FC3B0FE0/Elix/detectron2/datasets/cityscapes/preds"
    if not os.path.exists(pred_dir): os.makedirs(pred_dir)
    for img_panoptic, image_id in zip(panoptic_tensor, targets['image_id']):
        img_data = dict()
        img_data['image_id'] = image_id
        img_data['segment_info'] = dict()
        img_panoptic = img_panoptic.numpy()
        for ind, instance in enumerate(np.unique(img_panoptic)):
            img_data['segment_info'].update(
                {int(ind): {
                    'id': int(instance),
                    'category_id': int(instance) if instance < 1000 else int(instance % 1000)
                    }
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
