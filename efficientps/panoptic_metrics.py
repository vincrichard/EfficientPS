import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch.nn.functional as F
from panopticapi.utils import id2rgb

def generate_pred_panoptic(cfg, outputs):
    """
    Take all output of a model and save a json file with al predictions as
    well as all panoptic image prediction.
    This is done in order to use `pq_compute` function from panoptic api
    Args:
    - cfg (Config) : config object
    - outputs (list[dict]) : List of a full epoch of outputs
    """
    # Create prediction dir if needed
    pred_dir = os.path.join(cfg.DATASET_PATH, cfg.PRED_DIR)
    if not os.path.exists(pred_dir): os.makedirs(pred_dir)

    annotations = []
    print("Saving panoptic prediction to compute validation metrics")
    # Loop on each validation output
    for output in tqdm(outputs):
        # Loop on each image of the batch
        for img_panoptic, image_id in zip(output['panoptic'], output['image_id']):
            img_data = dict()
            img_data['image_id'] = image_id
            # Resize the image to original size
            img_panoptic = F.interpolate(
                img_panoptic.unsqueeze(0).unsqueeze(0).float(),
                size=(1024, 2048),
                mode='nearest'
            )[0,0,...]
            # Create segment_info data
            img_data['segments_info'] = []
            img_panoptic = img_panoptic.cpu().numpy()
            for instance in np.unique(img_panoptic):
                if instance == 0:
                    continue
                img_data['segments_info'].append(
                    {
                        'id': int(instance),
                        'category_id': int(instance)
                                       if instance < 1000
                                       else int(instance / 1000)
                    }
                )
            # Save panotic_pred
            img_data['file_name'] = "{}_preds_panoptic.png".format(image_id)
            img = id2rgb(img_panoptic)
            img_to_save = Image.fromarray(img)
            img_to_save.save(os.path.join(pred_dir, img_data['file_name']))
            # Add annotation of a one image
            annotations.append(img_data)

    save_json_file(cfg, annotations)

def save_json_file(cfg, annotations):
    """
    Load gt json file to have same architecture and replace annotations
    with the prediction annotations

    Args:
    - cfg (Config) : config object
    - annotations (List[dict]) : List containing prediction info for each image
    """
    gt_path = os.path.join(cfg.DATASET_PATH, cfg.VALID_JSON)
    pred_path = os.path.join(cfg.DATASET_PATH, cfg.PRED_JSON)

    # Save prediction file
    with open(gt_path, "r") as f:
        json_data = json.load(f)
    json_data['annotations'] = annotations
    with open(pred_path, "w") as f:
        f.write(json.dumps(json_data))
