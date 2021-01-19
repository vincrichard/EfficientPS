import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

class PanopticDataset(Dataset):
    """A dataset for Panotic task"""

    def __init__(self, path_json, root_dir, split, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        # Load json file containing information about the dataset
        path_file = os.path.join(self.root_dir, path_json)
        data = json.load(open(path_file))
        annotations = data['annotations']
        images = data['images']
        categories = data['categories']
        # TODO Possible problem with VOID label and train_id
        # Add mapper to training id and class id
        self.class_mapper = {cat['id']:{
                                        'train_id':i,
                                        'isthing':cat['isthing']
                                    } 
                                    for i, cat in enumerate(categories)}
        self.class_mapper.update({0:{
                        'train_id':len(categories) +1,
                        'isthing':False }})
        self.train_id_to_eval_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                        23, 24, 25, 26, 27, 28, 31, 32, 33, 0]

        # Generate a dictionary with all needed information with idx as key
        self.meta_data = {}
        for i in range(len(images)):
            self.meta_data.update({i:{}})
            #TODO Error Message
            assert annotations[i]['image_id'] == images[i]['id'] 
            self.meta_data[i].update({'labelfile_name': annotations[i]['file_name']})
            self.meta_data[i].update(annotations[i])
            self.meta_data[i].update(images[i])
    
    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        # Retrieve meta data of image
        img_data = self.meta_data[idx]

        # Load image
        path_img = os.path.join(self.root_dir,
                                'leftImg8bit',
                                self.split,
                                img_data['file_name'].split('_')[0],
                                img_data['file_name'].replace('gtFine_', ''))
        image = np.asarray(Image.open(path_img))

        # Get label info
        path_label = os.path.join(self.root_dir,
                                  'gtFine',
                                  'cityscapes_panoptic_'+self.split,
                                  img_data['labelfile_name'])
        panoptic = np.asarray(Image.open(path_label))
        panoptic = rgb2id(panoptic)
        
        # Create semantic segmentation target
        semantic = np.zeros_like(panoptic)
        instance_mask = []
        rpn_bbox = []
        rpn_mask = np.zeros_like(panoptic)
        for seg in img_data['segments_info']:
            # if seg['iscrowd']:
                #TODO
            seg_category = self.class_mapper[seg['category_id']]
            semantic[panoptic == seg["id"]] = seg_category['train_id']
            # If segmentation is an rpn generate a mask for maskrcnn target
            # Collect information for RPN targets
            if seg_category['isthing']:
                mask = np.zeros_like(panoptic)
                mask[panoptic == seg["id"]] = seg_category['train_id']
                instance_mask.append(mask)
                # RPN targets
                rpn_mask[panoptic == seg["id"]] = 1
                rpn_bbox.append(seg["bbox"])
            
        return {
            'image': image,
            'semantic': semantic,
            'instance_mask': np.stack([*instance_mask]),
            'rpn_mask': rpn_mask,
            'rpn_bbox': np.stack([*rpn_bbox])
        }


def rgb2id(color):
    """ Pass the image from RGB to the instance id value 
    See COCO format doc https://cocodataset.org/#format-data
    """
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


