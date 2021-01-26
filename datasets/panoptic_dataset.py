import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from detectron2.structures import Instances, BitMasks, Boxes

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
        self.transform = transform
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

        # Get bbox info
        rpn_bbox = []
        class_bbox = []
        for seg in img_data['segments_info']:
            seg_category = self.class_mapper[seg['category_id']]
            if seg_category['isthing']:
                rpn_bbox.append(seg["bbox"])
                class_bbox.append(seg_category['train_id'])

        # Apply augmentation with albumentations
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=panoptic,
                bboxes=rpn_bbox,
                class_labels=class_bbox
            )
            image = transformed['image']
            panoptic = transformed['mask']
            rpn_bbox = transformed['bboxes']
            class_bbox = transformed['class_labels']
        
        # Create instance class for detectron (Mask RCNN Head)
        instance = Instances(panoptic.shape)

        # Create semantic segmentation target with augmented data
        semantic = np.zeros_like(panoptic)
        rpn_mask = np.zeros_like(panoptic)
        instance_mask = []
        instance_cls = []

        for seg in img_data['segments_info']:
            # if seg['iscrowd']:
                #TODO
            seg_category = self.class_mapper[seg['category_id']]
            semantic[panoptic == seg["id"]] = seg_category['train_id']
            # If segmentation is a thing generate a mask for maskrcnn target
            # Collect information for RPN targets
            if seg_category['isthing']:
                mask = np.zeros_like(panoptic)
                mask[panoptic == seg["id"]] = 1 #seg_category['train_id']
                instance_cls.append(seg_category['train_id'])
                instance_mask.append(mask)
                # RPN targets
                rpn_mask[panoptic == seg["id"]] = 1
        
        # Create same size of bbox and mask instance
        #TODO if batch_size > 1
        rpn_bbox = coco_to_pascal_bbox(np.stack([*rpn_bbox]))

        instance.gt_masks = BitMasks(instance_mask)
        instance.gt_classes = torch.as_tensor(instance_cls)
        instance.gt_boxes = Boxes(rpn_bbox)

        return {
            'image': np.array(image),
            'semantic': semantic,
            # 'instance_mask': torch.as_tensor(np.stack([*instance_mask])),
            # 'rpn_mask': torch.as_tensor(rpn_mask),
            # # xmin, ymin, xmax, ymax
            # 'rpn_bbox': rpn_bbox,
            'instance': instance
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


def coco_to_pascal_bbox(bbox):
    return np.stack((bbox[:,0], bbox[:,1], 
            bbox[:,0]+bbox[:,2], bbox[:,1]+bbox[:,3]), axis=1)


def collate_fn(inputs):
    return {
        'image': torch.stack([F.to_tensor(i['image']) for i in inputs]),
        'semantic': torch.as_tensor([i['semantic'] for i in inputs]),
        'instance': [i['instance'] for i in inputs]
    }