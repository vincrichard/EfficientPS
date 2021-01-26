import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datasets.panoptic_dataset import PanopticDataset


def add_box(ax, box, color='b', thickness=2):
    """ Draws annotations in an image. 
    # Arguments
        ax          : The matplotlib ax to draw on.
        box         : A [1, 5] matrix (x1, y1, w, h, label).
        color       : The color of the boxes.
        thickness   : (optional) thickness of the bbox.
    """
    rect = Rectangle((box[0], box[1]), box[2], box[3], 
                        color=color, 
                        fill=False, 
                        linewidth=thickness
                    )
    ax.add_patch(rect)

def add_boxes(ax, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.
    # Arguments
        image     : The matplotlib ax to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        add_box(ax, b, color, thickness=thickness)

def vizualise_input_targets(dataset, seed=65):
    # Get a sample
    sample = dataset[seed]

    # Figure
    fig = plt.figure(figsize=(15,10))
    for i, (name, tensor) in enumerate(sample.items()):
        if name == 'rpn_bbox':
            continue
            
        ax = fig.add_subplot(2, 3, i+1)
        if name == 'image': 
            add_boxes(ax, sample['rpn_bbox'], 'g')
        
        if name == 'instance_mask':
            id_instance = np.random.choice(tensor.shape[0])
            tensor = tensor[id_instance]
        
        ax.set_title(name)
        plt.imshow(tensor)

    plt.show()


def main():
    base_path = "/media/vincent/C0FC3B20FC3B0FE0/Elix/detectron2/datasets/cityscapes"
    train_json = "gtFine/cityscapes_panoptic_train.json"

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomScale(scale_limit=[0.5, 2]),
        # A.RandomSizedCrop()
        # A.Resize(height=500, width=500)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    train_dataset = PanopticDataset(train_json, base_path, 'train', transform=transform)

    vizualise_input_targets(train_dataset)

if __name__ == '__main__':
    main()