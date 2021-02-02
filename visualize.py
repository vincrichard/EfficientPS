import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datasets.panoptic_dataset import PanopticDataset


def add_box(ax, box, color='b', thickness=2):
    """ Draws annotations in an image.
    # Arguments
        ax          : The matplotlib ax to draw on.
        box         : A [1, 5] matrix (x1, y1, x2, y2, label).
        color       : The color of the boxes.
        thickness   : (optional) thickness of the bbox.
    """
    rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
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
        if name in ['instance', 'image_id']:
            continue

        ax = fig.add_subplot(2, 3, i+1)
        if name == 'image':
            add_boxes(ax, sample['instance'].gt_boxes.tensor.numpy(), 'g')

        # if name == 'instance':
        #     id_instance = np.random.choice(tensor.shape[0])
        #     tensor = tensor[id_instance]

        ax.set_title(name)
        plt.imshow(tensor)

    plt.show()


def main():
    base_path = "/home/ubuntu/Elix/cityscapes"
    train_json = "gtFine/cityscapes_panoptic_train.json"

    transform = A.Compose([
        A.Resize(height=512, width=1024),
        A.RandomCrop(height=512, width=1024),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(106.433, 116.617, 119.559), std=(65.496, 67.6, 74.123)),
        # A.RandomScale(scale_limit=[0.5, 2]),
        # A.RandomSizedCrop()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    train_dataset = PanopticDataset(train_json, base_path, 'train', transform=transform)

    vizualise_input_targets(train_dataset)

if __name__ == '__main__':
    main()