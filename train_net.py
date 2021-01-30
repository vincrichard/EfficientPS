import albumentations as A
import pytorch_lightning as pl
from efficientps import EffificientPS
from detectron2.config import get_cfg
from torch.utils.data import DataLoader
from datasets.panoptic_dataset import PanopticDataset, collate_fn

from detectron2.utils.events import _CURRENT_STORAGE_STACK, EventStorage


# def parse_args():
#     parser = argparse.ArgumentParser(description='Train panoptic network')

#     parser.add_argument('--cfg',
#                         help='experiment configure file name',
#                         required=True,
#                         type=str)
#     # parser.add_argument("--local_rank", type=int, default=0)
#     # parser.add_argument('opts',
#     #                     help="Modify config options using the command-line",
#     #                     default=None,
#     #                     nargs=argparse.REMAINDER)

#     args = parser.parse_args()
#     update_config(config, args)

    # return args

def main():
    # args = parse_args()

    cfg = get_cfg()
    cfg.NUM_CLASS = 19
    cfg.EFFICIENTNET_ID = 5
    cfg.BATCH_SIZE = 1
    cfg.merge_from_file("config.yaml")
    # test = EffificientPS("5", nb_class=19, cfg=cfg)
    _CURRENT_STORAGE_STACK.append(EventStorage())

    # test(torch.rand(size=(1,3,256,512)))

    base_path = "/media/vincent/C0FC3B20FC3B0FE0/Elix/detectron2/datasets/cityscapes"
    train_json = "gtFine/cityscapes_panoptic_train.json"
    valid_json = "gtFine/cityscapes_panoptic_val.json"

    transform = A.Compose([
        # A.HorizontalFlip(p=0.5),
        # A.RandomScale(scale_limit=[0.5, 2]),
        # CINAPTIC_MEAN = (0.485, 0.456, 0.406) bizarre
        # CINAPTIC_STD = (0.229, 0.224, 0.225)
        # A.RandomSizedCrop()
        A.Resize(height=256, width=512)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    train_dataset = PanopticDataset(train_json, base_path, 'train', transform=transform)
    valid_dataset = PanopticDataset(valid_json, base_path, 'val', transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,#cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=2,#cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=False
    )

    # Init model
    efficientps = EffificientPS(cfg)
    # efficientps(train_dataset[0])
    # Init trainer pytorch lighting
    trainer = pl.Trainer(weights_summary='full', fast_dev_run=True)
    trainer.fit(efficientps, train_loader, val_dataloaders=valid_loader)

if __name__ == '__main__':
    main()