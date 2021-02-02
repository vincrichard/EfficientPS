import os
import albumentations as A
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping
)
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.events import _CURRENT_STORAGE_STACK, EventStorage

from efficientps import EffificientPS
from datasets.panoptic_dataset import PanopticDataset, collate_fn

def add_custom_param(cfg):
    """
    In order to add custom config parameter in the .yaml those parameter must
    be initialised
    """
    # Model
    cfg.EFFICIENTNET_ID = 5
    # DATASET
    cfg.NUM_CLASS = 19
    cfg.DATASET_PATH = "/home/ubuntu/Elix/cityscapes"
    cfg.TRAIN_JSON = "gtFine/cityscapes_panoptic_train.json"
    cfg.VALID_JSON = "gtFine/cityscapes_panoptic_val.json"
    cfg.PRED_DIR = "preds"
    cfg.PRED_JSON = "cityscapes_panoptic_preds.json"
    # Transfom
    cfg.TRANSFORM = CfgNode()
    cfg.TRANSFORM.NORMALIZE = CfgNode()
    cfg.TRANSFORM.NORMALIZE.MEAN = (106.433, 116.617, 119.559)
    cfg.TRANSFORM.NORMALIZE.STD = (65.496, 67.6, 74.123)
    cfg.TRANSFORM.RESIZE = CfgNode()
    cfg.TRANSFORM.RESIZE.HEIGHT = 512
    cfg.TRANSFORM.RESIZE.WIDTH = 1024
    cfg.TRANSFORM.RANDOMCROP = CfgNode()
    cfg.TRANSFORM.RANDOMCROP.HEIGHT = 512
    cfg.TRANSFORM.RANDOMCROP.WIDTH = 1024
    cfg.TRANSFORM.HFLIP = CfgNode()
    cfg.TRANSFORM.HFLIP.PROB = 0.5
    # Solver
    cfg.SOLVER.NAME = "SGD"
    # Runner
    cfg.BATCH_SIZE = 1
    cfg.CHECKPOINT_PATH = ""
    # Inference
    cfg.INFERENCE = CfgNode()
    cfg.INFERENCE.AREA_TRESH = 0

def main():
    # Retrieve Config and and custom base parameter
    cfg = get_cfg()
    add_custom_param(cfg)
    cfg.merge_from_file("config.yaml")

    # Initialise Custom storage to avoid error when using detectron 2
    _CURRENT_STORAGE_STACK.append(EventStorage())

    # Create transforms
    transform_train = A.Compose([
        A.Resize(height=cfg.TRANSFORM.RESIZE.HEIGHT,
                 width=cfg.TRANSFORM.RESIZE.WIDTH),
        A.RandomCrop(height=cfg.TRANSFORM.RANDOMCROP.HEIGHT,
                     width=cfg.TRANSFORM.RANDOMCROP.WIDTH),
        A.HorizontalFlip(p=cfg.TRANSFORM.HFLIP.PROB),
        A.Normalize(mean=cfg.TRANSFORM.NORMALIZE.MEAN,
                    std=cfg.TRANSFORM.NORMALIZE.STD),
        # A.RandomScale(scale_limit=[0.5, 2]),
        # A.RandomSizedCrop()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    transform_valid = A.Compose([
        A.Resize(height=512, width=1024),
        A.Normalize(mean=(106.433, 116.617, 119.559), std=(65.496, 67.6, 74.123)),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    # Create Dataset
    train_dataset = PanopticDataset(cfg.TRAIN_JSON,
                                    cfg.DATASET_PATH,
                                    'train',
                                    transform=transform_train)

    valid_dataset = PanopticDataset(cfg.VALID_JSON,
                                    cfg.DATASET_PATH,
                                    'val',
                                    transform=transform_valid)

    # Create Data Loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=False,
        num_workers=4
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False,
        num_workers=4
    )

    # Create model or load a checkpoint
    if os.path.exists(cfg.CHECKPOINT_PATH):
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        print("Loading model from {}".format(cfg.CHECKPOINT_PATH))
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        efficientps = EffificientPS.load_from_checkpoint(cfg=cfg,
            checkpoint_path=cfg.CHECKPOINT_PATH)
    else:
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        print("Creating a new model")
        print('""""""""""""""""""""""""""""""""""""""""""""""')
        efficientps = EffificientPS(cfg)
        cfg.CHECKPOINT_PATH = None

    # Callbacks / Hooks
    early_stopping = EarlyStopping('PQ', patience=5, mode='max')

    # Create a pytorch lighting trainer
    trainer = pl.Trainer(
        # weights_summary='full',
        gpus=1,
        num_sanity_val_steps=0,
        # fast_dev_run=True,
        callbacks=[early_stopping],
        precision=16,
        resume_from_checkpoint=cfg.CHECKPOINT_PATH,
        min_epochs=10,
        gradient_clip_val=35,
        # auto_lr_find=True
    )
    # trainer.tune(efficientps, train_loader)
    trainer.fit(efficientps, train_loader, val_dataloaders=valid_loader)

if __name__ == '__main__':
    main()