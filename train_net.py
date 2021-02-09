import os
import logging
import albumentations as A
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint
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
    cfg.MODEL_CUSTOM = CfgNode()
    cfg.MODEL_CUSTOM.BACKBONE = CfgNode()
    cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID = 5
    cfg.MODEL_CUSTOM.BACKBONE.LOAD_PRETRAIN = False
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
    cfg.SOLVER.ACCUMULATE_GRAD = 1
    # Runner
    cfg.BATCH_SIZE = 1
    cfg.CHECKPOINT_PATH = ""
    cfg.PRECISION = 32
    # Callbacks
    cfg.CALLBACKS = CfgNode()
    cfg.CALLBACKS.CHECKPOINT_DIR = None
    # Inference
    cfg.INFERENCE = CfgNode()
    cfg.INFERENCE.AREA_TRESH = 0

def main():
    # Retrieve Config and and custom base parameter
    cfg = get_cfg()
    add_custom_param(cfg)
    cfg.merge_from_file("config.yaml")

    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    logger = logging.getLogger("pytorch_lightning.core")
    if not os.path.exists(cfg.CALLBACKS.CHECKPOINT_DIR):
        os.makedirs(cfg.CALLBACKS.CHECKPOINT_DIR)
    logger.addHandler(logging.FileHandler(
        os.path.join(cfg.CALLBACKS.CHECKPOINT_DIR,"core.log")))
    with open("config.yaml") as file:
        logger.info(file.read())
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
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    transform_valid = A.Compose([
        A.Resize(height=512, width=1024),
        A.Normalize(mean=cfg.TRANSFORM.NORMALIZE.MEAN,
                    std=cfg.TRANSFORM.NORMALIZE.STD),
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

    logger.info(efficientps.print)
    # Callbacks / Hooks
    early_stopping = EarlyStopping('PQ', patience=5, mode='max')
    checkpoint = ModelCheckpoint(monitor='PQ',
                                 mode='max',
                                 dirpath=cfg.CALLBACKS.CHECKPOINT_DIR,
                                 save_last=True,
                                 verbose=True,)

    # Create a pytorch lighting trainer
    trainer = pl.Trainer(
        # weights_summary='full',
        gpus=1,
        num_sanity_val_steps=0,
        # fast_dev_run=True,
        callbacks=[early_stopping, checkpoint],
        precision=cfg.PRECISION,
        resume_from_checkpoint=cfg.CHECKPOINT_PATH,
        gradient_clip_val=15,
        accumulate_grad_batches=cfg.SOLVER.ACCUMULATE_GRAD
    )
    logger.addHandler(logging.StreamHandler())
    trainer.fit(efficientps, train_loader, val_dataloaders=valid_loader)

if __name__ == '__main__':
    main()