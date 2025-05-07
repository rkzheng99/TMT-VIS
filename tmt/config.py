# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_tmt_config(cfg):
    cfg.DATASETS.DATASET_RATIO = []

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 20
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    # Pseudo Data Use
    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ['rotation']
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = False
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (384, 600)

    # LSJ
    cfg.INPUT.LSJ_AUG = CN()
    cfg.INPUT.LSJ_AUG.ENABLED = False
    cfg.INPUT.LSJ_AUG.IMAGE_SIZE = 1024
    cfg.INPUT.LSJ_AUG.MIN_SCALE = 0.1
    cfg.INPUT.LSJ_AUG.MAX_SCALE = 2.0

    # tmt
    cfg.MODEL.tmt = CN()
    cfg.MODEL.tmt.NHEADS = 8
    cfg.MODEL.tmt.DROPOUT = 0.0
    cfg.MODEL.tmt.DIM_FEEDFORWARD = 2048
    cfg.MODEL.tmt.ENC_LAYERS = 6
    cfg.MODEL.tmt.DEC_LAYERS = 3
    cfg.MODEL.tmt.ENC_WINDOW_SIZE = 0
    cfg.MODEL.tmt.PRE_NORM = False
    cfg.MODEL.tmt.HIDDEN_DIM = 256
    cfg.MODEL.tmt.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.tmt.ENFORCE_INPUT_PROJ = True

    cfg.MODEL.tmt.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.tmt.DEEP_SUPERVISION = True
    cfg.MODEL.tmt.LAST_LAYER_NUM = 3
    cfg.MODEL.tmt.MULTI_CLS_ON = True
    cfg.MODEL.tmt.APPLY_CLS_THRES = 0.01

    cfg.MODEL.tmt.SIM_USE_CLIP = True
    cfg.MODEL.tmt.SIM_WEIGHT = 0.5

    cfg.MODEL.tmt.FREEZE_DETECTOR = False
    cfg.MODEL.tmt.TEST_RUN_CHUNK_SIZE = 18
    cfg.MODEL.tmt.TEST_INTERPOLATE_CHUNK_SIZE = 5
