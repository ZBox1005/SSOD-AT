#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

from ssod_at.config import add_ssod_at_config
from ssod_at.engine.trainer import SSOD_AT_Trainer, BaselineTrainer
from ssod_at.modeling.meta_arch.ts_ensemble import EnsembleTSModel

from ssod_at.utils.build_datasets import build_dota, build_dior

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ssod_at_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    
    # register dataset
    if cfg.DATASETS.TRAIN[0]=="dota_1024_coco_train":
        dota=build_dota("/data1/users/tuwenkai/ZBox/datasets/dota_v1.0_1024")
        dota.register_dota_dataset()
    elif cfg.DATASETS.TRAIN[0]=="dior_coco_train":
        dior=build_dior("/data1/users/tuwenkai/ZBox/datasets/DIOR-coco")
        dior.register_dior_dataset()
    
    # select trainer
    if cfg.SEMISUPNET.Trainer == "ssod_at":
        Trainer = SSOD_AT_Trainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    # eval only
    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ssod_at":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    # train
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = default_argument_parser().parse_args()
    # import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    # # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # args.config_file = 'configs/voc5/baseline/voc5_sup100_run1.yaml'
    # # # args.config_file = 'configs/voc/pick/voc07+voc12_sup25_run1.yaml'
    # args.config_file = 'configs/coco/faster_rcnn_R_50_FPN_sup5_run1.yaml'
    # # args.dist_url='tcp://127.0.0.1:1237'
    # args.num_gpus = 4
    # args.eval_only = True
    # #args.opts = ['SOLVER.IMG_PER_BATCH_LABEL', '8', 'SOLVER.IMG_PER_BATCH_UNLABEL', '8',  'MODEL.WEIGHTS', 'output/voc07+voc12/pick/voc07+voc12_sup25_pick_box_number20+random5_8/model_best.pth']
    # args.opts = ['SOLVER.IMG_PER_BATCH_LABEL', '16', 'SOLVER.IMG_PER_BATCH_UNLABEL', '16',  'MODEL.WEIGHTS', 'output/pick/imagenet_sup20_pick_all_indicator10+random10_16/model_best.pth']
    # args.opts = ['OUTPUT_DIR',"output/coco/faster_rcnn_R_50_FPN_sup5_run1_16b-9wit"]
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )