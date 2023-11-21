# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator, DatasetEvaluators
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.structures import pairwise_iou
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

from ssod_at.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from ssod_at.data.dataset_mapper import DatasetMapperTwoCropSeparate
from ssod_at.engine.hooks import LossEvalHook
from ssod_at.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ssod_at.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ssod_at.solver.build import build_lr_scheduler

# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


class SSOD_AT_Trainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        if cfg.TEST.EVALUATOR == 'COCOeval':
            evaluator_type = "coco"
        elif cfg.TEST.EVALUATOR == 'VOCeval':
            evaluator_type = "pascal_voc"
        else:
            raise NotImplementedError(
                "Evaluator for the dataset {} with the type {} not implemented".format(
                    dataset_name, evaluator_type
                )
            )

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.gt_origin_idx= proposal_bbox_inst.origin_idx[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def compare_ts_pred_class_consistence(
            self,
            teacher_proposal_bbox_inst,
            student_proposal_bbox_inst,
            t_proposal_scores,
            s_proposal_scores):

        teacher_boxes=teacher_proposal_bbox_inst.gt_boxes
        student_boxes=student_proposal_bbox_inst.gt_boxes

        teacher_classes=teacher_proposal_bbox_inst.gt_classes
        student_classes=student_proposal_bbox_inst.gt_classes

        if teacher_classes.shape[0]==0 or student_classes.shape[0]==0:
            return False, 0

        iou_matrix=pairwise_iou(teacher_boxes,student_boxes).to(torch.device("cpu"))
        # print("######打印iou_matrix######: " ,iou_matrix)
        iou_max_per_teacher_boxes_idx = torch.max(iou_matrix, 1)[1].numpy()
        # print("######打印iou_max_per_teacher_boxes_idx######: ",iou_max_per_teacher_boxes_idx)

        s_proposal_scores_new = torch.full_like(t_proposal_scores, 0)
        # print("######打印s_proposal_scores_new######: ",s_proposal_scores_new)
        for (i,item) in enumerate(s_proposal_scores_new):
            item = s_proposal_scores[iou_max_per_teacher_boxes_idx[i]]

        kl=F.kl_div(t_proposal_scores,s_proposal_scores_new, reduction="batchmean")

        flag=True

        for idx_teacher in range(teacher_classes.shape[0]):
            t_cls=teacher_classes[idx_teacher]
            s_cls=student_classes[iou_max_per_teacher_boxes_idx[idx_teacher]]
            flag &= t_cls==s_cls

        return flag, kl

    def process_proposal_scores_distribution(self,proposal_scores,proposal_bbox_inst,idx,thres=0.7):
        proposal_scores=proposal_scores[:,:-1]
        valid_map_step1=[]
        for i in range(proposal_scores.shape[0]):
            valid_map_step1.append(i in idx.cpu().detach().numpy())
        # print("打印idx: ", idx,'\n')
        proposal_scores=proposal_scores[valid_map_step1]
        return proposal_scores


    def process_pseudo_label(
        self,
        teacher_proposals_roih_unsup_k,
        teacher_roi_predictions_unsup_k,
        student_proposals_roih_unsup_k,
        student_roi_predictions_unsup_k,
        cur_threshold,
        proposal_type,
        psedo_label_method=""
    ):
        # 将proposal加工成instances
        list_instances = []
        valid_map=[]
        list_kl_div=[]
        # print("teacher的prediction: ",teacher_roi_predictions_unsup_k)
        # print("teacher的proposal_roi: ", teacher_proposals_roih_unsup_k)
        # print("teacher的idx: ", teacher_roih_idx_in_proposals)
        # num_proposal_output = 0.0
        for (t_proposal_bbox_inst,
             t_proposal_scores,
             s_proposal_bbox_inst,
             s_proposal_scores,
             ) in zip(
            teacher_proposals_roih_unsup_k,
            teacher_roi_predictions_unsup_k,
            student_proposals_roih_unsup_k,
            student_roi_predictions_unsup_k,
        ):
            # thresholding
            # t_proposal_scores=F.log_softmax(t_proposal_scores[:, :-1][t_idx], dim=-1)
            # s_proposal_scores = F.softmax(s_proposal_scores[:, :-1][s_idx], dim=-1)
            # print("打印scores:", t_proposal_scores, "scores的shape：", t_proposal_scores.shape)

            # print("打印inst:", t_proposal_bbox_inst, "inst的shape：", t_proposal_bbox_inst.shape)


            if psedo_label_method == "thresholding":
                t_proposal_bbox_inst = self.threshold_bbox(
                    t_proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
                s_proposal_bbox_inst = self.threshold_bbox(
                    s_proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            # num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(t_proposal_bbox_inst)

            t_idx=t_proposal_bbox_inst.gt_origin_idx
            s_idx=s_proposal_bbox_inst.gt_origin_idx

            t_proposal_scores = F.log_softmax(
                self.process_proposal_scores_distribution(t_proposal_scores, t_proposal_bbox_inst, t_idx),
                dim=-1)
            s_proposal_scores = F.softmax(
                self.process_proposal_scores_distribution(s_proposal_scores, s_proposal_bbox_inst, s_idx),
                dim=-1)
            flag, ts_kl_div= self.compare_ts_pred_class_consistence(t_proposal_bbox_inst,
                                                                    s_proposal_bbox_inst,
                                                                    t_proposal_scores,
                                                                    s_proposal_scores)
            valid_map.append(flag)
            list_kl_div.append(ts_kl_div)
        # num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        valid_map=torch.tensor(valid_map).numpy()
        list_kl_div=torch.tensor(list_kl_div)
        return list_instances, valid_map, list_kl_div

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def pick_valid_and_add_label(self, unlabled_data, label, valid_map):
        for (unlabel_datum, lab_inst ,flag) in zip(unlabled_data, label, valid_map):
            if not flag:
                continue
            unlabel_datum["instances"] = lab_inst
        unlabled_data=list(filter(lambda d: "instances" in d, unlabled_data))
        return unlabled_data

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[SSOD_AT_Trainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        # remove unlabeled data labels
        unlabel_data_q = self.remove_label(unlabel_data_q)
        unlabel_data_k = self.remove_label(unlabel_data_k)

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, _, _, _ = self.model(
                label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)

            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                (
                    _,
                    teacher_proposals_rpn_unsup_k,
                    teacher_proposals_roih_unsup_k,
                    teacher_roi_predictions_unsup_k,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

                (
                    _,
                    student_proposals_rpn_unsup_k,
                    student_proposals_roih_unsup_k,
                    student_roi_predictions_unsup_k,
                ) = self.model(unlabel_data_k, branch="unsup_data_weak")

            batch_size_unlabel_k=self.cfg.SOLVER.IMG_PER_BATCH_UNLABEL // 2
            num_classes=self.cfg.MODEL.ROI_HEADS.NUM_CLASSES+1

            # print("打印roi_predictions: ", teacher_roi_predictions_unsup_k, teacher_roi_predictions_unsup_k[0].shape, teacher_roi_predictions_unsup_k[1].shape )

            teacher_roi_predictions_unsup_k=teacher_roi_predictions_unsup_k[0].reshape(batch_size_unlabel_k,-1,num_classes)
            student_roi_predictions_unsup_k=student_roi_predictions_unsup_k[0].reshape(batch_size_unlabel_k,-1,num_classes)

            #  Pseudo-labeling
            teacher_proposals={
                "rpn": teacher_proposals_rpn_unsup_k,
                "roih": teacher_proposals_roih_unsup_k,
                "roi_predictions": teacher_roi_predictions_unsup_k,
            }

            student_proposals = {
                "rpn": student_proposals_rpn_unsup_k,
                "roih": student_proposals_roih_unsup_k,
                "roi_predictions": student_roi_predictions_unsup_k,
            }

            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}

            joint_proposal_dict["proposals_rpn"] = teacher_proposals_rpn_unsup_k

            # (pesudo_proposals_rpn_unsup_k, num_pseudo_bbox_rpn) = self.process_pseudo_label(
            #     teacher_proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            # )
            # joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k

            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup_k, valid_map, list_kl_div= self.process_pseudo_label(
                teacher_proposals_roih_unsup_k,
                teacher_roi_predictions_unsup_k,
                student_proposals_roih_unsup_k,
                student_roi_predictions_unsup_k,
                cur_threshold,
                "roih",
                "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

            kl_weights=torch.exp(list_kl_div[valid_map]).numpy()

            # print("========打印kl_weights==========", "\n")
            # print(kl_weights)
            # print("========打印list_kl_div==========", "\n")
            # print(list_kl_div)
            # print("========打印valid_map==========", "\n")
            # print(valid_map)

            #  add pseudo-label to unlabeled data

            # unlabel_data_q = self.add_label(
            #     unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            # )
            unlabel_data_q = self.pick_valid_and_add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"], valid_map
            )

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

            if len(all_unlabel_data) == 0:
                record_dict = {}
                record_all_label_data, _, _, _ = self.model(
                    all_label_data, branch="supervised")
                record_dict.update(record_all_label_data)
                # weight losses
                loss_dict = {}
                for key in record_dict.keys():
                    if key[:4] == "loss":
                        loss_dict[key] = record_dict[key] * 1
                losses = sum(loss_dict.values())
            else:
                record_dict = {}
                record_all_label_data, _, _, _ = self.model(
                    all_label_data, branch="supervised"
                )
                record_dict.update(record_all_label_data)
                record_all_unlabel_data, _, _, _ = self.model(
                    all_unlabel_data, branch="supervised"
                )
                new_record_all_unlabel_data = {}
                for key in record_all_unlabel_data.keys():
                    new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
                record_dict.update(new_record_all_unlabel_data)
                # weight losses
                loss_dict = {}
                for (key,kl_weight) in zip(record_dict.keys(), kl_weights):
                    if key[:4] == "loss":
                        if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                            # pseudo bbox regression <- 0
                            loss_dict[key] = record_dict[key] * 0
                        elif key[-6:] == "pseudo":  # unsupervised loss
                            loss_dict[key] = (
                                record_dict[key] *
                                self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT * kl_weight
                            )
                        else:  # supervised loss
                            loss_dict[key] = record_dict[key] * 1

                losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        # print("========打印metrics_dict========")
        # print(all_metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            # metrics_dict = {
            #     k: np.mean([x[k] for x in all_metrics_dict])
            #     for k in all_metrics_dict[0].keys()
            # }
            tag_metrics = all_metrics_dict[0]

            for item in all_metrics_dict:
                if len(item.keys())==9:
                    tag_metrics=item
                    break

            for k in tag_metrics.keys():
                key_list=[]
                for x in all_metrics_dict:
                    if k in x:
                        key_list.append(x[k])
                metrics_dict[k]=np.mean(key_list)

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )
            

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)

            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            ret.append(
                hooks.BestCheckpointer(cfg.TEST.EVAL_PERIOD,self.checkpointer,"bbox/AP")
            )
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

