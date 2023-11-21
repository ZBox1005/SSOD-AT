import json
import operator

import os

import torch
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import detection_utils as utils
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.engine import default_argument_parser, default_setup
from detectron2.structures import pairwise_iou, Boxes

from ssod_at.config import add_ssod_at_config
from ssod_at.data.build import divide_label_unlabel
from ssod_at.engine.trainer import SSOD_AT_Trainer
from ssod_at.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from tqdm import tqdm

from ssod_at.utils.build_datasets import build_dota, build_dior
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

global_prototype = torch.zeros(15, 1024, dtype=torch.float, device="cuda")
global_class_roi_features = [0]*15


def process_nan(a):
    for i in range(a.shape[0]):
        a[i] = torch.where(torch.isnan(a[i]), torch.tensor(0, dtype=a.dtype, device=a.device), a[i])
    return a


def update_global_prototype(origin, new, keep_rate=0.996):
    return keep_rate * origin + (1 - keep_rate) * new


def get_sim_scores_to_compute_diversity(roi_features_pgt, s_thr=0.6):
    roi_features_pgt_new = roi_features_pgt.unsqueeze(1)
    global_proto_tensor = global_prototype.unsqueeze(0)
    sim_matrix = torch.cosine_similarity(roi_features_pgt_new, global_proto_tensor, dim=-1)
    final_scores, final_idxes = torch.max(sim_matrix, dim=1)
    for i in range(final_scores.shape[0]):
        if final_scores[i] < s_thr:
            global_prototype[final_idxes[i]] = update_global_prototype(global_prototype[final_idxes[i]],
                                                                       roi_features_pgt[i])
    return final_scores


def get_roi_features(cfg, model_student, proposals_teacher, proposals_student, features,
                     stage='proto_init'):
    box_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
    features = [features[f] for f in box_in_features]

    box_pooler_student = model_student.roi_heads.box_pooler
    box_head_student = model_student.roi_heads.box_head

    if stage == 'proto_init':
        box_features = box_pooler_student(features, [proposals_student])
        roi_features = box_head_student(box_features)
    elif stage == 'proto_update':
        box_features = box_pooler_student(features, [x.pred_boxes for x in proposals_teacher])
        # box_features = box_pooler_student(features, [x.proposal_boxes for x in proposals_teacher])
        roi_features = box_head_student(box_features)
    else:
        raise ValueError('Invalid Parameter: stage, stage must in proto_init or ptoto_update')

    return roi_features


def get_t_iou_max_in_s(boxes_t,boxes_s):
    # compute iou among teacher boxes and student boxes and get a iou_matrix
    iou_matrix = pairwise_iou(boxes_t, boxes_s).to(torch.device("cpu"))
    # for each teacher box, find the box with max iou in student boxes
    iou_max_t_in_s_idx = torch.max(iou_matrix, 1)[1].numpy()
    return iou_max_t_in_s_idx


def compare_ts_pred_class_consistence(
    teacher_proposal_bbox_inst,
    student_proposal_bbox_inst
):

    teacher_boxes=teacher_proposal_bbox_inst.pred_boxes
    student_boxes=student_proposal_bbox_inst.pred_boxes

    teacher_classes=teacher_proposal_bbox_inst.pred_classes
    student_classes=student_proposal_bbox_inst.pred_classes

    if teacher_classes.shape[0]==0 or student_classes.shape[0]==0:
        return []

    iou_max_t_in_s_idx = get_t_iou_max_in_s(teacher_boxes,student_boxes)

    unc_inst_list=[]
    for idx_teacher in range(teacher_classes.shape[0]):
        t_cls=teacher_classes[idx_teacher]
        s_cls=student_classes[iou_max_t_in_s_idx[idx_teacher]]
        if t_cls!=s_cls:
            unc_inst_list.append(idx_teacher)

    return unc_inst_list


def get_t_scores_to_compute_unc(t_proposal_scores,t_bbox_inst,unc_inst_list):
    origin_idx=t_bbox_inst.origin_idx.cpu().detach().numpy()
    final_idx=[]
    for idx in unc_inst_list:
        final_idx.append(origin_idx[idx])
    valid_map_step1 = []
    for i in range(t_proposal_scores.shape[0]):
        valid_map_step1.append(i in final_idx)
    # print("打印idx: ", idx,'\n')
    final_scores = t_proposal_scores[valid_map_step1]
    return final_scores

# def compute_uncertainty(scores_teacher, scores_student, proposals_teacher, proposals_student):
#     scores_teacher = F.softmax(scores_teacher, dim=1)
#     scores_student = F.softmax(scores_student, dim=1)
#     scores_student_new = torch.full_like(scores_teacher, 0)
#     for i in range(scores_student_new.shape[0]):
#         scores_student_new[i] = scores_student[iou_max_per_teacher_instance_idx[i]]
#     scores_teacher_entropy = process_nan(-torch.log(scores_teacher) * scores_teacher)
#     scores_student_entropy = process_nan(torch.log(scores_student_new) * scores_student_new)
#     entropy_dif_teacher_and_student = torch.sum(torch.abs(torch.add(scores_teacher_entropy, scores_student_entropy)),
#                                                 dim=1)
#     uncertainty_image = torch.mean(entropy_dif_teacher_and_student)
#     return uncertainty_image


def uncertainty_entropy(p):
    # p.size() = num_instances of a image, num_classes
    p = F.softmax(p, dim=1)
    p = - torch.log2(p) * p
    unc_instances = torch.sum(p, dim=1)
    # set uncertainty of image eqs the mean uncertainty of instances
    unc_image = torch.mean(unc_instances)
    return unc_image


def compute_uncertainty(t_bbox_inst,s_bbox_inst,t_proposal_scores):
    unc_list=compare_ts_pred_class_consistence(t_bbox_inst,s_bbox_inst)
    if len(unc_list)==0:
        return torch.tensor(0)
    # p.size() = num_instances of an image, num_classes
    p=get_t_scores_to_compute_unc(t_proposal_scores,t_bbox_inst,unc_list)
    return uncertainty_entropy(p)


def compute_diversity(x):
    l = x.shape[0]
    return 1 - torch.sum(x) / l


data_hook_teacher = {}
data_hook_student = {}


def box_predictor_hooker_teacher(m, i, o):
    data_hook_teacher['scores_hooked'] = o[0].clone().detach()
    data_hook_teacher['boxes_hooked'] = o[1].clone().detach()


def box_predictor_hooker_student(m, i, o):
    data_hook_student['scores_hooked'] = o[0].clone().detach()
    data_hook_student['boxes_hooked'] = o[1].clone().detach()


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
    if cfg.DATASETS.TRAIN[0]=="dota_1024_coco_train":
        dota=build_dota("/data1/users/tuwenkai/ZBox/datasets/dota_v1.0_1024")
        dota.register_dota_dataset()
    elif cfg.DATASETS.TRAIN[0]=="vhr_coco_train":
        vhr=build_vhr("/data/users/zhangboxuan/datasets/NWPU-VHR-10")
        vhr.register_vhr_dataset()
    elif cfg.DATASETS.TRAIN[0]=="dior_coco_train":
        dior=build_dior("/data1/users/tuwenkai/ZBox/datasets/DIOR-coco")
        dior.register_dior_dataset()
    Trainer = SSOD_AT_Trainer
    assert args.eval_only is True, "Inference should be eval only."
    inference(Trainer, cfg)


@torch.no_grad()
def inference(Trainer, cfg):
    print('Loading Model named: ', cfg.MODEL.WEIGHTS)
    model = Trainer.build_model(cfg)
    model_teacher = Trainer.build_model(cfg)
    ensem_ts_model = EnsembleTSModel(model_teacher, model)

    DetectionCheckpointer(
        ensem_ts_model, save_dir=cfg.OUTPUT_DIR
    ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

    # 注册modelTeacher的hook
    ensem_ts_model.modelTeacher.roi_heads.box_predictor.register_forward_hook(box_predictor_hooker_teacher)
    # ensem_ts_model.modelTeacher.proposal_generator.register_forward_hook(proposal_generator_hooker_teacher)
    ensem_ts_model.modelTeacher.eval()
    ensem_ts_model.modelTeacher.training = False

    # 注册modelStudent的hook
    ensem_ts_model.modelStudent.roi_heads.box_predictor.register_forward_hook(box_predictor_hooker_student)
    # ensem_ts_model.modelStudent.proposal_generator.register_forward_hook(proposal_generator_hooker_student)
    ensem_ts_model.modelStudent.eval()
    ensem_ts_model.modelStudent.training = False

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    label_dicts, unlabel_dicts = divide_label_unlabel(
        dataset_dicts,
        cfg.DATALOADER.SUP_PERCENT,
        cfg.DATALOADER.RANDOM_DATA_SEED,
        cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
    )

    dic = {}
    final_dic={}


    # use labeled images to init global_prototype
    print('start init global class prototype')
    for j, item in enumerate(tqdm(label_dicts)):
        file_name = item['file_name']
        # print(j, file_name)
        image = utils.read_image(file_name, format='BGR')
        image = torch.from_numpy(image.copy()).permute(2, 0, 1)
        image = ensem_ts_model.modelStudent.preprocess_image([{'image': image}])
        features = ensem_ts_model.modelStudent.backbone(image.tensor)
        annotations = item['annotations']
        bboxes = []
        categories_id = []
        for anno in annotations:
            bbox = anno['bbox']
            category_id = anno['category_id']
            bboxes.append(bbox)
            categories_id.append(category_id)
        bboxes = Boxes(torch.tensor(bboxes, device="cuda"))
        roi_feature_gt = get_roi_features(cfg, ensem_ts_model.modelStudent, None, bboxes, features, stage='proto_init')
        for i, class_id in enumerate(categories_id):
            # 原型中还没有该类，用当前roi_feature进行初始化
            if global_prototype[class_id].equal(torch.zeros(1024, dtype=torch.float, device="cuda")):
                global_prototype[class_id] = roi_feature_gt[i]
                global_class_roi_features[class_id] = roi_feature_gt[i].unsqueeze(0)
            # 原型中已有该类，对原型进行更新，求平均原型
            else:
                origin_class_roi=global_class_roi_features[class_id]
                new_roi=roi_feature_gt[i].unsqueeze(0)
                global_class_roi_features[class_id] = torch.cat((origin_class_roi, new_roi), dim=0)
                global_prototype[class_id] =torch.mean(global_class_roi_features[class_id],dim=0)
        del image
        del features
        del anno
        del annotations
        del bbox
        del bboxes
        del category_id
        del categories_id
        del class_id
        # del proposals_student
        del roi_feature_gt
    print('labeled dicts for global class prototype init Done')



    # Using unlabeled images to ema update global prototype
    print('start compute unc and select unlabeled images with unc metric')
    for j, item in enumerate(tqdm(unlabel_dicts)):
        g = global_prototype
        file_name = item['file_name']
        # print(j, file_name)

        final_dic[file_name] = []

        # preprocess image
        image = utils.read_image(file_name, format='BGR')
        image = torch.from_numpy(image.copy()).permute(2, 0, 1)
        image = ensem_ts_model.modelTeacher.preprocess_image([{'image': image}])
        # get features of image
        features = ensem_ts_model.modelTeacher.backbone(image.tensor)

        proposals_teacher, _ = ensem_ts_model.modelTeacher.proposal_generator(image, features, None)
        proposals_student, _ = ensem_ts_model.modelStudent.proposal_generator(image, features, None)

        # roi info of teacher and student : pred_boxes , scores , pred_classes
        res_teacher, _ = ensem_ts_model.modelTeacher.roi_heads(image, features, proposals_teacher, None)
        res_student, _ = ensem_ts_model.modelStudent.roi_heads(image, features, proposals_student, None)

        scores_teacher = data_hook_teacher["scores_hooked"].to(torch.device("cpu"))

        if len(res_teacher[0]) == 0:
            image_info = {
                "uncertainty": 0,
                "diversity": 0
            }
            final_dic[file_name].append(image_info)
            continue

        uncertainty = compute_uncertainty(res_teacher[0], res_student[0], scores_teacher)

        if uncertainty==0:
            image_info = {
                "uncertainty": 0,
                "diversity": 0
            }
            final_dic[file_name].append(image_info)
            continue

        image_info = {
            "uncertainty": uncertainty.cpu().detach().clone().item(),
        }
        dic[file_name]=image_info


        del image
        del image_info
        del features
        del proposals_teacher
        del proposals_student
        del res_teacher
        del res_student
        del data_hook_teacher['scores_hooked']
        torch.cuda.empty_cache()
    print('unlabeled dicts with unc metric Done, len dic: ', len(dic))


    def sort_key(item):
        return item[1]["uncertainty"]
    sorted_dic=dic.items()
    sorted_dic=sorted(sorted_dic, key=sort_key)


    # get image info (uncertainty and diversity) of unlabeled images
    print('start compute diversity')
    for j, item in enumerate(tqdm(sorted_dic)):
        file_name = item[0]
        image_info=item[1]
        print(j, file_name, image_info)

        # preprocess image
        image = utils.read_image(file_name, format='BGR')
        image = torch.from_numpy(image.copy()).permute(2, 0, 1)
        image = ensem_ts_model.modelTeacher.preprocess_image([{'image': image}])
        # get features of image
        features = ensem_ts_model.modelTeacher.backbone(image.tensor)

        proposals_teacher, _ = ensem_ts_model.modelTeacher.proposal_generator(image, features, None)
        proposals_student, _ = ensem_ts_model.modelStudent.proposal_generator(image, features, None)

        # roi info of teacher and student : pred_boxes , scores , pred_classes
        res_teacher, _ = ensem_ts_model.modelTeacher.roi_heads(image, features, proposals_teacher, None)
        res_student, _ = ensem_ts_model.modelStudent.roi_heads(image, features, proposals_student, None)

        # [num_instances,roi_feature]
        roi_features_pgt = get_roi_features(cfg, ensem_ts_model.modelStudent, res_teacher, None, features,
                                            stage="proto_update")

        # get sim scores of features in global prototype
        max_sim_scores_in_global=get_sim_scores_to_compute_diversity(roi_features_pgt)

        # get diversity scores
        diversity = compute_diversity(max_sim_scores_in_global)

        
        image_info["diversity"] = diversity.cpu().detach().clone().item()

        final_dic[file_name].append(image_info)

        del image
        del features
        del image_info
        del proposals_teacher
        del proposals_student
        del roi_features_pgt
        del max_sim_scores_in_global
        del res_teacher
        del res_student
        # del data_hook_teacher['boxes_hooked']
        # del data_hook_student['scores_hooked']
        # del data_hook_student['boxes_hooked']
        torch.cuda.empty_cache()
    print('unlabeled dicts for compute uncertainty and diversity Done')



    with open(FILE_PATH, 'w') as f:
        f.write(json.dumps(final_dic))
    print('Done')


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument("--static-file", type=str,
                        default='./temp/dota_1/v6.0/static_by_active35.json')  # Json file of the intermediate process
    parser.add_argument("--model-weights", type=str, default='./output_ablations/dota_1/v6.0/dota_1024_sup35_run1_32bs/model_best.pth')
    args = parser.parse_args()
    args.eval_only = True
    args.resume = True
    args.num_gpus = 1

    # args.static_file='../temp/coco/static_by_random5.json'
    args.config_file = './configs/dota_coco_1024/dota_coco_1024_sup35_run1.yaml'  # the config file you used to train this inference model
    # args.model_weights = '../output/coco/model_best.pth'
    # args.opts = ['OUTPUT_DIR', "../output/coco/active_pick_strategies"]
    FILE_PATH = args.static_file
    # you should config MODEL.WEIGHTS and keep other hyperparameters default(Odd-numbered items are keys, even-numbered items are values)
    args.opts = ['MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5, 'MODEL.ROI_HEADS.NMS_THRESH_TEST', 0.5,
                 'TEST.DETECTIONS_PER_IMAGE', 20, 'INPUT.FORMAT', 'RGB', 'MODEL.WEIGHTS', args.model_weights,
                 'DATALOADER.RANDOM_DATA_SEED_PATH' , './dataseed/dota_pick/dota_1/v6.0/pick_maxnorm30+random5.txt']
    print("Command Line Args:", args)
    main(args)
