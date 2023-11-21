import os.path as osp
import os
from PIL import Image
import json
from matplotlib import pyplot as plt


# Get template from info, annos need to be type [x,y,x,y]
def coco_annotations(bbox, cid, bbox_id, img_id, iscrowd):
    x1, y1, x2, y2 = bbox
    return {'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
            'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
            'category_id': cid,
            'area': (y2 - y1 + 1) * (x2 - x1 + 1),
            'iscrowd': iscrowd,
            'image_id': img_id,
            'id': bbox_id}


def coco_images(file_name, height, width, img_id):
    return {'file_name': file_name,
            'height': height,
            'width': width,
            'id': img_id}


# Get info from annos of dota, parameters cls_name2id，categories are needed
def deal_with_txt(label_path, img_id, anno_id):
    annos = []

    with open(label_path, 'r') as gt:
        gt_lines = gt.readlines()
    for i in gt_lines:
        iscrowd = int(i[-2])
        i = i.split(' ')
        cls_name = i[-2]
        cid = cls_name2id[cls_name]

        cood = i[:8]
        cood = tuple(map(float, cood))
        x1 = min(cood[0], cood[2], cood[4], cood[6])
        x2 = max(cood[0], cood[2], cood[4], cood[6])
        y1 = min(cood[1], cood[3], cood[5], cood[7])
        y2 = max(cood[1], cood[3], cood[5], cood[7])
        b = (x1, y1, x2, y2)

        anno = coco_annotations(b, cid, anno_id, img_id, iscrowd)
        annos.append(anno)
        anno_id += 1
    return annos, anno_id


def generate_coco_fmt(data_root, anno_root, categories, img_root):
    '''
    data_root: path of dataset, other path are relative of data_root
    anno_root: path odf annos
    img_root: path of images
    categories have got before
    '''
    img_id, anno_id = 0, 0
    all_annos, all_images = [], []

    for anno_txt in os.listdir(osp.join(data_root, anno_root)):
        file_name = anno_txt.replace('txt', 'png')
        label_path = osp.join(data_root, anno_root, anno_txt)
        img_path = osp.join(data_root, img_root, file_name)
        if osp.exists(img_path):
            annos, anno_id = deal_with_txt(label_path, img_id, anno_id)
            all_annos.extend(annos)

            img = Image.open(img_path)
            all_images.append(coco_images(osp.join(img_root, file_name), img.height, img.width, img_id))
            img_id += 1
    return {
        'images': all_images,
        "annotations": all_annos,
        "categories": categories,
        "type": "instance"
    }

if __name__ == '__main__':
    '''
        Get categories and cls_name2id
    '''
    classes=('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                   'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                   'basketball-court', 'storage-tank', 'soccer-ball-field',
                   'roundabout', 'harbor', 'swimming-pool', 'helicopter')
    cls_name2id={}
    categories=[]
    for i in range(0,15):
        cls_name2id.update({classes[i]:i})
        dict_cate={'id':i,'name':classes[i],'supercategory':classes[i]}
        categories.append(dict_cate)

    # path of annos and images
    data_root='/media/zbox/文档/datasets/dota_1024'
    anno_root='train/labelTxt'
    img_root='train/images'
    dota_coco_fmt=generate_coco_fmt(data_root,anno_root,categories,img_root)
    json.dump(dota_coco_fmt,open(osp.join(data_root,'train.json'),'a'),ensure_ascii=False)

