import json
import argparse
import random
from detectron2.data.build import (
    get_detection_dataset_dicts,
)
from ssod_at.utils.build_datasets import build_dota, build_dior

def generate_random_merge_random(random_file, random_percent, pick_random_percent, save_file):

    datasets = []
    for d in args.datasets.split(','):
        if d == '':
            continue
        datasets.append(d)
    dataset_dicts = get_detection_dataset_dicts(datasets)
    total_imgs=len(dataset_dicts)
    print('total images: ',total_imgs)
    idx=range(total_imgs)

    dic={}
    dic[str(random_percent+pick_random_percent)]={}
    with open(random_file,'r') as f:
        table = json.load(f)
        for i in range(10):
            exist_idx = table[str(random_percent)][str(i)]
            iddx = []
            for item in idx:
                if item not in exist_idx:
                    iddx.append(item)
            num_pick_random= int(total_imgs*(random_percent+pick_random_percent)/100) - len(table[str(random_percent)][str(i)])
            pick_random=random.sample(iddx,num_pick_random)
            arr = pick_random + table[str(random_percent)][str(i)]
            print(num_pick_random,len(arr))
            dic[str(random_percent+pick_random_percent)][str(i)] = arr

    with open(save_file,'w') as f:
        f.write(json.dumps(dic))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pick and merge label data partition')
    parser.add_argument("--datasets",type=str,default='dota_1024_coco_train,') 
    parser.add_argument("--random_file",type=str,default='')
    parser.add_argument("--random_percent",type=float,default=10.0)
    parser.add_argument("--pick_percent",type=float,default=10.0)
    parser.add_argument("--save_file",type=str,default='')
    args = parser.parse_args()
    print(args.datasets)
    if args.datasets=="dota_1024_coco_train,":
        dota=build_dota("/data1/users/tuwenkai/ZBox/datasets/dota_v1.0_1024")
        dota.register_dota_dataset()
    elif args.datasets=="dior_coco_train,":
        dior=build_dior("/data1/users/tuwenkai/ZBox/datasets/DIOR-coco")
        dior.register_dior_dataset()
    generate_random_merge_random(
        args.random_file,
        args.random_percent,
        args.pick_percent,
        args.save_file,
    )