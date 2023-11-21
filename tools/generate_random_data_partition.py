import json
import argparse
from detectron2.data.build import (
    get_detection_dataset_dicts,
)
from ssod_at.utils.build_datasets import build_dota, build_dior

def generate(args):
    print(args.datasets)
    if args.datasets=="dota_1024_coco_train,":
        dota=build_dota("/data1/users/tuwenkai/ZBox/datasets/dota_v1.0_1024")
        dota.register_dota_dataset()
    elif args.datasets=="dior_coco_train,":
        dior=build_dior("/data1/users/tuwenkai/ZBox/datasets/DIOR-coco")
        dior.register_dior_dataset()
    datasets = []
    for d in args.datasets.split(','):
        if d == '':
            continue
        datasets.append(d)
    dataset_dicts = get_detection_dataset_dicts(datasets)
    print('len of dataset:',len(dataset_dicts))
    try:
        with open(args.random_file,'r') as f:
                dic=json.load(f)
    except:
        dic={}

    dic[str(args.random_percent)] = {}
    seeds = [int(i) for i in args.random_seeds.split(',')]
    for i in range(10):
        arr = generate_supervised_seed(
            dataset_dicts,
            args.random_percent,
            seeds[i]
        )
        print(len(arr))
        dic[str(args.random_percent)][str(i)] = arr
    with open(args.random_file,'w') as f:
        f.write(json.dumps(dic))


def generate_supervised_seed(
    dataset_dicts, SupPercent, seed
):
    num_all = len(dataset_dicts)
    num_label = int(SupPercent / 100.0 * num_all)

    arr = range(num_all)
    import random
    random.seed(seed)
    return random.sample(arr,num_label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--random-file",type=str,default='')
    parser.add_argument("--random-percent",type=float,default=10.0)
    parser.add_argument("--datasets",type=str,default='dota_1024_coco_train,') # default='voc_2007_trainval,voc_2012_trainval,'
    parser.add_argument("--random-seeds",type=str,default="0,1,2,3,4,5,6,7,8,9") # Need to set 10 random number seeds for experiments, divided by ','
    args = parser.parse_args()
    generate(args)