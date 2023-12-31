import json
import argparse

def generate_pick_merge_random(random_file, random_percent, indicator_file, pick_percent, save_file, reverse=True):
    P = []
    with open(indicator_file,'r') as f:
        items = f.readlines()
        P = [float(item) for item in items]

    idx = sorted(range(len(P)), key=lambda k: P[k], reverse=reverse)
    len_unlabeled=len(idx)
    # print(idx,len(idx))

    with open(random_file,'r') as f:
        table = json.load(f)
        len_labeled=len(table[str(random_percent)][str(0)])
    
    total_imgs = len_labeled+len_unlabeled
    print('total images: ',total_imgs)
    dic={}
    dic[str(random_percent+pick_percent)]={}
    with open(random_file,'r') as f:
        table = json.load(f)
        for i in range(10):
            exist_idx = table[str(random_percent)][str(i)]
            iddx = []
            for item in idx:
                if item not in exist_idx:
                    iddx.append(item)
            left = int(total_imgs*(random_percent+pick_percent)/100) - len(table[str(random_percent)][str(i)])
            arr = iddx[:left] + table[str(random_percent)][str(i)]
            print(left,len(arr))
            dic[str(random_percent+pick_percent)][str(i)] = arr

    with open(save_file,'w') as f:
        f.write(json.dumps(dic))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pick and merge label data partition')
    parser.add_argument("--random_file",type=str,default='')
    parser.add_argument("--random_percent",type=float,default=10.0)
    parser.add_argument("--indicator_file",type=str,default='')
    parser.add_argument("--pick_percent",type=float,default=10.0)
    parser.add_argument("--reverse",type=bool,default=True)
    parser.add_argument("--save_file",type=str,default='')
    args = parser.parse_args()
    generate_pick_merge_random(
        args.random_file,
        args.random_percent,
        args.indicator_file,
        args.pick_percent,
        args.save_file,
        args.reverse
    )