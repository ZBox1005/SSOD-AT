import os
import os.path as osp

from DOTA_devkit.ImgSplit_multi_process import splitbase as splitbase_trainval
from DOTA_devkit.SplitOnlyImage_multi_process import splitbase as splitbase_test

# from .ImgSplit_multi_process import splitbase as splitbase_trainval
# from .SplitOnlyImage_multi_process import splitbase as splitbase_test
from DOTA_devkit.convert_dota_to_mmdet import convert_dota_to_mmdet


def mkdir_if_not_exists(path):
    if not osp.exists(path):
        os.mkdir(path)


def prepare_multi_scale_data(src_path, dst_path, gap=200, subsize=1024, scales=[0.5, 1.0, 1.5], num_process=32):
    dst_train_path = osp.join(dst_path, 'train')
    dst_val_path=osp.join(dst_path,'val')
    dst_test_base_path = osp.join(dst_path, 'test')
    dst_test_path = osp.join(dst_path, 'test/images')
    # make dst path if not exist
    mkdir_if_not_exists(dst_path)
    mkdir_if_not_exists(dst_train_path)
    mkdir_if_not_exists(dst_val_path)
    mkdir_if_not_exists(dst_test_base_path)
    mkdir_if_not_exists(dst_test_path)

    # split train data
    print('split train data')
    split_train = splitbase_trainval(osp.join(src_path, 'train'), dst_train_path,
                                     gap=gap, subsize=subsize, num_process=num_process)
    for scale in scales:
        split_train.splitdata(scale)

    # split val data
    print('split val data')
    split_val = splitbase_trainval(osp.join(src_path, 'val'), dst_val_path,
                                   gap=gap, subsize=subsize, num_process=num_process)
    for scale in scales:
        split_val.splitdata(scale)

    # split test data
    print('split test data')
    split_test = splitbase_test(osp.join(src_path, 'test/images'), dst_test_path,
                                gap=gap, subsize=subsize, num_process=num_process)
    for scale in scales:
        split_test.splitdata(scale)
    print('done!')


if __name__ == '__main__':
    prepare_multi_scale_data('/data/zhangboxuan/datasets/dota_v1.0', '/data/zhangboxuan/datasets/dota_v1.0_1024', gap=200, subsize=1024, scales=[1.0],
                             num_process=32)
