import argparse
import json
import math
from collections import defaultdict


def norm_dict(_dict):
    v_max = max(_dict.values())
    for k, v in _dict.items():
        _dict[k] = v / v_max
    _dict['max'] = v_max
    return _dict


def PreprocessData(file_name):
    print('Loading File {}...'.format(file_name))
    with open(file_name, 'r') as f:
        data = json.load(f)
    uncertainty_indicators = defaultdict(float)
    diversity_indicators = defaultdict(float)

    for index, (image_path, image_info) in enumerate(data.items()):
        # boxes_info: list[box0, box1, ...]
        _uncertainty = image_info[0]["uncertainty"]
        _diversity = image_info[0]["diversity"]

        if math.isnan(_diversity):
            _diversity=0

        uncertainty_indicators[image_path] = _uncertainty
        diversity_indicators[image_path] = _diversity

    return data, uncertainty_indicators, diversity_indicators


def CombineMetrics(data,
                   uncertainty_indicators,
                   diversity_indicators,
                   weights,
                   file_name):
    f = open(file_name + '.txt', 'w')
    final_value = defaultdict(float)
    for image_path, _ in data.items():
        _final_value = uncertainty_indicators[image_path] * weights['uncertainty'] + \
                       diversity_indicators[image_path] * weights['diversity']
        final_value[image_path] = _final_value
        f.write(str(_final_value) + '\n')
    f.close()

    with open(file_name + '.json', 'w') as f:
        f.write(json.dumps(final_value))

    print('Finish {}'.format(file_name))


def special_(args):
    weights = {
        'uncertainty': 1.0,
        'diversity': 1.0,
    }  # weights to combine metrics

    data, uncertainty_indicators, diversity_indicators = PreprocessData(file_name=args.static_file)
    _uncertainty_indicators = norm_dict(uncertainty_indicators)
    _diversity_indicators = norm_dict(diversity_indicators)
    CombineMetrics(data,
                   _uncertainty_indicators,
                   _diversity_indicators,
                   weights=weights,
                   file_name=args.indicator_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score function')
    parser.add_argument("--static-file", type=str,
                        default='temp/coco/static_by_random10.json')  # Json file of the inferenced datasets info
    parser.add_argument("--indicator-file", type=str,
                        default='results/coco/15random_maxnorm')  # indicator file to be used in picking data
    args = parser.parse_args()
    special_(args)
