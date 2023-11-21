import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

class build_vhr():
    def __init__(self,
                dataset_root):
        self.CLASS_NAMES=['airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
              'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle']
        self.DATASET_ROOT=dataset_root
        self.ANN_ROOT=os.path.join(self.DATASET_ROOT, 'annotations')
        self.TRAIN_PATH=os.path.join(self.DATASET_ROOT, 'trainval-images')
        self.VAL_PATH=os.path.join(self.DATASET_ROOT, 'trainval-images')
        self.TRAIN_JSON = os.path.join(self.ANN_ROOT, 'train.json')
        self.VAL_JSON = os.path.join(self.ANN_ROOT, 'val.json')

        self.PREDEFINED_SPLITS_DATASET = {
            "vhr_coco_train": (self.TRAIN_PATH, self.TRAIN_JSON),
            "vhr_coco_val": (self.VAL_PATH, self.VAL_JSON),
        }
    
    def register_vhr_dataset(self):
        #register train set
        DatasetCatalog.register("vhr_coco_train", lambda: load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH))
        MetadataCatalog.get("vhr_coco_train").set(thing_classes=self.CLASS_NAMES, 
                                                        evaluator_type='coco', # evaluate method
                                                        json_file=self.TRAIN_JSON,
                                                        image_root=self.TRAIN_PATH)
        #register val set
        DatasetCatalog.register("vhr_coco_val", lambda: load_coco_json(self.VAL_JSON, self.VAL_PATH))
        MetadataCatalog.get("vhr_coco_val").set(thing_classes=self.CLASS_NAMES,
                                                    evaluator_type='coco', 
                                                    json_file=self.VAL_JSON,
                                                    image_root=self.VAL_PATH)

# if __name__ == "__main__":
#     register_vhr_dataset()
