import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

class build_dota():
    def __init__(self,
                dataset_root):
        self.CLASS_NAMES=['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                   'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                   'basketball-court', 'storage-tank', 'soccer-ball-field',
                   'roundabout', 'harbor', 'swimming-pool', 'helicopter']
        self.DATASET_ROOT=dataset_root
        self.ANN_ROOT=os.path.join(self.DATASET_ROOT, 'annotations')
        self.TRAIN_PATH=os.path.join(self.DATASET_ROOT, '')
        self.VAL_PATH=os.path.join(self.DATASET_ROOT, '')
        self.TRAIN_JSON = os.path.join(self.ANN_ROOT, 'train_HBB.json')
        self.VAL_JSON = os.path.join(self.ANN_ROOT, 'val_HBB.json')

        self.PREDEFINED_SPLITS_DATASET = {
            "dota_1024_coco_train": (self.TRAIN_PATH, self.TRAIN_JSON),
            "dota_1024_coco_val": (self.VAL_PATH, self.VAL_JSON),
        }
    
    def register_dota_dataset(self):
        #register train set
        DatasetCatalog.register("dota_1024_coco_train", lambda: load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH))
        MetadataCatalog.get("dota_1024_coco_train").set(thing_classes=self.CLASS_NAMES, 
                                                        evaluator_type='coco', # evaluate method
                                                        json_file=self.TRAIN_JSON,
                                                        image_root=self.TRAIN_PATH)
        #register val set
        DatasetCatalog.register("dota_1024_coco_val", lambda: load_coco_json(self.VAL_JSON, self.VAL_PATH))
        MetadataCatalog.get("dota_1024_coco_val").set(thing_classes=self.CLASS_NAMES,
                                                    evaluator_type='coco', 
                                                    json_file=self.VAL_JSON,
                                                    image_root=self.VAL_PATH)

# if __name__ == "__main__":
#     register_dota_dataset()
