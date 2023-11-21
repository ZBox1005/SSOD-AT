import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

class build_dior():
    def __init__(self,
                dataset_root):
        self.CLASS_NAMES=['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam',
                 'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
                 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']
        self.DATASET_ROOT=dataset_root
        self.ANN_ROOT=os.path.join(self.DATASET_ROOT, 'annotations')
        self.TRAIN_PATH=os.path.join(self.DATASET_ROOT, 'train/images')
        self.VAL_PATH=os.path.join(self.DATASET_ROOT, 'val/images')
        self.TEST_PATH=os.path.join(self.DATASET_ROOT, 'test/images')
        self.TRAIN_JSON = os.path.join(self.ANN_ROOT, 'train_HBB.json')
        self.VAL_JSON = os.path.join(self.ANN_ROOT, 'val_HBB.json')
        self.TEST_JSON = os.path.join(self.ANN_ROOT, 'test_HBB.json')

        self.PREDEFINED_SPLITS_DATASET = {
            "dior_coco_train": (self.TRAIN_PATH, self.TRAIN_JSON),
            "dior_coco_val": (self.VAL_PATH, self.VAL_JSON),
            "dior_coco_test": (self.TEST_PATH, self.TEST_JSON),
        }
    
    def register_dior_dataset(self):
        #register train set
        DatasetCatalog.register("dior_coco_train", lambda: load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH))
        MetadataCatalog.get("dior_coco_train").set(thing_classes=self.CLASS_NAMES, 
                                                        evaluator_type='coco', # evaluate method
                                                        json_file=self.TRAIN_JSON,
                                                        image_root=self.TRAIN_PATH)
        #register val set
        DatasetCatalog.register("dior_coco_val", lambda: load_coco_json(self.VAL_JSON, self.VAL_PATH))
        MetadataCatalog.get("dior_coco_val").set(thing_classes=self.CLASS_NAMES,
                                                    evaluator_type='coco', 
                                                    json_file=self.VAL_JSON,
                                                    image_root=self.VAL_PATH)
        #register val set
        DatasetCatalog.register("dior_coco_test", lambda: load_coco_json(self.TEST_JSON, self.TEST_PATH))
        MetadataCatalog.get("dior_coco_test").set(thing_classes=self.CLASS_NAMES,
                                                    evaluator_type='coco', 
                                                    json_file=self.TEST_JSON,
                                                    image_root=self.TEST_PATH)

# if __name__ == "__main__":
#     register_dior_dataset()
