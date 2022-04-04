import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.python.client import device_lib
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT_DIR = os.path.abspath("./Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn as mrcnn
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

import json


class CustomDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_custom(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """

        # Load json from file
        # print("Annotation json path: ", annotation_json)
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()

        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']

            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}

        len_images = len(coco_json['images'])
        img_range = [0, len_images]

        for i in range(img_range[0], img_range[1]):
            image = coco_json['images'][i]
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """

        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        # print("Class_ids, ", class_ids)
        return mask, class_ids

    def count_classes(self):
        class_ids = set()
        for image_id in self.image_ids:
            image_info = self.image_info[image_id]
            annotations = image_info['annotations']

            for annotation in annotations:
                class_id = annotation['category_id']
                class_ids.add(class_id)

        class_number = len(class_ids)
        return class_number


def load_image_dataset(annotation_path, dataset_path):
    dataset = CustomDataset()
    dataset.load_custom(annotation_path, dataset_path)
    dataset.prepare()
    return dataset


def load_training_model(config, path=COCO_MODEL_PATH):
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    # print(path)
    model.load_weights(path, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask", ])
    return model


def train(finetune=1, epochs=5, file='Mask_RCNN/mask_rcnn_coco.h5'):
    dataset_path = './Dataset/images'
    train_path = './Dataset/train.json'
    val_path = './Dataset/val.json'
    dataset_train = load_image_dataset(train_path, dataset_path)
    dataset_val = load_image_dataset(val_path, dataset_path)
    class_number = dataset_train.count_classes()

    print(str(type(class_number)) + str(class_number))
    print('Train: %d' % len(dataset_train.image_ids))
    print('Validation: %d' % len(dataset_val.image_ids))
    print("Classes: {}".format(class_number))
    annotation_count = int(class_number)

    class TrainConfig(Config):
        NAME = "object"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + annotation_count
        STEPS_PER_EPOCH = 200
        VALIDATION_STEPS = 10
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512

    config = TrainConfig()
    path = os.path.join('./', file)
    rcnn = load_training_model(config, path)
    rcnn.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / finetune, epochs=epochs, layers='heads')


def copy_newest(nr):
    path = MODEL_DIR
    dir = sorted(os.listdir(path), key=str.lower)[-1]
    fpath = path + "/" + dir + "/" + sorted(os.listdir(path + "/" + dir), key=str.lower)[-1]
    dstformat = "./h5/mask_rcnn_object_{epoch:04d}.h5"
    dstname = dstformat.format(epoch=nr)
    from shutil import copyfile
    copyfile(fpath, dstname)


from Scripts import cocosplit

cocosplit.main('./Dataset/annotations.json', 'Dataset/train.json', 'Dataset/val.json', 0.8)


def main():
    finetune = int(input("enter Finetune (1=coarse, >1=finer): "))
    epochs = int(input("enter epochs: "))
    version_name = int(input("enter version number: "))
    starting_point = input("Enter starting point file name or \"NONE\" for from scratch: ")
    if starting_point == "NONE":
        starting_point = "Mask_RCNN/mask_rcnn_coco.h5"
    else:
        starting_point = "h5/" + starting_point
    train(finetune, epochs, starting_point)
    copy_newest(version_name)


if __name__ == '__main__':
    repeat = True
    while repeat:
        main()
        ans = input("another? y/n: ")
        if "n" == ans or "N" == ans:
            repeat = False
        else:
            repeat = True
