import json
import os
import os.path
import warnings

import mrcnn.model as modellib
import numpy as np
from PIL import Image, ImageDraw
from Scripts import cocosplit
from mrcnn import utils
from mrcnn.config import Config

warnings.simplefilter(action='ignore', category=FutureWarning)
# Local path to trained weights file
COCO_MODEL_PATH = "/content/mask_rcnn_coco.h5"
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

MODEL_DIR = "/content/logs"


class ToothDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, images_dir):
        json_file = open(dataset_dir)
        coco_json = json.load(json_file)
        json_file.close()

        for category in coco_json['categories']:
            class_id = category['id']

            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return

            self.add_class('Dataset', class_id, class_name)

        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        seen_images = {}
        for image in coco_json['images']:
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
                    source='Dataset',
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    # load the masks for an image
    def load_mask(self, image_id):
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
        return mask, class_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


def load_weights(config):
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    return model


def train():
    dataset_path = '/content/Dataset/images'
    train_path = '/content/Dataset/train.json'
    val_path = '/content/Dataset/val.json'

    annotation_count = 1
    epochs = 5

    class TrainConfig(Config):
        NAME = "object"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 4
        NUM_CLASSES = 1 + annotation_count
        STEPS_PER_EPOCH = 20
        VALIDATION_STEPS = 2
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512
        DETECTION_MIN_CONFIDENCE = 0.9

    config = TrainConfig()
    # train set
    dataset_train = ToothDataset()
    dataset_train.load_dataset('/content/Dataset/train.json', '/content/Dataset/images')
    dataset_train.prepare()
    print('Train: %d' % len(dataset_train.image_ids))

    # test/val set
    dataset_val = ToothDataset()
    dataset_val.load_dataset('/content/Dataset/val.json', '/content/Dataset/images')
    dataset_val.prepare()
    print('Test: %d' % len(dataset_val.image_ids))

    model = load_weights(config)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=10,
                layers="all")


cocosplit.main('/content/Dataset/annotations.json', 'Dataset/train.json', 'Dataset/val.json', 0.8)

train()
