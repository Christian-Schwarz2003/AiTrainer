import json
import funcy
from sklearn.model_selection import train_test_split
import os.path


def save_coco(file, info, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'info': info, 'images': images,
                   'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def main(annotation_file, train, test, split, having_annotations=True):
    #print(os.path.isfile(annotation_file))
    with open(annotation_file, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=split)

        save_coco(train, info, x, filter_annotations(annotations, x), categories)
        save_coco(test, info, y, filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(len(x), train, len(y), test))

# main('Dataset/annotations.json', 'Dataset/train.json', 'Dataset/val.json', 0.8)
