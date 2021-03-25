import scipy.io
import os
import shutil
from PIL import Image
import numpy as np
import yaml

classes = [c.strip() for c in open('data/cub_200/lists/lists/classes.txt').readlines()]
class_dict = {c: i for i, c in enumerate(classes)}
files = [f.strip() for f in open('data/cub_200/lists/lists/files.txt').readlines()]
shfl_files = [f.strip() for f in open('data/cub_200/lists/lists/files.txt').readlines()]
np.random.shuffle(shfl_files)
train_test_split_idx = int(len(shfl_files) * 0.7)
train_set, test_set = shfl_files[:train_test_split_idx], shfl_files[train_test_split_idx:]
np.random.shuffle(train_set)
train_val_split_idx = int(len(train_set) * 0.8)
train_set, val_set = set(train_set[:train_val_split_idx]), set(train_set[train_val_split_idx:])

images_path = 'data/cub_200/images/images'
annotations_path = 'data/cub_200/annotations/annotations-mat'

if not os.path.isdir('yolov5/data/cub_200/images'):
    os.mkdir('yolov5/data/cub_200/images')
if not os.path.isdir('yolov5/data/cub_200/labels'):
    os.mkdir('yolov5/data/cub_200/labels')

with open('yolov5/data/cub_200/train.txt', 'w') as train_file:
    with open('yolov5/data/cub_200/test.txt', 'w') as test_file:
        with open('yolov5/data/cub_200/val.txt', 'w') as val_file:
            for f in files:
                img_path = os.path.normpath(os.path.join(images_path, f))
                mat_path = os.path.normpath(os.path.join(annotations_path, f)).replace('.jpg', '.mat')
                bb = np.vstack(scipy.io.loadmat(mat_path)['bbox'][0]).ravel()[0]
                bb = np.array([bb[0], bb[1], bb[2], bb[3]]).ravel().astype(np.float)
                img = Image.open(img_path)
                a = np.array([img.size[0], img.size[1]] * 2, dtype=np.float)
                cls_num = class_dict[os.path.split(f)[0]]
                yolo_bb = np.array([bb[0] + ((bb[2] - bb[0]) / 2), bb[1] + ((bb[3] - bb[1]) / 2), bb[2] - bb[0], bb[3] - bb[1]])
                yolo_bb /= a
                yolo_bb_str = f'{cls_num} {" ".join(map(str, yolo_bb))}'
                yolo_img_path = os.path.join('data/cub_200/images/', os.path.split(img_path)[-1])
                print(yolo_img_path)
                yolo_label_path = os.path.join('yolov5/data/cub_200/labels/', os.path.split(img_path)[-1].replace('.jpg', '.txt'))
                print(yolo_label_path)
                shutil.copy(img_path, os.path.join('yolov5', yolo_img_path))
                with open(yolo_label_path, 'w') as wf:
                    wf.write(yolo_bb_str)
                if f in train_set:
                    train_file.write(f'{yolo_img_path}\n')
                if f in test_set:
                    test_file.write(f'{yolo_img_path}\n')
                if f in val_set:
                    val_file.write(f'{yolo_img_path}\n')

config_dict = {
    'train': 'data/cub_200/train.txt',
    'test': 'data/cub_200/test.txt',
    'val': 'data/cub_200/val.txt',
    'nc': 200,
    'names': classes
}
with open('yolov5/data/cub_200/cub_200.yaml', 'w') as yf:
    yaml.dump(config_dict, yf)