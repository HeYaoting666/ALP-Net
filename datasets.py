import os
import cv2
import random
import numpy as np
from torch.utils.data import *
from imutils import paths
from config import *
from utils.utils import get_dir_paths, collate_fn


class LPRDataSet(Dataset):
    def __init__(self, img_dir, imgSize, PreprocFun=None):

        self.img_dir = img_dir
        self.img_size = imgSize
        self.img_paths = []
        random.seed(0)

        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        h, w, _ = image.shape
        if h != self.img_size[1] or w != self.img_size[0]:
            image = cv2.resize(image, self.img_size)
        image = self.PreprocFun(image)

        base_name = os.path.basename(img_path)
        img_name, suffix = os.path.splitext(base_name)
        img_name = img_name.split("-")[-1]
        label = list()
        for i, c in enumerate(img_name):
            if i == 0:
                label.append(CHARS_DICT_1[c])
            elif i == 1:
                label.append(CHARS_DICT_2[c])
            else:
                label.append(CHARS_DICT_3[c])

        return image, label

    @staticmethod
    def transform(img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img


def LPRDataLoader(dirs, img_size, batch_size=256, num_workers=0, shuffle=True):
    data_path = get_dir_paths(dirs)
    if len(data_path) == 0:
        data_path = [dirs]
    return DataLoader(
        LPRDataSet(data_path, img_size),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
