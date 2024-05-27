# This file is part of CATGCasFormer, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/CATGCasFormer/blob/master/LICENSE for details.

import random
import math
import torch
import numpy as np
from copy import deepcopy
from torchvision.transforms import functional as F
import torchvision.transforms.transforms as transforms

def mixup_data(images, alpha=0.8):
    if alpha > 0. and alpha < 1.:
        lam = random.uniform(alpha, 1)
    else:
        lam = 1.

    batch_size = len(images)
    min_x = 9999
    min_y = 9999
    for i in range(batch_size):
        min_x = min(min_x, images[i].shape[1])
        min_y = min(min_y, images[i].shape[2])

    shuffle_images = deepcopy(images)
    random.shuffle(shuffle_images)
    mixed_images = deepcopy(images)
    for i in range(batch_size):
        mixed_images[i][:, :min_x, :min_y] = lam * images[i][:, :min_x, :min_y] + (1 - lam) * shuffle_images[i][:, :min_x, :min_y]

    return mixed_images

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=2, length=100):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img, target):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img, target


class RandomErasing(object):
    '''
    https://github.com/zhunzhong07/CamStyle/blob/master/reid/utils/data/transforms.py
    '''
    def __init__(self, EPSILON=0.5, mean=[0.485, 0.456, 0.406]):
        self.EPSILON = EPSILON
        self.mean = mean

    def __call__(self, img, target):
        if random.uniform(0, 1) > self.EPSILON:
            return img, target

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(0.02, 0.2) * area
            aspect_ratio = random.uniform(0.3, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]

                return img, target

        return img, target

class LGT(object):

    def __init__(self, probability=0.2, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, image, target):
        # print(image.shape)  #(3,600,800)   C,H,W
        grayscale_transform = transforms.Grayscale(num_output_channels=1)
        grayscale_image_tensor = grayscale_transform(image) # Convert from here to the corresponding grayscale image  (1,600,800)
        image_gray = torch.cat((grayscale_image_tensor, grayscale_image_tensor, grayscale_image_tensor), 0)
        bbox = target["boxes"]

        if random.uniform(0, 1) >= self.probability:
            return image, target

        for attempt in range(50):
            bbox_xmin = bbox[:, 0]
            bbox_ymin = bbox[:, 1]
            bbox_width = bbox[:, 2]
            bbox_heigth = bbox[:, 3]
            for i in range(bbox.size(0)):
                bbox_area = bbox_width[i] * bbox_heigth[i]
                target_area = random.uniform(self.sl, self.sh) * bbox_area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < bbox_width[i] and h < bbox_heigth[i]:
                    x1 = random.randint(bbox_xmin[i], bbox_xmin[i] + bbox_width[i] - w)
                    y1 = random.randint(bbox_ymin[i], bbox_ymin[i] + bbox_heigth[i] - h)

                    image[0, y1:y1 + h, x1:x1 + w] = image_gray[0, y1:y1 + h, x1:x1 + w]
                    image[1, y1:y1 + h, x1:x1 + w] = image_gray[1, y1:y1 + h, x1:x1 + w]
                    image[2, y1:y1 + h, x1:x1 + w] = image_gray[2, y1:y1 + h, x1:x1 + w]
            return image, target

        return image, target


class ToTensor:
    def __call__(self, image, target):
        # convert [0, 255] to [0, 1]
        image = F.to_tensor(image)
        return image, target


def build_transforms(cfg, is_train):
    transforms = []
    transforms.append(ToTensor())
    if is_train:
        transforms.append(RandomHorizontalFlip())
        transforms.append(LGT())
        
        if cfg.INPUT.IMAGE_CUTOUT:
            transforms.append(Cutout())
        if cfg.INPUT.IMAGE_ERASE:
            transforms.append(RandomErasing())

    return Compose(transforms)
