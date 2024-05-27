import random
import math
from torchvision.transforms import functional as F
#import cv2
import numpy as np
import torchvision.transforms.transforms as transforms
import torch


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
    
class rgb_to_linear_rgb:  #线性RGB色彩空间中的值是线性的，即它们与场景中实际亮度成比例。线性RGB更适合于图像合成和处理，因为在这些RGB值上的算术运算能够更准确地反映实际的光照和颜色混合效果。
    def __init__(self, probrgb=0.4):
        self.probrgb = probrgb
    def __call__(self, image, target):
        if random.random() < self.probrgb:
            # if len(image.shape) < 3 or image.shape[-3] != 3:
            #     raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")
            image = torch.where(image > 0.04045, torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92)
        return image, target



class rgb_to_hsvv:   ###这个是yolov5中使用的
    def __init__(self, probhls=0.5, hgain=0.5, sgain=0.5, vgain=0.5):
        self.probhls = probhls
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
    def __call__(self, image, target):
        if random.random() < self.probhls:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains

            hue, sat, val = rgb_to_hsv(image)
            x = np.arange(0, 256, dtype = r.dtype)
            lut_hue = ((x * r[0]) % 180).astype('uint8')
            lut_sat = np.clip(x * r[1], 0, 255).astype('uint8')
            lut_val = np.clip(x * r[2], 0, 255).astype('uint8')
            im_hsv = cv2.merge((cv2.LUT(hue.numpy(), lut_hue), cv2.LUT(sat.numpy(), lut_sat), cv2.LUT(val.numpy(), lut_val)))
            im_hsv = torch.from_numpy(im_hsv).permute(2, 0, 1)

            #image = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)  # no return needed
            #将HSV转化为RGB
            image = hsv_to_rgb(im_hsv)

        return image, target



def rgb_to_hsv(image):   #且使得hue在0-179，其他在0-255
    max_val, _ = torch.max(image, dim = 0)
    min_val, _ = torch.min(image, dim = 0)
    delta = max_val - min_val
    delta = torch.where(delta == 0, torch.ones_like(delta), delta)
    #计算hue
    hue = torch.zeros_like(max_val)
    hue[max_val == image[0]] = ((image[1][max_val == image[0]]- image[2][max_val == image[0]]) / delta[max_val == image[0]]) % 6
    hue[max_val == image[1]] = (image[2][max_val == image[1]] - image[0][max_val == image[1]]) / delta[
        max_val == image[1]] + 2
    hue[max_val == image[2]] = (image[0][max_val == image[2]] - image[1][max_val == image[2]]) / delta[
        max_val == image[2]] + 4

    hue[hue != 0] /= 6

    #计算sat
    sat = torch.where(max_val==0, torch.zeros_like(delta), delta / max_val)

    #hue:0-179, S:0-255, V:0-255
    hue = (hue * 179).to(torch.uint8)
    sat = (sat * 255).to(torch.uint8)
    value = (max_val * 255).to(torch.uint8)

    return hue, sat, value

def hsv_to_rgb(image):

    h, s, v = image.unbind(0)   #拆分HSV通道
    #h从0-179映射到0-1
    h = h.float() / 179.0
    #s和v从0-255映射到0-1
    s = s.float() / 255.0
    v = v.float() / 255.0
    #计算RGB转换中使用的中间值
    hi = torch.floor(h * 6)
    f = (h * 6) - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    #计算6个不同的区间，以确定RGB值
    hi = hi.long() % 6
    #构建RGB通道
    r = torch.where(hi == 0, v, torch.where(hi == 1, q, torch.where(hi == 2, p, torch.where(hi == 3, p, torch.where(hi == 4, t, q)))))
    g = torch.where(hi == 1, v, torch.where(hi == 2, q, torch.where(hi == 3, p, torch.where(hi == 4, p, torch.where(hi == 5, t, q)))))
    b = torch.where(hi == 2, v, torch.where(hi == 3, q, torch.where(hi == 4, p, torch.where(hi == 5, p, torch.where(hi == 0, t, q)))))
    #栈叠RGB通道
    image = torch.stack([r, g, b], dim = 0).clamp(0,1)
    return image



#yolov5中用的直方图均衡

class hist_equalize:   ###这个是yolov5中使用的
    def __init__(self, probhist=0.6, clahe=True):
        self.probhist = probhist
        self.clahe = clahe
    def __call__(self, image, target):
        if random.random() < self.probhist:
            yuv = rgb_to_yuv(image)
            if self.clahe:
                #CLAHE（对比度受限自适应直方图均衡化）算法对YUV图像的亮度通道进行增强处理。创建一个CLAHE对象，
                yuv = yuv.permute(1,2,0).numpy()
                c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   #opencv处理的图形形状是（H，W，C）
                y_channel = (yuv[:,:,0] * 255).astype('uint8')
                y_channel_clahe = c.apply(y_channel)
                yuv[:,:,0] = y_channel_clahe

                # yuv[:, :, 0] = c.apply(yuv[:, :, 0])
            else:
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
            image = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            image = torch.from_numpy(image).permute(2,0,1).float()/255.0
        return image, target



def rgb_to_yuv(image):
    r_channel = image[0]
    g_channel = image[1]
    b_channel = image[2]

    y_channel = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel  #0-1
    u_channel = -0.147 * r_channel - 0.289 * g_channel + 0.436 * b_channel  #-0.5 ~ 0.5
    v_channel = 0.615 * r_channel - 0.515 * g_channel - 0.100 * b_channel   #-0.5 ~ 0.5

    yuv_image = torch.stack([y_channel, u_channel, v_channel], dim=0)

    return yuv_image



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
        # transforms.append(rgb_to_linear_rgb())
        # transforms.append(rgb_to_hsvv())
        # transforms.append(hist_equalize())
        if cfg.INPUT.IMAGE_CUTOUT:    
            transforms.append(Cutout())
        if cfg.INPUT.IMAGE_ERASE:
            transforms.append(RandomErasing())
        
    return Compose(transforms)
