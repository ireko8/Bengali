from pathlib import Path
from multiprocessing import Pool
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import conf

HEIGHT = conf.image_size[0]
WIDTH = conf.image_size[1]
width, height = conf.image_size
train_images = pd.read_parquet("input/train.parquet")
test_images = pd.read_parquet("input/test.parquet")


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=width, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))


def resize(images, index):
    print(index)
    row_image = images.iloc[index]
    image = 255 - row_image.drop('image_id').values.astype(np.uint8).reshape(137, 236)
    # image = cv2.resize(image, conf.image_size)
    image = (image*(255.0/image.max())).astype(np.uint8)
    # image = crop_resize(image)
    return image


def train_resize(index):
    return resize(train_images, index)


def test_resize(index):
    return resize(test_images, index)


def dump_resized():

    with Pool(8) as p:
        args = list(range(len(train_images)))
        train_resized = p.map(train_resize, args)        

    mean = 0
    std = 0
    for img in train_resized:
        mean += (img / 255).mean()
        std += ((img / 255) ** 2).mean()

    n = len(train_resized)
    mean /= n
    std = np.sqrt(std / n - mean**2)
    print(mean, std)
    train_resized = np.stack(train_resized)
    np.save(f"input/train_{width}x{height}.npy", train_resized)

    with Pool(8) as p:
        args = list(range(len(test_images)))
        test_resized = p.map(test_resize, args)        

    np.save(f"input/test_{width}x{height}.npy", np.stack(test_resized))


dump_resized()
