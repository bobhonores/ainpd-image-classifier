import torch

import numpy as np
from PIL import Image


def process_image(image):
    pil_image = Image.open(image)
    pil_image = pil_image.resize(
        calculate_newsize(pil_image.size), Image.ANTIALIAS)
    pil_image = pil_image.crop(calculate_crop(pil_image.size))
    
    np_image = np.array(pil_image)
    np_image = np_image / 255
    means = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / std
    np_image = np_image.transpose((2, 0, 1))

    img = torch.from_numpy(np_image).type(torch.FloatTensor)

    return img.unsqueeze(0)


def calculate_crop(size):
    newsize = 224.0
    width, height = size
    left = (width - newsize) / 2
    upper = (height - newsize) / 2
    right = (width + newsize) / 2
    lower = (height + newsize) / 2

    return left, upper, right, lower


def calculate_newsize(size):
    minvalue = 256.0
    width, height = size
    ratio = width / height

    if ratio < 1:
        width = minvalue
        height = minvalue * 1 / (ratio)
    else:
        height = minvalue
        width = minvalue * ratio

    return int(width), int(height)
