import cv2
import torch
import numpy as np


def get_test_input(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (416, 416))
    # img = img.reshape((416, 416))
    img = img[:,:,::-1].transpose((2,0,1))
    img = img[np.newaxis, ...] / 255.0
    img = torch.from_numpy(img).float()

    return img