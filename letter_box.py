import cv2
import numpy as np


def letter_box(img, inp_size):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_size, inp_size

    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    mask = np.full((h, w, 3), 128)
    mask[(h-new_h) // 2 : ((h-new_h) // 2 + new_h), (w-new_w) // 2 : ((w-new_w) // 2 + new_w), :] = img

    return mask