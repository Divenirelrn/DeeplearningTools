#-*- coding: utf-8 -*-
#Author: zhaolu
#Date: 2021/01/10

from util import *
from darknet import Darknet

import torch
import os
import cv2
import random
import pickle as pkl
import pandas as pd


# def plot_rectangle(out, load_imgs):
#     c1 = tuple(out[1:3].int())
#     c2 = tuple(out[3:5].int())
#     color = random.choice(colors)
#     img = load_imgs[int(out[0])]
#
#     cv2.rectangle(img, c1, c2, color, 1)
#     label = classes[int(out[-1])]
#     t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
#
#     c2 = c1[0] + t_size[0] +3, c1[1] + t_size[1] + 4
#     cv2.rectangle(img, c1, c2, color, -1)
#     cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
#
#     cv2.imshow("img", img)
#     cv2.waitKey(1000)
#     return img


def plot_rectangle(out, img_loaded):
    c1 = tuple(out[1:3].int())
    c2 = tuple(out[3:5].int())

    img = img_loaded[int(out[0])]
    label = classes[int(out[-1])]
    color = random.choice(colors)

    cv2.rectangle(img, c1, c2, color, 1)

    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + label_size[0] + 3, c1[1] + label_size[1] + 3
    cv2.rectangle(img, c1, c2, color, -1)

    c1 = c1[0], c1[1] + label_size[1] + 3
    cv2.putText(img, label, c1, cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    return img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#获得参数
args = args_parse()

#导入类别
class_num = 80
classes = load_classes("./data/coco.names")

#导入模型
net = Darknet("./yolov3.cfg")
net.load_weights("./yolov3.weights")
net = net.to(device)
net.eval()

#save_path
save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

#input size
inp_size = int(args.input_size)
assert inp_size > 32
assert inp_size % 32 == 0

#images
image = args.image
try:
    img_list = [os.path.join(os.path.realpath("."), image, img)  for img in os.listdir(image)]
except NotADirectoryError:
    img_list = [os.path.join(os.path.realpath("."), image)]
except FileNotFoundError:
    print("No path or directory named %s" % image)

img_loaded = [cv2.imread(img) for img in img_list]

img_prep = list(map(prep_img, img_loaded, [inp_size for i in range(len(img_loaded))]))

# print(img_prep[0])

#w, h
inp_dim_list = [(img.shape[1], img.shape[0]) for img in img_loaded]
inp_dim_list = torch.FloatTensor(inp_dim_list)

#batch
batch_size = int(args.batch_size)

if len(img_list) % batch_size == 0:
    num_batchs = int(len(img_list) // batch_size)
else:
    num_batchs = int(len(img_list) // batch_size) + 1

if len(img_list) > 1:
    img_batchs = [torch.cat(img_prep[i * batch_size : min((i + 1) * batch_size, len(img_list))], dim=0) for i in range(num_batchs)]
else:
    img_batchs = img_prep

write = 0
for i, img_batch in enumerate(img_batchs):
    img_batch = img_batch.to(device)

    prediction = net(img_batch)
    prediction = write_results(prediction, 0.5, 80, 0.4)
    print("write_results:", prediction)

    if type(prediction) == int:
        print("Nothing detected")
        continue

    prediction[:, 0] += i * batch_size

    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction), dim=0)

    for ind in range(len(img_list[i * batch_size : min((i + 1) * batch_size, len(img_list))])):
        img_ind = i * batch_size + ind
        objs = [classes[int(out[-1])] for out in output if int(out[0]) == img_ind]

        print("objection: {}".format(",".join(objs)))

try:
    output
except Exception as e:
    print("No detections")
    exit()

inp_dim_list = torch.index_select(inp_dim_list, 0, output[:, 0].long())
scale_factor = torch.min(inp_size / inp_dim_list, dim=1)[0].view(-1, 1)

output[:, [1,3]] = output[:, [1,3]] - (inp_size - scale_factor * inp_dim_list[:, 0].view(-1, 1)) / 2
output[:, [2,4]] = output[:, [2,4]] - (inp_size - scale_factor * inp_dim_list[:, 1].view(-1, 1)) / 2

for i in range(output.size(0)):
    output[:, [1,3]] = torch.clamp(output[:, [1,3]], min=0.0, max=inp_dim_list[i, 0])
    output[:, [2,4]] = torch.clamp(output[:, [2,4]], min=0.0, max=inp_dim_list[i, 1])

output[:, 1:5] /= scale_factor

colors = pkl.load(open("pallete", "rb"))
list(map(lambda x: plot_rectangle(x, img_loaded), output))

save_names = pd.Series(img_list).apply(lambda x: "{}/detect_{}".format(args.save_path, x.split("\\")[-1]))
list(map(cv2.imwrite, save_names, img_loaded))



