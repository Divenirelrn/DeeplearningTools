import torch
import cv2
import argparse
import random
import pickle as pkl

from darknet import Darknet
from util import *


def plot_rectangle(out, img_loaded):
    c1 = tuple(out[1:3].int())
    c2 = tuple(out[3:5].int())

    img = img_loaded
    label = classes[int(out[-1])]
    color = random.choice(colors)

    cv2.rectangle(img, c1, c2, color, 1)

    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + label_size[0] + 3, c1[1] + label_size[1] + 3
    cv2.rectangle(img, c1, c2, color, -1)

    c1 = c1[0], c1[1] + label_size[1] + 3
    cv2.putText(img, label, c1, cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    return img


def args_parse():
    parser = argparse.ArgumentParser(description="YOLOv3 video")

    parser.add_argument("--video", default=0)
    parser.add_argument("--input_size", default=416)
    parser.add_argument("--conf", default=0.5)
    parser.add_argument("--nms_conf", default=0.4)

    return parser.parse_args()


args = args_parse()

model = Darknet("./yolov3.cfg")
model.load_weights("./yolov3.weights")
model.eval()

source = args.video
cap = cv2.VideoCapture(int(source))
assert cap.isOpened(), "cap is not opened"

input_size = int(args.input_size)
conf = float(args.conf)
nms_conf = float(args.nms_conf)
num_classes = 80

classes = load_classes("data/coco.names")

while cap.isOpened():
    rval, frame = cap.read()
    if not rval:
        continue

    size = frame.shape[1], frame.shape[0]
    size = torch.FloatTensor(size) #[2]
    frame2 = prep_img(frame, input_size)

    prediction = model(frame2)
    prediction = write_results(prediction, conf, num_classes, nms_conf) #[1, 8]

    if type(prediction) == int:
        print("No detections")
        continue

    scale_factor = torch.min(input_size / size).unsqueeze(0)
    #[1,2]
    prediction[:, [1,3]] -= (input_size - size[0] * scale_factor.unsqueeze(0)) / 2
    prediction[:, [2,4]] -= (input_size - size[1] * scale_factor.unsqueeze(0)) / 2

    prediction[:, 1:5] /= scale_factor

    prediction[:, [1,3]] = torch.clamp(prediction[:, [1,3]], min=0.0, max=float(size[0]))
    prediction[:, [2,4]] = torch.clamp(prediction[:, [2,4]], min=0.0, max=float(size[1]))

    objs = [classes[int(obj[-1])] for obj in prediction]
    print("objects:", ",".join(objs))

    colors = pkl.load(open("pallete", "rb"))
    list(map(lambda x: plot_rectangle(x, frame), prediction))

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()

