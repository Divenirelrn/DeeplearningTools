import os, shutil
import glob
import json
from matplotlib import pyplot as plt


def compute_iou(bbox, bbox_gt):
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox_gt_x1, bbox_gt_y1, bbox_gt_x2, bbox_gt_y2 = bbox_gt[0], bbox_gt[1], bbox_gt[2], bbox_gt[3]

    inter_x1 = max(bbox_x1, bbox_gt_x1)
    inter_y1 = max(bbox_y1, bbox_gt_y1)
    inter_x2 = min(bbox_x2, bbox_gt_x2)
    inter_y2 = min(bbox_y2, bbox_gt_y2)

    inter_area = max(inter_x2 - inter_x1 + 1, 0) * max(inter_y2 - inter_y1 + 1, 0)

    bbox_area = (bbox_y2 - bbox_y1 + 1) * (bbox_x2 - bbox_x1 + 1)
    bbox_gt_area = (bbox_gt_y2 - bbox_gt_y1 + 1) * (bbox_gt_x2 - bbox_gt_x1 + 1)

    return inter_area / (bbox_area + bbox_gt_area - inter_area)


def plot_pr_curve(prec, rec):
    plt.plot(rec, prec)
    plt.show()


def compute_ap(prec, rec):
    prec.insert(0, 0.0)
    prec.append(0.0)
    mprec = prec[:]

    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])

    i_list = list()
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i-1]) * mprec[i]

    return ap


MIN_OVERLAP = 0.5

root_dir = os.path.join(os.getcwd(), "input")
gt_path = os.path.join(root_dir, "ground-truth")
dt_path = os.path.join(root_dir, "detection-results")
image_path = os.path.join(root_dir, "images-optional")

temp_dir = ".temp"
output_dir = "./output"

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

gt_file_list = glob.glob(gt_path + "/*.txt")
classes = list()
classes_num = dict()
for gt_file in gt_file_list:
    objs = list()
    with open(gt_file, "r") as fp:
        lines = fp.readlines()
    lines = [line.rstrip() for line in lines]

    for line in lines:
        temp = dict()
        temp["class_name"] = line.split(" ")[0]
        temp["bbox"] = line.split(" ")[1:]
        temp["bbox"] = [int(i) for i in temp["bbox"]]
        temp["used"] = False
        objs.append(temp)
        if temp["class_name"] not in classes:
            classes.append(temp["class_name"])

        if temp["class_name"] not in classes_num.keys():
            classes_num[temp["class_name"]] = 1
        else:
            classes_num[temp["class_name"]] += 1

    with open(os.path.join(temp_dir, "gt_" + os.path.basename(gt_file).split(".", 1)[0] + ".json"), "w") as fo:
        json.dump(objs, fo)

cls_dict = dict()
for cls in classes:
    cls_dict[cls] = list()

dt_file_list = glob.glob(dt_path + "/*.txt")
for dt_file in dt_file_list:
    with open(dt_file, "r") as fp:
        lines = fp.readlines()
    lines = [line.rstrip() for line in lines]

    for line in lines:
        temp = dict()
        name = line.split(" ")[0]
        score = float(line.split(" ")[1])
        bbox = line.split(" ")[2:]
        bbox = [int(i) for i in bbox]

        temp["score"] = score
        temp["bbox"] = bbox
        temp["image_id"] = os.path.basename(dt_file).split(".", 1)[0]

        if name in cls_dict.keys():
            cls_dict[name].append(temp)

for cls in cls_dict.keys():
    cls_dict[cls].sort(key= lambda k: k["score"], reverse=True)

    with open(os.path.join(temp_dir, "dt_" + cls + ".json"), "w") as fo:
        json.dump(cls_dict[cls], fo)

sum_ap = 0.0
for cls in classes:
    with open(os.path.join(temp_dir, "dt_" + cls + ".json"), "r") as fo:
        objs = json.load(fo)

    tp = [0] * len(objs)
    fp = [0] * len(objs)

    for i, obj in enumerate(objs):
        ovmax = 0.0
        ground_truth = None

        bbox = obj["bbox"]
        image_id = obj["image_id"]

        gt_file = os.path.join(temp_dir, "gt_" + image_id + ".json")
        with open(gt_file, "r") as fo:
            objs_gt = json.load(fo)

        for obj_gt in objs_gt:
            class_name = obj_gt["class_name"]

            if class_name == cls:
                bbox_gt = obj_gt["bbox"]

                iou = compute_iou(bbox, bbox_gt)
                if iou > ovmax:
                    ovmax = iou
                    ground_truth = obj_gt

        if ovmax > MIN_OVERLAP:
            if not bool(ground_truth["used"]):
                tp[i] = 1
                ground_truth["used"] = True
                obj_gt = ground_truth
            else:
                fp[i] = 1
        else:
            fp[i] = 1

        with open(gt_file, "w") as fo:
            json.dump(objs_gt, fo)

    for i in range(1, len(tp)):
        tp[i] = tp[i] + tp[i - 1]

    for i in range(1, len(fp)):
        fp[i] = fp[i] + fp[i - 1]

    gt_num = classes_num[cls]

    pre = [0] * len(objs)
    rec= [0] * len(objs)
    for i in range(len(tp)):
        pre[i] = tp[i] / (tp[i] + fp[i])
        rec[i] = tp[i] / gt_num

    ap = compute_ap(pre, rec)
    sum_ap += ap
    print(cls, ap)

map = sum_ap / len(classes)
print(map)
















