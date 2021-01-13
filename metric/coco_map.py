import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


parent = os.path.dirname(__file__)
gt_file = os.path.join(parent, "label", "instances_val2017.json")

# with open(gt_file, "r") as fp:
#     data = json.load(fp)
#
# annotations = data["annotations"]
# anno_list = []
# for anno in annotations:
#     temp = dict()
#     temp["score"] = 1.0
#     temp["category_id"] = anno["category_id"]
#     temp["image_id"] = anno["image_id"]
#     temp["bbox"] = anno["bbox"]
#     anno_list.append(temp)
#
# dt_file = os.path.join(parent, "label", "dt.json")
# with open(dt_file, "w") as fo:
#     json.dump(anno_list, fo)

dt_file = os.path.join(parent, "label", "dt.json")
coco = COCO(gt_file)
coco_gt = coco.loadRes(dt_file)
cocoEval = COCOeval(coco, coco_gt, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
