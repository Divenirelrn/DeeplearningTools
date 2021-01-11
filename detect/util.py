import torch
import numpy as np
import argparse
import cv2


def predict_transform(prediction, inp_dim, anchors, num_classes):
    #[1, 255, 13, 13]
    batch_size = prediction.size(0)
    grid_size = prediction.size(2)
    stride = inp_dim // grid_size
    num_attr = num_classes + 5
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, num_anchors * num_attr, grid_size * grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, num_attr)

    grid = np.arange(grid_size)
    x_offset, y_offset = np.meshgrid(grid, grid)
    x_offset, y_offset = torch.FloatTensor(x_offset).view(-1, 1), torch.FloatTensor(y_offset).view(-1, 1)
    x_y_offset = torch.cat((x_offset, y_offset), dim=1).repeat(1, num_anchors).view(-1, 2)

    # if cuda:
    #     x_y_offset = x_y_offset.cuda()

    prediction[:, :, :2] = torch.sigmoid(prediction[:, :, :2])
    prediction[:, :, :2] += x_y_offset

    anchors = [(anchor[0] / stride, anchor[1] / stride) for anchor in anchors]
    anchors = torch.FloatTensor(anchors)
    # if cuda:
    #     anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors

    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    prediction[:,:,5:] = torch.sigmoid(prediction[:,:,5:])

    prediction[:,:,:4] *= stride

    return prediction


def compute_ious(box1, box2):
    #[1,7], [6,7]
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_area = torch.clamp((inter_x2 - inter_x1), min=0) * torch.clamp((inter_y2 - inter_y1), min=0)
    
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    ious = inter_area / (box1_area + box2_area - inter_area)

    return ious


def write_results(prediction, conf, num_classes, nms_conf):
    #[1, 10647, 85]
    conf_mask = (prediction[:, :, 4] > conf).unsqueeze(2)
    prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    write = 0
    for ind in range(batch_size):
        prediction_ind = prediction[ind]
        #[10647, 85]
        nonzero_ind = torch.nonzero(prediction_ind[:, 4]).squeeze()
        prediction_ind = prediction_ind[nonzero_ind]
        #[15, 85]
        max_value, max_index = torch.max(prediction_ind[:, 5:], dim=1)
        # print(max_value) #[15]
        # print(max_index) #[15]
        max_value, max_index = max_value.unsqueeze(1), max_index.unsqueeze(1)

        prediction_ind = torch.cat((prediction_ind[:, :5], max_value, max_index), dim=1)
        #[15, 7]
        all_cls = torch.unique(prediction_ind[:, -1])
        for cls in all_cls:
            cls_mask = (prediction_ind[:, -1] == cls).unsqueeze(1)
            prediction_cls = prediction_ind * cls_mask
            nonzero_ind = torch.nonzero(prediction_cls[:, 4]).squeeze()
            prediction_cls = prediction_cls[nonzero_ind].view(-1, 7)
            #[7, 7]
            _, sorted_ind = torch.sort(prediction_cls[:, 4], descending=True)
            prediction_cls = prediction_cls[sorted_ind]

            all_bbox = prediction_cls.size(0)
            for i in range(all_bbox):
                try:
                    ious = compute_ious(prediction_cls[i].unsqueeze(0), prediction_cls[i + 1:])
                except ValueError:
                    break
                except IndexError:
                    break

                iou_mask = (ious < nms_conf).unsqueeze(1)
                prediction_cls[i + 1:] *= iou_mask
                nonzero_ind = torch.nonzero(prediction_cls[:, 4]).squeeze()
                prediction_cls = prediction_cls[nonzero_ind].view(-1, 7)

            #[1, 7]
            batch_ind = prediction_cls.new(prediction_cls.size(0), 1).fill_(ind)
            prediction_cls = torch.cat((batch_ind, prediction_cls), dim=1)

            if not write:
                output = prediction_cls
                write = 1
            else:
                output = torch.cat((output, prediction_cls), dim=0)

    try:
        return output
    except Exception as e:
        return 1


def args_parse():
    parser = argparse.ArgumentParser(description="YOLOV3 args")

    parser.add_argument("--image", default="./images", help="Path of image/directory")
    parser.add_argument('--save_path', help='save path', default='./results')
    parser.add_argument('--cfg', help='cfg_path', default='./yolov3.cfg')
    parser.add_argument('--weights', help='weights path', default='./yolov3.weights')
    parser.add_argument('--input_size', help='input size', default=416)
    parser.add_argument('--batch_size', default=2)
    parser.add_argument('--confidence', help='confidence to filter detections', default=0.5)
    parser.add_argument('--nms_conf', help='threshhold fpr nms', default=0.4)

    return parser.parse_args()


def load_classes(cls_file):
    with open(cls_file, "r") as fp:
        classes = fp.read()

    classes = classes.splitlines()
    classes = [cls.rstrip("\n") for cls in classes]

    return classes


def letter_box(img, inp_size):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_size, inp_size

    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    mask = np.full((h, w, 3), 128)
    mask[(h-new_h) // 2 : ((h-new_h) // 2 + new_h), (w-new_w) // 2 : ((w-new_w) // 2 + new_w), :] = img

    return mask


def prep_img(img, inp_size):
    img = letter_box(img, inp_size)
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = img[np.newaxis, ...] / 255.0
    img = torch.from_numpy(img).float()

    return img