import torch
import numpy as np

def predict_transform(prediction, inp_dim, anchors, num_classes, cuda):
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

    if cuda:
        x_y_offset = x_y_offset.cuda()

    prediction[:, :, :2] += x_y_offset

    anchors = [(anchor[0] / stride, anchor[1] / stride) for anchor in anchors]
    anchors = torch.FloatTensor(anchors)
    if cuda:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors

    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    prediction[:,:,5:] = torch.sigmoid(prediction[:,:,5:])

    prediction[:,:,:4] *= stride

    return prediction

