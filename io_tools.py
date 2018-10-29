import torch
import torchvision
import numpy as np
import os.path as osp
import os
import time
import pandas as pd

CUDA = torch.cuda.is_available()

vot_path = "/home/zabulskyy/data/vot2016/"
yolo_pred_path = "yolo_predictions/extended"  # TODO replace with realtime yolo_predictions

# coefficents of the parameters for computing energy
crit_vals = {
    "iou": 1,
    "cc": 1,  # class_correspondence
    "id": 1,  # initial dist
    "vd": 1  # vector dist
}

yolo_classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
           'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
           'teddy bear', 'hair drier', 'toothbrush']

dist = torch.nn.modules.PairwiseDistance()
vot_classes = [x for x in os.listdir(vot_path) if not x.endswith(".txt")]



def read_csv(file):
    with open(file, 'r') as file:
        l = file.read().split("\n")
        l.pop(-1)
        l = [[float(x) for x in y.split(",")[:-1]] for y in l]
        return torch.tensor(l)

cm = read_csv("data/correlations.csv")  # correlation matrix

# cm = read_csv("../data/correlations.csv")  # correlation matrix

def read_gt(vot_class, vot_path=vot_path, force_square=True):
    # read the groundtruth for a precise vot class
    gt_file = osp.join(vot_path, vot_class, "groundtruth.txt")
    with open(gt_file, 'r') as file:
        l = file.read().split("\n")
        l.pop(-1)
        l = [[float(x) for x in y.split(",")] for y in l]
        if force_square:
            l = [[min(m[::2]), min(m[1::2]), max(m[::2]), max(m[1::2])] for m in l]
        return torch.tensor(l)


def read_full_gt(vot_path=vot_path, force_square=True):
    # read the groundtruth for a full vot dataset
    vot_classes = [x for x in os.listdir(vot_path) if not x.endswith(".txt")]
    X = list()
    for vot_class in vot_classes:
        gt_file = osp.join(vot_path, vot_class, "groundtruth.txt")
        with open(gt_file, 'r') as file:
            l = file.read().split("\n")
            l.pop(-1)
            l = [[float(x) for x in y.split(",")] for y in l]
            if force_square:
                l = [[min(m[::2]), min(m[1::2]), max(m[::2]), max(m[1::2])] for m in l]
        X.append(l)
    return torch.tensor(X)


def read_yolo_pred(vot_class, yolo_pred_path="yolo_predictions/extended", pred_format="csv", first_len=True):
    # read offline-made YOLO predictions for a single vot class
    yolo_pred_file = osp.join(yolo_pred_path, vot_class + "." + pred_format)
    with open(yolo_pred_file, 'r') as file:
        l = file.read().split("\n")
        l.pop(-1)
        im_len = -1
        if first_len:
            im_len = float(l.pop(0))
        l = [[float(x) for x in y.split(",")] for y in l]
        return torch.tensor(l), im_len


def read_all_yolo_preds(path="yolo_predictions/extended", pred_format="csv"):
    X = []
    for vot_class in os.listdir(path):
        yolo_pred_file = osp.join(path, vot_class)
        with open(yolo_pred_file, 'r') as file:
            l = file.read().split("\n")
            l.pop(-1)
            l.pop(0)
            im_len = -1
            for i in l:
                X.append([float(x) for x in i.split(",")])
    X = torch.tensor(X)
    return X


def write_csv(matrix, file="correlations.csv"):
    for i in matrix.tolist():
        for j in i:
            print(j, sep=", ", end=", ", file=open(file, "a+"))
        print(file=open(file, "a+"))


def save(res, folder, ext="csv", nm_fr=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for name in res:
        if nm_fr is not None:
            print(nm_fr[name], file=open(osp.join(folder, name + "." + ext), 'w+'))
            print("\n".join([",".join([str(a) for a in x]) for x in res[name]]),
                  file=open(osp.join(folder, name + "." + ext), 'a'))
        else:
            print("\n".join([",".join([str(a) for a in x]) for x in res[name]]),
                  file=open(osp.join(folder, name + "." + ext), 'w+'))
