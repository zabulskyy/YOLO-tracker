import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import os.path as osp


def plot_single(im_path, coords_path, idx=0, force_square=False):
    """
    coords: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    with open(coords_path, 'r') as f:
        l = f.read().split("\n")
        pix = [float(x) for x in l[idx].split(',')]
    im = Image.open(im_path)
    plt.subplots(1)
    plt.imshow(im)
    if (force_square):
        X, Y = pix[::2], pix[1::2]
        pix = [min(X), min(Y), max(X),  max(Y)]
        plt.plot([pix[0], pix[2], pix[2], pix[0], pix[0], ],
                 [pix[1], pix[1], pix[3], pix[3], pix[1], ], 'ro-', lw=2)
    elif (len(pix) == 8):
        plt.plot(pix[::2] + [pix[0]], pix[1::2] + [pix[1]], 'ro-', lw=2)
    else:
        plt.plot([pix[0], pix[2], pix[2], pix[0], pix[0], ],
                 [pix[1], pix[1], pix[3], pix[3], pix[1], ], 'ro-', lw=2)
    plt.show()
    return plt


def iou(cords1, cords2):
    if (len(cords1) == 8):
        X1, Y1 = cords1[::2], cords1[1::2]
        cords1 = [min(X1), min(Y1), max(X1),  max(Y1)]
        X2, Y2 = cords2[::2], cords2[1::2]
        cords2 = [min(X2), min(Y2), max(X2),  max(Y2)]
    xA = max(cords1[0], cords2[0])
    yA = max(cords1[1], cords2[1])
    xB = min(cords1[2], cords2[2])
    yB = min(cords1[3], cords2[3])
    interArea = max(0, xB - xA + 1e-32) * max(0, yB - yA + 1e-32)
    cords1Area = (cords1[2] - cords1[0] + 1e-32) * \
        (cords1[3] - cords1[1] + 1e-32)
    cords2Area = (cords2[2] - cords2[0] + 1e-32) * \
        (cords2[3] - cords2[1] + 1e-32)
    return interArea / float(cords1Area + cords2Area - interArea)


def open_and_cast(file):
    with open(file, 'r') as f:
        pred = [s.split(',') for s in f.read().split("\n")]
        for i in range(len(pred)):
            if pred[i] == [""]:
                pred.pop(i)
                continue
            pred[i] = [float(x) for x in pred[i]]
    return pred


def grade(gt_path, pred_path, method="mean_iou"):
    gt = open_and_cast(gt_path)
    pred = open_and_cast(pred_path)
    grades = [iou(gt[i], pred[i]) for i in range(len(gt))]
    return sum(grades) / len(grades), grades

def grade_vot(vot_folder, res_folder, gt_filename="groundtruth.txt"):
    vot_folders = os.listdir(vot_folder)
    res = dict()
    for i in range(len(vot_folders)):
        if (vot_folders[i].endswith(".txt")):
            continue
        gt_file = osp.join(vot_folder, vot_folders[i], gt_filename)
        pred_file = osp.join(res_folder, vot_folders[i] + ".txt")
        single_grade = grade(gt_file, pred_file)[0]
        res[vot_folders[i]] = single_grade
    mean = 0
    for i in res:
        mean += res[i]
    return mean / len(res.keys()), res

print(grade_vot("/home/zabulskyy/Datasets/vot2016", "results/middle-box"))