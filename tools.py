import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import os.path as osp


def plot_single(im_path, pred_path=None, pred_idx=0, gt_path=None, gt_idx=0, force_square=False):
    """
    coords: [x1, y1, x2, y2, x3, y3, x4, y4]
    """

    im = Image.open(im_path)
    plt.imshow(im)

    if pred_path is not None:
        with open(pred_path, 'r') as f:
            l = f.read().split("\n")
            pred_bb = [float(x) for x in l[pred_idx].split(',')]
        
        if (force_square):
            X, Y = pred_bb[::2], pred_bb[1::2]
            pred_bb = [min(X), min(Y), max(X),  max(Y)]
            plt.plot([pred_bb[0], pred_bb[2], pred_bb[2], pred_bb[0], pred_bb[0], ],
                    [pred_bb[1], pred_bb[1], pred_bb[3], pred_bb[3], pred_bb[1], ], 'r-', lw=2)
        elif (len(pred_bb) == 8):
            plt.plot(pred_bb[::2] + [pred_bb[0]], pred_bb[1::2] + [pred_bb[1]], 'r-', lw=2)
        else:
            plt.plot([pred_bb[0], pred_bb[2], pred_bb[2], pred_bb[0], pred_bb[0], ],
                    [pred_bb[1], pred_bb[1], pred_bb[3], pred_bb[3], pred_bb[1], ], 'r-', lw=2)

    if gt_path is not None:
        with open(gt_path, 'r') as f:
            l = f.read().split("\n")
            gt_bb = [float(x) for x in l[gt_idx].split(',')]
            
        if (force_square):
            X, Y = gt_bb[::2], gt_bb[1::2]
            gt_bb = [min(X), min(Y), max(X),  max(Y)]
            plt.plot([gt_bb[0], gt_bb[2], gt_bb[2], gt_bb[0], gt_bb[0], ],
                    [gt_bb[1], gt_bb[1], gt_bb[3], gt_bb[3], gt_bb[1], ], 'b-', lw=2)
        elif (len(gt_bb) == 8):
            plt.plot(gt_bb[::2] + [gt_bb[0]], gt_bb[1::2] + [gt_bb[1]], 'b-', lw=2)
        else:
            plt.plot([gt_bb[0], gt_bb[2], gt_bb[2], gt_bb[0], gt_bb[0], ],
                    [gt_bb[1], gt_bb[1], gt_bb[3], gt_bb[3], gt_bb[1], ], 'b-', lw=2)

    return plt


def save_plot_single(im_path, name="plot.jpg", pred_path=None, pred_idx=0, gt_path=None, gt_idx=0, force_square=False):
    plt = plot_single(im_path, pred_path, pred_idx, gt_path, gt_idx, force_square)
    plt.savefig(name)
    plt.show()
    return plt


if __name__ == "__main__":
    save_plot_single("/home/zabulskyy/Datasets/vot2016/bag/00000001.jpg",
                     pred_path="/home/zabulskyy/Projects/CTU-Research/results/yolo_tracker/bag.txt",
                     gt_path="/home/zabulskyy/Datasets/vot2016/bag/groundtruth.txt",
                     force_square=True)
