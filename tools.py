import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import os.path as osp

def plot_single_arr(im_path, pr_arr=None, pr_idx=0, gt_arr=None, gt_idx=0, force_square=False):
    im = Image.open(im_path)
    plt.imshow(im)

    if pr_arr is not None:
        l = pr_arr[pr_idx]
        if l == "":
            return
        pr_bb = [float(x) for x in l.split(',')]

        if force_square:
            X, Y = pr_bb[::2], pr_bb[1::2]
            pr_bb = [min(X), min(Y), max(X),  max(Y)]
            plt.plot([pr_bb[0], pr_bb[2], pr_bb[2], pr_bb[0], pr_bb[0], ],
                     [pr_bb[1], pr_bb[1], pr_bb[3], pr_bb[3], pr_bb[1], ], 'r-', lw=2)
        elif (len(pr_bb) == 8):
            plt.plot(pr_bb[::2] + [pr_bb[0]],
                     pr_bb[1::2] + [pr_bb[1]], 'r-', lw=2)
        else:
            plt.plot([pr_bb[0], pr_bb[2], pr_bb[2], pr_bb[0], pr_bb[0], ],
                     [pr_bb[1], pr_bb[1], pr_bb[3], pr_bb[3], pr_bb[1], ], 'r-', lw=2)

    if gt_arr is not None:
        l = gt_arr[gt_idx]
        if l == "":
            return
        gt_bb = [float(x) for x in l.split(',')]

        if (force_square):
            X, Y = gt_bb[::2], gt_bb[1::2]
            gt_bb = [min(X), min(Y), max(X),  max(Y)]
            plt.plot([gt_bb[0], gt_bb[2], gt_bb[2], gt_bb[0], gt_bb[0], ],
                     [gt_bb[1], gt_bb[1], gt_bb[3], gt_bb[3], gt_bb[1], ], 'b-', lw=2)
        elif (len(gt_bb) == 8):
            plt.plot(gt_bb[::2] + [gt_bb[0]],
                     gt_bb[1::2] + [gt_bb[1]], 'b-', lw=2)
        else:
            plt.plot([gt_bb[0], gt_bb[2], gt_bb[2], gt_bb[0], gt_bb[0], ],
                     [gt_bb[1], gt_bb[1], gt_bb[3], gt_bb[3], gt_bb[1], ], 'b-', lw=2)

def plot_single(im_path, pr_path=None, pr_idx=0, gt_path=None, gt_idx=0, force_square=False):
    """
    coords: [x1, y1, x2, y2, x3, y3, x4, y4]
    """

    im = Image.open(im_path)
    plt.imshow(im)

    if pr_path is not None:
        with open(pr_path, 'r') as f:
            l = f.read().split("\n")[pr_idx]
            if l == "":
                return
            pr_bb = [float(x) for x in l.split(',')]

        if (force_square):
            X, Y = pr_bb[::2], pr_bb[1::2]
            pr_bb = [min(X), min(Y), max(X),  max(Y)]
            plt.plot([pr_bb[0], pr_bb[2], pr_bb[2], pr_bb[0], pr_bb[0], ],
                     [pr_bb[1], pr_bb[1], pr_bb[3], pr_bb[3], pr_bb[1], ], 'r-', lw=2)
        elif (len(pr_bb) == 8):
            plt.plot(pr_bb[::2] + [pr_bb[0]],
                     pr_bb[1::2] + [pr_bb[1]], 'r-', lw=2)
        else:
            plt.plot([pr_bb[0], pr_bb[2], pr_bb[2], pr_bb[0], pr_bb[0], ],
                     [pr_bb[1], pr_bb[1], pr_bb[3], pr_bb[3], pr_bb[1], ], 'r-', lw=2)

    if gt_path is not None:
        with open(gt_path, 'r') as f:
            l = f.read().split("\n")[gt_idx]
            gt_bb = [float(x) for x in l.split(',')]

        if (force_square):
            X, Y = gt_bb[::2], gt_bb[1::2]
            gt_bb = [min(X), min(Y), max(X),  max(Y)]
            plt.plot([gt_bb[0], gt_bb[2], gt_bb[2], gt_bb[0], gt_bb[0], ],
                     [gt_bb[1], gt_bb[1], gt_bb[3], gt_bb[3], gt_bb[1], ], 'b-', lw=2)
        elif (len(gt_bb) == 8):
            plt.plot(gt_bb[::2] + [gt_bb[0]],
                     gt_bb[1::2] + [gt_bb[1]], 'b-', lw=2)
        else:
            plt.plot([gt_bb[0], gt_bb[2], gt_bb[2], gt_bb[0], gt_bb[0], ],
                     [gt_bb[1], gt_bb[1], gt_bb[3], gt_bb[3], gt_bb[1], ], 'b-', lw=2)

def save_plot_single_arr(im_path, name="plot.jpg", pr_arr=None, pr_idx=0, gt_arr=None, gt_idx=0, force_square=False):
    plot_single_arr(im_path, pr_arr, pr_idx, gt_arr, gt_idx, force_square)
    plt.savefig(name)

def save_plot_single(im_path, name="plot.jpg", pr_path=None, pr_idx=0, gt_path=None, gt_idx=0, force_square=False):
    plot_single(im_path, pr_path, pr_idx, gt_path, gt_idx, force_square)
    plt.savefig(name)


def save_plot_folder(dir_path, saveto="results", pr_path=None, gt_path=None, force_square=False):
    if (not os.path.exists(saveto)):
        os.makedirs(saveto)
    files = sorted([x for x in os.listdir(dir_path) if x.endswith(".jpg")])

    if gt_path is not None:
        with open(gt_path, 'r') as f:
            gt_arr = f.read().split("\n")
    

    if pr_path is not None:
        with open(pr_path, 'r') as f:
            pr_arr = f.read().split("\n")

    for n, file in enumerate(files):
        name = osp.join(saveto, file)
        print("saving {}".format(name))
        save_plot_single_arr(osp.join(dir_path, file), name=name, pr_arr=pr_arr,
                         pr_idx=n, gt_arr=gt_arr, gt_idx=n, force_square=force_square)
        plt.close()


if __name__ == "__main__":
    # save_plot_single("/home/zabulskyy/Datasets/vot2016/birds2/00000100.jpg",
    #                  pr_path="/home/zabulskyy/Projects/CTU-Research/results/yolo-blind/birds2.txt",
    #                  gt_path="/home/zabulskyy/Datasets/vot2016/birds2/groundtruth.txt",
    #                  force_square=True, gt_idx=101, pr_idx=101)
    save_plot_folder("/home/zabulskyy/Datasets/vot2016/birds2", saveto="./plots/yolo-first/birds2",
                     pr_path="./results/yolo-first/birds2.txt",
                     gt_path="/home/zabulskyy/Datasets/vot2016/birds2/groundtruth.txt",
                     force_square=True)
    save_plot_folder("/home/zabulskyy/Datasets/vot2016/birds2", saveto="./plots/yolo-first-smart/birds2",
                    pr_path="./results/yolo-first-smart/birds2.txt",
                    gt_path="/home/zabulskyy/Datasets/vot2016/birds2/groundtruth.txt",
                    force_square=True)
    # for plt, _ in arr:
    #     plt.show()
    #     plt.close('all')
