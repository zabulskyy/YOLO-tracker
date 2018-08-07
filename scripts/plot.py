import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path as osp
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from random import randrange
import numpy as np


def arg_parse():
    """
    Parse arguements to the detect module
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--met", dest='method', help="method (yolo-smart, stupid-box, etc)",
                        default="", type=str)
    parser.add_argument("--cls", dest='cls', help="class (ball1, singer3, etc)",
                        default="ball1", type=str)

    return parser.parse_args()


def plot_single_arr(im_path, pr_arr=None, pr_idx=0, gt_arr=None, gt_idx=0, force_square=False):
    im = Image.open(im_path)
    plt.imshow(im)

    if pr_arr is not None:
        l = pr_arr[pr_idx]
        if l == "":
            return
        pr_bb = [float(x) for x in l.split(',')]
        if len(pr_bb) < 4:
            return
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


def csv2tensor(path_to_file, first_len=False):
    print("==========>>>>>>.", path_to_file)
    with open(path_to_file, 'r') as f:
        pr_arr = f.read().split("\n")
        if first_len:
            im_len = int(pr_arr.pop(0))
        else:
            im_len = -1
        pr_arr.pop(-1)
        pr_arr = [[float(y) for y in x.split(",")] for x in pr_arr]
        print(pr_arr[:4])
        return torch.tensor(pr_arr), im_len


def plot_single_yolo(im_path, pr_arr, pr_idx, gt_arr, gt_idx, force_square, gs=False):
    im = np.array(Image.open(im_path), dtype=np.uint8)
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    # Create a Rectangle patch
    for row in pr_arr[pr_arr[:, 0] == pr_idx]:
        color = (randrange(1, 99) / 100, randrange(1, 99) /
                 100, randrange(1, 99) / 100) if gs==False else (.7, .7, .7)
        label = int(row[-1])
        x, y = float(row[1]), float(row[2])
        h, w = float(row[4]) - y, float(row[3]) - x
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', label=label)
        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.annotate(label, (x, y), color=color, weight='bold',
                    fontsize=6, ha='left', va='top')
    # plt.show()


def save_plot_single_yolo(im_path, name="plot.jpg", pr_tensor=None, pr_idx=0, gt_tensor=None, gt_idx=0, force_square=False, gs=False):
    plot_single_yolo(im_path, pr_tensor, pr_idx,
                     gt_tensor, gt_idx, force_square, gs)
    plt.savefig(name)


def save_plot_yolo(class_dir, saveto="results", pr_path=None, gt_path=None, force_square=False):
    if (not os.path.exists(saveto)):
        os.makedirs(saveto)
    images = sorted([x for x in os.listdir(class_dir) if x.endswith(".jpg")])

    if gt_path is not None:
        gt_tensor, gt_len = csv2tensor(gt_path)

    if pr_path is not None:
        pr_tensor, pr_len = csv2tensor(pr_path)

    for n, file in enumerate(images):
        name = osp.join(saveto, file)
        print("saving {}".format(name))
        save_plot_single_yolo(osp.join(class_dir, file), name=name, gt_tensor=gt_tensor,
                              pr_idx=n, pr_tensor=pr_tensor, gt_idx=n, force_square=force_square)
        plt.close()


def save_plot_everything(imgs_folder=None, yolo_pred_file=None, pred_file=None, gt_file=None, saveto="results", force_square=False):
    if (not os.path.exists(saveto)):
        os.makedirs(saveto)
    files = sorted([x for x in os.listdir(imgs_folder) if x.endswith(".jpg")])

    if gt_file is not None:
        gt_tensor, gt_len = csv2tensor(gt_file)

    if pred_file is not None:
        pred_tensor, pred_len = csv2tensor(pred_file)

    if pred_file is not None:
        yolo_pred_tensor, yolo_pred_len = csv2tensor(yolo_pred_file, True)

    for n, file in enumerate(files):
        name = osp.join(saveto, file)
        print("saving {}".format(name))
        impath = osp.join(imgs_folder, file)

        plot_single_yolo(impath, yolo_pred_tensor, n, gt_tensor, n, force_square, gs=True)
        plot_single(impath, pred_file, n, gt_file, n, force_square=True)
        # plt.savefig(name)
        plt.show()
        plt.close()


if __name__ == "__main__":
    # save_plot_single("/home/zabulskyy/Datasets/vot2016/leaves/00000100.jpg",
    #                  pr_path="/home/zabulskyy/Projects/CTU-Research/results/yolo-blind/leaves.txt",
    #                  gt_path="/home/zabulskyy/Datasets/vot2016/leaves/groundtruth.txt",
    #                  force_square=True, gt_idx=101, pr_idx=101)
    args = arg_parse()
    cls = args.cls
    met = args.method

    # save_plot_folder(osp.join("/home/zabulskyy/Datasets/vot2016", cls), saveto=osp.join("plots", met, cls),
    #                  pr_path=osp.join("lololo/" + met, cls) + ".csv",
    #                  gt_path=osp.join(
    #                      "/home/zabulskyy/Datasets/vot2016", cls, "groundtruth.txt"),
    #                  force_square=True)
    save_plot_everything(imgs_folder=osp.join("/home/zabulskyy/Datasets/vot2016", cls), yolo_pred_file=osp.join("yolo_predictions/even_more", cls+".csv"), 
                        pred_file=osp.join("results/" + met, cls) + ".csv", saveto="lololo", 
                        force_square=True, gt_file=osp.join(
                         "/home/zabulskyy/Datasets/vot2016", cls, "groundtruth.txt"))
    # save_plot_folder("/home/zabulskyy/Datasets/vot2016/leaves", saveto="./plots/yolo-first-smart-smart/leaves",
    #                 pr_path="./results/yolo-first-smart-smart/leaves.txt",
    #                 gt_path="/home/zabulskyy/Datasets/vot2016/leaves/groundtruth.txt",
    #                 force_square=True)
    # for plt, _ in arr:
    #     plt.show()
    #     plt.close('all')
