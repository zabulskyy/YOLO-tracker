from detector import predict, arg_parse
import torch
import numpy as np
import os
import os.path as osp
from postprocessing import interpolate_blind, interpolate_with_first


class Args:
    def __init__(self):
        self.bs = 1
        self.confidence = .5
        self.cfgfile = "cfg/yolov3.cfg"
        self.nms_thresh = .4
        self.weightsfile = "yolov3.weights"
        self.images = None
        self.reso = "416"
        self.scales = "1,2,3"
        self.saveto = ""
        self.silent = None
        self.cuda = "3"
        self.det = "det"


def fill_zeros(folder):
    return [[0, 0, 0, 0] for i in os.listdir(folder) if i.endswith(".jpg")]


def read_first_gt(folder):
    with open(osp.join(folder, "groundtruth.txt"), 'r') as file:
        l = file.read().split("\n")[0].split(",")
        X, Y = l[::2], l[1::2]
        l = [min(X), min(Y), max(X),  max(Y)]
        l = torch.tensor([float(x) for x in l])
        return l


def run_vot_full(vot_path, postprocessor):
    args = arg_parse()
    res = dict()
    folders = sorted(os.listdir(vot_path))[:]
    for folder in folders[:]:
        print("data from {}".format(folder))
        if (folder.endswith(".txt")):
            continue
        first = read_first_gt(osp.join(vot_path, folder))
        args.images = osp.join(vot_path, folder)
        bbox = predict(args, first, postprocessor)
        if len(bbox) != 0:
            res[folder] = bbox[0][:, 1:5].tolist()
        else:
            res[folder] = fill_zeros(osp.join(vot_path, folder))

    return res


def save(res, folder):
    os.chdir(folder)
    for name in res:
        print("\n".join([",".join([str(a) for a in x])
                         for x in res[name]]), file=open(name + ".txt", 'w+'))


if __name__ == "__main__":
    res = run_vot_full(arg_parse().vot, interpolate_with_first)
    save(res, "../results/yolo-first")
