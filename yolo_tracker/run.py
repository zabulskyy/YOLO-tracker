from detector import predict, arg_parse
import torch
import numpy as np
import os
import os.path as osp

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


def run(vot_path):
    args = arg_parse()
    res = dict()
    folders = sorted(os.listdir(vot_path))[:]
    folders = ["bag", "ball1", "ball2", "birds1",
               "birds2", "bolt1", "basketball"]
    for folder in folders[:]:
        print("data from {}".format(folder))
        if (folder.endswith(".txt")):
            continue
        args.images = osp.join(vot_path, folder)
        bbox = predict(args)
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
    res = run(arg_parse().vot)
    save(res, "../results/yolo-blind")
