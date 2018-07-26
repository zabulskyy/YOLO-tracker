from detector import predict, arg_parse
import torch
import numpy as np
import os
import os.path as osp
from postprocessing import postprocess


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

def run_vot_full(vot_path, postprocessor=None):
    args = arg_parse()
    res = dict()
    folders = sorted(os.listdir(vot_path))[:]
    for n, folder in enumerate(folders[:]):
        print("data from {}".format(folder))
        if (folder.endswith(".txt")):
            continue
        args.images = osp.join(vot_path, folder)
        bbox = predict(args)
        if postprocessor is not None:
            bbox[0] = postprocess(bbox, osp.join(vot_path, folder), postprocessor)
        if len(bbox[0]) != 0:
            res[folder] = bbox[0][:, 1:5].tolist()
        else:
            res[folder] = fill_zeros(osp.join(vot_path, folder))

    return res


def save(res, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.chdir(folder)
    for name in res:
        print("\n".join([",".join([str(a) for a in x])
                         for x in res[name]]), file=open(name + ".txt", 'w+'))


if __name__ == "__main__":
    res = run_vot_full(arg_parse().vot, "interpolate_with_first_and_tmfc")
    save(res, "../results/yolo-first-smarter")
