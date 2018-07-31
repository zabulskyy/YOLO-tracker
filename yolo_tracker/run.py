from detector import predict
import torch
import numpy as np
import os
import os.path as osp
from postprocess import postprocess
from args import arg_parse


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
        self.vot = "../../../vot2016/"
        self.pp = "first_and_mfc_smart"
        self.saveto = "lol.txt"


def fill_zeros(folder):
    return [[0, 0, 0, 0] for i in os.listdir(folder) if i.endswith(".jpg")]


def do_full_postprop(predictions, postprocessor, vot_path):

    RESULTS = predictions["results"]
    NUM_FRAMES = predictions["num_frames"]
    CUDA = predictions["CUDA"]

    res = dict()

    for im_class in RESULTS:
        if postprocessor is not None:
            pp_result = postprocess((RESULTS[im_class], NUM_FRAMES[im_class], CUDA), osp.join(vot_path, im_class), postprocessor)
        if len(pp_result) != 0:
            res[im_class] = pp_result[:, 1:5].tolist()            
        else:
            res[im_class] = fill_zeros(osp.join(vot_path, im_class))

    return res


def save(res, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.chdir(folder)
    for name in res:
        print("\n".join([",".join([str(a) for a in x])
                         for x in res[name]]), file=open(name + ".txt", 'w+'))


if __name__ == "__main__":
    args = Args()
    vot_path = args.vot
    saveto = args.saveto
    pp = None if args.pp.lower() == "none" else args.pp

    predictions = predict(args)
    pp_predictions = do_full_postprop(predictions, pp, vot_path)
    save(pp_predictions, saveto)
