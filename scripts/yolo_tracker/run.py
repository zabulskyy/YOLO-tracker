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
        self.silent = None
        self.cuda = "3"
        self.det = "det"
        self.vot = "../../../vot2016/"
        self.pp = "first_and_mfc_smart"
        self.saveto = "lol"


def fill_zeros(folder):
    return [[0, 0, 0, 0] for i in os.listdir(folder) if i.endswith(".jpg")]


def do_full_postprop(predictions, postprocessor, vot_path):
    RESULTS = predictions["results"]
    NUM_FRAMES = predictions["num_frames"]
    CUDA = predictions["CUDA"]

    res = dict()
    nm_fr = dict()

    for im_class in RESULTS:
        pp_result = postprocess((RESULTS[im_class], NUM_FRAMES[im_class], CUDA), osp.join(vot_path, im_class),
                                postprocessor)
        if len(pp_result) != 0:
            res[im_class] = pp_result.tolist()
            nm_fr[im_class] = NUM_FRAMES[im_class]
        else:
            res[im_class] = fill_zeros(osp.join(vot_path, im_class))
            nm_fr[im_class] = NUM_FRAMES[im_class]

    return res, nm_fr


def read(folder="../../yolo_vot"):
    tensors = dict()
    lengths = dict()
    file_names = os.listdir(folder)
    for file_name in file_names:
        vot_class = file_name.split('.')[0]
        with open(osp.join(folder, file_name), "r") as f:
            l = f.read().split("\n")
            l.pop(-1)

            length = int(l.pop(0))
            lengths[vot_class] = length

            tensor = torch.tensor([[float(y) for y in x.split(',')] for x in l])
            tensors[vot_class] = tensor
    return {"results": tensors, "num_frames": lengths}


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




if __name__ == "__main__":
    args = Args()
    # vot_path = args.vot
    vot_path = "/home/zabulskyy/Datasets/vot2016/"
    saveto = args.saveto
    pp = None if args.pp.lower() == "none" else args.pp

    predictions = read()
    predictions["CUDA"] = torch.cuda.is_available()
    pp_predictions, nm_fr = do_full_postprop(predictions, "mfc", vot_path)
    save(pp_predictions, saveto)
