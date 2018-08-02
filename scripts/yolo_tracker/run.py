import torch
import numpy as np
import os
import os.path as osp
from postprocess import do_full_postprop
from tools import arg_parse
from tools import get_prj_path


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
        self.saveto = "yolo_god"
        self.fs = True



def read(folder=osp.join(get_prj_path(), "yolo_vot"), force_square=False):
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
    force_square = args.fs

    predictions = read(force_square=force_square)
    predictions["CUDA"] = torch.cuda.is_available()
    pp_predictions, nm_fr = do_full_postprop(predictions, ("god", "first", "mfc"), vot_path, force_square)
    save(pp_predictions, saveto)
