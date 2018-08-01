import os
import os.path as osp
from PIL import Image
import numpy as np
import torch

def run(vot_path, percentile=3):
    percentile = 1 / percentile
    res = dict()
    folders = sorted(os.listdir(vot_path))
    for folder in folders:
        if (folder.endswith(".txt")):
            continue
        li = sorted(os.listdir(osp.join(vot_path, folder)))
        for n, image in enumerate(li):
            if not image.endswith(".jpg"):
                continue
            im = Image.open(osp.join(vot_path, folder, image))
            width, height = im.size
            bbox = [percentile * width, percentile * height,
                    (1-percentile)*width,  (1-percentile)*height]
            try:
                res[folder].append(bbox)
            except:
                res[folder] = [bbox]
    return res

def file2tensor(file):
    with open(file, "r") as f:
        l = f.read().split("\n")[:-1]
        return torch.tensor([[float(y) for y in x.split(',')] for x in l])


def run_theoretical(vot_path):
    vot_classes = os.listdir(vot_path)
    res = dict()
    for vot_class in vot_classes:
        if vot_class.endswith(".txt"):
            continue
        file = osp.join(vot_path, vot_class, "groundtruth.txt")
        tensor = file2tensor(file)
        frames = [x for x in os.listdir(osp.join(vot_path, vot_class)) if x.endswith(".jpg")]
        res[vot_class] = [torch.mean(tensor.float(), dim=0).tolist() for _ in range(len(frames))]
    return res

def save(res, folder):
    os.makedirs(folder)
    for name in res:
        content = "\n".join([",".join([str(a) for a in x]) for x in res[name]])
        f = open(osp.join(folder, name + ".csv"), 'w+')
        print(content, file=f)


def save_all(dic, folder):
    os.makedirs(folder)
    for key in dic:
        res = dic[key]
        os.makedirs(osp.join(folder, "middle_box_{}".format(str(key))))
        for name in res:
            content = "\n".join([",".join([str(a) for a in x]) for x in res[name]])
            f = open(osp.join(folder, "middle_box_{}".format(str(key)), name + ".csv"), 'w+')
            print(content, file=f)


def run_fixed_boxes():
    a, b, step = 2., 10., .1
    res = dict()
    for i in np.arange(a, b, step):
        res[i] = run("../../../../vot2016/", i)
    save_all(res, "../../results/middle_box")

if __name__ == "__main__":
    res = run_theoretical("../../../../vot2016/")
    save(res, "../../results/theor_middle_box")
