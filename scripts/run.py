import torch
import torchvision
import numpy as np
import os.path as osp
import os
import argparse

# - given a folder of frames as arg (in a future - videoframe)
# - run YOLO on all frames
# - my super mega algorithm
# - save results, or plot as a gif (in a future - videoframe)


def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO-based tracker')
    parser.add_argument("--dir", dest='dir', help="directory containing images for detection",
                        default="data", type=str)
    parser.add_argument("--saveto", dest='saveto', help="directory to save results",
                        default="results", type=str)

    return parser.parse_args()

if __name__ != "__main__":
    exit(0)

coco_names_file = "data/coco.names"

with open(coco_names_file, "r") as f:
    classes = f.read().split("\n")[:-1]

args = arg_parse()
folder = args.dir
saveto = args.saveto

def read_csv(file):
    with open(file, 'r') as file:
        l = file.read().split("\n")
        l.pop(-1)
        l = [[float(x) for x in y.split(",")[:-1]] for y in l]   
        return torch.tensor(l)

corr = read_csv("data/correlations.csv")
