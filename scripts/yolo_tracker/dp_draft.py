import torch
import numpy as np
import os.path as osp
import os


def pp_dp(data):
    # do dynamic programming to find out the best chain of boxes
    # possible parameters to choose the next frame:

    # - closest by iou
    # - closest by dist2

    # - position of the box (the most centered is the best one)
    # - position if the box relatively to other boxes

    # - the same class
    # - similar class
    
    # - 

    num_frames = data["num_frames"]
    folder = data["folder"]
    output = data["output"]

