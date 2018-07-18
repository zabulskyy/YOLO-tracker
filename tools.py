import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import os.path as osp


def plot_single(im_path, coords_path, idx=0, force_square=False):
    """
    coords: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    with open(coords_path, 'r') as f:
        l = f.read().split("\n")
        pix = [float(x) for x in l[idx].split(',')]
    im = Image.open(im_path)
    plt.subplots(1)
    plt.imshow(im)
    if (force_square):
        X, Y = pix[::2], pix[1::2]
        pix = [min(X), min(Y), max(X),  max(Y)]
        plt.plot([pix[0], pix[2], pix[2], pix[0], pix[0], ],
                 [pix[1], pix[1], pix[3], pix[3], pix[1], ], 'ro-', lw=2)
    elif (len(pix) == 8):
        plt.plot(pix[::2] + [pix[0]], pix[1::2] + [pix[1]], 'ro-', lw=2)
    else:
        plt.plot([pix[0], pix[2], pix[2], pix[0], pix[0], ],
                 [pix[1], pix[1], pix[3], pix[3], pix[1], ], 'ro-', lw=2)
    plt.show()
    return plt
