import torch
import numpy as np
import os.path as osp
import os

def the_most_iou(row_to_compare, rows):
    def iou(boxA, boxB):
        boxA = boxA[1:5] if len(boxA) != 4 else boxA
        boxB = boxB[1:5] if len(boxB) != 4 else boxB

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1) 
        return float(interArea / float(boxAArea + boxBArea - interArea))


    m, i = 0, 0
    for n, row in enumerate(rows):
        d = iou(row, row_to_compare)
        print(d)
        if d > m:
            m = d
            i = n
    return rows[i], i


a = torch.tensor([[1, 0, 0, 3, 3,1,1,1],[213123, 2,2,5,5,1,1,1],[1,10,10,11,11,1,1,1]])
b = a[1]
print(b)
print(a)
print(the_most_iou(b, a))
b.shape