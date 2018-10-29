import torch
import torchvision
import numpy as np

# compute class correlation
def ccc(classes_probs):
    # classes_probs.shape = (bboxes, classes)
    corr_mat = torch.zeros((classes_probs.shape[1], classes_probs.shape[1],))

    for cls in range(len(classes_probs[0])):
        col = classes_probs[:, cls].view((-1, 1))
        tX = col * classes_probs
        sX = torch.sum(tX, dim=0)
        corr_mat[cls] = sX
    return corr_mat


def corr(col1, col2):
    mean1, mean2 = torch.mean(col1), torch.mean(col2)
    cm1, cm2 = col1 - mean1, col2 - mean2
    top = sum(cm1 * cm2)
    bot = torch.sqrt(sum(cm1 ** 2) * sum(cm2 ** 2))
    return top / bot


# compute class correlation coefficient
def cccc(classes_probs):
    # classes_probs.shape = (bboxes, classes)
    corr_mat = torch.zeros((classes_probs.shape[1], classes_probs.shape[1],))
    num_classes = len(classes_probs[0])
    for cls1 in range(num_classes):
        col1 = classes_probs[:, cls1].view((-1, 1))
        for cls2 in range(cls1 + 1, num_classes):
            print("{}.{} / {}".format(cls1 + 1, cls2 + 1, num_classes))
            col2 = classes_probs[:, cls2].view((-1, 1))
            _corr = corr(col1, col2)
            #             _corr = np.corrcoef(col1, col2)
            corr_mat[cls1, cls2] = _corr
    return corr_mat
