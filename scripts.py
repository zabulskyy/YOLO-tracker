import torch
import torchvision
import numpy as np
import os.path as osp
import os
import time
import pandas as pd

from io_tools import *


def dist2(box1, box2):
    # euclidean distance between two boxes
    x11, y11, x12, y12 = box1[1:5] if len(box1) != 4 else box1[:]
    x21, y21, x22, y22 = box2[1:5] if len(box2) != 4 else box2[:]
    x1, y1 = (x11 + x12) / 2, (y11 + y12) / 2
    x2, y2 = (x21 + x22) / 2, (y21 + y22) / 2
    X = (x1 - x2) ** 2
    Y = (y1 - y2) ** 2
    res = float((X + Y) ** 0.5)
    return res


def the_closest(box_to_compare, boxes):
    # returns the closest box and its index
    m, i = np.infty, 0
    for n, box in enumerate(boxes):
        d = dist2(box, box_to_compare)
        if d < m:
            m = float(d)
            i = n
    return boxes[i], i


def iou(boxA, boxB):
    # IoU between two boxes
    boxA = boxA[1:5] if len(boxA) != 4 else boxA
    boxB = boxB[1:5] if len(boxB) != 4 else boxB

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    res = interArea / float(boxAArea + boxBArea - interArea)
    return res


def the_most_iou(box_to_compare, boxes):
    # returns the box with greatest IoU and its index
    M, i = 0, 0
    for n, box in enumerate(boxes):
        v = iou(box, box_to_compare)
        if v > M:
            M = float(v)
            i = n
    return boxes[i], i


def generate_subframes(row1, row2, width):
    frame1, frame2 = int(row1[0]), int(row2[0])
    diff = frame1 - frame2
    if diff == 1:
        return None
    result = torch.zeros((frame2 - frame1 - 1, width))
    # result[:, -1] = row1[-1]
    step = (row1 - row2) / diff
    for frame_n in range(frame2 - frame1 - 1):
        result[frame_n] = row1 + (step) * (frame_n + 1)
    return result


def merge_frames(*rows):
    result = torch.zeros((1, 8))
    n = len(rows)
    for row in rows:
        result += row
    return result / n


def fill_first(tensor):
    FIRST = tensor[0]
    for i in range(int(FIRST[0]) - 1, -1, -1):
        row = FIRST
        row[0] = i
        row = row.view((1, -1))
        tensor = torch.cat((row, tensor))
    return tensor


def fill_last(tensor, NUM_FRAMES):
    last = tensor[tensor[:, 0] != 0][-1]
    for i in range(int(last[0]) + 1, NUM_FRAMES):
        row = last
        row[0] = i
        row = row.view((1, -1))
        tensor[i] = row
    return tensor


def most_frequent_class(results):
    counts = np.bincount(results[:, -1])
    return np.argmax(counts)


def interpolate(data):
    # TODO: replace with built-in torch functionality
    """
    interpolate gaps
    choose one detection among multiple, by picking the closest one by the euclidean distance
    ignore class labels
    :param data: dict: {OUTPUT: tensor, NUM_FRAMES: int, CUDA: bool}
    :return: interpolated tensor
    """
    OUTPUT = data["output"]
    NUM_FRAMES = data["num_frames"]
    CUDA = data["CUDA"]

    width = OUTPUT.size()[1]
    result = torch.zeros((NUM_FRAMES, width))
    if CUDA:
        result = result.cuda()

    if OUTPUT[0][0] != 0:
        OUTPUT = fill_first(OUTPUT)

    output_iter = 0
    res_iter = 0
    while res_iter != NUM_FRAMES:
        if (output_iter > len(OUTPUT) - 1):
            # break
            result = fill_last(result, NUM_FRAMES)
            break
        if output_iter == OUTPUT.shape[0] - 1:
            result[res_iter] = OUTPUT[output_iter]
            output_iter += 1
            res_iter += 1
            continue

        # next frame is interpolated
        if OUTPUT[output_iter][0] == OUTPUT[output_iter + 1][0] - 1:
            result[res_iter] = OUTPUT[output_iter]
            output_iter += 1
            res_iter += 1
            continue

        # missing detection on some frames, fill with missing
        if OUTPUT[output_iter][0] + 1 < OUTPUT[output_iter + 1][0]:
            subframes = generate_subframes(
                OUTPUT[output_iter], OUTPUT[output_iter + 1], width)
            result[res_iter] = OUTPUT[output_iter]
            for subframe in subframes:
                res_iter += 1
                result[res_iter] = subframe
            output_iter += 1
            res_iter += 1
            continue

        # this frame contains multiple detections, have to choose one
        if OUTPUT[output_iter][0] == OUTPUT[output_iter + 1][0]:
            frame = OUTPUT[output_iter][0]
            to_cut = OUTPUT[OUTPUT[:, 0] == frame]
            closest, _ = the_most_iou(result[res_iter - 1], to_cut)
            result[res_iter] = closest

            output_iter += len(to_cut)
            res_iter += 1

            if output_iter >= OUTPUT.shape[0] - 1:
                break

            # missing detection on some frames, fill with missing
            if OUTPUT[output_iter - 1][0] + 1 < OUTPUT[output_iter][0]:
                subframes = generate_subframes(
                    result[res_iter - 1], OUTPUT[output_iter], width)
                for subframe in subframes:
                    result[res_iter] = subframe
                    res_iter += 1
            continue
    return result


def eval_single_class_corr(cls1: int, cls2: int, correlations=cm):
    # calculate correlations between two classes
    return float(cm[cls1][cls2])


def eval_energy(bb1, bb2, cv=crit_vals, ban_negative_cc=True):
    box1, box2 = bb1[1:5], bb2[1:5]
    cls1, cls2 = bb1[6:], bb2[6:]  # 5'th column is the precision of the bbox
    class1, class2 = torch.argmax(cls1), torch.argmax(cls2)

    _iou = iou(box1, box2)
    _cc = eval_single_class_corr(class1, class2)
    _vd = vec_dist(bb1, bb2)

    return {"iou": _iou, "cc": _cc, "vd": 1 / _vd}


def cebtf(frame1, frame2, cv=crit_vals, ban_negative_cc=True):
    # calculate_energies_between_two_frames
    _boxes = torch.zeros((len(frame1)))
    _energies = torch.zeros((len(frame1)))
    for i in range(len(frame1)):
        max_energy = 0
        max_energy_index = 0

        ious = torch.zeros(len(frame2))
        ccs = torch.zeros(len(frame2))
        vds = torch.zeros(len(frame2))

        for j in range(len(frame2)):
            energy = eval_energy(frame1[i], frame2[j], cv=crit_vals, ban_negative_cc=ban_negative_cc)

            if ban_negative_cc and energy["cc"] < 0:
                pass  # values are already 0
            else:
                ious[j] = energy["iou"] * crit_vals["iou"]
                ccs[j] = energy["cc"] * crit_vals["cc"]
                vds[j] = energy["vd"] * crit_vals["vd"]

        # normalize
        ious = ious / (torch.max(ious) + 1e-10)
        ccs = ccs / (torch.max(ccs) + 1e-10)
        vds = vds / (torch.max(vds) + 1e-10)

        energies = sum((ious, ccs, vds))

        _boxes[i] = torch.argmax(energies)
        _energies[i] = torch.max(energies)
    return _boxes, _energies


def initial_energy(first_boxes, method="ltd2", cv=crit_vals, ):
    l = first_boxes.shape[0]
    ies = torch.zeros(l)  # initial_energies
    for i in range(l):
        for j in range(l):
            ie = dist2(first_boxes[i], first_boxes[j])
            ies[i] += ie
    ies = 1 / ies * crit_vals["id"]
    return ies


def vec_dist(vec1, vec2, dist=dist):
    vd = None

    if len(vec1) > 80:
        vec1[torch.argmax(vec1[6:]) + 6] = 1
        vec2[torch.argmax(vec2[6:]) + 6] = 1
        vd = dist(vec1[6:].view([1, -1]), vec2[6:].view([1, -1]))
    else:
        vec1[torch.argmax(vec1)] = 1
        vec2[torch.argmax(vec2)] = 1
        vd = dist(vec1.view([1, -1]), vec2.view([1, -1]))
    return vd


def the_closest_vec(row_to_compare, rows):
    m, i = np.infty, 0
    for n, row in enumerate(rows):
        d = vec_dist(row, row_to_compare)
        if d < m:
            m = float(d)
            i = n
    return rows[i], i


def do_dp(total_res, yolo_pred, cv=crit_vals):
    # do dynamic programming
    # core function 
    yp, imlen = yolo_pred[vot_class]  # yolo predictions
    imlen = int(imlen)

    data = dict()
    data["output"] = yp
    data["num_frames"] = int(imlen)
    data["CUDA"] = CUDA
    data["first"] = get_first(vot_class)

    if CUDA:
        data["output"] = data["output"].cuda()
        data["first"] = data["first"].cuda()

    frames_with_detection = sorted(list(set(data["output"][:, 0].tolist())))
    ffwdi = frames_with_detection[0]  # first_frame_with_detection_index
    first_boxes = data["output"][data["output"][:, 0] == ffwdi]
    path = torch.zeros((len(first_boxes), imlen))
    energies = torch.zeros((len(first_boxes), imlen))

    prev_boxes = first_boxes
    path[:, 0] = first_boxes[0][0]

    # calculate energies
    for n, i in enumerate(frames_with_detection[:]):
        i_boxes = data["output"][data["output"][:, 0] == i]

        boxes, energy = cebtf(prev_boxes, i_boxes, cv=cv)

        path[:, int(n)] = boxes
        energies[:, int(n)] += energy

        prev_boxes = i_boxes[boxes.long()]

    verdict = torch.sum(energies, dim=1)

    # init energies   
    ie = initial_energy(first_boxes, method="ltd2")
    ie = ie / max(ie) * max(verdict) * crit_vals["id"]

    if verdict.shape != torch.Size([1]):
        verdict += ie

    first_bb_candidates = (verdict == torch.max(verdict)).nonzero().view(-1).tolist()
    res_i = path[first_bb_candidates].view(-1).int()
    res = torch.zeros((len(frames_with_detection), 5))

    # prepare for interpolation
    for n, i in enumerate(frames_with_detection[:]):
        cfp = data["output"][data["output"][:, 0] == i]  # current_frame_predictions
        index = res_i[n]
        cp = cfp[index]  # chosen_prediction
        res[int(n)] = cp[:5]

    data["output"] = res
    total_res[vot_class] = interpolate(data)[:, 1:5].tolist()

    return res
