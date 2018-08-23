import torch
import torchvision
import numpy as np
import os.path as osp
import os


def dist2(box1, box2):
    # Euclidian distance between two boxes
    x11, y11, x12, y12 = box1[1:5] if len(box1) != 4 else box1[:]
    x21, y21, x22, y22 = box2[1:5] if len(box2) != 4 else box2[:]
    x1, y1 = (x11 + x12) / 2, (y11 + y12) / 2
    x2, y2 = (x21 + x22) / 2, (y21 + y22) / 2
    X = (x1 - x2) ** 2
    Y = (y1 - y2) ** 2
    res = float((X + Y) ** 0.5)
    return res


def the_closest(box_to_compare, boxes):
    # return the closest box to the boxes
    m, i = np.infty, 0
    for n, box in enumerate(boxes):
        d = dist2(box, box_to_compare)
        if d < m:
            m = float(d)
            i = n
    return boxes[i], i


def iou(box1, box2):
    # return IoU between two boxes (inf if boxes are equal)
    box1 = box1[1:5] if len(box1) != 4 else box1
    box2 = box2[1:5] if len(box2) != 4 else box2

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    res = interArea / float(box1Area + box2Area - interArea)
    return res


def the_most_iou(box, boxes):
    # return the closest box by IoU
    M, i = 0, 0
    for n, box in enumerate(boxes):
        v = iou(box, box)
        if v > M:
            M = float(v)
            i = n
    return boxes[i], i


def generate_subframes(row1, row2, width):
    # interpolate subframes between two boxes
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
    # merge boxes into the average box
    result = torch.zeros((1, 8))
    n = len(rows)
    for row in rows:
        result += row
    return result / n


def fill_first(tensor):
    # fill missing first frames
    first = tensor[0]
    for i in range(int(first[0]) - 1, -1, -1):
        row = first
        row[0] = i
        row = row.view((1, -1))
        tensor = torch.cat((row, tensor))
    return tensor


def fill_last(tensor, num_frames):
    # fill missing last frames
    last = tensor[tensor[:, 0] != 0][-1]
    for i in range(int(last[0]) + 1, num_frames):
        row = last
        row[0] = i
        row = row.view((1, -1))
        tensor[i] = row
    return tensor


def most_frequent_class(results):
    # return the most frequent class ocuuring in results
    counts = np.bincount(results[:, -1])
    return np.argmax(counts)


def interpolate(data):
    """
    - interpolate gaps
    - choose one detection among multiple, by picking the closest one by the euclidean distance
    - ignore class labels
    :param output: tensor
    :param num_frames:
    :param CUDA: bool - if cuda is available
    :return: interpolated tensor
    """
    output = data["output"]
    num_frames = data["num_frames"]
    CUDA = data["CUDA"]

    width = output.size()[1]
    result = torch.zeros((num_frames, width))

    if CUDA:
        result = result.cuda()

    if output[0][0] != 0:
        output = fill_first(output)

    output_iter = 0
    res_iter = 0
    while res_iter != num_frames:
        if (output_iter > len(output) - 1):
            # break
            result = fill_last(result, num_frames)
            break
        if output_iter == output.shape[0] - 1:
            result[res_iter] = output[output_iter]
            output_iter += 1
            res_iter += 1
            continue

        # next frame is interpolated
        if output[output_iter][0] == output[output_iter + 1][0] - 1:
            result[res_iter] = output[output_iter]
            output_iter += 1
            res_iter += 1
            continue

        # missing detection on some frames, fill with missing
        if output[output_iter][0] + 1 < output[output_iter + 1][0]:
            subframes = generate_subframes(
                output[output_iter], output[output_iter + 1], width)
            result[res_iter] = output[output_iter]
            for subframe in subframes:
                res_iter += 1
                result[res_iter] = subframe
            output_iter += 1
            res_iter += 1
            continue

        # this frame contains multiple detections, have to choose one
        if output[output_iter][0] == output[output_iter + 1][0]:
            frame = output[output_iter][0]
            to_cut = output[output[:, 0] == frame]
            # closest, _ = the_closest(result[res_iter - 1], to_cut)
            # result[res_iter] = closest
            closest, _ = the_most_iou(result[res_iter - 1], to_cut)
            result[res_iter] = closest

            output_iter += len(to_cut)
            res_iter += 1

            if output_iter >= output.shape[0] - 1:
                break

            # missing detection on some frames, fill with missing
            if output[output_iter - 1][0] + 1 < output[output_iter][0]:
                subframes = generate_subframes(
                    result[res_iter - 1], output[output_iter], width)
                # result[res_iter] = output[output_iter]
                for subframe in subframes:
                    result[res_iter] = subframe
                    res_iter += 1
            continue
    return result


def read_gt(gt_file, force_square=True):
    # read a specific groundtruth file and return coordinates as a tensor
    with open(gt_file, 'r') as file:
        l = file.read().split("\n")
        l.pop(-1)
        l = [[float(x) for x in y.split(",")] for y in l]
        if force_square:
            l = [[min(m[::2]), min(m[1::2]), max(m[::2]), max(m[1::2])]
                 for m in l]
        return torch.tensor(l)


def read_full_gt(folder=folder, force_square=True):
    # read all groundtruth files from dataset and return coordinates as a tensor
    all_classes = [x for x in os.listdir(folder) if not x.endswith(".txt")]
    X = list()
    for single_class in all_classes:
        gt_file = osp.join(folder, single_class, "groundtruth.txt")
        with open(gt_file, 'r') as file:
            l = file.read().split("\n")
            l.pop(-1)
            l = [[float(x) for x in y.split(",")] for y in l]
            if force_square:
                l = [[min(m[::2]), min(m[1::2]), max(m[::2]), max(m[1::2])]
                     for m in l]
        X.append(l)
    return torch.tensor(X)

def read_yolo_pred(vot_class, yolo_pred_path=yolo_pred_path, pred_format="csv", first_len=True):
    yolo_pred_file = osp.join(yolo_pred_path, vot_class + "." + pred_format)
    with open(yolo_pred_file, 'r') as file:
        l =file.read().split("\n")
        l.pop(-1)
        im_len = -1
        if first_len:
            im_len = float(l.pop(0))
        l = [[float(x) for x in y.split(",")] for y in l]            
        return torch.tensor(l), im_len
    
    