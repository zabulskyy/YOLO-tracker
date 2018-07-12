from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import pickle as pkl
import itertools


class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers = num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5, 5)
                                     for x in range(num_layers)])
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)


def get_test_input(input_dim, CUDA, cuda_n):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda(cuda_n)
    return img_


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                        default="1,2,3", type=str)
    parser.add_argument("--save", dest='saveto', help="Save results to the chosen file",
                        default="", type=str)
    parser.add_argument("--silent", dest='silent', help="[all] - Run in silent mode",
                        default="", type=str)
    parser.add_argument("--cuda", dest='cuda', help="cuda [0-9] select a cuda device",
                        default="0", type=str)

    return parser.parse_args()
def IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def most_frequent_class(results, max_num=1):
    """
    results: torch.tensor
    TODO max_num: notes how mush should we return
    """
    counts = np.bincount(results[:, -1])
    return np.argmax(counts)


def load_results(file="results.txt"):
    with open(file, mode='r') as f:
        return f.readlines()


def generate_subframes(row1, row2):
    frame1, frame2 = int(row1[0]), int(row2[0])
    diff = frame1 - frame2
    if diff == 1:
        return None
    result = torch.zeros((frame2 - frame1 - 1, 8))
    # result[:, -1] = row1[-1]
    step = (row1 - row2) / diff
    for i, frame_n in enumerate(range(frame1 + 1, frame2)):
        result[i] = row1 + (step) * (i + 1)
    return result


def merge_frames(*rows):
    result = torch.zeros((1, 8))
    n = len(rows)
    for row in rows:
        result += row
    return result / n


def the_closest(row_to_compare, rows):
    def dist(row1, row2):
        x11, y11, x12, y12 = row1[1:5]
        x21, y21, x22, y22 = row2[1:5]
        x1, y1 = (x11 + x12) / 2, (y11 + y12) / 2
        x2, y2 = (x21 + x22) / 2, (y21 + y22) / 2
        return float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
    m, i = np.infty, 0
    for n, row in enumerate(rows):
        d = dist(row, row_to_compare)
        if d < m:
            m = float(d)
            i = n
    return rows[i]


# def merge_frames_in_tensor(tensor):
#     row = 0
#     for row in tensor:
#         frame = row[0]
#         if sum(tensor[:, 0] == frame) == 1:
#             continue


def fullfill(output, num_frames):
    mfc = most_frequent_class(output)
    mfc_data = output[output[:, -1] == float(mfc)]  # only detections with the most frequent class
    result = torch.zeros((num_frames, 8))

    mfc_iter = 0
    res_iter = 0
    while res_iter != num_frames:
        # check1 = mfc_data[row]
        # chekc2 = mfc_data[row + 1]

        if mfc_iter == mfc_data.shape[0] - 1:
            result[res_iter] = mfc_data[mfc_iter]
            mfc_iter += 1
            res_iter += 1
            continue

        # next frame is fulfilled
        if mfc_data[mfc_iter][0] == mfc_data[mfc_iter + 1][0] - 1:
            result[res_iter] = mfc_data[mfc_iter]
            mfc_iter += 1
            res_iter += 1
            continue

        # missing detection on some frames, fill with missing
        if mfc_data[mfc_iter][0] + 1 < mfc_data[mfc_iter + 1][0]:
            subframes = generate_subframes(mfc_data[mfc_iter], mfc_data[mfc_iter + 1])
            result[res_iter] = mfc_data[mfc_iter]
            for subframe in subframes:
                res_iter += 1
                result[res_iter] = subframe
            mfc_iter += 1
            res_iter += 1
            continue

        # this frame contains multiple detections, have to choose one
        if mfc_data[mfc_iter][0] == mfc_data[mfc_iter + 1][0]:
            frame = mfc_data[mfc_iter][0]
            to_cut = mfc_data[mfc_data[:, 0] == frame]
            closest = the_closest(result[res_iter - 1], to_cut)
            result[res_iter] = closest
            mfc_iter += len(to_cut)
            res_iter += 1

            if mfc_iter >= mfc_data.shape[0] - 1:
                break

            # missing detection on some frames, fill with missing
            if mfc_data[mfc_iter - 1][0] + 1 < mfc_data[mfc_iter][0]:
                subframes = generate_subframes(result[res_iter - 1], mfc_data[mfc_iter])
                # result[res_iter] = mfc_data[mfc_iter]
                for subframe in subframes:
                    result[res_iter] = subframe
                    res_iter += 1
            continue
    return result
    

def run():
    args = arg_parse()
    cuda_n = int(args.cuda)
    silent = args.silent == "all"
    if (silent):
        import sys
        sys.stdout = open(os.devnull, 'w')
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    classes = load_classes('data/coco.names')
    num_classes = len(classes)

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda(cuda_n)

    # Set the model in evaluation mode
    model.eval()

    read_dir = time.time()
    # Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in sorted(os.listdir(images))[:] if os.path.splitext(  # sorted() is my modification
            img)[1] == '.png' or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[1] == '.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()

    num_frames = len(imlist)

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    load_batch = time.time()

    batches = list(
        map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        im_dim_list = im_dim_list.cuda(cuda_n)

    leftover = 0

    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i*batch_size: min((i + 1)*batch_size,
                                                              len(im_batches))])) for i in range(num_batches)]

    i = 0

    write = False
    model(get_test_input(inp_dim, CUDA, cuda_n), CUDA)

    start_det_loop = time.time()

    objs = {}

    for batch in im_batches:
        # load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda(cuda_n)

        # Apply offsets to the result predictions
        # Tranform the predictions as described in the YOLO paper
        # flatten the prediction vector
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes)
        # Put every proposed box as a row.
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

#        prediction = prediction[:,scale_indices]

        # get the boxes with object confidence > threshold
        # Convert the cordinates to absolute coordinates
        # perform NMS on these boxes, and save the results
        # I could have done NMS and saving seperately to have a better abstraction
        # But both these operations require looping, hence
        # clubbing these ops in one loop instead of two.
        # loops are slower than vectorised operations.

        prediction = write_results(
            prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        if type(prediction) == int:
            i += 1
            continue

        end = time.time()


#        print(end - start)

        prediction[:, 0] += i*batch_size

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(imlist[i*batch_size: min((i + 1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(
                image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")
        i += 1

        if CUDA:
            torch.cuda.synchronize()

    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(inp_dim/im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor *
                          im_dim_list[:, 0].view(-1, 1))/2
    output[:, [2, 4]] -= (inp_dim - scaling_factor *
                          im_dim_list[:, 1].view(-1, 1))/2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(
            output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(
            output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    output_recast = time.time()

    class_load = time.time()

    colors = pkl.load(open("pallete", "rb"))

    draw = time.time()

    def write(x, batches, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())

        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img

    # # print(output)
    list(map(lambda x: write(x, im_batches, orig_ims), output))
    det_names = pd.Series(imlist).apply(lambda x: "{}/yolo/{}".format(args.det, x.split("/")[-1]))
    list(map(cv2.imwrite, det_names, orig_ims))
    # print(output)
    # output = fullfill(output, num_frames)
    # print(output)
    list(map(lambda x: write(x, im_batches, orig_ims), output))
    det_names = pd.Series(imlist).apply(lambda x: "{}/rec/{}".format(args.det, x.split("/")[-1]))
    list(map(cv2.imwrite, det_names, orig_ims))


    end = time.time()

    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format(
        "Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format(
        "Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format(
        "Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format(
        "Average time_per_img", (end - load_batch)/len(imlist)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()

    def report(x, num_frames):
        lx = x.tolist()
        results = list
        for pred in lx:
            results.append(pred)
        return results

    def save_report(rep, file, format=""):
        if format == "":
            with open(file, mode='w') as f:
                print(rep, file=f)
        elif format == "csv":
            import csv
            with open(file, 'wb') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow(rep)

    save_to = args.saveto
    rep = report(output, len(imlist))
    if save_to != "":
        print("saving results to", save_to)
        save_report(rep, save_to, "csv")

    return output, len(imlist), (im_batches, orig_ims)





if __name__ == '__main__':
    o = torch.tensor([
        [   0.0000,     1.,  1.,  1.,  1.,      0.0,  0.9955,    1.0000],
        [   1.0000,     2.,  2.,  2.,  2.,      0.1,  0.9173,    1.0000],
        [   1.0000,     10., 10., 10., 10.,     0.2,  0.8216,    1.0000],
        [   2.0000,     3.,  3.,  3.,  3.,      0.3,  0.6765,    1.0000],
        [   2.0000,     30., 30., 30., 30.,     0.4,  0.7964,    1.0000],
        [   2.0000,     1.,  1.,  1.,  1.,      0.5,  0.7626,    1.0000],
        [   4.0000,     4.,  4.,  4.,  4.,      0.6,  0.9043,    1.0000],
        [   5.0000,     4.,  4.,  4.,  4.,      0.7,  0.7252,    1.0000],
        [   6.0000,     5.,  4.,  4.,  4.,      0.8,  0.9907,    1.0000],
        [   10.0000,    10.,  4.,  4.,  4.,      0.9,  0.6294,    1.0000],
        [   15.0000,    4.,  4.,  4.,  4.,      0.91,  0.4758,    1.0000]])

    output, num_frames, (im_batches, orig_ims) = run()
