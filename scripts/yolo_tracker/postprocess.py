import torch
import numpy as np
import os.path as osp
import os


def generate_subframes(row1, row2):
    frame1, frame2 = int(row1[0]), int(row2[0])
    diff = frame1 - frame2
    if diff == 1:
        return None
    result = torch.zeros((frame2 - frame1 - 1, 8))
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


def the_closest(row_to_compare, rows):
    def dist(row1, row2):
        x11, y11, x12, y12 = row1[1:5] if len(row1) != 4 else row1[:]
        x21, y21, x22, y22 = row2[1:5] if len(row2) != 4 else row2[:]
        x1, y1 = (x11 + x12) / 2, (y11 + y12) / 2
        x2, y2 = (x21 + x22) / 2, (y21 + y22) / 2
        X = (x1 - x2) ** 2
        Y = (y1 - y2) ** 2
        res = float((X + Y) ** 0.5)
        return res

    m, i = np.infty, 0
    for n, row in enumerate(rows):
        d = dist(row, row_to_compare)
        if d < m:
            m = float(d)
            i = n
    return rows[i], i


# def merge_frames_in_tensor(tensor):
#     row = 0
#     for row in tensor:
#         frame = row[0]
#         if sum(tensor[:, 0] == frame) == 1:
#             continue


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
    """
    interpolate gaps
    choose one detection among multiple, by picking the closest one by the euclidean distance
    ignore class labels
    :param OUTPUT: tensor
    :param NUM_FRAMES:
    :param CUDA: bool - if cuda is available
    :return: interpolated tensor
    """
    OUTPUT = data["output"]
    NUM_FRAMES = data["num_frames"]
    CUDA = data["CUDA"]

    result = torch.zeros((NUM_FRAMES, 8))
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
                OUTPUT[output_iter], OUTPUT[output_iter + 1])
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
            closest, _ = the_closest(result[res_iter - 1], to_cut)
            result[res_iter] = closest
            output_iter += len(to_cut)
            res_iter += 1

            if output_iter >= OUTPUT.shape[0] - 1:
                break

            # missing detection on some frames, fill with missing
            if OUTPUT[output_iter - 1][0] + 1 < OUTPUT[output_iter][0]:
                subframes = generate_subframes(
                    result[res_iter - 1], OUTPUT[output_iter])
                # result[res_iter] = OUTPUT[output_iter]
                for subframe in subframes:
                    result[res_iter] = subframe
                    res_iter += 1
            continue
    return result


def the_closest_class(row_to_compare, rows):
    first_frame_rows = rows[rows[:, 0] == rows[0][0]]
    closest, _ = the_closest(row_to_compare, first_frame_rows)
    return closest


def replace_first_frame(to_replace, OUTPUT):
    OUTPUT = OUTPUT[OUTPUT[:, 0] != 0]
    return torch.cat((to_replace.view((1, -1)), OUTPUT))


def replace_i_frame(to_replace, OUTPUT, i):
    a = OUTPUT[OUTPUT[:, 0] < i]
    b = to_replace.view((1, -1))
    c = OUTPUT[OUTPUT[:, 0] > i]
    return torch.cat((a, b, c))


def read_spec_gt(folder, i):
    # reads the specific line in the groundtruth txt file and returns it as a tensor
    # 1x4
    with open(osp.join(folder, "groundtruth.txt"), 'r') as file:
        l = file.read().split("\n")[i].split(",")
        X, Y = l[::2], l[1::2]
        l = [min(X), min(Y), max(X), max(Y)]
        l = torch.tensor([float(x) for x in l])
        return l


def pp_mfc(data):
    # picks up the most frequent class and removes the different ones
    # mfc = the Most Frequent Class
    output = data["output"]
    mfc = most_frequent_class(output)

    # only detections with the most frequent class
    output = output[output[:, -1] == float(mfc)]
    return output


def pp_first(data):
    # detects the closest object to FIRST gt box and removes the rest on the FIRST frame
    CUDA = data["CUDA"]
    output = data["output"]
    first = data["first"]

    if CUDA:
        first = first.cuda()
    tcc = the_closest_class(first, output)
    # only detections close to the true FIRST frame
    output = replace_first_frame(tcc, output)
    return output


def pp_god(data):

    num_frames = data["num_frames"]
    folder = data["folder"]
    output = data["output"]

    for i in range(num_frames):
        truth = read_spec_gt(folder, i)
        to_compare = output[output[:, 0] == i]
        if to_compare.size()[0] == 0:
            continue
        tcc, idx = the_closest(truth, to_compare)
        output = replace_i_frame(tcc, output, i)
    return output


def fill_zeros(folder):
    return [[0, 0, 0, 0] for i in os.listdir(folder) if i.endswith(".jpg")]


def postprocess(output, folder, postprocessors):
    """
    :param data: (tensor, num_frames, CUDA)
    :param folder: absulute path to VOT specific class directory
    :param postprocessors: [mfc, god, first]
    :return:
    """

    data = {
        "first": read_spec_gt(folder, 0),
        "output": output[0],
        "num_frames": output[1],
        "CUDA": output[2],
        "folder": folder
    }

    _mfc = "mfc" in postprocessors
    _first = "first" in postprocessors
    _god = "god" in postprocessors

    if _god:
        data["output"] = pp_god(data)

    if _first:
        data["output"] = pp_first(data)

    if _mfc:
        data["output"] = pp_mfc(data)

    return interpolate(data)


def do_full_postprop(predictions, postprocessors, vot_path, force_square):
    RESULTS = predictions["results"]
    NUM_FRAMES = predictions["num_frames"]
    CUDA = predictions["CUDA"]

    res = dict()
    nm_fr = dict()

    for im_class in RESULTS:
        pp_result = postprocess((RESULTS[im_class], NUM_FRAMES[im_class], CUDA), osp.join(vot_path, im_class),
                                postprocessors)
        if force_square:
            pp_result = pp_result[:, 1:5]
        if len(pp_result) != 0:
            res[im_class] = pp_result.tolist()
            nm_fr[im_class] = NUM_FRAMES[im_class]
        else:
            res[im_class] = fill_zeros(osp.join(vot_path, im_class))
            nm_fr[im_class] = NUM_FRAMES[im_class]

    return res, nm_fr
