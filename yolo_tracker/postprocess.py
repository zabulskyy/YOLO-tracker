import torch
import numpy as np
import os.path as osp


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
    first = tensor[0]
    for i in range(int(first[0]) - 1, -1, -1):
        row = first
        row[0] = i
        row = row.view((1, -1))
        tensor = torch.cat((row, tensor))
    return tensor


def fill_last(tensor, num_frames):
    last = tensor[tensor[:, 0] != 0][-1]
    for i in range(int(last[0]) + 1, num_frames):
        row = last
        row[0] = i
        row = row.view((1, -1))
        tensor[i] = row
    return tensor


def most_frequent_class(results):
    counts = np.bincount(results[:, -1])
    return np.argmax(counts)


def interpolate(output, num_frames, CUDA):
    """
    interpolate gaps
    choose one detection among multiple, by picking the closest one by the euclidean distance
    ignore class labels
    :param output: tensor
    :param num_frames:
    :param CUDA: bool - if cuda is available
    :return: interpolated tensor
    """
    result = torch.zeros((num_frames, 8))
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
                output[output_iter], output[output_iter + 1])
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
            closest, _ = the_closest(result[res_iter - 1], to_cut)
            result[res_iter] = closest
            output_iter += len(to_cut)
            res_iter += 1

            if output_iter >= output.shape[0] - 1:
                break

            # missing detection on some frames, fill with missing
            if output[output_iter - 1][0] + 1 < output[output_iter][0]:
                subframes = generate_subframes(
                    result[res_iter - 1], output[output_iter])
                # result[res_iter] = output[output_iter]
                for subframe in subframes:
                    result[res_iter] = subframe
                    res_iter += 1
            continue
    return result


def the_closest_class(row_to_compare, rows):
    first_frame_rows = rows[rows[:, 0] == rows[0][0]]
    closest, _ = the_closest(row_to_compare, first_frame_rows)
    return closest


def replace_first_frame(to_replace, output):
    output = output[output[:, 0] != 0]
    return torch.cat((to_replace.view((1, -1)), output))


def read_spec_gt(folder, i):
    # reads the specific line in the groundtruth txt file and returns it as a tensorcl
    with open(osp.join(folder, "groundtruth.txt"), 'r') as file:
        l = file.read().split("\n")[i].split(",")
        X, Y = l[::2], l[1::2]
        l = [min(X), min(Y), max(X), max(Y)]
        l = torch.tensor([float(x) for x in l])
        return l


def interpolate_with_first_and_mfc(first, output, num_frames, CUDA):
    if CUDA:
        first = first.cuda()
    tcc = the_closest_class(first, output)  # tensor
    true_class = tcc[-1]  # number

    # only detections with the true class
    output = output[output[:, -1] == float(true_class)]
    output = replace_first_frame(tcc, output)
    result = interpolate(output, num_frames, CUDA)
    return result


def interpolate_mfc(output, num_frames, CUDA):
    # picks up the most frequent class and removes the different ones
    # mfc = the Most Frequent Class
    mfc = most_frequent_class(output)

    # only detections with the most frequent class
    output = output[output[:, -1] == float(mfc)]
    return interpolate(output, num_frames, CUDA)


def interpolate_with_first(first, output, num_frames, CUDA):
    # detects the closest object to first gt box and removes the rest on the first frame
    if CUDA:
        first = first.cuda()
    tcc = the_closest_class(first, output)
    # only detections close to the true first frame
    output = replace_first_frame(tcc, output)
    return interpolate(output, num_frames, CUDA)


def postprocess(data, folder, pp):
    output = data[0]
    num_frames = data[1]
    CUDA = data[3]
    first = read_spec_gt(folder, 0)

    if pp == "mfc":
        return interpolate_mfc(output, num_frames, CUDA)
    elif pp == "first":
        first = read_spec_gt(folder, 0)
        return interpolate_with_first(first, output, num_frames, CUDA)
    elif pp == "first_and_mfc":
        first = read_spec_gt(folder, 0)
        return interpolate_with_first_and_mfc(first, output, num_frames, CUDA)
    else:
        raise Exception("pp was not cpecified correctly")
