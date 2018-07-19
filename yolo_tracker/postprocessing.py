import torch
import numpy as np


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


def interpolate_blind(output, num_frames, CUDA):
    mfc = most_frequent_class(output)
    # only detections with the most frequent class
    mfc_data = output[output[:, -1] == float(mfc)]
    result = torch.zeros((num_frames, 8))
    if CUDA:
        result = result.cuda()

    if mfc_data[0][0] != 0:
        mfc_data = fill_first(mfc_data)

    mfc_iter = 0
    res_iter = 0
    while res_iter != num_frames:
        try:
            if mfc_iter == mfc_data.shape[0] - 1:
                result[res_iter] = mfc_data[mfc_iter]
                mfc_iter += 1
                res_iter += 1
                continue

            # next frame is interpolated
            if mfc_data[mfc_iter][0] == mfc_data[mfc_iter + 1][0] - 1:
                result[res_iter] = mfc_data[mfc_iter]
                mfc_iter += 1
                res_iter += 1
                continue

            # missing detection on some frames, fill with missing
            if mfc_data[mfc_iter][0] + 1 < mfc_data[mfc_iter + 1][0]:
                subframes = generate_subframes(
                    mfc_data[mfc_iter], mfc_data[mfc_iter + 1])
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
                closest, _ = the_closest(result[res_iter - 1], to_cut)
                result[res_iter] = closest
                mfc_iter += len(to_cut)
                res_iter += 1

                if mfc_iter >= mfc_data.shape[0] - 1:
                    break

                # missing detection on some frames, fill with missing
                if mfc_data[mfc_iter - 1][0] + 1 < mfc_data[mfc_iter][0]:
                    subframes = generate_subframes(
                        result[res_iter - 1], mfc_data[mfc_iter])
                    # result[res_iter] = mfc_data[mfc_iter]
                    for subframe in subframes:
                        result[res_iter] = subframe
                        res_iter += 1
                continue
        except:
            return result
    return result


def the_closest_class(row_to_compare, rows):
    first_frame_rows = rows[rows[:, 0] == rows[0][0]]
    closest, _ = the_closest(row_to_compare, first_frame_rows)
    return closest[-1]


def interpolate_with_first(first, output, num_frames, CUDA):
    tcc = the_closest_class(first, output)
    # print(output[tcc].tolist(), file=open("fuck/output_tcc.txt", 'w+'))

    # only detections with the common with gt class class
    tcc_data = output[output[:, -1] == float(tcc)]
    # print(tcc_data.tolist(), file=open("fuck/tcc_data.txt", 'w+'))

    result = torch.zeros((num_frames, 8))
    if CUDA:
        result = result.cuda()

    if tcc_data[0][0] != 0:
        tcc_data = fill_first(tcc_data)

    tcc_iter = 0
    res_iter = 0
    while res_iter != num_frames:
        if (tcc_iter > len(tcc_data) - 1):
            # break
            result = fill_last(result, num_frames)
            break
        # try:
        if tcc_iter == tcc_data.shape[0] - 1:
            result[res_iter] = tcc_data[tcc_iter]
            tcc_iter += 1
            res_iter += 1
            continue

        # next frame is interpolated
        if tcc_data[tcc_iter][0] == tcc_data[tcc_iter + 1][0] - 1:
            result[res_iter] = tcc_data[tcc_iter]
            tcc_iter += 1
            res_iter += 1
            continue

        # missing detection on some frames, fill with missing
        if tcc_data[tcc_iter][0] + 1 < tcc_data[tcc_iter + 1][0]:
            subframes = generate_subframes(
                tcc_data[tcc_iter], tcc_data[tcc_iter + 1])
            result[res_iter] = tcc_data[tcc_iter]
            for subframe in subframes:
                res_iter += 1
                result[res_iter] = subframe
            tcc_iter += 1
            res_iter += 1
            continue

        # this frame contains multiple detections, have to choose one
        if tcc_data[tcc_iter][0] == tcc_data[tcc_iter + 1][0]:
            frame = tcc_data[tcc_iter][0]
            to_cut = tcc_data[tcc_data[:, 0] == frame]
            closest, _ = the_closest(result[res_iter - 1], to_cut)
            result[res_iter] = closest
            tcc_iter += len(to_cut)
            res_iter += 1

            if tcc_iter >= tcc_data.shape[0] - 1:
                break

            # missing detection on some frames, fill with missing
            if tcc_data[tcc_iter - 1][0] + 1 < tcc_data[tcc_iter][0]:
                subframes = generate_subframes(
                    result[res_iter - 1], tcc_data[tcc_iter])
                # result[res_iter] = tcc_data[tcc_iter]
                for subframe in subframes:
                    result[res_iter] = subframe
                    res_iter += 1
            continue
        # except:
        return result
    return result
