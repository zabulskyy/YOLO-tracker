import torch

a = torch.tensor([[4.0,27.7522029876709,41.05860900878906,391.4895935058594,318.52587890625,0.0015927463537082076,0.2963290810585022,14.0],[
4.0,297.18267822265625,68.29659271240234,389.58087158203125,177.93865966796875,0.009883671998977661,0.2447674721479416,25.0],[
4.0,300.8069152832031,69.85529327392578,382.0081787109375,199.25563049316406,0.9253581166267395,0.5963649153709412,29.0],[
4.0,295.9216613769531,77.04573059082031,389.44622802734375,195.33233642578125,0.7198722958564758,0.2401549369096756,33.0],[
4.0,298.96746826171875,76.67626190185547,383.6321105957031,256.77752685546875,0.002787762088701129,0.1909792423248291,34.0]])

gt = torch.tensor([292.64,81.098,357.49,64.544,388.99,187.95,324.14,204.5])
gt = to4(gt)

def to4(l):
    X, Y = l[::2], l[1::2]
    l = [min(X), min(Y), max(X), max(Y)]
    l = torch.tensor([float(x) for x in l])
    return l

def the_most_iou(row_to_compare, rows):
    def iou(boxA, boxB):
        boxA = boxA[1:5] if len(boxA) != 4 else boxA
        boxB = boxB[1:5] if len(boxB) != 4 else boxB
        print(boxA, boxB)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)

    M, i = 0, 0
    for n, row in enumerate(rows):
        v = iou(row, row_to_compare)
        print(v)
        if v > M:
            M = float(v)
            i = n
    return rows[i], i

the_most_iou(gt, a)
