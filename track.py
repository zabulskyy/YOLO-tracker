import argparse
import numpy as np

def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='Tracker')

    parser.add_argument("--input", dest='input', help="Input dict file",
                        default="", type=str)


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

    print(interArea, float(boxAArea + boxBArea - interArea))
    return iou

if __name__ == '__main__':
    c1 = [1, 2, 3, 4]
    c2 = [4, 3, 2, 1]
    print(IoU(c1, c2))

    args = arg_parse()
    if (args.input == ""):
        raise Exception("No --input specified")
        
