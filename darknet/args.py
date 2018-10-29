import argparse

def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--vot", dest='vot', help="Image / Directory containing vot dataset to perform detection upon",
                        default="", type=str)
    parser.add_argument("--pp", dest='pp', help="Postprocessing method",
                        default="none", type=str)
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
    parser.add_argument("--saveto", dest='saveto', help="Save results to the chosen file",
                        default="", type=str)
    parser.add_argument("--silent", dest='silent', help="[all] - Run in silent mode",
                        default="", type=str)
    parser.add_argument("--cuda", dest='cuda', help="cuda [0-9] select a cuda device",
                        default="0", type=str)

    return parser.parse_args()