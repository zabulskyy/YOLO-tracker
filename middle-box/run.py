import os
import os.path as osp
from PIL import Image


def run(vot_path, percentile=3):
    percentile = 1 / percentile
    res = dict()
    folders = sorted(os.listdir(vot_path))
    for folder in folders:
        if (folder.endswith(".txt")):
            continue
        li = sorted(os.listdir(osp.join(vot_path, folder)))
        for n, image in enumerate(li):
            if not image.endswith(".jpg"):
                continue
            im = Image.open(osp.join(vot_path, folder, image))
            width, height = im.size
            bbox = [percentile * width, percentile * height,
                    (1-percentile)*width,  (1-percentile)*height]
            try:
                res[folder].append(bbox)
            except:
                res[folder] = [bbox]
    return res


def save(res, folder):
    os.chdir(folder)
    for name in res:
        print("\n".join([",".join([str(a) for a in x])
                         for x in res[name]]), file=open(name + ".txt", 'w+'))


if __name__ == "__main__":
    res = run("/home/zabulskyy/Datasets/vot2016", 3)
    save(res, "results/middle-box")
