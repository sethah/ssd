import numpy as np
from pathlib import Path
from matplotlib import patches, patheffects
import matplotlib.pyplot as plt
import json

from PIL import Image

DATA_DIR = Path("../data/")
train_ids = DATA_DIR/'processed'/'PascalVOC'/'train'/'VOCDevkit'/'VOC2007'/'ImageSets'/'Layout'/'train.txt'
valid_ids = DATA_DIR/'processed'/'PascalVOC'/'train'/'VOCDevkit'/'VOC2007'/'ImageSets'/'Layout'/'val.txt'
jpgs = DATA_DIR/'processed'/'PascalVOC'/'train'/'VOCDevkit'/'VOC2007'/'JPEGImages'
json_paths = {'train': DATA_DIR/'processed'/'PascalVOC'/'train'/'pascal_train2007.json',
         'valid': DATA_DIR/'processed'/'PascalVOC'/'train'/'pascal_val2007.json'}


def get_jpgs(id_file):
    jpgs = []
    with open(id_file) as f:
        for l in f.readlines():
            jpgs.append(l.strip() + ".jpg")
    return jpgs


def draw_rec(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)


def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


def hw_bb(bb):
    return np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1])


def show_img(img_id, annos):
    fig, ax = plt.subplots()
    im = Image.open(imgs[img_id])
    bboxs = [anno['bbox'] for anno in annos[img_id]]
    ax.imshow(im)
    for bbox in bboxs:
        draw_rec(ax, bbox)
    im.close()
