# Tested in Python 3.5.2
# Authors: shivanthan.yohanandan@rmit.edu.au

from pycocotools.coco import COCO
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.misc import toimage
import numpy as np
import os

def blur(image):
    blurImage = cv.GaussianBlur(image, (5, 5), 0) # 5x5 kernal
    return blurImage

def downsample(image, width, height):
    scaledImage = cv.resize(image, (width, height), \
        interpolation=cv.INTER_CUBIC)
    return scaledImage

def upsample(image, width, height):
    scaledImage = cv.resize(image, (width, height), \
        interpolation=cv.INTER_CUBIC)
    return scaledImage

def grayscale(image):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return grayImage

RESOLUTIONS = [16, 32, 64, 128, 256, 512]
COCO_TRAIN_ANN_PATH = '/coco/coco2017/annotations/instances_train2017.json'
COCO_VAL_ANN_PATH = '/coco/coco2017/annotations/instances_val2017.json'
COCO_TRAIN_DATA_PATH = '/coco/coco2017/train2017/'
COCO_VAL_DATA_PATH = '/coco/coco2017/val2017/'
annFile = COCO_TRAIN_ANN_PATH
coco=COCO(annFile)

A = 'traffic light'
B = 'stop sign'
C = 'fire hydrant'

COCO_FAMILY = 'TSF'

all_catIds = coco.getCatIds(catNms=[A, B, C])
all_imgIds = coco.getImgIds(catIds=all_catIds)

pc_catIds = coco.getCatIds(catNms=[A, B])
pc_imgIds = coco.getImgIds(catIds=pc_catIds)

pd_catIds = coco.getCatIds(catNms=[A, C])
pd_imgIds = coco.getImgIds(catIds=pd_catIds)

cd_catIds = coco.getCatIds(catNms=[B, C])
cd_imgIds = coco.getImgIds(catIds=cd_catIds)

p_catIds = coco.getCatIds(catNms=[A])
p_imgIds = coco.getImgIds(catIds=p_catIds)

c_catIds = coco.getCatIds(catNms=[B])
c_imgIds = coco.getImgIds(catIds=c_catIds)

d_catIds = coco.getCatIds(catNms=[C])
d_imgIds = coco.getImgIds(catIds=d_catIds)

d = {} # Dict for storing unique imgIds
gbl_idx = 0

print(len(all_imgIds), len(pc_imgIds), len(cd_imgIds), len(p_imgIds), len(c_imgIds), len(d_imgIds))

try:

    for res in RESOLUTIONS:
        DATA_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_data/'
        LABEL_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_labels/'
        # create data and label folders
        os.mkdir(DATA_PATH)
        os.mkdir(LABEL_PATH)

except:
    print('Folders already exist!')
    pass

for i in range(len(all_imgIds)):
    img = coco.loadImgs(all_imgIds[i])[0]
    if img['file_name'] in d: d[img['file_name']] += 1
    else:
        d[img['file_name']] = 1
        orig = cv.imread(COCO_TRAIN_DATA_PATH + img['file_name']) # load original image
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=all_catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = coco.annToMask(anns[0])
        for j in range(len(anns)):
            mask += coco.annToMask(anns[j])

        O = (mask > 0).astype(int)
        I = np.asarray(toimage(O))

        for res in RESOLUTIONS:
            DOWNSAMPLE_HEIGHT = res
            DOWNSAMPLE_WIDTH = res
            DATA_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_data/'
            LABEL_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_labels/'
            orig_lg = downsample(blur(grayscale(orig)), DOWNSAMPLE_WIDTH, DOWNSAMPLE_HEIGHT)
            cv.imwrite(DATA_PATH + str(gbl_idx).zfill(5) + '.jpg', orig_lg)
            O_lr = downsample(I, DOWNSAMPLE_HEIGHT, DOWNSAMPLE_WIDTH)
            cv.imwrite(LABEL_PATH + str(gbl_idx).zfill(5) + '.jpg', O_lr)

        print(gbl_idx)
        gbl_idx += 1

for i in range(len(pc_imgIds)):
    img = coco.loadImgs(pc_imgIds[i])[0]
    if img['file_name'] in d:
        d[img['file_name']] += 1
    else:
        d[img['file_name']] = 1
        orig = cv.imread(COCO_TRAIN_DATA_PATH + img['file_name'])  # load original image
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=pc_catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = coco.annToMask(anns[0])
        for j in range(len(anns)):
            mask += coco.annToMask(anns[j])

        O = (mask > 0).astype(int)
        I = np.asarray(toimage(O))

        for res in RESOLUTIONS:
            DOWNSAMPLE_HEIGHT = res
            DOWNSAMPLE_WIDTH = res
            DATA_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_data/'
            LABEL_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_labels/'
            orig_lg = downsample(blur(grayscale(orig)), DOWNSAMPLE_WIDTH, DOWNSAMPLE_HEIGHT)
            cv.imwrite(DATA_PATH + str(gbl_idx).zfill(5) + '.jpg', orig_lg)
            O_lr = downsample(I, DOWNSAMPLE_HEIGHT, DOWNSAMPLE_WIDTH)
            cv.imwrite(LABEL_PATH + str(gbl_idx).zfill(5) + '.jpg', O_lr)

        print(gbl_idx)
        gbl_idx += 1

for i in range(len(pd_imgIds)):
    img = coco.loadImgs(pd_imgIds[i])[0]
    if img['file_name'] in d:
        d[img['file_name']] += 1
    else:
        d[img['file_name']] = 1
        orig = cv.imread(COCO_TRAIN_DATA_PATH + img['file_name'])  # load original image
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=pd_catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = coco.annToMask(anns[0])
        for j in range(len(anns)):
            mask += coco.annToMask(anns[j])

        O = (mask > 0).astype(int)
        I = np.asarray(toimage(O))

        for res in RESOLUTIONS:
            DOWNSAMPLE_HEIGHT = res
            DOWNSAMPLE_WIDTH = res
            DATA_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_data/'
            LABEL_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_labels/'
            orig_lg = downsample(blur(grayscale(orig)), DOWNSAMPLE_WIDTH, DOWNSAMPLE_HEIGHT)
            cv.imwrite(DATA_PATH + str(gbl_idx).zfill(5) + '.jpg', orig_lg)
            O_lr = downsample(I, DOWNSAMPLE_HEIGHT, DOWNSAMPLE_WIDTH)
            cv.imwrite(LABEL_PATH + str(gbl_idx).zfill(5) + '.jpg', O_lr)

        print(gbl_idx)
        gbl_idx += 1

for i in range(len(cd_imgIds)):
    img = coco.loadImgs(cd_imgIds[i])[0]
    if img['file_name'] in d:
        d[img['file_name']] += 1
    else:
        d[img['file_name']] = 1
        orig = cv.imread(COCO_TRAIN_DATA_PATH + img['file_name'])  # load original image
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=cd_catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = coco.annToMask(anns[0])
        for j in range(len(anns)):
            mask += coco.annToMask(anns[j])

        O = (mask > 0).astype(int)
        I = np.asarray(toimage(O))

        for res in RESOLUTIONS:
            DOWNSAMPLE_HEIGHT = res
            DOWNSAMPLE_WIDTH = res
            DATA_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_data/'
            LABEL_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_labels/'
            orig_lg = downsample(blur(grayscale(orig)), DOWNSAMPLE_WIDTH, DOWNSAMPLE_HEIGHT)
            cv.imwrite(DATA_PATH + str(gbl_idx).zfill(5) + '.jpg', orig_lg)
            O_lr = downsample(I, DOWNSAMPLE_HEIGHT, DOWNSAMPLE_WIDTH)
            cv.imwrite(LABEL_PATH + str(gbl_idx).zfill(5) + '.jpg', O_lr)

        print(gbl_idx)
        gbl_idx += 1

for i in range(len(p_imgIds)):
    img = coco.loadImgs(p_imgIds[i])[0]
    if img['file_name'] in d:
        d[img['file_name']] += 1
    else:
        d[img['file_name']] = 1
        orig = cv.imread(COCO_TRAIN_DATA_PATH + img['file_name'])  # load original image
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=p_catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = coco.annToMask(anns[0])
        for j in range(len(anns)):
            mask += coco.annToMask(anns[j])

        O = (mask > 0).astype(int)
        I = np.asarray(toimage(O))

        for res in RESOLUTIONS:
            DOWNSAMPLE_HEIGHT = res
            DOWNSAMPLE_WIDTH = res
            DATA_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_data/'
            LABEL_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_labels/'
            orig_lg = downsample(blur(grayscale(orig)), DOWNSAMPLE_WIDTH, DOWNSAMPLE_HEIGHT)
            cv.imwrite(DATA_PATH + str(gbl_idx).zfill(5) + '.jpg', orig_lg)
            O_lr = downsample(I, DOWNSAMPLE_HEIGHT, DOWNSAMPLE_WIDTH)
            cv.imwrite(LABEL_PATH + str(gbl_idx).zfill(5) + '.jpg', O_lr)

        print(gbl_idx)
        gbl_idx += 1

for i in range(len(c_imgIds)):
    img = coco.loadImgs(c_imgIds[i])[0]
    if img['file_name'] in d:
        d[img['file_name']] += 1
    else:
        d[img['file_name']] = 1
        orig = cv.imread(COCO_TRAIN_DATA_PATH + img['file_name'])  # load original image
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=c_catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = coco.annToMask(anns[0])
        for j in range(len(anns)):
            mask += coco.annToMask(anns[j])

        O = (mask > 0).astype(int)
        I = np.asarray(toimage(O))

        for res in RESOLUTIONS:
            DOWNSAMPLE_HEIGHT = res
            DOWNSAMPLE_WIDTH = res
            DATA_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_data/'
            LABEL_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_labels/'
            orig_lg = downsample(blur(grayscale(orig)), DOWNSAMPLE_WIDTH, DOWNSAMPLE_HEIGHT)
            cv.imwrite(DATA_PATH + str(gbl_idx).zfill(5) + '.jpg', orig_lg)
            O_lr = downsample(I, DOWNSAMPLE_HEIGHT, DOWNSAMPLE_WIDTH)
            cv.imwrite(LABEL_PATH + str(gbl_idx).zfill(5) + '.jpg', O_lr)

        print(gbl_idx)
        gbl_idx += 1

for i in range(len(d_imgIds)):
    img = coco.loadImgs(d_imgIds[i])[0]
    if img['file_name'] in d:
        d[img['file_name']] += 1
    else:
        d[img['file_name']] = 1
        orig = cv.imread(COCO_TRAIN_DATA_PATH + img['file_name'])  # load original image
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=d_catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = coco.annToMask(anns[0])
        for j in range(len(anns)):
            mask += coco.annToMask(anns[j])

        O = (mask > 0).astype(int)
        I = np.asarray(toimage(O))

        for res in RESOLUTIONS:
            DOWNSAMPLE_HEIGHT = res
            DOWNSAMPLE_WIDTH = res
            DATA_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_data/'
            LABEL_PATH = COCO_FAMILY + '_COCO_LG_' + str(res) + '_labels/'
            orig_lg = downsample(blur(grayscale(orig)), DOWNSAMPLE_WIDTH, DOWNSAMPLE_HEIGHT)
            cv.imwrite(DATA_PATH + str(gbl_idx).zfill(5) + '.jpg', orig_lg)
            O_lr = downsample(I, DOWNSAMPLE_HEIGHT, DOWNSAMPLE_WIDTH)
            cv.imwrite(LABEL_PATH + str(gbl_idx).zfill(5) + '.jpg', O_lr)

        print(gbl_idx)
        gbl_idx += 1

print('###############################################################')
print('###############################################################')
print('###############################################################')
# for key in d:
#     print(key, d[key])