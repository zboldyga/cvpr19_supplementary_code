# Tested in Python 3.5.2
# Authors: shivanthan.yohanandan@rmit.edu.au

from __future__ import division, print_function, absolute_import
import numpy as np
from keras.optimizers import RMSprop
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from model_def_hourglass_lg_shallow_4 import model_def
from PIL import Image, ImageChops
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.io import loadmat
from metrics import AUC_Judd, AUC_Borji, AUC_shuffled, NSS, CC, SIM
import timeit
import tensorflow as tf

def upsample(image, width, height):
    scaledImage = cv.resize(image, (width, height), \
        interpolation=cv.INTER_CUBIC)
    return scaledImage

def iou(a, b): # Compute intersection over union
    overlap = a * b # logical AND
    union = a + b # logical OR
    return overlap.sum() / float(union.sum())

def binarize(a, threshold):
    return a >= threshold

def flatten_2D(a):
    if len(a.shape) > 2: return a[:, :, 0]
    else: return a

def iou_eval(prediction, label, scale, threshold):
    if prediction.shape[0] != scale: scaled_pred = upsample(prediction, scale, scale)
    else: scaled_pred = prediction
    flat_pred = flatten_2D(scaled_pred)
    flat_lbl = flatten_2D(label)
    bin_pred = binarize(flat_pred, threshold)
    bin_lbl = binarize(flat_lbl, 128)
    return iou(bin_pred, bin_lbl)

EVAL_SET = [] # Load held-out test set IDs (~20% of dataset)
RES = [16, 32, 64, 128, 256, 512]
results = np.zeros((len(EVAL_SET), len(RES) + 1))

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1 # Set GPU limit
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

res_count = 1

for r in RES:

    MODEL_WEIGHTS_PATH = str(r) + '_sc-fcn.hdf5'
    IMG_WIDTH = r
    IMG_HEIGHT = r
    CHANNELS = 1
    CLASSES = 1

    hr_mod = model_def(IMG_WIDTH, IMG_HEIGHT, CHANNELS, CLASSES)  # Load model definition
    hr_mod.load_weights(MODEL_WEIGHTS_PATH)
    hr_mod.compile(optimizer=RMSprop(lr=1e-4), loss='mse')  # mean square error

    eval_count = 0

    for img in EVAL_SET:

        IMG_PATH = 'COCO_LG_' + str(r) + '_data/' + str(img).zfill(5) + '.jpg'
        LABEL_PATH = 'COCO_LG_512_labels/' + str(img).zfill(5)+ '.jpg'

        curr_image = Image.open(IMG_PATH, 'r')
        label_image = Image.open(LABEL_PATH, 'r')
        label_image = np.asarray(label_image).astype(np.float32) # NumPy array
        curr_image = np.asarray(curr_image).astype(np.float32) # NumPy array
        curr_image /= 255.  # scale to [0, 1] since this is what the model has been trained on
        tmp = np.zeros((1, IMG_WIDTH, IMG_HEIGHT, CHANNELS))
        tmp[0, :, :, 0] = curr_image # Use this for grayscale

        start_time = timeit.default_timer()
        prediction = hr_mod.predict(tmp)
        elapsed = timeit.default_timer() - start_time
        prediction = np.array(prediction)
        prediction = prediction[0, :, :, 0] # Slice

        # print(r, img, elapsed, iou_eval(prediction, label_image, 512, (84 / 255.0)))

        results[eval_count][0] = img
        # results[eval_count][res_count] = iou_eval(prediction, label_image, 512, (84 / 255.0))
        results[eval_count][res_count] = iou_eval(prediction, label_image, 512, (84 / 255.0))
        eval_count += 1

    res_count += 1

    print(str(r) + ' done!')

for i in range(len(EVAL_SET)):
    print(results[i, 0], results[i, 1], results[i, 2], results[i, 3], results[i, 4], results[i, 5], results[i, 6])
    # print(results[i, 0], results[i, 1], results[i, 2])