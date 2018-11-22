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
import os

def downsample(image, width, height):
    scaledImage = cv.resize(image, (width, height), \
        interpolation=cv.INTER_CUBIC)
    return scaledImage

def grayscale(image):
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return grayImage

L = ['EZG', 'PBK', 'PCB', 'TSF', 'CBB']
R = [16, 32, 64, 128, 256, 512]
F = [0, 1, 2, 3]

# L = ['EZG']
# R = [16, 32]
# F = [0, 1, 2, 3]

for subset in L:

    for resolution in R:

        MODEL_WEIGHTS_PATH = 'snapshots/coco_' + subset + '_lg_' + str(
            resolution) + '_hourglass_net.hdf5'
        IMG_WIDTH = resolution
        IMG_HEIGHT = resolution
        CHANNELS = 1  # Toggle 1/3 (LG/HC)
        CLASSES = 1

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1  # Set GPU limit
        K.set_session(
            K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

        hr_mod = model_def(IMG_WIDTH, IMG_HEIGHT, CHANNELS, CLASSES)  # Load model definition
        hr_mod.load_weights(MODEL_WEIGHTS_PATH)
        hr_mod.compile(optimizer=RMSprop(lr=1e-4), loss='mse')  # mean square error

        for f in F:

            INPUT_IMG_PATH = subset + '_' + str(f) + '.jpg'
            curr_image = Image.open(INPUT_IMG_PATH, 'r')
            curr_image = np.asarray(curr_image).astype(np.float32) # NumPy array
            curr_image = downsample(grayscale(curr_image), IMG_WIDTH, IMG_HEIGHT)
            curr_image /= 255.  # scale to [0, 1] since this is what the model has been trained on
            tmp = np.zeros((1, IMG_WIDTH, IMG_HEIGHT, CHANNELS))
            tmp[0, :, :, 0] = curr_image # Use this for grayscale

            start_time = timeit.default_timer()
            prediction = hr_mod.predict(tmp)
            elapsed = timeit.default_timer() - start_time
            prediction = np.array(prediction)
            prediction = prediction[0, :, :, 0]

            try:
                DATA_PATH = 'preds/' + subset + '_' +  str(f) + '_' + str(resolution) + '/'
                os.mkdir(DATA_PATH) # create data folders
            except:
                print('Folders already exist!')
                pass

            for t in range(1, 255):

                pred_new = np.array(prediction > (float(t) / 255.0), dtype=float) # Enable for binary mask
                plt.imsave(DATA_PATH + subset + '_' +  str(f) + '_' + str(resolution) + '_' + str(t) + '.png', pred_new, cmap='gray') # Save predicted output

            print(subset + '_' +  str(f) + '_' + str(resolution) + ' done!')
