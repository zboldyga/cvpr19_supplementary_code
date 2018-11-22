# Tested in Python 3.5.2
# Authors: shivanthan.yohanandan@rmit.edu.au
# Source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

from __future__ import division, print_function, absolute_import
import os
import numpy as np
import random

# Seed the random generators for reproducable results
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# os.environ['PYTHONHASHSEED'] = '0'
# np.random.seed(0)
random.seed(0)
import tensorflow as tf
# tf.set_random_seed(0)

from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint  # Required for saving model states
from keras.optimizers import RMSprop
from model_def_hourglass_lg_shallow_4 import model_def
from data_generator import DataGenerator
import sys
import csv
import time

# # Load config info
# if len(sys.argv) < 2:
#     # Need config file path
#     exit('Error: please specify config file path as argument.')
#
# config_file = sys.argv[1]

config_file = '003.cfg'

L = []
with open(config_file, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        L.append(row)

#####################################
########## HYPERPARAMETERS ##########
#####################################

SNAPSHOT_PATH = L[0][1]
DATA_PATH = L[1][1]
LABELS_PATH = L[2][1]
DATASET_SIZE = int(L[3][1])
TRAIN_VAL_SPLIT = float(L[4][1])
CLASSES = int(L[5][1])
INPUT_IMG_SIZE = int(L[6][1])
LABEL_SIZE = int(L[7][1])
CHANNELS = int(L[8][1])
BATCH_SIZE = int(L[9][1])
EPOCHS = int(L[10][1])
GPU_MEM_FRAC = float(L[11][1])
VAL_SIZE = int(DATASET_SIZE * TRAIN_VAL_SPLIT)
TRAIN_SIZE = DATASET_SIZE - VAL_SIZE
STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE # Steps

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRAC # Set GPU limit
set_session(tf.Session(config=config))

###############################
########## LOAD DATA ##########
###############################

def load_labels():
    labels = []
    for key in range(DATASET_SIZE):
        labels.append(key)
    return labels

t0 = time.time()

labels = load_labels()
random.shuffle(labels) # Random shuffle to ensure representative distribution of categories
training_data = labels[0:TRAIN_SIZE]
validation_data = labels[TRAIN_SIZE:DATASET_SIZE]

for i in validation_data:
    print(i)

#######################################
########## PREPARE GENERATOR ##########
#######################################

# Generator parameters
params = {'dim_x': INPUT_IMG_SIZE,
          'dim_y': INPUT_IMG_SIZE,
          'lbl_x': LABEL_SIZE,
          'lbl_y': LABEL_SIZE,
          'batch_size': BATCH_SIZE,
          'classes': CLASSES,
          'channels': CHANNELS,
          'shuffle': False,
          'data_path': DATA_PATH,
          'label_path': LABELS_PATH}

# Generators
training_generator = DataGenerator(**params).generate(labels, training_data)
validation_generator = DataGenerator(**params).generate(labels, validation_data)

#################################
########## TRAIN MODEL ##########
#################################

# checkpoint (model snapshots)
checkpoint = ModelCheckpoint(SNAPSHOT_PATH, monitor='loss', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
hr_mod = model_def(INPUT_IMG_SIZE, INPUT_IMG_SIZE, CHANNELS, CLASSES) # Load model definition
hr_mod.compile(optimizer=RMSprop(lr=1e-4), loss='mse') # mean squared error loss function
hr_mod.fit_generator(generator = training_generator,
                    steps_per_epoch = STEPS_PER_EPOCH,
                    nb_epoch = EPOCHS,
                    validation_data = validation_generator,
                    shuffle=False,
                    callbacks=callbacks_list,
                    validation_steps = len(validation_data) // BATCH_SIZE) # Integer div by BATCH_SIZE

t1 = time.time()

total = t1-t0

print(total)
