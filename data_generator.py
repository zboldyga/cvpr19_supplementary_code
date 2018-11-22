# Tested in Python 3.5.2
# Authors: shivanthan.yohanandan@rmit.edu.au
# Source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

import numpy as np
from PIL import Image
import cv2 as cv
from scipy import misc, ndimage
import matplotlib.pyplot as plt

class DataGenerator(object):
  'Generates data for Keras'
  def __init__(self, dim_x = 32, dim_y = 32, lbl_x = 32, lbl_y = 32,
               batch_size = 32, classes = 2, channels = 3, shuffle = False,
               data_path = '', label_path = ''):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.lbl_x = lbl_x
      self.lbl_y = lbl_y
      self.batch_size = batch_size
      self.classes = classes
      self.channels = channels
      self.shuffle = shuffle
      self.data_path = data_path
      self.label_path = label_path

  def generate(self, labels, list_IDs):
      'Generates batches of samples'
      # Infinite loop
      while 1:
          # Generate order of exploration of dataset
          indexes = self.__get_exploration_order(list_IDs)

          # Generate batches
          imax = int(len(indexes)/self.batch_size)
          for i in range(imax):
              # Find list of IDs
              list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

              # Generate data
              X, y = self.__data_generation(labels, list_IDs_temp)

              yield X, y

  def __get_exploration_order(self, list_IDs):
      'Generates order of exploration'
      # Find exploration order
      indexes = np.arange(len(list_IDs))
      if self.shuffle == True:
          np.random.shuffle(indexes)

      return indexes

  def __data_generation(self, labels, list_IDs_temp):
      'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
      # Initialization
      X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.channels))
      y = np.empty((self.batch_size, self.lbl_x, self.lbl_y, self.classes), dtype = float)

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          # Store volume
          filename = self.data_path + str(ID).zfill(5) + '.jpg' # png/jpg and zfill=4/5
          curr_image = cv.imread(filename)
          curr_image = np.asarray(curr_image).astype(np.float32)
          curr_image /= 255  # scale to [0, 1]
          # print(curr_image.shape)
          curr_image = curr_image[:, :, 0:self.channels]
          X[i] = curr_image

          # Store class
          tmp = np.zeros((self.lbl_x, self.lbl_y, self.classes))
          label_filename = self.label_path + str(ID).zfill(5) + '.jpg' # png/jpg and zfill=4/5
          label_image = Image.open(label_filename, 'r')
          label_image = np.asarray(label_image).astype(np.float32)
          # label_image = label_image[:, :, 0]
          label_image /= 255  # Standardise (scale) to [0, 1]
          tmp[:, :, 0] = label_image
          y[i] = tmp

      return X, y