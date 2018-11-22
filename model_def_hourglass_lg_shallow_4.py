# Tested in Python 3.5.2
# Authors: shivanthan.yohanandan@rmit.edu.au

from keras.layers import Input, Activation, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, BatchNormalization, concatenate
from keras.models import Model
import keras

#########################################
########## SC-RPN ARCHITECTURE ##########
#########################################

def model_def(input_img_width, input_img_height, n_channels, n_classes):

    input_img1 = Input(shape=(input_img_width, input_img_height, n_channels))
    # e.g. 64

    ae1_down0 = Conv2D(32, (3, 3), padding='same')(input_img1)
    ae1_down0 = BatchNormalization()(ae1_down0)
    ae1_down0 = Activation('relu')(ae1_down0)
    ae1_down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(ae1_down0)
    # e.g. 32

    ae1_down1 = Conv2D(32, (3, 3), padding='same')(ae1_down0_pool)
    ae1_down1 = BatchNormalization()(ae1_down1)
    ae1_down1 = Activation('relu')(ae1_down1)
    ae1_down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(ae1_down1)
    # e.g. 16

    ae1_down2 = Conv2D(32, (3, 3), padding='same')(ae1_down1_pool)
    ae1_down2 = BatchNormalization()(ae1_down2)
    ae1_down2 = Activation('relu')(ae1_down2)
    ae1_down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(ae1_down2)
    # e.g. 8

    center = Conv2D(32, (3, 3), padding='same')(ae1_down2_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # e.g. 8 (center)

    ae1_up2 = UpSampling2D((2, 2))(center)
    ae1_up2 = Conv2D(32, (3, 3), padding='same')(ae1_up2)
    ae1_up2 = BatchNormalization()(ae1_up2)
    ae1_up2 = Activation('relu')(ae1_up2)
    # e.g. 16

    ae1_up1 = UpSampling2D((2, 2))(ae1_up2)
    ae1_up1 = Conv2D(32, (3, 3), padding='same')(ae1_up1)
    ae1_up1 = BatchNormalization()(ae1_up1)
    ae1_up1 = Activation('relu')(ae1_up1)
    # e.g. 32

    ae1_up0 = UpSampling2D((2, 2))(ae1_up1)
    ae1_up0 = Conv2D(32, (3, 3), padding='same')(ae1_up0)
    ae1_up0 = BatchNormalization()(ae1_up0)
    ae1_up0 = Activation('relu')(ae1_up0)
    # e.g. 64

    ae1_decoded = Conv2D(n_classes, (1, 1), activation='sigmoid', padding='same')(ae1_up0)
    stackedModel = Model(input_img1, [ae1_decoded])
    print(stackedModel.summary())

    return stackedModel