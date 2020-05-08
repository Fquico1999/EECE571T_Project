#!/usr/bin/python

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow.keras import datasets, layers, Model, utils, optimizers, models
from tensorflow.keras.layers import Conv2D,concatenate, Activation,UpSampling2D, BatchNormalization, MaxPooling2D, Dropout,Conv2DTranspose
import json
from tqdm import tqdm
import pickle


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

  return 1 - (numerator + 1) / (denominator + 1)

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

    # Expansive Path
    #u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = UpSampling2D((2,2), interpolation='bilinear')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

    #u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = UpSampling2D((2,2), interpolation='bilinear')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    #u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = UpSampling2D((2,2), interpolation='bilinear')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

    #u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = UpSampling2D((2,2), interpolation='bilinear')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

if __name__ == '__main__':

    #Read configuration file
    with open('config.json') as f:
        config = json.load(f)

    IMG_W = config['image_width']
    IMG_H = config['image_height']

    win_w = config['window_width']
    win_h = config['window_height']

    scaled_win_w = config['model_input_width']
    scaled_win_h = config['model_input_height']

    num_win_x = IMG_W/win_w
    num_win_y = IMG_H/win_h

    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--images', type=str, required=True, help='path to image dataset')
    ap.add_argument('-m', '--masks', type=str, required=True, help='path to mask dataset')
    ap.add_argument('-c','--collect_data', type=str2bool, required=True, help='Whether to create dataset or use existing data.')
    ap.add_argument('-e', '--epochs', type=int, required=False, default=50, help='number of epochs to train on')
    ap.add_argument('-b', '--num_batches', type=int, required=False, default=16, help='number of batches to train on')
    ap.add_argument('-mp', '--model_path', type=str, required=False, default=None, help='path of model to train')
    ap.add_argument('-vs', '--validation_split', type=float, required=False, default=0.15, help='Training validation_split')
    ap.add_argument('-ri', '--rename_images', type=str2bool, required=False, default=False, help='Whether to rename images to lowercase for mask match')
    args = vars(ap.parse_args())

    print(args['collect_data'])
    if args['collect_data']:

        img_names = os.listdir(args['images'])
        mask_names = os.listdir(args['masks'])

        X = []
        y = []

        #Because of annotation software, there may be a mismatch in naming
        if args['rename_images']:
            for img_name in img_names:
                img_path_curr = os.path.join(args['images'], img_name)
                img_path_lower = os.path.join(args['images'], img_name.lower())
                os.rename(img_path_curr, img_path_lower)
            #After renaming, get img_names again
            img_names = os.listdir(args['images'])

        #Data Collection
        for i in tqdm(range(len(img_names)), desc='Collecting Data'):

            #Obtain mask and image paths
            mask_path = os.path.join(args['masks'], mask_names[i])
            img_name = mask_names[i].split('_mask')[0]+'.jpg'
            assert img_name in img_names, 'Expected {} in {}, if image filename is uppercase, use --ri = True'.format(img_name, args['images'])
            img_path = os.path.join(args['images'], img_name)

            #Read mask and img in
            mask = cv2.imread(mask_path, 0)
            ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            img = cv2.imread(img_path)
            scaled_img = cv2.resize(img, (128, 128))/255.0
            scaled_mask = cv2.resize(thresh, (128, 128))/255.0

            cv2.imshow('Image', scaled_img)
            cv2.waitKey(300)
            cv2.imshow('Mask', scaled_mask)
            cv2.waitKey(300)

            X.append(scaled_img)
            y.append(scaled_mask)
        
        cv2.destroyAllWindows()
        X = np.asarray(X)
        y = np.asarray(y)

        #reshape y
        if len(y.shape) < 4:
            y = np.expand_dims(y, axis=3)

        #Shuffle
        idx = np.random.permutation(len(X))

        #Split into train and test
        #Do this instead of validation split because if we want to train further
        #we make sure we dont use the same testing data for training
        split = int(args['validation_split']*len(X))
        X_train = X[:-split]
        y_train = y[:-split]

        X_test = X[-split:]
        y_test = y[-split:]

        print('Saving Data\n')
        pickle.dump(X_train, open("X_train", "wb"))
        pickle.dump(y_train, open("y_train", "wb"))

        pickle.dump(X_test, open("X_test", "wb"))
        pickle.dump(y_test, open("y_test", "wb"))
        print('Data Saved\n\n')

    else:
        print('Loading Data\n')
        X_train = pickle.load( open( "X_train", "rb" ) )
        y_train = pickle.load( open( "y_train", "rb" ) )

        X_test = pickle.load( open( "X_test", "rb" ) )
        y_test = pickle.load( open( "y_test", "rb" ) )

        print('Data Loaded\n\n')

    print('Training on %i samples\nTesting on %i samples\n\n' %(len(X_train), len(X_test)))

    if not args['model_path']:
        print('No model path given, creating new model')

        #Model Definition
        num_channels = 3 #bgr

        inputs = layers.Input((128, 128, num_channels))

        model = get_unet(inputs)

        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=dice_loss, metrics=[dice_loss,'accuracy'])

    else:
        print('Model path specified, loading model')

        try:
            model = models.load_model(args['model_path'], compile=False) 
            model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss=dice_loss, metrics=[dice_loss,'accuracy'])

        except Exception as e:
            print(e)

    history = model.fit(X_train, y_train, validation_data=[X_test,y_test], epochs=args['epochs'], batch_size=args['num_batches'])

    model.save('unet_model')

    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)
    ax.plot(history.history['acc'], label='accuracy')
    ax.plot(history.history['val_acc'], label='validation_accuracy')
    ax.plot(history.history['dice_loss'], label='dice_loss')
    ax.plot(history.history['val_dice_loss'], label='validation_dice_loss')
    ax.set_xlabel('Epoch')
    plt.legend()
    plt.savefig('history.png')
    
    print('Model saved')
