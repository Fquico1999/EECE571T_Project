#!/usr/bin/python

import os
import argparse
import numpy as np
import pickle
import cv2
from tqdm import tqdm
from tensorflow.keras import datasets, layers, Model, utils, optimizers, models
import matplotlib.pyplot as plt


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
    ap.add_argument('-e', '--epochs', type=int, required=False, default=3, help='number of epochs to train on')
    ap.add_argument('-b', '--num_batches', type=int, required=False, default=16, help='number of batches to train on')
    ap.add_argument('-mp', '--model_path', type=str, required=False, default=None, help='path of model to train')
    ap.add_argument('-vs', '--validation_split', type=float, required=False, default=0.15, help='Training validation_split')
    args = vars(ap.parse_args())

    img_names = os.listdir(args['images'])
    mask_names = os.listdir(args['masks'])

    X = []
    y = []

    #Data Collection
    for i in tqdm(range(len(img_names)), desc='Collecting Data'):

        #Obtain mask and image paths
        mask_path = os.path.join(args['masks'], mask_names[i])
        img_name = mask_names[i].split('_mask')[0]+'.jpg'
        assert img_name in img_names, 'Expected {} in {}'.format(img_name, args['images'])
        img_path = os.path.join(args['images'], img_name)

        #Read mask and img in
        mask = cv2.imread(mask_path, 0)
        ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        img = cv2.imread(img_path)

        #Create image and mask patches
        x_count = 0
        y_count = 0

        for i in range(num_win_y):
            for j in range(num_win_x):
                win_img = img[y_count*win_h:(y_count+1)*win_h,x_count*win_w:(x_count+1)*win_w,:]/255.0
                win_mask = mask[y_count*win_h:(y_count+1)*win_h,x_count*win_w:(x_count+1)*win_w]/255.0
                
                scaled_win_img = cv2.resize(win_img, (scaled_win_w, scaled_win_h))
                scaled_win_mask = cv2.resize(win_mask, (scaled_win_w, scaled_win_h))
                
                X.append(scaled_win_img)
                y.append(scaled_win_mask)
                y_count+=1
            x_count+=1
            y_count=0
    
    X = np.asarray(X)
    y = np.asarray(y)

    if not args['model_path']:
        print('No model path given, creating new model')

        #Model Definition
        num_channels = 3 #bgr

        inputs = layers.Input((scaled_win_w, scaled_win_h, num_channels))

        c1 = layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = layers.Dropout(0.1)(c1)
        c1 = layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)

        p1 = layers.MaxPooling2D((2,2))(c1)
        c2 = layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = layers.Dropout(0.1)(c2)
        c2 = layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)

        p2 = layers.MaxPooling2D((2,2))(c2)
        c3 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = layers.Dropout(0.2)(c3)
        c3 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

        p3 = layers.MaxPooling2D((2,2))(c3)
        c4 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = layers.Dropout(0.2)(c4)
        c4 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

        p4 = layers.MaxPooling2D((2,2))(c4)
        c5 = layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = layers.Dropout(0.3)(c5)
        c5 = layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
        u6 = layers.concatenate([u6,c4])
        c6 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = layers.Dropout(0.2)(c6)
        c6 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
        u7 = layers.concatenate([u7,c3])
        c7 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = layers.Dropout(0.2)(c7)
        c7 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
        u8 = layers.concatenate([u8,c2])
        c8 = layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = layers.Dropout(0.1)(c8)
        c8 = layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
        u9 = layers.concatenate([u9,c1])
        c9 = layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = layers.Dropout(0.1)(c9)
        c9 = layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = layers.Conv2D(1,(1,1), activation='sigmoid')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    else:
        print('Model path specified, loading model')

        try:
            model = models.load_model(args['model_path'])
        except Exception as e:
            print(e)

    #reshape y
    if len(y.shape) < 4:
        y = np.expand_dims(y, axis=3)

    #Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    history = model.fit(X, y, epochs=args['epochs'], batch_size=args['num_batches'], validation_split=args['validation_split'])

    model.save('unet_model')

    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)
    ax.plot(history.history['acc'], label='accuracy')
    ax.plot(history.history['val_acc'], label='validation_accuracy')
    ax.set_xlabel('Epoch')
    plt.legend()
    plt.savefig('history.png')
    
    print('Model saved')
