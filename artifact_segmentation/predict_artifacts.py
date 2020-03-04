#!/usr/bin/python

import os
import numpy as np
import cv2
from tensorflow.keras import models
import argparse
from tqdm import tqdm

IMG_W = 2886
IMG_H = 2886

win_w = 481
win_h = 481

scaled_win_w = 384
scaled_win_h = 384

num_win_x = IMG_W/win_w
num_win_y = IMG_H/win_h

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--images', type=str, required=True, help='path to image dataset')
    ap.add_argument('-mp', '--model_path', type=str, required=True, help='path of model')
    ap.add_argument('-o', '--output_folder', type=str, required=False, default='./predicted_masks', help='Path to mask output folder')
    args = vars(ap.parse_args())

    model = models.load_model(args['model_path'])

    img_names = os.listdir(args['images'])

    #Data Collection
    for img_name in tqdm(img_names, desc='Generating Masks'):
        X = []

        #Obtain image path
        img_path = os.path.join(args['images'], img_name)

        #Read image
        img = cv2.imread(img_path)

        #Create image patches
        x_count = 0
        y_count = 0

        for i in range(num_win_y):
            for j in range(num_win_x):
                win_img = img[y_count*win_h:(y_count+1)*win_h,x_count*win_w:(x_count+1)*win_w,:]/255.0
                
                scaled_win_img = cv2.resize(win_img, (scaled_win_w, scaled_win_h))
                
                X.append(scaled_win_img)
                y_count+=1
            x_count+=1
            y_count=0
    
        X = np.asarray(X)

        #Get y patches
        y = model.predict(X)

        #Create mask
        mask = np.zeros((IMG_W, IMG_H))

        #build output mask
        x_count=0
        y_count=0
        for i in range(num_win_y):
            for j in range(num_win_x):
                patch = y[(i*6)+j]*255
                scaled_patch = cv2.resize(patch, (win_w, win_h)) #scale back up to size
                scaled_patch = np.reshape(scaled_patch, (win_w, win_h)) #remove last dimension

                mask[y_count*win_h:(y_count+1)*win_h,x_count*win_w:(x_count+1)*win_w] = scaled_patch
                y_count+=1
            x_count+=1
            y_count=0

        mask_path = os.path.join(args['output_folder'],img_name.split('.')[0] + '_mask.jpg')
        cv2.imwrite(mask_path, mask)
