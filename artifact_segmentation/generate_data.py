#!/usr/bin/python

import os
import argparse
import numpy as np
import pickle
import cv2
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
    ap.add_argument('-m', '--masks', type=str, required=True, help='path to mask dataset')

    args = vars(ap.parse_args())

    img_names = os.listdir(args['images'])
    mask_names = os.listdir(args['masks'])

    X = []
    y = []

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

    with open('data.pickle','wb') as out_file:
        pickle.dump(np.asarray(X), out_file)
        pickle.dump(np.asarray(y),out_file)
