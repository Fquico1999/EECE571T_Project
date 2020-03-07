#!/usr/bin/python

import os
import numpy as np
import cv2
from tensorflow.keras import models
import argparse
from tqdm import tqdm
import json

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

    padding = config['padding']

    padless_win_w = scaled_win_w-2*padding
    padless_win_h = scaled_win_h-2*padding

    num_win_x = IMG_W/win_w
    num_win_y = IMG_H/win_h

    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--images', type=str, required=True, help='path to image dataset')
    ap.add_argument('-mp', '--model_path', type=str, required=True, help='path of model')
    ap.add_argument('-o', '--output_folder', type=str, required=False, default='./artifactless', help='Path to filtered image output folder')
    args = vars(ap.parse_args())

    model = models.load_model(args['model_path'])

    img_names = os.listdir(args['images'])
    

    #Data Collection
    for img_name in tqdm(img_names, desc='Removing Artifacts'):
        X = []

        #Obtain image path
        img_path = os.path.join(args['images'], img_name)

        #Read image
        img = cv2.imread(img_path)

        #Ensure image size is IMG_WxIMG_H
        if img.shape[0:2] != (IMG_W, IMG_H):
            img = cv2.resize(img, (IMG_W, IMG_H))

        #Create image patches
        cols = np.arange(0, num_win_x*win_w, win_w)
        rows = np.arange(0, num_win_y*win_h, win_h)
        for row in rows:
            for col in cols:
                win_img = img[col:col+win_w,row:row+win_w,:]/255.0

                padless_win_img = cv2.resize(win_img, (padless_win_w, padless_win_h))
                scaled_win_img = cv2.copyMakeBorder( padless_win_img, padding, padding, padding, padding, cv2.BORDER_REPLICATE)
        
                X.append(scaled_win_img)
    
        X = np.asarray(X)

        #Get y patches
        y = model.predict(X)

        #Create mask
        mask = np.zeros((num_win_x*padless_win_w, num_win_y*padless_win_h))

        cols = np.arange(0, num_win_x*padless_win_w, padless_win_w)
        rows = np.arange(0, num_win_y*padless_win_h, padless_win_h)

        for i in range(len(rows)):
            for j in range(len(cols)):
                patch = y[(i*6)+j][padding:-padding,padding:-padding] #remove padding
                patch = np.reshape(patch, (padless_win_w, padless_win_h)) #remove last dimension

                mask[cols[j]:cols[j]+padless_win_h,rows[i]:rows[i]+padless_win_w] = patch*255

        resized_mask = cv2.resize(mask, (IMG_W, IMG_H))

        int_mask = np.uint8(resized_mask)

        ret, thresh = cv2.threshold(int_mask, 0.15*np.max(int_mask), 255, cv2.THRESH_BINARY)

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        min_size = 500

        img_o = img.copy()
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img_o[output == i + 1,:] = [255,255,255]

        output_path = os.path.join(args['output_folder'],img_name.split('.')[0] + '_artifactless.jpg')
        cv2.imwrite(output_path, img_o)
