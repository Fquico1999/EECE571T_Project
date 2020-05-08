#!/usr/bin/python

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models
import argparse
from tqdm import tqdm
import json

def post_proc(mask, std):
    ret, thresh = cv2.threshold(mask, std, 255, cv2.THRESH_BINARY)
    
    #Close gaps in mask and dilate
    kernel_c = np.ones((2,2),np.uint8)
    kernel_d = np.ones((5,5),np.uint8)

    mask_proc = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_c)
    mask_proc = cv2.dilate(mask_proc,kernel_d,iterations = 1)
    return mask_proc


if __name__ == '__main__':

    #Read configuration file
    with open('config.json') as f:
        config = json.load(f)

    IMG_W = config['image_width']
    IMG_H = config['image_height']

    win_w = config['window_width']
    win_h = config['window_height']

    padding = config['padding']

    padless_win_w = win_w-2*padding
    padless_win_h = win_h-2*padding

    num_win_x = int(IMG_W/win_w)
    num_win_y = int(IMG_H/win_h)

    std_thresh = config['std_thresh']

    num_win_x = int(IMG_W/win_w)
    num_win_y = int(IMG_H/win_h)

    #Ensure Image size divides evenly by win_w and win_h
    scaled_img_w = win_w*num_win_x
    scaled_img_h = win_h*num_win_y

    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--images', type=str, required=True, help='path to image dataset')
    ap.add_argument('-mp', '--model_path', type=str, required=True, help='path of model')
    ap.add_argument('-o', '--output_folder', type=str, required=False, default='./markers', help='Path to filtered image output folder')
    ap.add_argument('-c','--color', type=list, required=False, default=[0,0,255], help='Marker label color in BGR ([B,G,R])')
    ap.add_argument('-m','--output_masks',type=bool, required=False, default=False, help='Whether to also output mask')
    args = vars(ap.parse_args())

    #Load model in with custom loss function
    try:
        model = models.load_model(args['model_path'], compile=False) 
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=dice_loss, metrics=[dice_loss,'accuracy'])

    except Exception as e:
        print(e)

    img_names = os.listdir(args['images'])
    
    #Data Collection
    for img_name in tqdm(img_names, desc='Segmenting Markers'):
        X = []

        #Obtain image path
        img_path = os.path.join(args['images'], img_name)

        #Read image
        img = cv2.imread(img_path)

        #Resize image to ensure integer number of patches fit within
        img_scaled = cv2.resize(img, (scaled_img_w, scaled_img_h))

        #Create image patches
        cols = np.arange(0, num_win_x*win_w, win_w)
        rows = np.arange(0, num_win_y*win_h, win_h)
        for row in rows:
            for col in cols:
                win_img = img_scaled[col:col+win_w,row:row+win_w,:]/255.0
                
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

        #Get scaled standard deviation of predictions
        std = np.std(y)*255.0

        for i in range(len(rows)):
            for j in range(len(cols)):
                patch = y[(i*num_win_x)+j][padding:-padding,padding:-padding] #Remove padding
                patch = np.reshape(patch, (padless_win_w, padless_win_h)) #remove last dimension
                patch = post_proc(patch*255, max(std_thresh, std))
                mask[cols[j]:cols[j]+padless_win_w,rows[i]:rows[i]+padless_win_h] = patch

        resized_mask = cv2.resize(mask, (IMG_W, IMG_H))

        #Make output image copy
        img_o = img.copy()
        idx = np.where(resized_mask == 255)
        img_o[idx] = args['color']

        if args['output_masks']:
            output_path = os.path.join(args['output_folder'],img_name.split('.')[0] + '_mask.jpg')
            cv2.imwrite(output_path, resized_mask)

        output_path = os.path.join(args['output_folder'],img_name.split('.')[0] + '_labelled.jpg')
        cv2.imwrite(output_path, img_o)
