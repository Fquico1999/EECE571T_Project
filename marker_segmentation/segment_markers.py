#!/usr/bin/python

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models
import argparse
from tqdm import tqdm
import json
import pandas as pd

def post_proc(mask, std):
    ret, thresh = cv2.threshold(mask, std, 1, cv2.THRESH_BINARY)
    
    #Close gaps in mask and dilate
    kernel_c = np.ones((2,2),np.uint8)
    kernel_d = np.ones((5,5),np.uint8)

    mask_proc = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_c)
    mask_proc = cv2.dilate(mask_proc,kernel_d,iterations = 1)
    return mask_proc

def marker_ratio(img, marker_mask):
    
    #Filter background, obtain mask of just tma
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0 #Normalize
    img_inv = 1.0 - img_gray #Invert
    ret, thresh = cv2.threshold(img_inv, np.std(img_inv), 1.0, cv2.THRESH_BINARY)
    kernel_c = np.ones((25,25),np.uint8)
    background_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_c)
    background_count = np.sum(background_mask)

    #Get marker count
    assert np.max(marker_mask) <= 1.0 , "Expected marker_mask to be under 1.0 but got %f" %(np.max(marker_mask))
    assert img.shape[:2] == marker_mask.shape
    marker_count = np.sum(marker_mask)

    return marker_count/background_count

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
    ap.add_argument('-of', '--output_file', type=str, required=False, default='./markers/output.txt', help='Output file name')
    args = vars(ap.parse_args())

    #Load model in with custom loss function
    try:
        model = models.load_model(args['model_path'], compile=False) 
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=dice_loss, metrics=[dice_loss,'accuracy'])

    except Exception as e:
        print(e)

    img_names = os.listdir(args['images'])

    #Setup output file
    columns = ['Grouping','Block','Sector','Row','Col','Positivity']
    output_df = pd.DataFrame(columns=columns)
    
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
                patch = post_proc(patch*255, max(std_thresh, std)) #patch is normalized to 1.0
                mask[cols[j]:cols[j]+padless_win_w,rows[i]:rows[i]+padless_win_h] = patch

        resized_mask = cv2.resize(mask, (IMG_W, IMG_H))

        #Make output image copy
        img_o = img.copy()
        idx = np.where(resized_mask == 1)
        img_o[idx] = args['color']

        if args['output_masks']:
            output_path = os.path.join(args['output_folder'],img_name.split('.')[0] + '_mask.jpg')
            cv2.imwrite(output_path, resized_mask*255.0) #Ouput image needs to be scaled back to 255.0

        output_path = os.path.join(args['output_folder'],img_name.split('.')[0] + '_labelled.jpg')
        cv2.imwrite(output_path, img_o)

        #Get marker percentage
        ratio = marker_ratio(img, resized_mask)
        #output.append(img_name.split('.')[0] + ": %f" % (ratio))

        img_name = img_name.replace(' ',"_")

        elems = img_name.split('_')
        grouping = elems[0]
        block = elems[1]
        sector = elems[2]
        row = elems[3]
        col = elems[4]
        output_df = output_df.append({'Grouping':grouping,'Block':block,'Sector':sector,'Row':row,'Col':col,'Positivity':ratio}, ignore_index=True)
    # with open(args["output_file"],"w") as outfile:
    #     for line in output:
    #         outfile.write(line + '\n')

    output_df.to_pickle('./markers/output.pickle')
    output_df.to_csv('./markers/output.csv')



