#!/usr/bin/env python3

import numpy as np
import cv2 as cv

from agent import PurePursuitPolicy
from utils import launch_env, seed
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask

DATASET_DIR="../../dataset"

npz_index = 0
def save_npz(img, boxes, classes):
    global npz_index
    with makedirs(DATASET_DIR):
        np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1

def clean_segmented_image(seg_img):
    # cv.imshow('original image', seg_img)
    # cv.waitKey(1)
    mask = np.zeros_like(seg_img)
    # TODO
    colors = {}
    colors[0] = np.array([255, 0, 255]) # background color
    colors[1] = np.array([100, 117, 226]) # duckie color
    colors[2] = np.array([226, 111, 101]) # cone color
    colors[3] = np.array([116, 114, 117]) # truck color
    colors[4] = np.array([216, 171, 15]) # bus color
    boxes = []
    labels = np.array([], dtype=np.int)
    for i in range(5):
        im = seg_img == colors[i]
        im = im[:,:,0] & im[:,:,1] & im[:,:,2]
        im = np.multiply(im, 1.0)
        im = np.uint8(im)
        boxes, labels, mask = findBoxes(im, boxes, labels, i, seg_img, mask)
    mask = plotWithBoundingBoxes(mask, boxes, labels)
    # cv.imshow('image sans snow', mask)
    # cv.waitKey(1)
    # # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes
    return boxes, labels

def findBoxes(im, boxes, labels, class_label, segmented_im, mask):
    num_labels, labels_im = cv.connectedComponents(im)
    obj_ids = np.unique(labels_im)
    obj_ids = obj_ids[1:]
    for j in range(len(obj_ids)):
        pos = np.where(labels_im == obj_ids[j])
        x_min = np.min(pos[1])
        x_max = np.max(pos[1])
        y_min = np.min(pos[0])
        y_max = np.max(pos[0])
        if (len(pos[0]) <= 34) & (x_max-x_min <= 10) & (y_max-y_min <= 10):
            # snow
            pass
        else :
            # not snow
            boxes.append([x_min,y_min,x_max,y_max])
            labels = np.append(labels,class_label)
            mask[pos[0],pos[1],:] = segmented_im[pos[0],pos[1],:]
    return boxes, labels, mask

def plotWithBoundingBoxes(seg_im,boxes,labels):
    for i in range(len(labels)):
        cv.rectangle(seg_im, (boxes[i][0],boxes[i][1]), (boxes[i][2],boxes[i][3]), (255,255,255),1)
        cv.rectangle(seg_im,(boxes[i][0],boxes[i][1]),(boxes[i][0]+10,boxes[i][1]-12),(255,255,255),cv.FILLED)
        cv.putText(seg_im,str(labels[i]),(boxes[i][0],boxes[i][1]),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
    return seg_im
        


seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

while True:
    obs = environment.reset() # create first segmented image
    environment.render(segment=True) # visualize first segmented image
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs)) # predicts action (v and omega using purepursuit controller)

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        # clean_segmented_image(segmented_obs)
        if np.mod(nb_of_steps, 5) == 0: 
            boxes, classes = clean_segmented_image(segmented_obs)
            output = plotWithBoundingBoxes(obs, boxes, classes)
            output = cv.cvtColor(output, cv.COLOR_BGR2RGB)
            # cv.imshow('result of clean_segmented_image', output)
            # cv.waitKey(1)
            save_npz(obs, boxes, classes)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break