#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
# ================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.config import cfg
from sys import argv
import shutil
import os
import colorsys
import random
#to use sort.py
import sys
sys.path.insert(0,'./sort')
from sort import Sort
mot_tracker = Sort(max_age=8,min_hits=3) #create an instance of the SORT tracker
memory = {}
#checkpoint line
line = [(448,454),(845,524)]

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0",
                   "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file = "./pb/yolov3_coco_4_coco1.pb"

# give the path to the video here. eg.: ./video/video_20190812.mp4
video_path = argv[1]
print("processing video {}".format(video_path))

#uncomment to save the frames drawn with bbox into a folder '<video_name>_<model_used>' under ./out_frame
#the model used - for the folder name
#model = argv[2]

#create a folder to store out_frames
#out_folder = './out_frame/{}_{}'.format(video_path.split('/')[-1].strip('.mp4'),model)
#print(out_folder)
#shutil.rmtree(out_folder,ignore_errors=True)
#os.makedirs(out_folder)

#get average fps
total_time = 0
classes = utils.read_class_names(cfg.YOLO.CLASSES)
num_classes = len(classes)
#colors = np.random.rand(32,3)
#colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
#define colors list
#hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
#colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
#colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
#random.seed(0)
#random.shuffle(colors)
#random.seed(4)
#np.random.seed(20)
#colors = np.random.randint(0, 255, size=(7, 3))
colors = [(255,218,0), (223,156,128), (224,118,40),(99,218,15),(0,145,255),(145,0,255),(255,204,153)]
input_size = 416

#counter dict: class_id:counting
counter_dict = dict()
for i in range(len(classes)-2):
    counter_dict[i] = 0
#object id dict: object_id:class_id 
obj_dic = {}

graph = tf.Graph()
return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
out = cv2.VideoWriter('out_{}.avi'.format(video_path.split("/")[-1].split(".")[0]),cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720))

with tf.Session(graph=graph) as sess:
    vid = cv2.VideoCapture(video_path)
    counter = 0 #count the frame
    while True:
        prev_time = time.time()
        return_value, frame = vid.read() # grab a single frame of video
        if return_value:
            counter += 1 #increment the counter 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (1280, 720))
            #crop_frame = frame[0:720,0:853]
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('out', gray)
            # img2 = np.zeros_like(frame)
            # img2[:, :, 0] = gray
            # img2[:, :, 1] = gray
            # img2[:, :, 2] = gray
            # image = Image.fromarray(img2)
            # cv2.imshow('out', img2)
        else:
            raise ValueError("No image!")
        #frame_size = crop_frame.shape[:2]
        frame_size = frame.shape[:2]
        #image_data = utils.image_preporcess(np.copy(crop_frame), [input_size, input_size])
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
            feed_dict={return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
        
        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, cfg.TEST.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, 0.45, method='nms')

        # CHANGE THIS FOR GRAYSCALE
        #image = utils.draw_bbox(frame, bboxes)
        #bbox:[x_min, y_min, x_max, y_max, probability, cls_id]
        
        #get bboxes from tracker
        dets = np.array(bboxes)
        trackers = mot_tracker.update(dets)  #trackers:[x1,y1,x2,y2,obj_id,class_id]
        previous = memory.copy()
        memory = {}
        objIDs = []
        for tracker in trackers:
            objIDs.append(tracker[4])
            memory[objIDs[-1]] = [tracker[0],tracker[1],tracker[2],tracker[3]]
            #update the object dict with new object id - class id pair
            if tracker[4] not in obj_dic:
                obj_dic[int(tracker[4])] = int(tracker[5])
        
        image_h, image_w, _ = frame.shape
        fontScale = 0.5
        bbox_thick = int(0.6 * (image_h + image_w) / 700)
         
        if len(trackers) > 0 :
            for i in range(len(trackers)):
                d = trackers[i]
                d = d.astype(np.int32)
                class_id = d[5]
                class_nm = classes[class_id]
                coor1, coor2 = (d[0], d[1]), (d[2], d[3])
                #the center of the bottom line of the bbox
                p0 = ((coor1[0] + coor2[0])//2,coor2[1])
                color = [int(c) for c in colors[class_id]]
                cv2.rectangle(frame, coor1, coor2, color, bbox_thick)
                cv2.circle(frame,p0,2,color,-1,0)
                #show label
                bbox_mess = '%s' % (class_nm)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
                cv2.rectangle(frame, coor1, (coor1[0] + t_size[0], coor1[1] - t_size[1] - 3),color, -1)  # filled
                cv2.putText(frame, bbox_mess, (coor1[0], coor1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0,0,0), bbox_thick//2, lineType=cv2.LINE_AA)
                
                if objIDs[i] in previous:
                    previous_box = previous[objIDs[i]]
                    pre_coor1 = (int(previous_box[0]),int(previous_box[1]))
                    pre_coor2 = (int(previous_box[2]),int(previous_box[3]))
                    #the center of bottom line of the bbox
                    p1 = ((pre_coor1[0] + pre_coor2[0])//2,pre_coor2[1])
                    if intersect(p0,p1,line[0],line[1]):
                        if obj_dic[objIDs[i]] in counter_dict:
                            counter_dict[obj_dic[objIDs[i]]] += 1
                            cv2.rectangle(frame, coor1, coor2, color, bbox_thick*4)
        
        # draw checkpoint line
        cv2.line(frame, line[0], line[1], (255, 255, 0), 3)

        #to display vehicle counting
        cv2.rectangle(frame, (0,0),(1280,46), (0, 0, 0), -1)
        x = 0
        wid = 1280//5
        for class_id, counter in counter_dict.items():
            color = [int(c) for c in colors[class_id]]
            cv2.putText(frame,'{} : {}'.format(classes[class_id],counter),(x+wid//4,26),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
            x += wid
        curr_time = time.time()
        exec_time = curr_time - prev_time
        #update total time
        #total_time += exec_time
        
        #result = np.asarray(frame)
        info = "time: %.2f ms" % (1000*exec_time)
        #   * WINDOW_NORMAL：window can be resized
        #   * WINDOW_KEEPRATIO：keep a fixed ratio of the window
        #   * WINDOW_GUI_EXPANDED： to use an enhanced GUI window 
        #   *cv2.WINDOW_AUTOSIZE: use the image size (fixed,cannot be resized)
        cv2.namedWindow("result", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

        # COMMENT THIS FOR GRAYSCALE
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fps = 1 / exec_time

        #to display fps
        cv2.rectangle(result, (0, 690), (180, 720), (0, 0, 0), -1)
        cv2.putText(result, "FPS = {:.2f}".format(fps), (20, 710),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("result", result)
        #uncomment to save frame drawn with predicted bbox
        #cv2.imwrite("{}/{}.jpg".format(out_folder,counter),result)
        #counter += 1
        #print('writing image: '+"{}/{}.jpg".format(out_folder,counter))'''
        
        out.write(result)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()
    #avg_fps = counter / total_time
    #print('Average fps is :  ',avg_fps)

