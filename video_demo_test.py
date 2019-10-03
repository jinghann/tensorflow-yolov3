#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor       : SJH
#   Former-Editor: VIM
#   File name    : video_demo.py
#   Author       : YunYang1994
#   Created date : 2018-11-30 15:56:37
#   Description  :
#
# ================================================================

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
from core.config import cfg
from sys import argv
import shutil
import os 
# MVI_39211.avi  MVI_39311.avi   MVI_39501.avi   MVI_40711.avi   MVI_40742.avi   DAY TIME
# MVI_40771.avi  MVI_40772.avi   MVI_40775.avi

#get class names
classes = utils.read_class_names(cfg.YOLO.CLASSES)

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0",
                   "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file = "./pb/yolov3_coco_4_coco1.pb"
# "../ITSWC/video3_20190702_112939.mp4"

video_path = argv[1]
print("processing video {}".format(video_path))
# if not os.path.exists(video_path):
#     print(os.path.exists(video_path))
#     print("..."+video_path)
#     print("video doesnt exist")
# video_path      = 0
num_classes = 7
input_size = 416
graph = tf.Graph()
return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
out = cv2.VideoWriter('out_{}.avi'.format(video_path.split("/")[-1].split(".")[0]),
                      cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720))

#create a folder to save the frame images
video_name = video_path.split("/")[-1].split(".")[0]
dst_path = './video_frame/{}'.format(video_name)
shutil.rmtree(dst_path,ignore_errors=True)
os.makedirs(dst_path)
#create a dataset txt file
txt_path = './org_anno/{}.txt'.format(video_name)
f = open(txt_path,'w')

with tf.Session(graph=graph) as sess:
    cap = cv2.VideoCapture(video_path)
    #get the number of frames of the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_digit = len(str(frame_count // 30))
    
    #format the image name
    padding = "{:0"+str(num_digit)+"}"

    prev_time = time.time()
    success, frame = cap.read()
    #count the frame
    frame_num = 0
    img_num = 1
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1280, 720))
        image = Image.fromarray(frame)
        
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('out', gray)
            # img2 = np.zeros_like(frame)
            # img2[:, :, 0] = gray
            # img2[:, :, 1] = gray
            # img2[:, :, 2] = gray
            # image = Image.fromarray(img2)
            # cv2.imshow('out', img2)   

        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(
            np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
            feed_dict={return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(
                                        pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(
            pred_bbox, frame_size, input_size, cfg.TEST.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, 0.45, method='nms')

        #save an image every 30 frames
        if frame_num % 30 == 0: 
            padded_frame = padding.format(img_num)
            img_name = "{}_{}.jpg".format(video_path.split("/")[-1].split(".")[0],padded_frame)
            #save the frame as an image
            cv2.imwrite("{}/{}".format(dst_path,img_name),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            img_num += 1
            #write bbox information
            f.write("{}/{} ".format(dst_path,img_name))
            for i, bbox in enumerate(bboxes):
                coor = np.array(bbox[:4], dtype=np.int32)
                class_ind = int(bbox[5])
                f.write("{},{},{},{},{} ".format(coor[0],coor[1],coor[2],coor[3],classes[class_ind]))
            f.write("\n")
        
            
        # CHANGE THIS FOR GRAYSCALE
        image = utils.draw_bbox(frame, bboxes)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" % (1000*exec_time)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)

        # COMMENT THIS FOR GRAYSCALE
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        

        fps = 1 / exec_time
        cv2.rectangle(result, (0, 0), (180, 30), (0, 0, 0), -1)
        cv2.putText(result, "FPS = {:.2f}".format(fps), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("result", result)

        out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #read the next frame
        frame_num += 1
        success, frame = cap.read()
    cap.release()
    out.release()
f.close()
