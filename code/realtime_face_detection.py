#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:54:03 2022

@author: payammohammadi
"""

import cv2
import face_recognition

#capture the video from default camera 
webcam_video_stream = cv2.VideoCapture(0)


#initialize the array variable to hold all face locations in the frame
all_face_locations = []


#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image
    ret,current_frame = webcam_video_stream.read()
    #resize the current frame to 1/4 size to proces faster
    current_frame_small = cv2.resize(current_frame, (0,0),fx=0.25,fy=0.25)
    #detect all face in the picture
    #argumants are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small,model = 'hog')
    #looping through the face locations 
    for index , current_face_location in enumerate(all_face_locations):
        #find llocation the face
        top_position, right_position, bottom_position, left_position = current_face_location
        #change the size
        top_position = top_position*4
        right_position = right_position * 4
        bottom_position = bottom_position * 4
        left_position = left_position * 4
        
        cv2.rectangle(current_frame, (left_position, top_position),(right_position,bottom_position), (0,0,255),2)
        cv2.imshow("title", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()