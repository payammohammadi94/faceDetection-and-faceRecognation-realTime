#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:49:13 2022

@author: payammohammadi
"""

import cv2
import face_recognition
import gtts  
from playsound import playsound  
import time
webcam_video_stream = cv2.VideoCapture(0)

messi_image = "../image/messi_sample.jpg"
xavi_image = "../image/xavi_sample.jpg"



messi_image_load = face_recognition.load_image_file(messi_image)
xavi_image_load = face_recognition.load_image_file(xavi_image)


messi_image_encoding = face_recognition.face_encodings(messi_image_load)[0]
xavi_image_encoding = face_recognition.face_encodings(xavi_image_load)[0]



list_of_encoding = [messi_image_encoding, xavi_image_encoding]
name_of_image = ["messi","xavi"]


all_face_locations = []
all_face_encoding = []
all_face_name = []

while True:
    #get the current frame from the video stream as an image
    ret,current_frame = webcam_video_stream.read()
    #resize the current frame to 1/4 size to proces faster
    current_frame_small = cv2.resize(current_frame, (0,0),fx=0.25,fy=0.25)
    #detect all face in the picture
    #argumants are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small,model = 'hog')
    
    all_face_encoding = face_recognition.face_encodings(current_frame_small,all_face_locations)
    all_face_name=[]
    
    for current_face_location , current_face_encoding in zip(all_face_locations,all_face_encoding):
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        top_pos = top_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos*4
        right_pos = right_pos *4
        
        all_matches = face_recognition.compare_faces(list_of_encoding, current_face_encoding)
        
        name = "unknown face"
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name = name_of_image[first_match_index]
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name, (left_pos,bottom_pos-10), font, .5, (255,255,255))

    cv2.imshow("title", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()
