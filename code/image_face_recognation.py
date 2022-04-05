#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:08:42 2022

@author: payammohammadi
"""

import cv2
import face_recognition

image_recognation = "../image/messi_xavi_inista.jpg" 

messi_image = "../image/messi_sample.jpg"
xavi_image = "../image/xavi_sample.jpg"

messi_image_load = face_recognition.load_image_file(messi_image)
xavi_image_load = face_recognition.load_image_file(xavi_image)

messi_image_encoding = face_recognition.face_encodings(messi_image_load)[0]
xavi_image_encoding = face_recognition.face_encodings(xavi_image_load)[0]

list_of_encoding = [messi_image_encoding, xavi_image_encoding]
name_of_image = ["messi","xavi"]

image_to_recognation = face_recognition.load_image_file(image_recognation)

all_face_locations = face_recognition.face_locations(image_to_recognation,model = "hog")
all_face_encoding = face_recognition.face_encodings(image_to_recognation,all_face_locations)


for current_face_location , current_face_encoding in zip(all_face_locations,all_face_encoding):
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    all_matches = face_recognition.compare_faces(list_of_encoding, current_face_encoding)
    name = "unknown face"
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name = name_of_image[first_match_index]
    cv2.rectangle(image_to_recognation,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_recognation, name, (left_pos,bottom_pos-10), font, .5, (255,255,255))
    
    cv2.imshow("face_recognation",image_to_recognation)