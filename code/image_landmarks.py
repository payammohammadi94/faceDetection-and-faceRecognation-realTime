#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:39:04 2022

@author: payammohammadi
"""
import face_recognition
from PIL import Image,ImageDraw

image_path = "../image/messi_sample.jpg"

face_image = face_recognition.load_image_file(image_path)
face_landmark_list = face_recognition.face_landmarks(face_image)
print(face_landmark_list)

for i , face_landmark in enumerate(face_landmark_list):
    face_pillow = Image.fromarray(face_image)
    d = ImageDraw.Draw(face_pillow)
    
    d.line(face_landmark['chin'],fill=(255,255,255),width=2)
    d.line(face_landmark['left_eyebrow'],fill=(255,255,255),width=2)
    d.line(face_landmark['right_eyebrow'],fill=(255,255,255),width=2)
    d.line(face_landmark['nose_bridge'],fill=(255,255,255),width=2)
    d.line(face_landmark['left_eye'],fill=(255,255,255),width=2)
    d.line(face_landmark['right_eye'],fill=(255,255,255),width=2)
    d.line(face_landmark['top_lip'],fill=(255,255,255),width=2)
    d.line(face_landmark['bottom_lip'],fill=(255,255,255),width=1)

face_pillow.show()