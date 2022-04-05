# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import face_recognition


#path of image
test_image = "../image/nasim.jpg"

#loading the image to detect
image_to_detect = cv2.imread(test_image)


#detect all faces in the image 
#arguments ate image, no_of_times_to_upsample, model

all_face_locations = face_recognition.face_locations(image_to_detect,model="hog")


#print the number of faces detected
print("there are {} no of faces in this image".format(len(all_face_locations)))

#looping through the face locations
for index , curent_face_location in enumerate(all_face_locations):
    
    #dplitting the tuple to get the four position values
    
    top_position, right_position, bottom_position, left_position = curent_face_location
    
    print("Found face {} at top:{}, right:{}, bottom:{}, left:{}".format(index,top_position, right_position, bottom_position, left_position))
    
    curent_face_detection = image_to_detect[top_position:bottom_position,left_position:right_position]
    
    cv2.imshow("face of " + str(index+1),image_to_detect)