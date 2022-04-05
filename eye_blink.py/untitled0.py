#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 18:43:49 2022

@author: payammohammadi
"""

import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
from cvzone import stackImages


cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640,360,[0,40])


idList = [22,23,24,26,110,157,158,159,160,161,130,243]
ratioList = []
countBlink = 0
counter = 0
while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img,draw=False)
    
    if faces:
        face = faces[0]
        
        for id in idList:
            cv2.circle(img,face[id],5,(255,0,0),cv2.FILLED)
        
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        
        cv2.line(img,leftUp,leftDown,(0,255,0),2)
        cv2.line(img,leftLeft,leftRight,(0,255,0),2)
        
        lengthUPDown , _ = detector.findDistance(leftUp,leftDown)
        lengthLeftRight , _ = detector.findDistance(leftLeft,leftRight)
        
        img = cv2.resize(img,(640,360))
        
        radio = int((lengthUPDown/lengthLeftRight)*100)
        ratioList.append(radio)
        
        if len(ratioList) > 3:
            ratioList.pop(0)
        
        avrageRadio = sum(ratioList)/len(ratioList)
        if avrageRadio < 32 and counter == 0 :
            countBlink+=1
            counter=1
        if counter>0 :
            counter+=1
            if counter ==10:
                counter=0
        
        cv2.putText(img, f'count = {countBlink}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0))
            
        
        imgPlot=plotY.update(avrageRadio , color=(255,0,255))
        imageStack=stackImages([img,imgPlot], 1, 1)
        
        #print(lengthUPDown)
        
    cv2.imshow("Image", imageStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()