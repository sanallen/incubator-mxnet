import cv2
import numpy as np
img=cv2.imread('/opt/data/plate/test.jpg',cv2.IMREAD_COLOR)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# lower_blue=np.array([110,50,50])
# upper_blue=np.array([130,255,255])
# mask=cv2.inRange(hsv,lower_blue,upper_blue)
# res=cv2.bitwise_and(img,img,mask=mask)
cv2.imshow('h',hsv[:,:, 0])
cv2.imshow('s',hsv[:,:, 1])
cv2.imshow('v',hsv[:,:, 2])
# cv2.imshow('img',img)
k=res=cv2.waitKey(5)&0xFF
while k==27:
    break
cv2.waitKey(0)