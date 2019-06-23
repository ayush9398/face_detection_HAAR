import numpy as np
import cv2

face_cascade=cv2.CascadeClassifier('/home/naruto/Documents/face_recognition_HAAR/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('/home/naruto/Documents/face_recognition_HAAR/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while(True):
    ret,frame =cap.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        gray2=gray[y:y+h,x:x+w]
        color_img=frame[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(gray2)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(color_img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('LIVE',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()