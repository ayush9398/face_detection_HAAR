import cv2
import numpy as np
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        face_cascade=cv2.CascadeClassifier('/home/naruto/Documents/face_recognition_HAAR/haarcascade_frontalface_default.xml')
        eye_cascade=cv2.CascadeClassifier('/home/naruto/Documents/face_recognition_HAAR/haarcascade_eye.xml')
        success, frame = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            gray2=gray[y:y+h,x:x+w]
            color_img=frame[y:y+h,x:x+w]
            eyes=eye_cascade.detectMultiScale(gray2)
            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(color_img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()