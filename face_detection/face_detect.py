
import cv2
# load the photograph
#pixels = cv2.imread('test1.jpg')
# load the pre-trained model
'''classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while cap.isOpened():
        frame = cap.read()
        # perform face detection
        bboxes = classifier.detectMultiScale(frame)
        # print bounding box for each detected face
        for box in bboxes:
            # extract
            x, y, width, height = box
            x2, y2 = x + width, y + height
            # draw a rectangle over the pixels
            cv2.rectangle(frame, (x, y), (x2, y2), (0,0,255), 1)
        # show the image
        cv2.imshow('face detection', pixels)
# keep the window open until we press a key
        if cv2.waitKey(25) and 0xff == ord(q):
            break
# close the window
cv2.destroyAllWindows()'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#mg = cv2.imread('xfiles4.jpg')
cap = cv2.VideoCapture('test.mp4')
while cap.isOpened():
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)


        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
        
        if cv2.waitKey(25) & 0xff == ord('q'):
            break    
        cv2.imshow('img',img)

cv2.destroyAllWindows()
