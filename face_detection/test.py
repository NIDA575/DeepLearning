import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from scipy.spatial import distance

face_model = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
#import matplotlib.pyplot as plt
#trying it out on a sample image
#img = cv2.imread('../input/face-mask-detection/images/maksssksksss244.png')

#img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

#faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4) #returns a list of (x,y,w,h) tuples

 #colored output image
cap = cv2.VideoCapture('./test.mp4')

#plotting
while cap.isOpened():

    ret,img = cap.read()
    new_img = img.copy()
    #print(ret)
    out_img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    #out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faces = face_model.detectMultiScale(out_img,scaleFactor=1.3, minNeighbors=4)
     #returns a list of (x,y,w,h) tuples

    for (x,y,w,h) in faces:
        cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,0,255),1)
    #plt.figure(figsize=(12,12))
    #plt.imshow(out_img)
    MIN_DISTANCE = 100
    if len(faces)>=2:
        label = [0 for i in range(len(faces))]
        for i in range(len(faces)-1):
            for j in range(i+1, len(faces)):
                dist = distance.euclidean(faces[i][:2],faces[j][:2])
                if dist<MIN_DISTANCE:
                    label[i] = 1
                    label[j] = 1
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
        for i in range(len(faces)):
            (x,y,w,h) = faces[i]
            if label[i]==1:
                cv2.rectangle(new_img,(x,y),(x+w,y+h),(255,0,0),1)
            else:
                cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,255,0),1)
    cv2.imshow('img',new_img)
    if cv2.waitKey(2) & 0xff == ord('q'):
        break
 #   plt.figure(figsize=(10,10))
 #   plt.imshow(new_img)

#    else:
#        print("No. of faces detected is less than 2")


