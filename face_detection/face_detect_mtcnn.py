from mtcnn.mtcnn import MTCNN
import cv2
from scipy.spatial import distance 
detector = MTCNN()
# detect faces in the image
cap = cv2.VideoCapture('test.mp4')
while cap.isOpened():
    ret, pixels = cap.read()
    faces = detector.detect_faces(pixels)
    new_img = pixels.copy()
    if len(faces)>=2:
        label = [0 for i in range(len(faces))]
        for i in range(len(faces)-1):
            for j in range(i+1, len(faces)):
                dist = distance.euclidean(faces[i][:2],faces[j][:2])
                if dist<MIN_DISTANCE:
                    label[i] = 1
                    label[j] = 1
        new_img = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR) #colored output image
        for i in range(len(faces)):
            (x,y,w,h) = faces[i]
            if label[i]==1:
                cv2.rectangle(new_img,(x,y),(x+w,y+h),(255,0,0),1)
            else:
                cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,255,0),1)
    elif faces:
        x, y, width, height = result['box']
        cv2.rectangle(pixels, (x,y),(x+width,y+height),(255,0,0),2)
    cv2.imshow('img',new_img)
    if cv2.waitKey(2) & 0xff == ord('q'):
        break 
    """for result in faces:
        x, y, width, height = result['box']
        cv2.rectangle(pixels, (x,y),(x+width,y+height),(255,0,0),2)
        if cv2.waitKey(2) & 0xff == ord('q'):
            break  """  
    #cv2.imshow('img',new_img)

        #rect = Rectangle((x, y), width, height, fill=False, color='red')
cv2.destroyAllWindows()	
	
