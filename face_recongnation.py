import cv2 as cv
import numpy as np
p=['brucelee', 'mike_tyson', 'mj', 'mohamad_ali', 'modi']

haar_cascade = cv.CascadeClassifier('face_detection.xml')
# features = np.load("/home/santhosh_chidipothu/Documents/idt/features.npy")
# labels = np.load("/home/santhosh_chidipothu/Documents/idt/labels.npy")
face_recongnizer = cv.face.LBPHFaceRecognizer_create()
face_recongnizer.read("/home/santhosh_chidipothu/Documents/idt/face_trained.yml")
img = cv.imread("/home/santhosh_chidipothu/Documents/idt/validation/mj3.jpeg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('person',gray)

face_rect = haar_cascade.detectMultiScale(gray,1.1,4)

for(x,y,w,h) in face_rect:
    face_roi = gray[y:y+h,x:x+w]

    label,confidence = face_recongnizer.predict(face_roi)
    print(f"label = {p[label]} with a confidence = {confidence}")

    cv.putText(img,str(p[label]),(20,20),cv.FONT_HERSHEY_SCRIPT_COMPLEX,1.0,(0,0,255),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow("detected image",img)
cv.waitKey(0)