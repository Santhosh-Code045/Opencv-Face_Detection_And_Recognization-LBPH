import os
import cv2 as cv
import numpy as np
p=[]
for i in os.listdir(r'/home/santhosh_chidipothu/Documents/idt/images'):
    p.append(i)
print(p)
dir = "/home/santhosh_chidipothu/Documents/idt/images"
features = []
labels = []
face_cascade = cv.CascadeClassifier("face_detection.xml")
def create_train():
    for person in p:
        path = os.path.join(dir,person)
        label = p.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            face_rect = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            for (x,y,w,h) in face_rect:
                face_roi = gray[y:y+h,x:x+w]
                features.append(face_roi)
                labels.append(label)

create_train()
print(f"features: {len(features)}")
print(f"labels: {len(labels)}")
print("---------------------training completed -------------------------")
features = np.array(features,dtype='object')
labels = np.array(labels)
face_recongnizer = cv.face.LBPHFaceRecognizer_create()
face_recongnizer.train(features,labels)
face_recongnizer.save("face_trained.yml")
np.save("features.npy",features)
np.save("labels.npy",labels)