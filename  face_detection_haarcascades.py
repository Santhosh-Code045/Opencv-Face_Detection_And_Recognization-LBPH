import cv2 as cv
# img = cv.imread("many_people.jpeg")
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("GRAY",gray)
# face_cascade = cv.CascadeClassifier("face_detection.xml")
# face_rect = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)
# for x,y,w,h in face_rect:
#     cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),thickness=3)
# cv.imshow("detected image",img)
cap = cv.VideoCapture("/home/santhosh_chidipothu/Documents/idt/people_video.mp4")
face_cascade = cv.CascadeClassifier("face_detection.xml")
while True:
    isTrue,frame=cap.read()
    gframe = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    face_rect = face_cascade.detectMultiScale(gframe,scaleFactor=1.05,minNeighbors=15)
    for x,y,w,h in face_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),thickness=2)
    cv.imshow("FACE_IDENTIFIED VIDEO",frame)
    if cv.waitKey(20) & 0xFF == ord("d"):
        break
    if not isTrue:
        break
cap.release()
cv.destroyAllWindows()