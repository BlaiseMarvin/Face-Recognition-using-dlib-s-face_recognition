import face_recognition
import cv2

cap=cv2.VideoCapture(0)

while(True):
    _,image=cap.read()

    locations=face_recognition.face_locations(image,model="hog")
    if len(locations)>0:
        image=cv2.rectangle(image,(locations[3],locations[0]),(locations[1],locations[2]),(0,255,255),10,cv2.LINE_AA)

        cv2.imshow("blaise",image)

        k=cv2.waitKey(1)&0xFF
        if k==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()