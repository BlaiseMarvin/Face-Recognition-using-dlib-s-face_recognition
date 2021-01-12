import face_recognition
import os
import cv2

KNOWN_FACES_DIR="known_faces"
#UNKNOWN_FACES_DIR="unknown_faces"

TOLERANCE=0.8
FRAME_THICKNESS=3
FONT_THICKNESS=2
MODEL="hog" #hog is the traditional model, try out cnn, cnn is slow on CPU

video=cv2.VideoCapture(0)


print("Loading known faces")

known_faces=[]
known_names=[]

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image=face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        if len(face_recognition.face_encodings(image))>0:
            encoding=face_recognition.face_encodings(image)[0]
        #print(encoding)
            known_faces.append(encoding)
            known_names.append(name)


print("Processing unknown faces")

while(True):

    ret,image=video.read()
    locations=face_recognition.face_locations(image,model=MODEL)
    encodings=face_recognition.face_encodings(image,locations)


    for face_encoding,face_location in zip(encodings,locations):
        results=face_recognition.compare_faces(known_faces,face_encoding,TOLERANCE)
        #print(results)

        match=None
        if True in results:
            match=known_names[results.index(True)]
            print(f"Match found:{match}")

            top_left=(face_location[3],face_location[0])
            bottom_right=(face_location[1],face_location[2])

            color=[0,255,0]
            cv2.rectangle(image,top_left,bottom_right,color,FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image,match,(face_location[3]+10, face_location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),FONT_THICKNESS)

    cv2.imshow("Image",image)
    #cv2.waitKey(0)
    k=cv2.waitKey(1) & 0xFF
    if k==ord('q'):
        break
cv2.destroyWindow(filename)







