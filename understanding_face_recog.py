import face_recognition
import numpy as np
import cv2

model="hog"

image=face_recognition.load_image_file("test2.jpg")

locations=face_recognition.face_locations(image,model=model)
print(locations)
encodings=face_recognition.face_encodings(image,locations)

image=cv2.rectangle(image,(429,50),(465,86),(255,0,0),46,cv2.LINE_AA)
image=cv2.rectangle(image,(369,162),(405,198),(255,0,0),46,cv2.LINE_AA)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(locations)