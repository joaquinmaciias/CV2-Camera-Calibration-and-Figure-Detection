import cv2, sys, time, os

from pprint import pprint as pp
from picamera2 import Picamera2

cascPath = 'lbpcascade_frontalface.xml'
faceCascade = cv2.CascadeClassifier(cascPath)


PICAM = Picamera2()
    
PICAM.preview_configuration.main.size=(720, 540)

PICAM.preview_configuration.main.format="RGB888"

PICAM.preview_configuration.align()

PICAM.configure("preview")

PICAM.start()

while True:
    frame = PICAM.capture_array()

    # Convert to greyscale for easier faster accurate face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist( gray )

    # Do face detection to search for faces from these captures frames
    faces = faceCascade.detectMultiScale(frame, 1.1, 3, 0, (10, 10))
 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        break

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()