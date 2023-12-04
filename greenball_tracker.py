import cv2
import numpy as np

from pprint import pprint as pp
from picamera2 import Picamera2


if __name__ == '__main__':

    PICAM = Picamera2()
    
    PICAM.preview_configuration.main.size=(720, 540)

    PICAM.preview_configuration.main.format="RGB888"

    PICAM.preview_configuration.align()

    PICAM.configure("preview")

    PICAM.start()

    while True:
        frame = PICAM.capture_array()

        # Blur the image to reduce noise
        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # Convert BGR to HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image for only green colors
        lower_green = np.array([40, 70, 70])
        upper_green = np.array([80, 200, 200])

        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Blur the mask
        bmask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Take the moments to get the centroid
        moments = cv2.moments(bmask)
        m00 = moments['m00']
        centroid_x, centroid_y = None, None
        if m00 != 0:
            centroid_x = int(moments['m10']/m00)
            centroid_y = int(moments['m01']/m00)

        # Assume no centroid
        ctr = (-1, -1)

        # Use centroid if it exists
        if centroid_x is not None and centroid_y is not None:

            ctr = (centroid_x, centroid_y)

            # Put black circle in at centroid in image
            cv2.circle(frame, ctr, 40, (0, 0, 0), 5)

        # Display full-color image
        cv2.imshow("Green Ball Tracker", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
