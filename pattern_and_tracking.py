import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from datetime import datetime
import copy
import mediapipe as mp

from pprint import pprint as pp
from picamera2 import Picamera2

from collections import Counter

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def distance(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    

def identify_figure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    # Encontrar contornos en la imagen
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    figure = None
    for contour in contours:
        # Calcular los momentos del contorno
        moments = cv2.moments(contour)
        
        # Calcular el centro del contorno
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            # Aproximar el contorno a una forma geométrica (triángulo, cuadrado, círculo)
            aprox = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

            # Dibujar el contorno y el centro en la imagen original
            cv2.drawContours(image, [aprox], 0, (0, 255, 0), 2)
            cv2.circle(image, (cx, cy), 5, (255, 255, 255), -1)

            area = cv2.contourArea(contour)

            if area < 1000:
                if area > 60:
                    if len(aprox) <= 4:
                        figure = "TRIANGLE"
                    else:
                        figure = "PENTAGON"
                else:
                    figure = "SQUARE"

    return figure


def count_fingers(multi_hand_landmarks, state):
    hand = multi_hand_landmarks[0]
    count = 0

    thumb_tip = hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip = hand.landmark[mp_hands.HandLandmark.THUMB_IP].y

    index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

    middle_tip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_pip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

    ring_tip = hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_pip = hand.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y

    pinky_tip = hand.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_pip = hand.landmark[mp_hands.HandLandmark.PINKY_PIP].y


    if abs(thumb_tip - thumb_ip) > 0.05 and state != "menu":
        count += 1
    if index_tip < index_pip:
        count += 1
    if middle_tip < middle_pip:
        count += 1
    if ring_tip < ring_pip:
        count += 1
    if pinky_tip < pinky_pip:
        count += 1
    
    return count


def write_text(image, text, position=(10, 50), color=(0, 0, 0)):
    image_to_write = copy.deepcopy(image)
    image_to_write = cv2.putText(image_to_write, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)
    return image_to_write


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def choose_from(accumulated):
    options_dict = {}
    for value in accumulated:
        if value in options_dict:
            options_dict[value] += 1
        else:
            options_dict[value] = 1
    
    maximum = 0
    selected = None
    for option, times in options_dict.items():
        if times > maximum:
            maximum = times
            selected = option
    
    return selected
    


def identify_pattern_and_track_hands():

    PICAM = Picamera2()
        
    PICAM.preview_configuration.main.size=(720, 540)

    PICAM.preview_configuration.main.format="RGB888"

    PICAM.preview_configuration.align()

    PICAM.configure("preview")

    PICAM.start()

    to_match = ["TRIANGLE", "SQUARE", "PENTAGON", "TRIANGLE"]
    current = 0
    safety_number = 30
    figures = [None for _ in range(safety_number)]

    previous_def_figure = None
    def_figure = None

    figure = None

    state = "pattern"

    gestures = [None for _ in range(safety_number)]

    legend = cv2.imread("leyenda_dedos.png")
    legend = cv2.resize(legend, (300, 120))
    leg_h, leg_w, leg_c = legend.shape

    previous_def_command = None
    def_command = None

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:

        while True:

            frame = PICAM.capture_array()

            if state == "pattern":

                figure = identify_figure(frame)

                figures.pop(0)
                figures.append(figure)

                previous_def_figure = copy.deepcopy(def_figure)
                def_figure = choose_from(figures)

                if previous_def_figure != def_figure:
                    if def_figure is not None:
                        if def_figure == to_match[current]:
                            current += 1
                        else:
                            current = 0
                
                phrase = f"Current Pattern Index: {current}. Detected: {def_figure}"
                image = write_text(frame, phrase)
            
            else:
                # BGR 2 RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Set flag to False. This improves performance as set the image
                # to Write-Only
                image.flags.writeable = False
                
                # Hand detection
                results = hands.process(image)

                # Set flag to True
                image.flags.writeable = True

                # RGB 2 BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Rendering results
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),      # points
                                                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))     # lines
                

                if state == "menu":
                    image[0:leg_h, 0:leg_w] = legend

                    if results.multi_hand_landmarks:
                        fingers = count_fingers(results.multi_hand_landmarks, state)
                        gestures.pop(0)
                        gestures.append(fingers)

                        previous_def_command = copy.deepcopy(def_command)
                        def_command = choose_from(gestures)

                        if def_command != previous_def_command:
                            if def_command == 1:
                                state = "count_fingers"
                            elif def_command == 2:
                                state = "exit"
                

                elif state == "count_fingers":
                    if results.multi_hand_landmarks:
                        fingers = count_fingers(results.multi_hand_landmarks, state)
                        gestures.pop(0)
                        gestures.append(fingers)

                        previous_def_command = copy.deepcopy(def_command)
                        def_command = choose_from(gestures)

                        phrase = f"Fingers Up: {def_command}"
                        image = write_text(image, phrase)

                        if def_command != previous_def_command:
                            if def_command == 0:
                                state = "menu"
            
            cv2.imshow("Camera", image)

            if state == "pattern" and current == len(to_match):
                print("Sucessful Pattern")
                state = "menu"

            if (cv2.waitKey(1) & 0xFF == ord('q')) or (state == "exit"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    identify_pattern_and_track_hands()
    