# CV2 Camera Calibration, Figure Detection and Hand Tracker
In this project we are going to implement a camera calibration algorithm using OpenCV. You have the option to decide
if you want to calibrate via snapshots previously taken or in real time. This is done by running calibration.py.

The file pattern_and_tracking.py includes the algorithm to detect the pattern TRIANGLE-SQUARE-PENTAGON-TRIANGLE and, if
introduced in the correct order, proceed to the hand tracker feature. When this happens, you will see a new pop up menu 
in the upper left part of the screen. Here, if you point up with your finger, you will enter the fingers count feature, 
which, as it name syas, counts how many fingers are pointing up. When you close your hnad, or put down all the fingers, 
you come back to the previous menu. You can exit the program by making the Peace sign (pointing up index and middle fingers).

The file greenball_tracker.py starts the greenball tracker algorthim, which applies a green mask to the frames to detect only
that color and draw a circle around it.

Lastly, the file face_tracker.py uses a Cascade Classifier to detect one face on the frame and draw a rectangle around it.

Links to the videos demosntrations:

- Real time calibration: https://youtube.com/shorts/z6nh6qLJsPI?feature=share
- Pattern detection and hands tracking: https://youtu.be/cWBRQQpCUWo
- Green ball tracker: https://youtu.be/nGupjvWsvj4
- Face tracker: https://youtu.be/Tb_ZimOae3Y