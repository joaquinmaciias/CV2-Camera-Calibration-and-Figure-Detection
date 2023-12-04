import cv2
import glob
import copy
import math
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import sys
import time

from pprint import pprint as pp

from picamera2 import Picamera2


def load_images(filenames):
    return [imageio.imread(filename) for filename in filenames]


def obtain_images():
    program_path = os.getcwd()

    filenames = []
    for i in range(15):
        filenames.append(os.path.join(program_path, "snapshots_folder", f"snapshot_640_480_{str(i)}.jpg"))
        
    images = load_images(filenames)
    return images

def obtain_corners(images, chessboard_shape):
    corners = []

    for image in images:
        corners.append(cv2.findChessboardCorners(image, chessboard_shape, None))
    
    return corners


def refine_corners(corners, images, chessboard_shape):
    corners_2 = copy.deepcopy(corners)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    greyscale_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    corners_refined = [[cor[0], cv2.cornerSubPix(i, cor[1], chessboard_shape, (-1, -1), criteria)] if cor[0] else [cor[0], None] for i, cor in zip(greyscale_images, corners_2)]

    return corners_refined


def draw_corners(corners_refined, images, chessboard_shape, real_time=False):
    images_2 = copy.deepcopy(images)

    for i in range(len(images_2)):
        # We specify the image, the dimensions, the location of the corners and whether it could find them
        cv2.drawChessboardCorners(images_2[i], chessboard_shape, corners_refined[i][1], corners_refined[i][0])

    if not real_time:
        # Print images
        for i in range(6):
            # We only want to check the succesful ones
            if corners_refined[i][0] != 0:
                # Show painted image
                cv2.imwrite(f"painted_image_{i}.jpg", images_2[i])
    
    else:
        return images_2[0]


def get_chessboard_points(chessboard_shape, dx, dy):
    # First we create the emoty (0) matrix
    cb_points = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), np.float32)
    i = 0
    
    # Now we fill the matrix with the corners coordinates
    for j in range(chessboard_shape[1]):
        for k in range(chessboard_shape[0]):
            cb_points[i] = np.array([k*dx, j*dy, 0])
            i += 1

    return cb_points


def obtain_parameters(corners_refined, chessboard_shape, dx, dy):

    valid_corners = [cor[1] for cor in corners_refined if cor[0]]

    num_valid_images = len(valid_corners)

    cb_points = get_chessboard_points(chessboard_shape, dx, dy)

    object_points = np.asarray([cb_points for i in range(num_valid_images)], dtype=np.float32)

    image_points = np.asarray(valid_corners, dtype=np.float32)

    # ASIGNMENT: Calibrate the left camera
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, chessboard_shape, None, None)

    # extrinsics = [np.hstack((cv2.Rodrigues(rvec)[0].flatten(), tvec.flatten())) for rvec, tvec in zip(rvecs, tvecs)]
    extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

    return intrinsics, extrinsics, dist_coeffs, rms


def snapshots_calibration(images, chessboard_shape, dx, dy):
    corners = obtain_corners(images, chessboard_shape)

    corners_refined = refine_corners(corners, images, chessboard_shape)

    draw_corners(corners_refined, images, chessboard_shape)
    
    intrinsics, extrinsics, dist_coeffs, rms = obtain_parameters(corners_refined, chessboard_shape, dx, dy)

    np.savez("calib_data", intrinsic=intrinsics, extrinsic=extrinsics)

    # Lets print some outputs
    print("Corners standard intrinsics:\n", intrinsics)
    print("Corners standard dist_coefs:\n", dist_coeffs)
    print("Root mean square reprojection error:\n", rms)


def draw_frame(images, chessboard_shape, dx, dy):
    
    corners = obtain_corners(images, chessboard_shape)

    corners_refined = refine_corners(corners, images, chessboard_shape)

    frame = draw_corners(corners_refined, images, chessboard_shape, real_time=True)

    return frame, corners_refined




def liveCalibration(chessboard_shape, dx, dy):

    PICAM = Picamera2()
        
    PICAM.preview_configuration.main.size=(640, 480)

    PICAM.preview_configuration.main.format="RGB888"

    PICAM.preview_configuration.align()

    PICAM.configure("preview")

    PICAM.start()

    i = 0
    corners_refined = []
    while True:

        frame = PICAM.capture_array()

        frame, new_corners_refined = draw_frame([frame], chessboard_shape, dx, dy)

        # frame_calibrated, corners_refined = calibration(frame)

        if new_corners_refined[0][0]:
            corners_refined += new_corners_refined
        
        # Show painted image
        cv2.imshow("picam", frame)
        time.sleep(0.2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            intrinsics, extrinsics, dist_coeffs, rms = obtain_parameters(corners_refined, chessboard_shape, dx, dy)

            # Lets print some outputs
            print("Corners standard intrinsics:\n", intrinsics)
            print("Corners standard dist_coefs:\n", dist_coeffs)
            print("Root mean square reprojection error:\n", rms)
            print("Corners standard extrinsics:\n", extrinsics)

            np.savez("calibration_parameters", intrinsics=intrinsics, dist_coeffs=dist_coeffs, rms=rms)
            break

    cv2.destroyAllWindows()

    



if __name__ == "__main__":
    chessboard_shape = (4, 6)
    dx = 25
    dy = 25

    # chessboard_shape = (7, 7)
    # dx = 20
    # dy = 20

    images = obtain_images()
    # snapshots_calibration(images, chessboard_shape, dx, dy)
    liveCalibration(chessboard_shape, dx, dy)



    