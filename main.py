import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# initialize the WindowCapture class
wincap = WindowCapture()
# initialize the Vision class
vision_object = Vision('img/empty_field_edges.png')
# initialize the trackbar window
vision_object.init_control_gui()

# limestone HSV filter
hsv_filter = HsvFilter(0, 180, 129, 15, 229, 243, 143, 0, 67, 0)

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()
    # pre-process the image
    processed_image = vision_object.apply_hsv_filter(screenshot)

    # do edge detection
    edges_image = vision_object.apply_edge_filter(processed_image)

    # do object detection
    rectangles = vision_object.find(processed_image, 0.46)

    # draw the detection results onto the original image
    output_image = vision_object.draw_rectangles(screenshot, rectangles)

    # keypoint searching
    keypoint_image = edges_image
    # crop the image to remove the ui elements
    x, w, y, h = [200, 1130, 70, 750]
    keypoint_image = keypoint_image[y:y+h, x:x+w]

    kp1, kp2, matches, match_points = vision_object.match_keypoints(keypoint_image)
    match_image = cv.drawMatches(
        vision_object.needle_img, 
        kp1, 
        keypoint_image, 
        kp2, 
        matches, 
        None)

    if match_points:
        # find the center point of all the matched features
        center_point = vision_object.centeroid(match_points)
        # account for the width of the needle image that appears on the left
        center_point[0] += vision_object.needle_w
        # drawn the found center point on the output image
        match_image = vision_object.draw_crosshairs(match_image, [center_point])

    # Load the individual images

    # Resize the images to width
    single_widht = 480
    image1 = cv.resize(match_image, (480,360), fx=0.5, fy=0.5)
    image2 = cv.resize(processed_image,(480,360), fx=0.5, fy=0.5)
    image3 = cv.resize(edges_image, (480,360), fx=0.5, fy=0.5)
    image4 = cv.resize(output_image,(480,360), fx=0.5, fy=0.5)
    cv.imshow("edges", image3)
    #combined1 = np.concatenate((image3, image1), axis=0)
    #combined2 = np.concatenate((image2, image4), axis=0)
    #cv.imshow("Edge + Match", combined1)
    #cv.imshow("Process + Output", combined2)
    # Display the combined image

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
