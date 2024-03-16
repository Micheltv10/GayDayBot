import numpy as np
import cv2 
import pyscreenshot as ImageGrab


class WindowCapture:

    # properties
    w = 1920
    h = 1080
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self):
        self.cropped_x = 0
        self.cropped_y = 0
        self.offset_x = 0
        self.offset_y = 0
        self.w = 1920
        self.h = 1080

    def get_screenshot(self):
        frame = ImageGrab.grab(bbox=(self.offset_x , self.cropped_y, self.w, self.h))
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame