import numpy as np
import cv2
from ppadb.client import Client as AdbClient


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
        self.adb = AdbClient(host="127.0.0.1", port=5037)
        self.devices = self.adb.devices()

        if len(self.devices) == 0:
            # create exception
            print("No device connected")

        self.device = self.devices[0]

    def get_frame(self):
        frame = self.device.screencap()  # Take a screenshot
        # Convert the screenshot to a format that OpenCV can use
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
