import cv2 as cv
import numpy as np
import os
from time import time
from vision import Vision
from windowcapture import WindowCapture
# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def click(x, y):
    print(f"Clicking {x}, {y}")
    wc.device.shell(f"input tap {x} {y}")

def swipe(x1, y1, x2, y2, duration):
    print(f"Swiping {x1}, {y1} to {x2}, {y2}")
    wc.device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")

# initialize the Vision class
vo = Vision()
wc = WindowCapture()
loop_time = time()
while True:
    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()
    # get Fields
    # img, fields = vo.find("1.png")
    # cv.imshow('Matches', img)
    screen, field_positions = vo.find("1.png", threshold=0.5, max_results=42)
    if field_positions is None:
        print("Field not found")
    else:
        for field_pos in field_positions:
            field = cv.imread("img/1.png")
            w, h = field.shape[1], field.shape[0]
            x, y = field_pos[0], field_pos[1]
            x, y = x + w // 2, y + h // 2
            # draw the click point
            cv.circle(screen, (x, y), 5, (0, 255, 0), -1)
    cv.resize(screen)
    cv.imshow("Screen", screen)
        
        
    screen, market_pos, maxval = vo.findOne("3.png")
    if maxval < 0.8:
        print("Market not found")
    else:
        print(f"Market found at {market_pos} with confidence {maxval}")
        market = cv.imread("img/3.png")
        w, h = market.shape[1], market.shape[0]
        x, y = market_pos
        x, y = x + w // 2, y + h // 2
        # draw the click point
        cv.circle(screen, (x, y), 5, (0, 255, 0), -1)
        cv.imshow("Screen", screen)
        
        click(x, y)
        
        
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
