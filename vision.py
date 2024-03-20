import cv2 as cv
import numpy as np
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter
import windowcapture as wc


class Vision:
    TRACKBAR_WINDOW = "Trackbars"

    def __init__(self, method=cv.TM_CCOEFF_NORMED):
        self.method = method
        self.device = wc.WindowCapture()

    def findOne(self, template_image_path: str, threshold: float = 0.4):
        template_image_path = "img/" + template_image_path
        template_image = cv.imread(template_image_path, cv.IMREAD_COLOR)
        game_screenshot = self.device.get_frame()

        template_image = template_image.astype(np.uint8)

        if len(template_image.shape) != len(game_screenshot.shape) or template_image.shape[2] != game_screenshot.shape[2]:
            raise ValueError("Template image and game screenshot have incompatible dimensions or number of channels")

        search_result = cv.matchTemplate(game_screenshot, template_image, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(search_result)
        rect = [max_loc[0], max_loc[1], template_image.shape[1], template_image.shape[0]]
        return rect
    def find(self, template_image_path: str, threshold: float = 0.45, max_results: int=42):
            template_image_path = "img/" + template_image_path
            print(f"{template_image_path} search")
            template_image = cv.imread(template_image_path, cv.IMREAD_COLOR)  # Load template image in color
            game_screenshot = self.device.get_frame()  # Take a screenshot
            # Ensure both images have the same data type
            template_image = template_image.astype(np.uint8)
            # Ensure both images have the same number of color channels
            if len(template_image.shape) != len(game_screenshot.shape) or template_image.shape[2] != game_screenshot.shape[
                2]:
                raise ValueError("Template image and game screenshot have incompatible dimensions or number of channels")

            # https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
            search_result = cv.matchTemplate(game_screenshot, template_image, cv.TM_CCOEFF_NORMED)
            locations = np.where(search_result >= threshold)
            locations = list(zip(*locations[::-1]))

            if not locations:
                print("No results found.")
                return game_screenshot, np.array([], dtype=np.int32).reshape(0, 4)

            rectangles = []

            for loc in locations:
                # get width and height
                rect = [int(loc[0]), int(loc[1])]
                rectangles.append(rect + [rect[0] + template_image.shape[1], rect[1] + template_image.shape[0]])
            rectangles, weights = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.01)
            # for performance reasons, return a limited number of results.
            # these aren't necessarily the best results.
            if len(rectangles) > max_results:
                print('Warning: too many results, raise the threshold.' + str(len(rectangles)))
                rectangles = rectangles[:max_results]
                # draw rectangles on the game screenshot
            for (startX, startY, endX, endY) in rectangles:
                cv.rectangle(game_screenshot, (startX, startY), (endX, endY), (255, 255, 0), 2)
                # show the output image with the rectangle drawn on it
            return game_screenshot, rectangles
