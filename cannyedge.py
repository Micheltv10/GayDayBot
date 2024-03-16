import cv2 
import pyscreenshot as ImageGrab
import cv2
import numpy as np

def main(): 
	
	while True: 
		# take screenshot
		frame = cv2.imread("img/Gnome.png")
		print(str(frame))
		frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
		frame = np.array(frame)
		
		# Perform Canny edge detection on the frame 
		blurred, edges = canny_edge_detection(frame) 
		# Save the frame to a file
		cv2.imshow("Edges", edges)
		cv2.imshow("Blurred", blurred)
		# Display the frame
		# Exit the loop when 'q' key is pressed 
		if cv2.waitKey(0): 
			cv2.imshow("Edges", edges)
			break
	cv2.destroyAllWindows()

def canny_edge_detection(frame):
    # Convert the frame to grayscale for edge detection 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # Apply Gaussian blur to reduce noise and smoothen edges 
	blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5) 
      
    # Perform Canny edge detection 
	edges = cv2.Canny(blurred, 70, 135) 
      
	return blurred, edges

if __name__ == "__main__":
    main()