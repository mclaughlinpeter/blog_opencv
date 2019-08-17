import cv2
from matplotlib import pyplot as plt
import imutils

# read image and take first channel only
image_3_channel = cv2.imread("./images/open.png")
image_gray = cv2.split(image_3_channel)[0]
cv2.imshow("Image", image_gray)
cv2.waitKey(0)

# draw histogram
plt.hist(image_gray.ravel(), 256,[0, 256]); plt.show()

# apply Canny edge detection
canny = imutils.auto_canny(image_gray)

# show the edges
cv2.imshow("Canny", canny)
cv2.waitKey(0)
