import numpy as np
import cv2

# load the image and convert it to grayscale
image = cv2.imread("./logo-cropped.png")
orig = image.copy()
gray = cv2.split(image)[0]

detector = cv2.xfeatures2d.SURF_create()
(kps, _) = detector.detectAndCompute(gray, None)

print("# of keypoints: {}".format(len(kps)))

# loop over the keypoints and draw them
for kp in kps:
	r = int(0.5 * kp.size)
	(x, y) = np.int0(kp.pt)
	cv2.circle(image, (x, y), r, (0, 255, 255), 2)

# show the image
cv2.imshow("Original", orig)
cv2.imshow("Keypoints", image)
cv2.waitKey(0)
