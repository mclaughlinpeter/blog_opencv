import cv2
from matplotlib import pyplot as plt

wrench = cv2.imread("wrench.png", cv2.IMREAD_GRAYSCALE)
#print(wrench.shape)
#wrench = cv2.cvtColor(wrench, cv2.COLOR_BGR2GRAY)
#cv2.imwrite("wrench_grayscale.png", wrench)
cv2.imshow("Wrench", wrench)
cv2.waitKey(0)

plt.hist(wrench.ravel(),256,[0,256]); plt.show()

#manual threshold
(T, wrench_50) = cv2.threshold(wrench, 50, 255, cv2.THRESH_BINARY)
cv2.imshow("Wrench Threshold 50", wrench_50)
cv2.waitKey(0)
#cv2.imwrite("wrench_grayscale_threshold_50.png", wrench_50)

wrench_illum = cv2.imread("wrench_illumination.png", cv2.IMREAD_GRAYSCALE)
#wrench_illum = cv2.cvtColor(wrench_illum, cv2.COLOR_BGR2GRAY)
cv2.imshow("Wrench Illumination", wrench_illum)
cv2.waitKey(0)
#cv2.imwrite("wrench_grayscale_illum.png", wrench_illum)

plt.hist(wrench_illum.ravel(),256,[0,256]); plt.show()

#(T, wrench_illum_50) = cv2.threshold(wrench_illum, 50, 255, cv2.THRESH_BINARY)
#cv2.imshow("Wrench Illuminated Threshold 50", wrench_illum_50)
#cv2.waitKey(0)
#cv2.imwrite("wrench_grayscale_illum_threshold_50.png", wrench_illum_50)

# Otsu's method
(T, wrench_illum_otsu) = cv2.threshold(wrench_illum, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Wrench Illuminated Otsu", wrench_illum_otsu)
print("Otsu's threshold: {}".format(T))
cv2.waitKey(0)
#cv2.imwrite("wrench_grayscale_illum_otsu.png", wrench_illum_otsu)

