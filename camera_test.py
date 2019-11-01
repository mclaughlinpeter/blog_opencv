import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(4)

if (cap.isOpened):
    print("Camera Open")

ret, frame = cap.read()
#cv2.imwrite("chess.png", frame)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow("Frame", frame)
plt.hist(frame.ravel(),256,[0,256]); plt.show()

print(type(frame))
print(frame.ndim)
print(frame.shape)


cap.release()

cv2.waitKey(0)

#(T, thresh) = cv2.threshold(frame, 130, 255, cv2.THRESH_BINARY)
(T, thresh) = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print("Otsu's thresholding value: {}".format(T))

cv2.imshow("Otsu's", thresh)
cv2.waitKey(0)

#adaptive threshold
thresh = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 15)
cv2.imshow("Adaptive", thresh)
cv2.waitKey(0)
#image = cv2.imread("image.png")
#cv2.imshow("Reopened", image)
#print(image.shape)

#cv2.waitKey(0)

#image = cv2.imread("/dev/video4")
#cv2.imshow("Original", image)
#cv2.waitKey(0)
