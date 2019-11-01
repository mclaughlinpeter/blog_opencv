import cv2

image = cv2.imread("./close-up-1.png")
print("Original image shape: ", image.shape)
orig = image.copy()
cv2.rectangle(image, (530, 450), (780, 770), (0, 255, 0))
cv2.imshow("Original", image)
cv2.waitKey(0)

cropped = orig[450:770, 530:780]
cv2.imshow("Cropped", cropped)
cv2.waitKey(0)

cv2.imwrite("close-up-1-cropped.png", cropped)
