import cv2
from matplotlib import pyplot as plt

chess = cv2.imread("chess.png", cv2.IMREAD_GRAYSCALE)
#chess = cv2.cvtColor(chess, cv2.COLOR_BGR2GRAY)
plt.hist(chess.ravel(),256,[0,256]); plt.show()
cv2.imshow("Chess", chess)
cv2.waitKey(0)
#cv2.imwrite("chess.png", chess)

#Otsu
(T, chess_otsu) = cv2.threshold(chess, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Chess Otsu", chess_otsu)
cv2.waitKey(0)
#cv2.imwrite("chess_otsu.png", chess_otsu)

#adaptive threshold
chess_adaptive = cv2.adaptiveThreshold(chess, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, -15)
cv2.imshow("Chess Adaptive", chess_adaptive)
cv2.waitKey(0)
#cv2.imwrite("chess_adaptive.png", chess_adaptive)