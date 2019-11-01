import numpy as np
import cv2
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create, DescriptorMatcher_create

# setup keypoint detector, extractor and matcher
detector = FeatureDetector_create("SURF")
extractor = DescriptorExtractor_create("RootSIFT")
matcher = DescriptorMatcher_create("BruteForce")

# load the two grayscale images and take the first channel
logo = cv2.imread("./logo-cropped.png")
scene = cv2.imread("./cluttered-scene.png")
grayLogo = cv2.split(logo)[0]
grayScene = cv2.split(scene)[0]

# detect keypoints
kpsLogo = detector.detect(grayLogo)
kpsScene = detector.detect(grayScene)

# extract features
(kpsLogo, featuresLogo) = extractor.compute(grayLogo, kpsLogo)
(kpsScene, featureScene) = extractor.compute(grayScene, kpsScene)

# match the keypoints
rawMatches = matcher.knnMatch(featuresLogo, featureScene, 2)
matches = []

if rawMatches is not None:
	for match in rawMatches:
		# ensure the distance passes David Lowe's ratio test
		if len(match) == 2 and match[0].distance < match[1].distance * 0.8:
			matches.append((match[0].trainIdx, match[0].queryIdx))

	# show some diagnostic information
	print("keypoints in logo: {}".format(len(kpsLogo)))
	print("keypoints in scene: {}".format(len(kpsScene)))
	print("matched keypoints: {}".format(len(matches)))

	# initialize the output visualization image
	(hA, wA) = logo.shape[:2]
	(hB, wB) = scene.shape[:2]
	vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
	vis[0:hA, 0:wA] = logo
	vis[0:hB, wA:] = scene

	# loop over the matches
	for (trainIdx, queryIdx) in matches:
		# generate a random color and draw the match
		color = np.random.randint(0, high=255, size=(3,))
		color = tuple(map(int, color))
		ptA = (int(kpsLogo[queryIdx].pt[0]), int(kpsLogo[queryIdx].pt[1]))
		ptB = (int(kpsScene[trainIdx].pt[0] + wA), int(kpsScene[trainIdx].pt[1]))
		cv2.line(vis, ptA, ptB, color, 2)
	# show the visualization
	cv2.imshow("Matched Keypoints", vis)
	cv2.waitKey(0)
