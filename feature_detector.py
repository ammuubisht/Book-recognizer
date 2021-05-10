import cv2 as cv
import numpy as np

img1 = cv.imread('ImageData/the_subtle_art.jpg', 0)
img2 = cv.imread('TrainImages/tsangf_train.jpeg', 0)

orb = cv.ORB_create(nfeatures=1000)

keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# keypoint1 = cv.drawKeypoints(img1, keypoints1, None)
# keypoint2 = cv.drawKeypoints(img2, keypoints2, None)

bf = cv.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print(len(good))
img3 = cv.drawMatchesKnn(img1, keypoints1, img2,
                         keypoints2, good, None, flags=2)

# cv.imshow('Keypoint1', keypoint1)
# cv.imshow('Keypoint2', keypoint2)
cv.imshow('Image3', img3)
cv.waitKey(0)
