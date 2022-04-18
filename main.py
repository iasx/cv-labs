import sys
import pcasift
import cv2 as cv
from matplotlib import pyplot as pl
from numpy import int32, float32, uint8, zeros
from pcasift import computeKeypointsAndDescriptors


def load(x): return cv.imread(x, cv.IMREAD_GRAYSCALE)


def show(x): pl.imshow(x, cmap='gray'); pl.show()


############################################################################### DESCRIPTORS

if len(sys.argv) != 3:
    print("Usage: main.py <image> <image>")
    sys.exit(1)

I1 = load(sys.argv[1])
I2 = load(sys.argv[2])

K1, D1 = computeKeypointsAndDescriptors(I1)
K2, D2 = computeKeypointsAndDescriptors(I2)

############################################################################### MATCHING

flann = cv.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
matches = flann.knnMatch(D1, D2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) < 10:
    print("Not enough matches")

############################################################################### HOMOGRAPHY

srcPts = float32([K1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dstPts = float32([K2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

M = cv.findHomography(srcPts, dstPts, cv.RANSAC, 5.0)[0]

############################################################################### DRAWING

h, w = I1.shape
pts = float32([[0, 0],
               [0, h - 1],
               [w - 1, h - 1],
               [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv.perspectiveTransform(pts, M)

I2 = cv.polylines(I2, [int32(dst)], True, 255, 3, cv.LINE_AA)

h1, w1 = I1.shape
h2, w2 = I2.shape
w3 = w1 + w2
h3 = max(h1, h2)
hd = int((h2 - h1) / 2)
result = zeros((h3, w3, 3), uint8)

for i in range(3):
    result[hd:hd + h1, :w1, i] = I1
    result[:h2, w1:w1 + w2, i] = I2

for m in good:
    pt1 = (int(K1[m.queryIdx].pt[0]), int(K1[m.queryIdx].pt[1] + hd))
    pt2 = (int(K2[m.trainIdx].pt[0] + w1), int(K2[m.trainIdx].pt[1]))
    cv.line(result, pt1, pt2, (255, 0, 0))

show(result)
