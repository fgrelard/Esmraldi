from __future__ import print_function
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import math

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.1

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fixed", help="Fixed image")
parser.add_argument("-m", "--moving", help="Moving image")
parser.add_argument("-o", "--output", help="Output")
args = parser.parse_args()

fixedname = args.fixed
movingname = args.moving
outputname = args.output

def edge(im):
    dx =  cv2.Sobel(im,cv2.CV_32F,1,0,ksize=5)
    dy = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=5)
    magnitude = cv2.magnitude(dx, dy)
    normalizedImg = np.uint8(cv2.normalize(magnitude,  None, 0, 255, cv2.NORM_MINMAX))
    return normalizedImg

def drawKeypoints(img, kp, color, flags):
    img2 = np.zeros((img.shape[1],img.shape[0],3), np.uint8)
    for marker in kp:
        img2 = cv2.drawMarker(img2, tuple(int(i) for i in marker.pt), color=color)
    plt.imshow(img2)
    plt.show()
    return img2

def sift(im1, im2):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(im1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matchesFull = flann.knnMatch(descriptors1,descriptors2,k=2)

    # Need to draw only good matches, so create a mask
    matches = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matchesFull):
        if m.distance < 0.8*n.distance:
            matches.append(n)
    print(len(matches))
    imMatches = cv2.drawMatches(im1,keypoints1,im2,keypoints2,matches,None)
    cv2.imwrite("matches.jpg", imMatches)
    return matches, keypoints1, keypoints2

def orb(im1, im2):
    orb = cv2.ORB_create(nfeatures=MAX_FEATURES, scoreType=cv2.ORB_FAST_SCORE)

    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)
    img2_kp = cv2.drawKeypoints(im2, keypoints2, None, color=(0,255,0), \
    flags=cv2.DrawMatchesFlags_DEFAULT)

    plt.figure()
    plt.imshow(img2_kp)
    plt.show()
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
#    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2, None)
    print(len(matches))
    # Sort matches by score
    matches.sort(key=lambda x:x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches))
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
    return matches, keypoints1, keypoints2

def alignImages(im1, im2):

  # Convert images to grayscale
  im1Gray = edge(im1)
  im2Gray = edge(im2)

  im1Gray = im1.copy()
  im2Gray = im2.copy()

  # Detect ORB features and compute descriptors.
  matches, keypoints1, keypoints2 = sift(im1Gray, im2Gray)
  print(len(matches))
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
  for i, match in enumerate(matches):
      points1[i, :] = keypoints1[match.queryIdx].pt
      points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  h, mask = cv2.estimateAffinePartial2D(points1, points2)
  # theta = -math.atan2(h[0,1], h[0,0]) * 180 / math.pi
  # print(theta)
  # Use homography
  height, width = im2.shape
  print(height, " ", width, " ")
  h = h.astype(np.float32)
  im1Reg = cv2.warpAffine(im1, h, (width, height))


  return im1Reg, h


if __name__ == '__main__':

  # Read reference image
  imReference = cv2.imread(fixedname, cv2.IMREAD_GRAYSCALE)

  # Read image to be aligned
  im = cv2.imread(movingname, cv2.IMREAD_GRAYSCALE)

  print("Aligning images ...")
  # Registered image will be resotred in imReg.
  # The estimated homography will be stored in h.
  imReg, h = alignImages(im, imReference)

  fig, ax=plt.subplots(1,2)
  ax[0].imshow(imReference)
  ax[1].imshow(imReg)

  plt.show()
  # Write aligned image to disk.
  outFilename = "aligned.jpg"
  # print("Saving aligned image : ", outFilename);
  #cv2.imwrite(outFilename, imReg)

  # Print estimated homography
  print("Estimated homography : \n",  h)
