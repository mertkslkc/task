
# Kodu lütfen terminalden çalıştırınız örnek / Please run the code from the terminal example
#  python starmap.py --image small.png

#Kodun ve resimlerin aynı klasörde olduğundan emin olunuz...
#Make sure the code and images are in the same folder.

import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import argparse
import os

MIN_MATCH_COUNT = 10

paths = os.getcwd()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = paths)
args = vars(ap.parse_args())
img1 = cv2.imread(args["image"],0)

for degrees in range(360, 0,-1):

    img2 = cv2.imread("map.png", 0)
    img2 = imutils.rotate_bound(img2, degrees)

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,1, cv2.LINE_AA)

        cv2.imwrite("result.png",img2)
        break

    else:
        #print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        #print(f"Deg:{degrees}")
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = None,
                       matchesMask = matchesMask,
                       flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    img3 = cv2.resize(img3, (960, 540))
    cv2.imshow("screen",img3)
    cv2.waitKey(1)
cv2.destroyAllWindows()