import os
import re
import time
import cv2
import numpy as np
from evaluate import evaluate

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def getImageList(dir):
    """
    Searches through a directory and fetches all files
    """
    fileList = os.listdir(dir)

    rgbList = []
    for file in fileList:
        rgbList.append(cv2.cvtColor(cv2.imread(
            dir + file), cv2.COLOR_BGR2RGB))

    return rgbList


def alignImageList(originalList):
    """
    Take all images and get aligned versions in reference to the first image
    """
    alignedList = []

    for image in originalList:
        imReg, h = alignImages(image, originalList[0])
        alignedList.append(imReg)

    return alignedList


def alignImages(im1, im2):
    """
    Align Images based on features
    https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
    """

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(
        im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


def blend(rgbList):
    """
    Median blending function
    """
    h = rgbList[0].shape[0]
    w = rgbList[0].shape[1]
    blended = np.copy(rgbList[0])

    # Initialize lists
    r = []
    g = []
    b = []

    start = time.time()
    for y in range(0, h):
        for x in range(0, w):
            # Loop and append to coresponding list
            for rgbImg in rgbList:
                # Don't include pure black pixels since they are usually from cropped/transformed images
                if all(pixelVal != 0 for pixelVal in rgbImg[y][x]):
                    r.append(rgbImg[y][x][0])
                    g.append(rgbImg[y][x][1])
                    b.append(rgbImg[y][x][2])

            # If there was a value that was appended, else don't change orignal
            if not (len(r) == 0 and len(g) == 0 and len(b) == 0):
                # Set current pixel to the median
                blended[y][x] = [np.median(r), np.median(g), np.median(b)]
                r.clear()
                b.clear()
                g.clear()

    end = time.time()

    # Convert back to BGR
    blendedBGR = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

    # Write image
    return blendedBGR, end-start


if __name__ == '__main__':
    # Change to desired directory
    dir = "./blur/"

    # Retrieve list
    print("Retrieving list from %s" % dir)
    originalList = getImageList(dir)

    # Align images
    print("Aligning images...")
    rgbList = alignImageList(originalList)

    # Blend images
    print("Blending images...")
    print("(Warning: algorithm is VERY slow, please wait)")
    blendedBGR, timeElap = blend(rgbList)
    print("Process completed in %d seconds." % timeElap)

    # Output file
    filename = re.sub('[./]', '', dir)
    cv2.imwrite("%s_blended.jpg" % filename, blendedBGR)
    print("File generated.")

    # Check if a ground truth directory exists
    grndTDir = dir.replace('./', '').replace('/', '_grndT')
    if os.path.isdir(grndTDir):
        grndT = cv2.imread(grndTDir + "/grndtruth.jpg")
        evaluate(blendedBGR, grndT)  # Run test (in evaluate.py)
    else:
        print("> There was no ground truth folder found")

    # Display result and a input file
    cv2.imshow("Reference Image", cv2.cvtColor(rgbList[0], cv2.COLOR_RGB2BGR))
    cv2.imshow("Result", blendedBGR)
    print("Images displayed, press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
