import cv2
import numpy as np


def evaluate(res, grndT):
    """
    Get the image difference, threshold and binarize before calculating
    the percent accuracy 
    """
    print("Evaluating result with groundtruth...")
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    grndT = cv2.cvtColor(grndT, cv2.COLOR_BGR2GRAY)

    # Subtract ground truth from result
    imgDiff = cv2.subtract(grndT, res)

    # Threshold
    retval2, threshold2 = cv2.threshold(
        imgDiff, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Calculate percentage
    accuracy = 100 - (cv2.countNonZero(
        threshold2) / (threshold2.shape[0] * threshold2.shape[1]) * 100)
    print("> There is a {:.2f}% accuracy between the result and ground truth".format(
        accuracy))

    # Display threshold
    cv2.imshow("Threshold", threshold2)


if __name__ == '__main__':
    # Change to desired image filepaths
    resFilePath = "./synth2_blended.jpg"
    grndTFilePath = "./synth2_grndT/grndtruth.jpg"

    res = cv2.imread(resFilePath)
    grndT = cv2.imread(grndTFilePath)

    evaluate(res, grndT)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
