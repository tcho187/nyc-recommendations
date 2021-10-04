import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)


def thresh_callback(val, source):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(source, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # Draw contours
    # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # for i in range(len(contours)):
    #     color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    #     cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    # # Show in a window
    # cv.imshow('Contours', drawing)
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        area = w * h
        if area > 10000:
            cv.rectangle(source, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cropped = source[y:y + h, x:x + w]
            yield cropped
            # cv.imshow('Cropped', cropped)
    # cv.imshow('Contours2', source)

if __name__ == '__main__':
    src = cv.imread("/Users/thomascho/code/tiktokvideos/consumingcouple/6958953030347672837/frame0.jpg")

    # Convert image to gray and blur it
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))

    source_window = 'Source'
    cv.namedWindow(source_window)
    cv.imshow(source_window, src)
    max_thresh = 255

    thresh = 100
    cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
    # Detect edges using Canny
    thresh_callback(thresh, src_gray)
    cv.waitKey()
    print ("End")
