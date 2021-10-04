import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
import os
from matplotlib import pyplot as plt

from imutils.video import FPS
import imutils

from find_contour import thresh_callback


def load(img, newW, newH, **kwargs):
    # resize the original image to new dimensions
    image = cv2.resize(img, (newW, newH))
    (H, W) = image.shape[:2]

    # construct a blob from the image to forward pass it to EAST model
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    return blob


# Returns a bounding box and probability score if it is more than minimum confidence
def predictions(prob_score, geo, args, **kwargs):
    (numR, numC) = prob_score.shape[2:4]
    boxes_list = []
    confidence_val_list = []

    # loop over rows
    for y in range(0, numR):
        scoresData = prob_score[0, 0, y]
        x0 = geo[0, 0, y]
        x1 = geo[0, 1, y]
        x2 = geo[0, 2, y]
        x3 = geo[0, 3, y]
        anglesData = geo[0, 4, y]

        # loop over the number of columns
        for i in range(0, numC):
            if scoresData[i] < args["min_confidence"]:
                continue

            (offX, offY) = (i * 4.0, y * 4.0)

            # extracting the rotation angle for the prediction and computing the sine and cosine
            angle = anglesData[i]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # using the geo volume to get the dimensions of the bounding box
            h = x0[i] + x2[i]
            w = x1[i] + x3[i]

            # compute start and end for the text pred bbox
            endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
            endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
            startX = int(endX - w)
            startY = int(endY - h)

            boxes_list.append((startX, startY, endX, endY))
            confidence_val_list.append(scoresData[i])

    # return bounding boxes and associated confidence_val
    return boxes_list, confidence_val_list


def decode_predictions(scores, geometry, min_confidence):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < min_confidence:
                continue
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    # return a tuple of the bounding boxes and associated confidences
    return rects, confidences


def read_video(args, vid_file, **kwargs):
    # initialize the original frame dimensions, new frame dimensions,
    # and ratio between the dimensions
    (W, H) = (None, None)
    (newW, newH) = (args["width"], args["height"])
    (rW, rH) = (None, None)
    # load the pre-trained EAST model for text detection
    net = cv2.dnn.readNet(args["east"])

    # The following two layer need to pulled from EAST model for achieving this.
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    vs = cv2.VideoCapture(vid_file)

    # start the FPS throughput estimator
    fps = FPS().start()

    # loop over frames from the video stream
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = vs.read()
        frame = frame[1]
        # check to see if we have reached the end of the stream
        if frame is None:
            break
        plt.imshow(frame)
        plt.title('gray')
        plt.show()
        # resize the frame, maintaining the aspect ratio
        frame = imutils.resize(frame, width=1000)
        plt.imshow(frame)
        plt.title('resize')
        plt.show()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        #otsu
        threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        invert = 255 - threshold
        frame = np.dstack([invert] * 3)
        # Invert the image
        #threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        plt.imshow(frame)
        plt.title('invert')
        plt.show()
        # frame = np.dstack([gray, gray, gray])

        kernel = np.ones((1, 1), np.uint8)
        dilate = cv2.dilate(threshold, kernel, iterations=1)
        erode = cv2.erode(dilate, kernel, iterations=1)

        #thresholding

        frame = np.dstack([invert] * 3)

        plt.imshow(frame)
        plt.title('Preprocessed')
        plt.show()
        orig = frame.copy()

        # if our frame dimensions are None, we still need to compute the
        # ratio of old frame dimensions to new frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            rW = W / float(newW)
            rH = H / float(newH)
        # resize the frame, this time ignoring aspect ratio
        frame = cv2.resize(frame, (newW, newH))

        # construct a blob from the frame and then perform a forward pass
        # of the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = decode_predictions(scores, geometry, args['min_confidence'])
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # initialize the list of results
        results = []
        texts = []
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # extract the region of interest
            r = orig[startY:endY, startX:endX]

            # configuration setting to convert image to string.
            configuration = ("-l eng --oem 1 --psm 6")
            ##This will recognize the text from the image of bounding box
            text = pytesseract.image_to_string(r, config=configuration)

            # append bbox coordinate and associated text to the list of results
            results.append(((startX, startY, endX, endY), text))

            texts.append(text)

            # Display the image with bounding box and recognized text
            orig_image = orig.copy()

            # Moving over the results and display on the image
            for ((start_X, start_Y, end_X, end_Y), text) in results:
                # display the text detected by Tesseract
                print("{}\n".format(text))

                # Displaying text
                text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
                cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
                              (0, 0, 255), 2)
                cv2.putText(orig_image, text, (start_X, start_Y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # draw the bounding box on the frame
            # cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # update the FPS counter
            fps.update()

            # show the output frame
            plt.imshow(orig_image)
            plt.title('Output')
            plt.show()
            concat = ','.join(texts)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        vs.release()

        # close all windows
        cv2.destroyAllWindows()


def read_frame(args, cropped, **kwargs):
    # load the pre-trained EAST model for text detection
    plt.imshow(cropped)
    plt.title('Cropped')
    plt.show()
    threshold = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    invert = 255 - threshold

    cnts = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for contour in cnts:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(invert, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(invert)
    plt.title('invert Image')
    plt.show()
    cnts = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if (aspect_ratio >= 2.5 or area < 75):
            cv2.drawContours(threshold, [c], -1, (255, 255, 255), -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    image = np.dstack([invert] * 3)
    # kernel = np.ones((1, 1), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)
    # image = cv2.erode(image, kernel, iterations=1)

    plt.imshow(image)
    plt.title('Preprocessed Image')
    plt.show()

    # Saving a original image and shape
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # set the new height and width to default 320 by using args #dictionary.
    (newW, newH) = (args["width"], args["height"])

    # Forward pass the blob from the image to get the desired output layers
    net.setInput(load(img=image, newW=newW, newH=newH))
    (scores, geometry) = net.forward(layerNames)
    # Find predictions and  apply non-maxima suppression
    (boxes, confidence_val) = predictions(scores, geometry, args)
    boxes = non_max_suppression(np.array(boxes), probs=confidence_val)

    ##Text Detection and Recognition

    # initialize the list of results
    results = []

    # Calculate the ratio between original and new image for both height and weight.
    # This ratio will be used to translate bounding box location on the original image.
    rW = origW / float(newW)
    rH = origH / float(newH)

    texts = []
    # loop over the bounding boxes to find the coordinate of bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # extract the region of interest
        r = orig[startY:endY, startX:endX]

        # configuration setting to convert image to string.
        configuration = ("-l eng --oem 1 --psm 6")
        try:
            ##This will recognize the text from the image of bounding box
            text = pytesseract.image_to_string(r, config=configuration)

            # append bbox coordinate and associated text to the list of results
            results.append(((startX, startY, endX, endY), text))

            texts.append(text)

            # Display the image with bounding box and recognized text
            orig_image = orig.copy()

            # Moving over the results and display on the image
            for ((start_X, start_Y, end_X, end_Y), text) in results:
                # display the text detected by Tesseract
                print("{}\n".format(text))

                # Displaying text
                text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
                cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
                              (0, 0, 255), 2)
                cv2.putText(orig_image, text, (start_X, start_Y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

                plt.imshow(orig_image)
                plt.title('Output')
                plt.show()
                concat = ','.join(texts)
        except Exception as e:
            print(e)
    return True


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    user = "consumingcouple"
    folder = f"/Users/thomascho/code/tiktokvideos/{user}"

    default_args = {
        "east": "/Users/thomascho/code/frozen_east_text_detection.pb",
        "min_confidence": 0.5,
        "width": 320,
        "height": 320}

    for root, dirs, files in os.walk(folder, topdown=False):
        for index, file in enumerate(files):
            if file.endswith('frame0.jpg'):
                full_path = os.path.join(root, file)
                net = cv2.dnn.readNet(default_args["east"])

                # The following two layer need to pulled from EAST model for achieving this.
                layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
                image = cv2.imread(full_path)
                plt.imshow(image)
                plt.title('Image')
                plt.show()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                cropped = thresh_callback(100, blur)
                for crop in cropped:
                    read_frame(default_args, cropped=crop)
                print(index)
