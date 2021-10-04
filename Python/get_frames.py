import cv2
import os
from pathlib import Path


user = "consumingcouple"
folder = f"/Users/thomascho/code/tiktokvideos/{user}"

temp = []
for root, dirs, files in os.walk(folder, topdown=False):
    for file in files:
        if file.endswith('.mp4'):
            temp.append(os.path.join(root, file))
            vidcap = cv2.VideoCapture(os.path.join(root, file))
            success, image = vidcap.read()
            count = 0
            print (os.path.join(root, file))
            print (root)
            while success:
                cv2.imwrite(f"{root}/frame{count}.jpg", image)  # save frame as JPEG file
                success, image = vidcap.read()
                # print('Read a new frame: ', success)
                count += 1
