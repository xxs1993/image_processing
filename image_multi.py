## flip images in 0,1,2,3,5

import cv2
import os

from PIL.Image import ROTATE_90

path = "E:/BaiduNetdiskDownload/5"
def flip_image(dir):
    new_path = os.path.join(path,dir)
    for file in os.listdir(new_path):
        img = cv2.imread(os.path.join(new_path, file))
        img4 = cv2.rotate(src=img,rotateCode=ROTATE_90)
        cv2.imwrite(os.path.join(new_path, "4_" + file),img4)
        img1 = cv2.flip(img, 0)
        img2 = cv2.flip(img, 1)
        img3 = cv2.flip(img,-1)
        cv2.imwrite(os.path.join(new_path,"1_"+file),img1)
        cv2.imwrite(os.path.join(new_path, "2_" + file), img2)
        cv2.imwrite(os.path.join(new_path, "3_" + file), img3)
flip_image("0")
flip_image("5")
flip_image("1")
flip_image("2")
flip_image("3")