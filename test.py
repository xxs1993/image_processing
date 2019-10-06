import os

import cv2

path = "D:/git/data/HuSHem"
file_path = os.path.join(path,"0")

for file in os.listdir(file_path):
    if os.path.isdir(os.path.join(file_path,file)):
        continue
    img = cv2.imread(os.path.join(file_path,file),cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(file_path,"n",file.replace("BMP",'tiff')),img)
