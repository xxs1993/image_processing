from collections import Counter

import cv2;
import numpy as np
import matplotlib.pyplot as plt
def calcGrayHist(I):
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist

Counter()
img = cv2.imread("450.tiff",0)
## normalization
# Imin, Imax = cv2.minMaxLoc(img)[:2]
#
# Omin, Omax = 0, 255
# # 计算a和b的值
# a = float(Omax - Omin) / (Imax - Imin)
# b = Omin - a * Imin
# out = a * img + b
# out = out.astype(np.uint8)






img = cv2.resize(img,(213,213))
out = cv2.resize(out,(213,213))
cv2.imshow("img",img)
cv2.imshow("out",out)

cv2.waitKey()

