# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:45:23 2024

@author: 0210s
"""
import numpy as np
import cv2
if __name__ == '__main__':
    for i in range(10):
        img = cv2.imread('data/ChineseNumber/'+str(i)+'.bmp')
        img = (255 - img).astype(np.uint8)
        cv2.imwrite('data/ChineseNumber/'+str(i)+'.bmp', img)