import cv2
import numpy as np
import matplotlib.pyplot as plt

#me
import torch
from numpy import random
from urllib.request import urlopen


cap = cv2.VideoCapture("http://192.168.0.211:8080/stream?topic=/csi_cam_0/image_raw")




if cap.isOpened() :
    while True : 
        ret, frame = cap.read()



        if ret :
            

            


            cv2.imshow("FRAME", frame)
            key = cv2.waitKey(30)

            if key & 0xFF == 27 :
                break

cap.release()
cv2.destroyAllWindows()