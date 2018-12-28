import os
import cv2

for filename in os.listdir("data_cl/wait"):
    img = cv2.imread("data_cl/wait/" + filename, 0)
    img = cv2.resize(img, (32, 32))
    cv2.imwrite("data_cl/wait/" + filename, img)

for filename in os.listdir("data_cl/up"):
    img = cv2.imread("data_cl/up/" + filename, 0)
    img = cv2.resize(img, (32, 32))
    cv2.imwrite("data_cl/up/" + filename, img)
