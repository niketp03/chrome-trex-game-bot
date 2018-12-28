from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import ImageGrab
import keyboard

model = load_model('final.h5')

tick = 0
while 1:
    scr = np.array(ImageGrab.grab(bbox=(175, 315, 775, 450)))
    cv2_scr = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)
    cv2_scr = cv2.resize(cv2_scr, (0,0), fx=0.2, fy=0.2)
    cv2_scr = cv2.resize(cv2_scr, (32, 32))
    x = np.interp(cv2_scr, (cv2_scr.min(), cv2_scr.max()), (-1, +1))
    pred = (model.predict(x.reshape(1, 32, 32, 1)))
    if pred[0][1] > pred[0][0]:
        print("up")
        keyboard.press('up')
        if (tick % 29) == 0:
            keyboard.send('up')
    else:
        print("wait")
        keyboard.release('up')
    tick += 1
