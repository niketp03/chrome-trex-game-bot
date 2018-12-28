from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import ImageGrab
import keyboard

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 600)

model = load_model('final.h5')

tick = 0
while 1:
    scr = np.array(ImageGrab.grab(bbox=(175, 315, 775, 450)))
    cv2_scr = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)
    cv2_scr = cv2.resize(cv2_scr, (0,0), fx=0.1, fy=0.1)
    cv2_scr = cv2.resize(cv2_scr, (32, 32))
    cv2.imshow('image', cv2_scr)
    cv2.waitKey(1)
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

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
