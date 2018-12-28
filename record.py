import numpy as np
import cv2
from PIL import ImageGrab
import keyboard

tick = 0;
while 1:
    scr = np.array(ImageGrab.grab(bbox=(175, 315, 775, 450)))
    cv2_scr = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)
    cv2_scr = cv2.resize(cv2_scr, (0,0), fx=0.2, fy=0.2)
    cv2.imshow('display', cv2_scr)
    if keyboard.is_pressed('up'):
        print("up")
        cv2.imwrite(f'data/up/{tick}.png', cv2_scr)
    elif keyboard.is_pressed('down'):
        print("down")
        cv2.imwrite(f'data/down/{tick}.png', cv2_scr)
    else:
        print("wait")
        cv2.imwrite(f'data/wait/{tick}.png', cv2_scr)

    tick += 1;
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
