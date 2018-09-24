import cv2
import numpy as np
import time

import screen_watcher

watcher = screen_watcher.ScreenWatcher()

while(True):
    # Capture frame-by-frame

    start = time.time() * 1000
    frame = next(watcher)
    frame = cv2.resize(np.array(frame), (84, 84), interpolation=cv2.INTER_LINEAR)[:,:,:3].flatten()
    end = time.time() * 1000
    print(end - start)