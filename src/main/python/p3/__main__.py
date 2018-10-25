import p3.p4
import time
import cv2

call = p3.p4.P4(True)
call.start()
while True:
    start = time.time()
    frame = call.get_frame_fast()
    reward = call.get_frame_reward()
    if reward != 0:
        print(reward)
    call.get_frame_reward()
    end = time.time()
    #print(end * 1000 - start * 1000)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if 1 / 60 > (end - start):
        time.sleep(1 / 60 - (end - start))

# When everything done, release the capture
cv2.destroyAllWindows()
