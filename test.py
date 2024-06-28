import cv2
import time

cam = cv2.VideoCapture(0)

pTime = 0
while True:
    _, frame = cam.read()

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,255,0), 2)

    cv2.imshow("Original", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
