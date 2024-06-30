import cv2
import time
import os
cap = cv2.VideoCapture(0)

pTime = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame from webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # faces = detector.detect_faces(rgb_frame)

    # FPS
    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime
    # for face_percent in faces:
    #     percent = face_percent["confidence"]
    #     cv2.putText(frame, f"Percent: {int(percent*100)} %", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # for face in faces:
    #     x, y, w, h = face['box']
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) == ord('v'):
        cv2.imwrite('Images\\Input_Image\\input_image.jpg', frame)
    elif cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()