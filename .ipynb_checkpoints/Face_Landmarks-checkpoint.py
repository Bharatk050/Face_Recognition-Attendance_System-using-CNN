import cv2
import mediapipe as mp
import time
import numpy as np


webcam = True
cap = cv2.VideoCapture(0)
          
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.holistic
# FaceMesh =  mpFaceMesh.FaceMesh(max_num_faces=1)
holistic_model = mpFaceMesh.Holistic(
    min_tracking_confidence= 0.5,
    min_detection_confidence=0.5
)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

pTime = 0
while cap.isOpened():
    if webcam:
        success, frame = cap.read()
    else:
        frame = cv2.imread("<path>")

    # Frame Size
    h_frame = frame.shape[0]
    b_frame = frame.shape[1]

    lst = []
    blank_Image = np.zeros((h_frame, b_frame, 3), np.uint8)

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(imgRGB)

    if results.face_landmarks:
        mpDraw.draw_landmarks(blank_Image, results.face_landmarks, mpFaceMesh.FACEMESH_CONTOURS,
            mpDraw.DrawingSpec(
                # Inner Face
                color=(0,0,0),
                thickness=1,
                circle_radius=1
            ),
            # Outer Face
            mpDraw.DrawingSpec(
                color=(255, 255, 255),
                thickness=1,
                circle_radius=1
            )
        )

        # Drawing Right hand Land_Marks
        mpDraw.draw_landmarks(
            blank_Image,
            results.right_hand_landmarks,
            mpFaceMesh.HAND_CONNECTIONS
            # landmark_drawing_spec= drawSpec
        )

        # Drawing Left hand Land_Marks
        mpDraw.draw_landmarks(
            blank_Image,
            results.left_hand_landmarks,
            mpFaceMesh.HAND_CONNECTIONS
            # landmark_drawing_spec= drawSpec
        )
        # for facelms in mpFaceMesh.FACEMESH_CONTOURS:
        #     print(facelms)

        # print(results.face_landmarks)
        for id, landmark in enumerate(results.face_landmarks.landmark):
            ih, iw, ic = frame.shape
            x, y, z = int(landmark.x * iw), int(landmark.y * ih), int(landmark.z * ic)
            # print(id, x, y)
            lst.append((x, y, z))
    np_lst = np.array(lst)
    if len(np_lst) != 0:
        print(np_lst)

    # FPS
    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)


    cv2.imshow('Image', frame)
    cv2.imshow("BlankImage", blank_Image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()