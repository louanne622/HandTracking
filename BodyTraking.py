import cv2
import mediapipe as mp
import time

# Initialisation de la caméra
cap = cv2.VideoCapture(0)

# Initialisation de MediaPipe Pose avec mode multi-personnes
mpPose = mp.solutions.pose
pose = mpPose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        for landmarks in results.pose_landmarks:
            mpDraw.draw_landmarks(img, landmarks, mpPose.POSE_CONNECTIONS)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Pose Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
