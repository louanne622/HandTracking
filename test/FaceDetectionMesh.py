import cv2
import mediapipe as mp
import time

# Initialisation de la caméra
cap = cv2.VideoCapture(0)


mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_LEFT_EYE)


    cv2.imshow('Image', img)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
