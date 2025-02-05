import cv2
import mediapipe as mp
import time

# Initialisation de la caméra
cap = cv2.VideoCapture(0)

# Initialisation de MediaPipe Face Detection
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)

            cv2.rectangle(img, bbox, (255, 0, 255), 3)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
