import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key

# Initialisation de MediaPipe Pose et du clavier virtuel
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
keyboard = Controller()

# Capture de la webcam
cap = cv2.VideoCapture(0)

# Seuils pour détecter la marche avec synchronisation des bras et jambes
walk_threshold = 0.1  # Seuil de différence verticale pour détecter un pas

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Conversion en RGB pour MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Vérifier si des postures corporelles sont détectées
    if results.pose_landmarks:
        # Dessiner les repères sur l'image
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Points clés pour la détection de la marche
        left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calcul de la différence verticale pour la marche
        left_knee_moving_up = abs(left_knee.y - left_hip.y) < walk_threshold
        right_knee_moving_up = abs(right_knee.y - right_hip.y) < walk_threshold

        # Détection de la marche avec synchronisation des bras et jambes opposées
        if left_knee_moving_up and right_knee_moving_up:
            # Si les jambes bougent et les bras aussi, il y a synchronisation
            # Détection des bras opposés pour la marche
            if left_shoulder.y < right_shoulder.y:  # Bras gauche en haut -> Jambe droite en bas
                print("🚶‍♂️ Marche avec bras gauche et jambe droite")
                keyboard.press('z')
            elif right_shoulder.y < left_shoulder.y:  # Bras droit en haut -> Jambe gauche en bas
                print("🚶‍♂️ Marche avec bras droit et jambe gauche")
                keyboard.press('z')
            else:
                keyboard.release('z')  # Aucun mouvement ou mouvement désynchronisé
        else:
            keyboard.release('z')  # Si les jambes ne bougent pas, pas de marche

    # Affichage de la webcam avec tracking du corps
    cv2.imshow("Minecraft Body Tracking", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermer la capture et les fenêtres
cap.release()
cv2.destroyAllWindows()