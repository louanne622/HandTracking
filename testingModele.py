import cv2
import mediapipe as mp
import pickle
import numpy as np

# Charger le modèle et l'encodeur de labels
with open("modele_mouvement.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Initialisation de Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialisation de la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append(landmark.x)
            landmarks.append(landmark.y)
            landmarks.append(landmark.z)

        # Transformer en tableau numpy
        X_test = np.array([landmarks])

        # Prédire le mouvement
        prediction = model.predict(X_test)
        mouvement_pred = label_encoder.inverse_transform(prediction)[0]

        # Afficher la prédiction
        cv2.putText(frame, f"Mouvement: {mouvement_pred}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Affichage de la webcam
    cv2.imshow("Détection de mouvements", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quitter avec 'q'
        break

cap.release()
cv2.destroyAllWindows()
