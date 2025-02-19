import cv2
import mediapipe as mp
import pandas as pd
import time
import os

# Initialisation de Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialisation de la webcam
cap = cv2.VideoCapture(0)

# Paramètres d'enregistrement
mouvement = input("Entrez le nom du mouvement (ex: immobile, marche, course) : ").strip().lower()
temps_preparation = 5  # Temps pour se placer
temps_enregistrement = 10  # Temps d'enregistrement en secondes

# Liste pour stocker les données
data = []

# Compte à rebours avant d'enregistrer
print(f"Préparez-vous pour le mouvement '{mouvement}' dans {temps_preparation} secondes...")
for i in range(temps_preparation, 0, -1):
    print(i)
    time.sleep(1)

print("Enregistrement en cours...")

start_time = time.time()
while time.time() - start_time < temps_enregistrement:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Vérifier si des landmarks sont détectés
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append(landmark.x)
            landmarks.append(landmark.y)
            landmarks.append(landmark.z)

        # Ajouter les données avec le label du mouvement
        data.append([mouvement] + landmarks)

    # Affichage en direct
    cv2.imshow("Capture des mouvements", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quitter avec 'q'
        break

# Vérifier si des données ont été collectées
if data:
    # Déterminer le nombre de colonnes dynamiquement
    num_landmarks = len(results.pose_landmarks.landmark) * 3  # (x, y, z) pour chaque landmark
    column_names = ["mouvement"] + [f"coord_{i}" for i in range(1, num_landmarks + 1)]

    # Créer un DataFrame
    df = pd.DataFrame(data, columns=column_names)

    # Vérifier si le fichier existe déjà
    file_exists = os.path.isfile("donnees_mouvements.csv")

    # Sauvegarde des données dans un fichier CSV
    df.to_csv("donnees_mouvements.csv", mode='a', index=False, header=not file_exists)

    print(f"Enregistrement terminé. Données sauvegardées dans 'donnees_mouvements.csv'.")
else:
    print("Aucune donnée enregistrée.")

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
