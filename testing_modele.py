import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Charger les données
df = pd.read_csv("donnees_mouvements.csv")

# Séparer les features (X) et les labels (y)
X = df.drop(columns=["mouvement"])  # Supprime la colonne "mouvement" pour ne garder que les coordonnées
y = df["mouvement"]  # Labels des mouvements

# Encoder les labels (marche, course, saut, immobile, etc.)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Séparer en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Entraîner un modèle RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tester le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy * 100:.2f}%")

# Sauvegarder le modèle et l'encodeur de labels
with open("modele_mouvement.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Modèle entraîné et sauvegardé sous 'modele_mouvement.pkl'")
