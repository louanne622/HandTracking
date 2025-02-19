import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Charger les données
df = pd.read_csv("donnees_mouvements.csv")

# Vérifier les premières lignes
print(df.head())

# Séparer les labels (Action) et les features (coordonnées des articulations)
X = df.drop(columns=["Action"])  # Coordonnées des articulations
y = df["Action"]  # Labels (Marcher, Courir, Sauter)

# Convertir les labels en nombres pour le modèle
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Diviser les données en ensemble d'entraînement et de test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Données prétraitées et prêtes pour l'entraînement !")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Créer et entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tester le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Précision du modèle : {accuracy:.2f}")
