import pandas as pd

# Créer un DataFrame test avec des colonnes
df = pd.DataFrame({
    "x": [0.5, 0.6, 0.7],
    "y": [0.8, 0.9, 0.85],
    "z": [0.1, 0.2, 0.15],
    "visibility": [0.99, 0.98, 0.97],
    "label": ["marche", "marche", "marche"]
})

# Sauvegarder en CSV (sans index)
df.to_csv("mouvements.csv", index=False)

print("Fichier 'mouvements.csv' créé avec succès !")
