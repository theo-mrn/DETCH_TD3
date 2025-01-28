import json

# Gestion des balances, des poids et des modèles
db_file = "model_data.json"

# Charger la base de données
def load_db():
    try:
        with open(db_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Sauvegarder la base de données
def save_db(data):
    with open(db_file, "w") as f:
        json.dump(data, f)

# Enregistrer un modèle avec un dépôt initial
def register_model(model_name, deposit):
    db = load_db()
    if model_name in db:
        return {"error": "Model already registered"}

    db[model_name] = {"balance": deposit, "weight": 1.0}
    save_db(db)
    return {"message": f"Model {model_name} registered with deposit {deposit}"}

# Mettre à jour les poids et les balances après une prédiction
def update_weights(predictions, target_class, reward=10, penalty=5):
    db = load_db()
    for model_name, model_prediction in predictions.items():
        accuracy = 1 if model_prediction == target_class else 0
        db[model_name]["balance"] += (accuracy * reward) - ((1 - accuracy) * penalty)
        db[model_name]["weight"] = max(0.1, db[model_name]["balance"] / 1000)  # Normalisation du poids

    save_db(db)
    return db