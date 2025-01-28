from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialiser l'application Flask
app = Flask(__name__)

# Charger les modèles entraînés
models = {
    "random_forest": joblib.load("models/random_forest.joblib"),
    "logistic_regression": joblib.load("models/logistic_regression.joblib"),
    "svm": joblib.load("models/svm.joblib")  # SVM doit avoir `probability=True`
}

# Mapper les classes aux noms des espèces
iris_classes = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

@app.route('/')
def home():
    return "Bienvenue dans l'API Flask ! Utilisez l'endpoint /predict pour obtenir les probabilités des classes."

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Récupérer les paramètres depuis la requête
        sepal_length = float(request.args.get("sepal_length"))
        sepal_width = float(request.args.get("sepal_width"))
        petal_length = float(request.args.get("petal_length"))
        petal_width = float(request.args.get("petal_width"))

        # Créer un tableau numpy pour les prédictions
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Obtenir les probabilités de chaque modèle
        predictions = {}
        for name, model in models.items():
            # Vérifier si le modèle supporte predict_proba
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]  # Probabilités pour chaque classe
                predictions[name] = {iris_classes[i]: round(prob, 4) for i, prob in enumerate(probabilities)}
            else:
                predictions[name] = "Probabilities not available for this model"
        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)