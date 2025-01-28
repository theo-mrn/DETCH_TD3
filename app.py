from flask import Flask, request, jsonify
import joblib
import numpy as np
from pos import register_model, update_weights

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

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    result = register_model(data['model_name'], data['deposit'])
    return jsonify(result)

@app.route('/predict_consensus', methods=['POST'])
def predict_consensus():
    data = request.json
    features = data['features']
    target_class = data['target_class']
    
    input_data = np.array([features])
    predictions = {}
    
    for name, model in models.items():
        pred = model.predict(input_data)[0]
        predictions[name] = int(pred)
    
    # Mettre à jour les poids basés sur les prédictions
    updated_weights = update_weights(predictions, target_class)
    
    return jsonify({
        "predictions": predictions,
        "updated_weights": updated_weights
    })

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Récupérer les paramètres
        params = {
            "sepal_length": float(request.args.get("sepal_length")),
            "sepal_width": float(request.args.get("sepal_width")),
            "petal_length": float(request.args.get("petal_length")),
            "petal_width": float(request.args.get("petal_width"))
        }

        input_data = np.array([[params["sepal_length"], params["sepal_width"], 
                              params["petal_length"], params["petal_width"]]])

        # Trouver le meilleur modèle et sa prédiction
        best_confidence = 0
        best_prediction = None
        best_model_type = None

        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0]
                confidence = np.max(proba)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_prediction = model.predict(input_data)[0]
                    best_model_type = name

        # Construire la réponse
        response = {
            "confidence": float(best_confidence),
            "model_type": best_model_type,
            "parameters": {
                "sepal_length": params["sepal_length"],
                "sepal_width": params["sepal_width"],
                "petal_length": params["petal_length"],
                "petal_width": params["petal_width"]
            },
            "prediction": int(best_prediction),
            "species": iris_classes[int(best_prediction)]
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)