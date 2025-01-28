import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

def wait_for_server(timeout=30, interval=2):
    print("Waiting for server to start...")
    start_time = time.time()
    while True:
        try:
            response = requests.get(f"{BASE_URL}/")
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            if time.time() - start_time > timeout:
                print("Server failed to start within timeout period")
                return False
            print(".", end="", flush=True)
            time.sleep(interval)

# Enregistrer un modèle
def register_model(model_name, deposit=1000):
    try:
        response = requests.post(
            f"{BASE_URL}/register", 
            json={"model_name": model_name, "deposit": deposit},
            timeout=5
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to server. Make sure the Flask app is running."}
    except Exception as e:
        return {"error": str(e)}

# Effectuer une prédiction avec consensus
def predict_consensus(features, target_class):
    response = requests.post(
        f"{BASE_URL}/predict_consensus",
        json={"features": features, "target_class": target_class}
    )
    return response.json()

# Mettre à jour les poids (optionnel, si manuel)
def update_weights(updates):
    response = requests.post(f"{BASE_URL}/update_weights", json={"updates": updates})
    return response.json()

def predict(sepal_length, sepal_width, petal_length, petal_width):
    try:
        response = requests.get(
            f"{BASE_URL}/predict",
            params={
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            }
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Tester les fonctions
if __name__ == "__main__":
    if not wait_for_server():
        sys.exit(1)
    
    try:
        # Exemple d'enregistrement de modèle
        for model in ["random_forest", "logistic_regression", "svm"]:
            result = register_model(model, deposit=1000)
            print(f"Registration result for {model}:", result)

        # Exemple de prédiction avec consensus
        features = [5.1, 3.5, 1.4, 0.2]
        target_class = 0
        result = predict_consensus(features, target_class)
        print("Prediction result:", result)
        
        # Test de prédiction
        result = predict(5.1, 3.5, 1.4, 0.2)
        print("Prediction result:")
        for key, value in result.items():
            print(f"{key:<12} {value}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {e}")
        sys.exit(1)