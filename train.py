from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import os

# Charger le dataset Iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Créer un répertoire pour sauvegarder les modèles
if not os.path.exists("models"):
    os.mkdir("models")

# Entraîner et sauvegarder plusieurs modèles
models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=200),
    "svm": SVC(probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/{name}.joblib")
    print(f"{name} model trained and saved.")