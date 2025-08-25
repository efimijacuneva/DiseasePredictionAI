from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_neural_network(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler