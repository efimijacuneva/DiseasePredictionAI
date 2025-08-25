from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Loading saved models and data
if os.path.exists('src/models/trained_models.pkl'):
    try:
        models = joblib.load('src/models/trained_models.pkl')
        label_encoders = joblib.load('src/models/label_encoders.pkl')
        feature_names = joblib.load('src/models/feature_names.pkl')
        results = joblib.load('src/models/results.pkl')
        print("\nAccuracy of models:")
        for name, res in results.items():
            print(f"{name}: {res['test_accuracy']:.4f}")
        best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
        print(f"Best model: {best_model_name}, Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        raise
else:
    raise FileNotFoundError("Models not found. First run train_models.py")

# Choosing an ensemble model for prediction
model = models["Ensemble"]

def predict_disease(model, label_encoders, selected_symptoms, feature_names, top_k=3):
    print("Selected symptoms:", selected_symptoms)
    input_vector = np.zeros(len(feature_names))
    unrecognized = []
    for sym in selected_symptoms:
        if sym in feature_names:
            idx = feature_names.index(sym)
            input_vector[idx] = 1
        else:
            unrecognized.append(sym)
    if unrecognized:
        print(f"Warning: Unrecognized symptoms: {unrecognized}")
    
    input_vector = np.array([input_vector])
    print("Input vector: shape:", input_vector.shape, "Number 1:", np.sum(input_vector))
    
    try:
        probs = model.predict_proba(input_vector)[0]
        top_k_indices = np.argsort(probs)[-top_k:][::-1]
        top_k_diseases = label_encoders['diseases'].inverse_transform(top_k_indices)
        top_k_probs = probs[top_k_indices]
        predictions = [(disease, prob) for disease, prob in zip(top_k_diseases, top_k_probs)]
        print("Predictions:", predictions)
        return predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        return [("Error", 0.0)]

@app.route('/')
def index():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_symptoms = request.form.getlist('symptoms[]')
        print("Symptoms obtained:", selected_symptoms)
        if not selected_symptoms:
            return jsonify({'error': 'No symptoms selected'})
        results = predict_disease(model, label_encoders, selected_symptoms, feature_names)
        if results[0][0] == "Error":
            return jsonify({'error': 'The prediction failed'})
        return jsonify({'predictions': [{'disease': disease, 'probability': f"{prob:.2%}"} for disease, prob in results]})
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': f'Server error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=False)