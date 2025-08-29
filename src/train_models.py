import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from models.decision_tree import train_decision_tree
from models.naive_bayes import train_naive_bayes
from models.neural_network import train_neural_network
from models.random_forest import train_random_forest
from models.xgboost_model import train_xgboost
from preprocess import load_and_preprocess_data

# Data loading and preprocessing
X, y, label_encoders, feature_names = load_and_preprocess_data()

# Stratified sampling of 120,000 rows
print("Taking a stratified subset of the data (120,000 rows)...")
subset_size = 120000 
if X.shape[0] > subset_size:
    X_subset, _, y_subset, _ = train_test_split(
        X, y, train_size=subset_size, stratify=y, random_state=42
    )
    X = X_subset
    y = y_subset
    unique_diseases = len(np.unique(y))
    print(f"Number of unique diseases in the subset: {unique_diseases}")
else:
    print("The dataset is smaller than 120,000 rows, the entire dataset is used.")
print(f"Subset: {X.shape[0]} rows, {X.shape[1]} symptoms")

# Merging rare diseases into the subset and recoding
print("Combining rare diseases into the subset...")
y_series = pd.Series(y)
disease_counts = y_series.value_counts()
rare_diseases = disease_counts[disease_counts < 15].index
if len(rare_diseases) > 0:
    print(f"Number of diseases with less than 15 samples in the subset: {len(rare_diseases)}")
    y_decoded = label_encoders['diseases'].inverse_transform(y)
    other_label = label_encoders['diseases'].transform(['Other'])[0]
    y = np.where(np.isin(y, rare_diseases), other_label, y)
    le = LabelEncoder()
    y = le.fit_transform(label_encoders['diseases'].inverse_transform(y))
    label_encoders['diseases'] = le
    new_disease_counts = pd.Series(y).value_counts()
    min_class_count = new_disease_counts.min()
    print(f"Number of unique diseases after merging: {len(np.unique(y))}")
    print(f"Minimum number of samples per class: {min_class_count}")
    print("Distribution of diseases after the merger:")
    print(new_disease_counts)
    if min_class_count < 2:
        raise ValueError(f"After the merge, there is a class with only {min_class_count} sample. Increase the merging threshold.")
else:
    print("There are no diseases with fewer than 15 samples in the subset.")

# Splitting the data into training and test sets
print("Splitting the data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Training individual models
print("Training the models...")
models = {}
try:
    models["Naive Bayes"] = train_naive_bayes(X_train, y_train)
    models["Decision Tree"] = train_decision_tree(X_train, y_train)
    models["Random Forest"] = train_random_forest(X_train, y_train)
    models["XGBoost"] = train_xgboost(X_train, y_train)
    nn_model, scaler = train_neural_network(X_train, y_train)
    models["Neural Network"] = (nn_model, scaler)
except Exception as e:
    print(f"Error while training model: {e}")
    raise

#  Evaluation of individual models
results = {}
print("\n Model accuracy (test set):")
for name, model in models.items():
    try:
        if name == "Neural Network":
            model, scaler = model
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'test_accuracy': acc}
        print(f"{name}: {acc:.4f}")
    except Exception as e:
        print(f"Error while evaluating {name}: {e}")

# Cross-validation for stability
print("\nCross-validation (3-fold):")
for name, model in models.items():
    try:
        print(f"Cross-validation for {name}...")
        if name in ["Neural Network", "XGBoost"]:  # Skipping Neural Network and XGBoost
            continue
        scores = cross_val_score(model, X, y, cv=3)
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    except Exception as e:
        print(f"Cross-validation error {name}: {e}")

# Creating and training an ensemble model
print("Ensemble model training...")
estimators = [
    ("Random Forest", models["Random Forest"]),
    ("Neural Network", models["Neural Network"][0]),
    ("XGBoost", models["XGBoost"])
]
try:
    ensemble = VotingClassifier(estimators, voting='soft', weights=[0.3, 0.5, 0.2])
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
    results["Ensemble"] = {'test_accuracy': ensemble_acc}
    print(f"Ensemble: {ensemble_acc:.4f}")
except Exception as e:
    print(f"Ensemble training error: {e}")
    raise

# Finding the best model
best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
print(f"\nBest model: {best_model_name}, Accuracy: {results[best_model_name]['test_accuracy']:.4f}")

# Saving models and data
print("Saving models...")
models["Ensemble"] = ensemble
joblib.dump(models, 'src/models/trained_models.pkl')
joblib.dump(label_encoders, 'src/models/label_encoders.pkl')
joblib.dump(feature_names, 'src/models/feature_names.pkl')
joblib.dump(results, 'src/models/results.pkl')

print("Training is over!")