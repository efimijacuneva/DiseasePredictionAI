from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_random_forest(X_train, y_train):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    model = RandomForestClassifier(random_state=42, class_weight=class_weight_dict, n_estimators=50, n_jobs=1)
    model.fit(X_train, y_train)
    return model