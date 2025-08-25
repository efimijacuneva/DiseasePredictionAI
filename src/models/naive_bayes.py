from sklearn.naive_bayes import ComplementNB
from sklearn.utils.class_weight import compute_sample_weight

def train_naive_bayes(X_train, y_train):
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model = ComplementNB()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model