from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=50,  
        max_depth=8,      
        n_jobs=1,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    return model