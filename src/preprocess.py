import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_and_preprocess_data(path='data/dataset.csv'):
    # Loading the dataset with optimized data types
    print("Loading dataset...")
    dtypes = {col: 'int8' for col in pd.read_csv(path, nrows=1).columns if col != 'diseases'}
    dtypes['diseases'] = 'category'
    df = pd.read_csv(path, dtype=dtypes)
    
    # Check for empty values
    # print("Check for empty values...")
    if df.isnull().sum().sum() > 0:
        # print("Empty values ​​found. Filling with 0...")
        df.fillna(0, inplace=True)
    
    # Check for non-binary values ​​in symptoms
    print("Checking for the validity of the symptoms...")
    for column in df.columns[1:]:
        unique_vals = df[column].unique()
        if not set(unique_vals).issubset({0, 1}):
            print(f"Warning: Non-binary values ​​in {column}: {unique_vals}")
            df[column] = df[column].apply(lambda x: 1 if x != 0 else 0)
    
    # Merging rare diseases (<2 instances) into a category "Other"
    print("Merging rare diseases...")
    disease_counts = df['diseases'].value_counts()
    rare_diseases = disease_counts[disease_counts < 2].index
    print(f"Number of diseases with less than 2 samples: {len(rare_diseases)}")
    df['diseases'] = df['diseases'].apply(lambda x: 'Other' if x in rare_diseases else x)
    print(f"Number of unique diseases after the merger: {df['diseases'].nunique()}")
    
    # Encoding the 'diseases' column
    print("Encoding of diagnoses...")
    label_encoders = {}
    le = LabelEncoder()
    df['diseases'] = le.fit_transform(df['diseases'])
    label_encoders['diseases'] = le
    
    # Separation of features and target variable
    X = df.drop('diseases', axis=1)
    y = df['diseases']
    feature_names = X.columns.tolist()
    
    print(f"Data prepared: {X.shape[0]} rows, {X.shape[1]} symptoms, {len(np.unique(y))} дијагнози")
    return X, y, label_encoders, feature_names