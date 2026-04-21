import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_data(filepath):
    """Load CSV, print shape and columns, and return dataframe."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded from {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def clean_data(df):
    """Drop unnecessary columns and duplicate rows."""
    cols_to_drop = ['UDI', 'Product ID']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    initial_shape = df.shape
    df = df.drop_duplicates()
    if df.shape != initial_shape:
        print(f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows.")
    
    print(f"Remaining shape after cleaning: {df.shape}")
    return df

def encode_features(df):
    """Encode categorical features and the target variable."""
    os.makedirs('models', exist_ok=True)
    
    le_type = LabelEncoder()
    df['Type'] = le_type.fit_transform(df['Type'])
    print(f"'Type' classes: {le_type.classes_}")
    
    le_target = LabelEncoder()
    df['Failure Type'] = le_target.fit_transform(df['Failure Type'])
    print(f"'Failure Type' classes: {le_target.classes_}")
    
    joblib.dump(le_target, 'models/label_encoder.pkl')
    print("Label encoder saved to models/label_encoder.pkl")
    
    return df

def engineer_features(df):
    """Create new domain-specific features."""
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    df['wear_torque'] = df['Tool wear [min]'] * df['Torque [Nm]']
    
    print("Feature engineering complete: added 'temp_diff', 'power', 'wear_torque'")
    return df

def split_and_balance(df):
    """Split data and apply SMOTE to the training set."""
    X = df.drop(columns=['Failure Type', 'Target'], errors='ignore')
    target_col = 'Failure Type' if 'Failure Type' in df.columns else 'Target'
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nClass distribution before SMOTE:")
    print(y_train.value_counts().sort_index())
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_train_res).value_counts().sort_index())
    
    return X_train_res, X_test, y_train_res, y_test

def scale_features(X_train, X_test):
    """Scale features using StandardScaler and save the scaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nScaler saved to models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled

def run_preprocessing():
    """Execute the full preprocessing pipeline."""
    data_path = 'data/predictive_maintenance.csv'
        
    df = load_data(data_path)
    df = clean_data(df)
    df = encode_features(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_and_balance(df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    print("\nPreprocessing complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    run_preprocessing()
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_data(filepath):
    """Load CSV, print shape and columns, and return dataframe."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded from {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def clean_data(df):
    """Drop unnecessary columns and duplicate rows."""
    # Drop UDI and Product ID
    cols_to_drop = ['UDI', 'Product ID']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Drop duplicate rows
    initial_shape = df.shape
    df = df.drop_duplicates()
    if df.shape != initial_shape:
        print(f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows.")
    
    print(f"Remaining shape after cleaning: {df.shape}")
    return df

def encode_features(df):
    """Encode categorical features and the target variable."""
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Encode 'Type' feature (L, M, H)
    le_type = LabelEncoder()
    df['Type'] = le_type.fit_transform(df['Type'])
    print(f"'Type' classes: {le_type.classes_}")
    
    # Encode 'Failure Type' target
    le_target = LabelEncoder()
    df['Failure Type'] = le_target.fit_transform(df['Failure Type'])
    print(f"'Failure Type' classes: {le_target.classes_}")
    
    # Save the Failure Type label encoder
    joblib.dump(le_target, 'models/label_encoder.pkl')
    print("Label encoder saved to models/label_encoder.pkl")
    
    return df

def engineer_features(df):
    """Create new domain-specific features."""
    # Create temp_diff
    df['temp_diff'] = df[df.columns[df.columns.str.contains('Process temperature')][0]] - df[df.columns[df.columns.str.contains('Air temperature')][0]]
    
    # Create power (approximate)
    df['power'] = df[df.columns[df.columns.str.contains('Torque')][0]] * df[df.columns[df.columns.str.contains('Rotational speed')][0]]
    
    # Create wear_torque
    df['wear_torque'] = df[df.columns[df.columns.str.contains('Tool wear')][0]] * df[df.columns[df.columns.str.contains('Torque')][0]]
    
    print("Feature engineering complete: added 'temp_diff', 'power', 'wear_torque'")
    return df

def split_and_balance(df):
    """Split data and apply SMOTE to the training set."""
    # Features X (excluding targets)
    X = df.drop(columns=['Failure Type', 'Target'])
    y = df['Failure Type']
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nClass distribution before SMOTE:")
    print(y_train.value_counts().sort_index())
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_train_res).value_counts().sort_index())
    
    return X_train_res, X_test, y_train_res, y_test

def scale_features(X_train, X_test):
    """Scale features using StandardScaler and save the scaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Target path for scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nScaler saved to models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled

def run_preprocessing():
    """Execute the full preprocessing pipeline."""
    # Path relative to project root
    data_path = 'data/predictive_maintenance.csv'
        
    df = load_data(data_path)
    df = clean_data(df)
    df = encode_features(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_and_balance(df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    print("\nPreprocessing complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    run_preprocessing()

