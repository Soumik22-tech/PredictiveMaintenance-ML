from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend, no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import os
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import run_preprocessing

def train_random_forest(X_train, y_train):
    """Train RandomForestClassifier with specified hyperparameters."""
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        random_state=42, 
        class_weight='balanced', 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Random Forest training complete")
    return model

def train_xgboost(X_train, y_train):
    """Train XGBClassifier with specified hyperparameters."""
    model = XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1,
        random_state=42, 
        eval_metric='mlogloss', 
        verbosity=0
    )
    model.fit(X_train, y_train)
    print("XGBoost training complete")
    return model

def evaluate_model(model, X_test, y_test, model_name, label_encoder):
    """Evaluate model performance and save confusion matrix plot."""
    os.makedirs('models', exist_ok=True)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Overall Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'models/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def save_best_model(rf_model, xgb_model, rf_metrics, xgb_metrics):
    """Compare models based on F1-Macro and save the winner."""
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(xgb_model, 'models/xgb_model.pkl')
    
    print("\nComparing models based on F1-Macro Score...")
    if rf_metrics['f1_macro'] >= xgb_metrics['f1_macro']:
        best_model = rf_model
        winner_name = "Random Forest"
        why = f"F1-Macro {rf_metrics['f1_macro']:.4f} >= XGBoost's {xgb_metrics['f1_macro']:.4f}"
    else:
        best_model = xgb_model
        winner_name = "XGBoost"
        why = f"F1-Macro {xgb_metrics['f1_macro']:.4f} > Random Forest's {rf_metrics['f1_macro']:.4f}"
    
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"Winner: {winner_name}")
    print(f"Reason: {why}")
    print("Best model saved to models/best_model.pkl")

def run_training():
    """Run the complete training and evaluation pipeline."""
    X_train, X_test, y_train, y_test = run_preprocessing()
    
    le = joblib.load('models/label_encoder.pkl')
    
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest", le)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost", le)
    
    save_best_model(rf_model, xgb_model, rf_metrics, xgb_metrics)
    
    summary = pd.DataFrame({
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics
    }).T
    print("\n--- Final Model Comparison ---")
    print(summary)

if __name__ == "__main__":
    run_training()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend, no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import os
import pandas as pd

# Import from local module
#from src.preprocess import run_preprocessing

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import run_preprocessing

def train_random_forest(X_train, y_train):
    """Train RandomForestClassifier with specified hyperparameters."""
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        random_state=42, 
        class_weight='balanced', 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Random Forest training complete")
    return model

def train_xgboost(X_train, y_train):
    """Train XGBClassifier with specified hyperparameters."""
    model = XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1,
        random_state=42, 
        eval_metric='mlogloss', 
        verbosity=0
    )
    model.fit(X_train, y_train)
    print("XGBoost training complete")
    return model

def evaluate_model(model, X_test, y_test, model_name, label_encoder):
    """Evaluate model performance and save confusion matrix plot."""
    # Ensure models directory exists for plots
    os.makedirs('models', exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Overall Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'models/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close() # Close figure to free memory
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def save_best_model(rf_model, xgb_model, rf_metrics, xgb_metrics):
    """Compare models based on F1-Macro and save the winner."""
    # Save individual models first
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(xgb_model, 'models/xgb_model.pkl')
    
    print("\nComparing models based on F1-Macro Score...")
    if rf_metrics['f1_macro'] >= xgb_metrics['f1_macro']:
        best_model = rf_model
        winner_name = "Random Forest"
        why = f"F1-Macro {rf_metrics['f1_macro']:.4f} >= XGBoost's {xgb_metrics['f1_macro']:.4f}"
    else:
        best_model = xgb_model
        winner_name = "XGBoost"
        why = f"F1-Macro {xgb_metrics['f1_macro']:.4f} > Random Forest's {rf_metrics['f1_macro']:.4f}"
    
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"Winner: {winner_name}")
    print(f"Reason: {why}")
    print("Best model saved to models/best_model.pkl")

def run_training():
    """Run the complete training and evaluation pipeline."""
    # 1. Get preprocessed data
    X_train, X_test, y_train, y_test = run_preprocessing()
    
    # Load label encoder for target names
    le = joblib.load('models/label_encoder.pkl')
    
    # 2. Train models
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # 3. Evaluate models
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest", le)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost", le)
    
    # 4. Save best model
    save_best_model(rf_model, xgb_model, rf_metrics, xgb_metrics)
    
    # 5. Final summary table
    summary = pd.DataFrame({
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics
    }).T
    print("\n--- Final Model Comparison ---")
    print(summary)

if __name__ == "__main__":
    run_training()

