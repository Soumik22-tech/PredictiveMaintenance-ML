import joblib
import numpy as np

def load_models():
    """Load the trained model, scaler, and label encoder."""
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    return model, scaler, label_encoder

def predict_failure(input_data):
    """Predict failure type for a given input array."""
    model, scaler, label_encoder = load_models()
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction_enc = model.predict(input_scaled)[0]
    prediction_label = label_encoder.inverse_transform([prediction_enc])[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    return prediction_label, probabilities

if __name__ == "__main__":
    # Example test
    # Order: Type(encoded), Air temp, Process temp, Rot speed, Torque, Tool wear, temp_diff, power, wear_torque
    sample_input = np.array([[1, 300.0, 310.0, 1500, 40.0, 100, 10.0, 60000.0, 4000.0]])
    try:
        label, probs = predict_failure(sample_input)
        print(f"Predicted Label: {label}")
        print(f"Probabilities: {probs}")
    except Exception as e:
        print(f"Error making prediction: {e}")
