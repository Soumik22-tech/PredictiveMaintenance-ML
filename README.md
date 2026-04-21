# Predictive Maintenance of Industrial Machinery

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.32+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.4+-orange.svg)

Predictive maintenance is a key component of Industry 4.0, using machine learning to predict equipment failures before they occur. This project implements a multiclass classification system to identify specific types of machine failures using sensor data.

## 🚀 Features

- **EDA**: Comprehensive data exploration in Jupyter Notebooks.
- **Preprocessing**: Automated cleaning, encoding, and scaling pipeline.
- **Feature Engineering**: Creation of domain-specific features (`temp_diff`, `power`, `wear_torque`).
- **Class Imbalance**: Handled using **SMOTE** (Synthetic Minority Over-sampling Technique) to address highly imbalanced failure classes.
- **Model Comparison**: Automated training and evaluation of **Random Forest** and **XGBoost**.
- **Interactive App**: A **Streamlit** dashboard for real-time failure prediction.

## 📊 Models & Performance

| Model | Accuracy | F1-Macro | Status |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **94.3%** | **0.664** | **Winner** |
| XGBoost | 93.8% | 0.642 | - |

The Random Forest model was selected as the winner due to better generalization across minority failure classes.

## 📂 Project Structure

```text
PredictiveMaintenance-ML/
├── app/
│   └── app.py              # Streamlit Web Application
├── data/
│   └── predictive_maintenance.csv  # Dataset
├── models/                 # Saved models, scalers, and plots
├── notebooks/
│   ├── 01_EDA.ipynb        # Exploratory Data Analysis
│   └── 02_model_training.ipynb     # Model training experiments
├── src/
│   ├── preprocess.py       # Data cleaning and balancing
│   ├── train.py            # Model training and selection
│   └── predict.py          # Inference logic
├── requirements.txt        # Project dependencies
└── README.md
```

## 🛠️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/PredictiveMaintenance-ML.git
cd PredictiveMaintenance-ML
```

### 2. Set up virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the models
```bash
python -m src.train
```

### 5. Run the Streamlit app
```bash
streamlit run app/app.py
```

## 📝 Dataset
The project uses the [Kaggle Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) dataset consisting of 10,000 data points with 10 features.
