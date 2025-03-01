import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Flatten

import tsfel

# Project: Active Noise Cancellation using Machine Learning and Deep Learning
# Author: Avishi Semwal
# Description: This project utilizes the UCI HAR dataset to predict human activity classes
# using deep learning (LSTM, CNN) and machine learning models (Random Forest, SVM, Logistic Regression).
# TSFEL is used for feature extraction.

# Function to load and preprocess the UCI HAR dataset (Replace with actual dataset loading code)
def load_data():
    """
    Generates a simulated dataset mimicking accelerometer data.
    - 1000 samples
    - 128 time-steps per sample
    - 3 sensor axes (x, y, z)
    Returns: Training and testing data split.
    """
    np.random.seed(42)
    X = np.random.randn(1000, 128, 3)  # Simulated dataset: 1000 samples, 128 time-steps, 3 sensor axes
    y = np.random.randint(0, 6, 1000)  # 6 activity classes (0 to 5)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

# --- Deep Learning Model (LSTM) ---
def build_lstm_model():
    """
    Builds an LSTM model for activity recognition.
    - 2 LSTM layers
    - Fully connected dense layers with dropout for regularization
    - Softmax activation for multi-class classification
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(128, 3)),
        LSTM(32),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(6, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create LSTM model
lstm_model = build_lstm_model()

# Placeholder for training (Uncomment to train)
# lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# --- Feature Extraction using TSFEL ---
def extract_features(X_train, X_test):
    """
    Extracts features from raw time-series data using TSFEL.
    - Uses predefined feature configuration
    - Applies to only one axis (e.g., X-axis) for simplicity
    """
    cfg = tsfel.get_features_by_domain()
    X_train_features = tsfel.time_series_features_extractor(cfg, X_train[:, :, 0], fs=50)
    X_test_features = tsfel.time_series_features_extractor(cfg, X_test[:, :, 0], fs=50)
    return X_train_features, X_test_features

# Generate features using TSFEL
X_train_features, X_test_features = extract_features(X_train, X_test)

# --- Machine Learning Models ---
def train_ml_models():
    """
    Trains multiple machine learning models on extracted features.
    - Random Forest
    - Support Vector Machine (SVM)
    - Logistic Regression
    Returns: Dictionary of model names and accuracy scores.
    """
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_features, y_train)
        y_pred = model.predict(X_test_features)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    return results

# Train machine learning models and store results
ml_results = train_ml_models()

# Print Accuracy Scores
print("Machine Learning Model Performance:")
for model, acc in ml_results.items():
    print(f"{model} Accuracy: {acc:.4f}")
