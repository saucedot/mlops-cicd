import pytest
import pandas as pd
from train import load_data, train_model, evaluate_model
import json

def test_data_loading():
    """Test that data loads correctly"""
    X, y = load_data()
    assert X.shape[0] == 150, "Expected 150 samples"
    assert X.shape[1] == 4, "Expected 4 features"
    assert len(y) == 150, "Target should have 150 values"

def test_data_quality():
    """Test data quality constraints"""
    X, y = load_data()
    assert not X.isnull().any().any(), "Data should not contain null values"
    assert (X >= 0).all().all(), "Features should be non-negative"
    assert y.nunique() == 3, "Should have 3 classes"

def test_model_training():
    """Test that model can be trained"""
    X, y = load_data()
    X_train = X.iloc[:100]
    y_train = y.iloc[:100]
    model = train_model(X_train, y_train)
    assert model is not None
    assert hasattr(model, 'predict')

def test_model_performance():
    """Test that model meets minimum performance threshold"""
    X, y = load_data()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    
    # Assert minimum performance thresholds
    assert metrics['accuracy'] >= 0.85, f"Accuracy {metrics['accuracy']} below threshold 0.85"
    assert metrics['f1_score'] >= 0.85, f"F1 {metrics['f1_score']} below threshold 0.85"
