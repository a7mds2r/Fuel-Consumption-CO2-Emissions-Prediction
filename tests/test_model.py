import sys
import os
import numpy as np
import pytest

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import NumPyMLP

def test_mlp_initialization():
    mlp = NumPyMLP()
    assert mlp.W1.shape == (3, 64)
    assert mlp.b1.shape == (64,)
    assert mlp.W2.shape == (64, 32)
    assert mlp.b2.shape == (32,)
    assert mlp.W3.shape == (32, 1)
    assert mlp.b3.shape == (1,)

def test_mlp_forward_pass():
    mlp = NumPyMLP()
    X = np.random.randn(10, 3)
    preds = mlp.forward(X)
    assert preds.shape == (10, 1)

def test_mlp_get_set_weights():
    mlp = NumPyMLP()
    weights = mlp.get_weights_vector()
    assert len(weights) == 2369

    new_weights = np.random.randn(2369)
    mlp.set_weights_vector(new_weights)
    assert np.allclose(mlp.get_weights_vector(), new_weights)

def test_mlp_training():
    mlp = NumPyMLP()
    X_train = np.random.randn(100, 3)
    y_train = np.random.randn(100)
    X_val = np.random.randn(20, 3)
    y_val = np.random.randn(20)

    train_hist, val_hist = mlp.train_bp(X_train, y_train, X_val, y_val, epochs=5, batch_size=10)
    assert len(train_hist) > 0
    assert len(val_hist) > 0
    assert len(train_hist) == len(val_hist)
