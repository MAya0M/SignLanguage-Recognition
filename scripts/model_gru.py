"""
Backward compatibility module for model_gru.
This module was replaced with model_cnn_lstm, but this file provides
compatibility for any code that still imports model_gru.
"""

# Import everything from model_cnn_lstm
from scripts.model_cnn_lstm import (
    build_cnn_lstm_model,
    compile_model
)

# For backward compatibility, export the same functions
__all__ = ['build_cnn_lstm_model', 'compile_model']

