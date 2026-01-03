"""
CNN + LSTM Model for Sign Language Recognition
Combines spatial (CNN) and temporal (LSTM) pattern recognition
"""

from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


def build_cnn_lstm_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    cnn_filters: int = 64,
    lstm_units: int = 128,
    dropout_rate: float = 0.3,
    num_cnn_layers: int = 2
) -> keras.Model:
    """
    Build a CNN + LSTM model for sign language recognition
    
    Args:
        input_shape: (sequence_length, num_features)
        num_classes: Number of classes to predict
        cnn_filters: Number of filters in CNN layers
        lstm_units: Number of units in LSTM layer
        dropout_rate: Dropout rate
        num_cnn_layers: Number of CNN layers
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    # Input shape: (batch, sequence_length, num_features)
    # Example: (batch, 96, 126)
    
    # CNN layers for spatial pattern recognition
    # Conv1D works on the last dimension (features), preserving time dimension
    # This learns spatial relationships between keypoints in each frame
    x = inputs
    for i in range(num_cnn_layers):
        # 1D Convolution over features dimension
        # Input: (batch, time, features) -> Output: (batch, time, filters)
        x = layers.Conv1D(
            filters=cnn_filters * (2 ** i),  # Increase filters: 64, 128, ...
            kernel_size=3,
            padding='same',  # Keep same time length
            activation='relu',
            name=f'conv1d_{i+1}'
        )(x)
        
        # Batch normalization for stable training
        x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
        
        # Max pooling reduces time dimension (not features!)
        # This is OK - we still have temporal information, just less granular
        # Input: (batch, time, filters) -> Output: (batch, time//2, filters)
        x = layers.MaxPooling1D(pool_size=2, name=f'pool_{i+1}')(x)
        
        # Dropout to prevent overfitting
        x = layers.Dropout(dropout_rate * 0.5, name=f'dropout_conv_{i+1}')(x)
    
    # After CNN: x shape is (batch, reduced_time, cnn_features)
    # Now LSTM processes the sequence of CNN outputs
    # This captures temporal patterns in the spatial features
    
    # Bidirectional LSTM for temporal pattern recognition
    # Bidirectional = sees both past and future context
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate * 0.5,
            return_sequences=False,  # Only return final output
            name='lstm_1'
        ),
        name='bidirectional_lstm'
    )(x)
    
    # Additional LSTM layer (optional, can be removed if overfitting)
    # x = layers.Bidirectional(
    #     layers.LSTM(
    #         lstm_units // 2,
    #         dropout=dropout_rate,
    #         recurrent_dropout=dropout_rate * 0.5,
    #         return_sequences=False,
    #         name='lstm_2'
    # ),
    #     name='bidirectional_lstm_2'
    # )(x)
    
    # Dense layers for classification
    x = layers.Dense(lstm_units, activation='relu', name='dense_1')(x)
    x = layers.Dropout(dropout_rate, name='dropout_dense_1')(x)
    
    x = layers.Dense(lstm_units // 2, activation='relu', name='dense_2')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='dropout_dense_2')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='sign_language_cnn_lstm')
    
    return model


def compile_model(model: keras.Model, learning_rate: float = 0.001) -> keras.Model:
    """
    Compile the model with optimizer and loss
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing CNN + LSTM model creation...")
    
    # Example input shape: (sequence_length, num_features)
    input_shape = (96, 126)  # max 96 frames, 126 features per frame
    num_classes = 8  # 8 sign language classes
    
    model = build_cnn_lstm_model(input_shape, num_classes)
    model = compile_model(model)
    
    print("\nModel summary:")
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")

