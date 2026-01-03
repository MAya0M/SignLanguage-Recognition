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
    
    # Reshape for 1D convolution: (batch, sequence_length, num_features, 1)
    # This allows CNN to process spatial patterns in each frame
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # CNN layers for spatial pattern recognition
    # Each layer learns different spatial patterns in keypoints
    for i in range(num_cnn_layers):
        # 1D Convolution over the feature dimension
        # This learns spatial relationships between keypoints
        x = layers.Conv1D(
            filters=cnn_filters * (2 ** i),  # Increase filters: 64, 128, ...
            kernel_size=3,
            padding='same',
            activation='relu',
            name=f'conv1d_{i+1}'
        )(x)
        
        # Batch normalization for stable training
        x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
        
        # Max pooling to reduce dimensionality
        x = layers.MaxPooling1D(pool_size=2, name=f'pool_{i+1}')(x)
        
        # Dropout to prevent overfitting
        x = layers.Dropout(dropout_rate * 0.5, name=f'dropout_conv_{i+1}')(x)
    
    # Global Average Pooling over time dimension
    # This reduces (batch, time_steps, features) to (batch, features)
    # After CNN layers, we have reduced time dimension, so we pool over it
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # Reshape for LSTM: (batch, 1, features)
    # LSTM expects sequence input, so we create a single timestep
    x = layers.Reshape((1, -1))(x)
    
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

