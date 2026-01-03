"""
GRU Model for Sign Language Recognition
"""

from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


def build_gru_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    gru_units: int = 128,
    dropout_rate: float = 0.3,
    recurrent_dropout: float = 0.2,
    num_gru_layers: int = 2
) -> keras.Model:
    """
    Build a GRU model for sign language recognition
    
    Args:
        input_shape: (sequence_length, num_features)
        num_classes: Number of classes to predict
        gru_units: Number of units in each GRU layer
        dropout_rate: Dropout rate for dense layers
        recurrent_dropout: Dropout rate for recurrent connections
        num_gru_layers: Number of GRU layers
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # First GRU layer (return sequences for stacked layers)
    # Use normal dropout - the issue might be too little regularization
    x = layers.GRU(
        gru_units,
        return_sequences=(num_gru_layers > 1),
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        name='gru_1'
    )(inputs)
    
    # Additional GRU layers
    for i in range(2, num_gru_layers + 1):
        return_sequences = (i < num_gru_layers)
        x = layers.GRU(
            gru_units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            name=f'gru_{i}'
        )(x)
    
    # Dense layers - removed batch normalization as it might interfere with small dataset
    x = layers.Dense(gru_units, activation='relu', name='dense_1')(x)
    x = layers.Dropout(dropout_rate * 0.7, name='dropout_1')(x)  # Slightly less dropout
    
    x = layers.Dense(gru_units // 2, activation='relu', name='dense_2')(x)
    x = layers.Dropout(dropout_rate * 0.7, name='dropout_2')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='sign_language_gru')
    
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
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing GRU model creation...")
    
    # Example input shape: (sequence_length, num_features)
    # sequence_length = max_length from padding
    # num_features = 2 * 21 * 3 = 126 (flattened keypoints)
    input_shape = (100, 126)  # Example: max 100 frames, 126 features per frame
    num_classes = 8  # Example: 8 sign language classes
    
    model = build_gru_model(input_shape, num_classes)
    model = compile_model(model)
    
    print("\nModel summary:")
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")

