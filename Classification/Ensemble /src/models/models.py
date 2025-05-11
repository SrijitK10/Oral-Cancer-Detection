"""
Model definitions and training utilities for the Oral Cancer Classification Pipeline.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, 
    Dropout, Average, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import Dict, Tuple, List, Any


def create_callbacks(config: Dict[str, Any], model_name: str) -> List:
    """
    Create callbacks for model training.
    
    Args:
        config: Configuration dictionary
        model_name: Name of the model for saving checkpoints
        
    Returns:
        List of callbacks
    """
    models_dir = config["models"]["save_dir"]
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, f"{model_name}_best.h5")
    
    callbacks = []
    
    # Model checkpoint
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    if config["training"].get("early_stopping_patience"):
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=config["training"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
    
    # Learning rate reduction
    if config["training"].get("reduce_lr_patience"):
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=config["training"]["reduce_lr_patience"],
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
    
    return callbacks


def create_baseline_model(config: Dict[str, Any], num_classes: int) -> Model:
    """
    Create a baseline CNN model.
    
    Args:
        config: Configuration dictionary
        num_classes: Number of output classes
        
    Returns:
        Compiled model
    """
    input_shape = (
        config["dataset"]["img_height"], 
        config["dataset"]["img_width"], 
        3
    )

    model = Sequential(name='Baseline')

    # First block
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    # Second block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    
    # Third block
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))

    # Fourth block
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.4))
    
    # Classification block
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile
    optimizer = Adam(learning_rate=config["training"]["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_transfer_learning_model(config: Dict[str, Any], base_model_name: str, num_classes: int) -> Model:
    """
    Create a transfer learning model.
    
    Args:
        config: Configuration dictionary
        base_model_name: Name of the pre-trained model to use
        num_classes: Number of output classes
        
    Returns:
        Compiled model
    """
    input_shape = (
        config["dataset"]["img_height"], 
        config["dataset"]["img_width"], 
        3
    )
    
    # Dictionary of available pre-trained models
    pretrained_models = {
        'ResNet50': tf.keras.applications.ResNet50,
        'EfficientNetB3': tf.keras.applications.EfficientNetB3,
        'MobileNetV2': tf.keras.applications.MobileNetV2,
        'DenseNet121': tf.keras.applications.DenseNet121,
        'InceptionV3': tf.keras.applications.InceptionV3
    }
    
    if base_model_name not in pretrained_models:
        raise ValueError(f"Model {base_model_name} not supported. Available models: {list(pretrained_models.keys())}")
    
    # Create base model
    base_model = pretrained_models[base_model_name](
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create model
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name=base_model_name)
    
    # Compile
    optimizer = Adam(learning_rate=config["training"]["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(config: Dict[str, Any], model: Model, train_gen, val_gen, fine_tune: bool = False) -> Dict:
    """
    Train a model.
    
    Args:
        config: Configuration dictionary
        model: Model to train
        train_gen: Training data generator
        val_gen: Validation data generator
        fine_tune: Whether to fine-tune the model
        
    Returns:
        History object
    """
    callbacks = create_callbacks(config, model.name)
    epochs = config["training"]["epochs"]
    
    print(f"\nTraining model: {model.name}")
    
    # Initial training with frozen base layers if it's a transfer learning model
    if not fine_tune or model.name == 'Baseline':
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
    
    # Fine-tuning for transfer learning models
    if fine_tune and model.name != 'Baseline':
        print(f"\nFine-tuning model: {model.name}")
        
        # Make the base model trainable
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                # Unfreeze the last 20% of layers in the base model
                num_layers = len(layer.layers)
                for i, sublayer in enumerate(layer.layers):
                    sublayer.trainable = i > 0.8 * num_layers
        
        # Use a lower learning rate for fine-tuning
        ft_lr = config["training"]["learning_rate"] * 0.1
        optimizer = Adam(learning_rate=ft_lr)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create new callbacks for fine-tuning
        ft_callbacks = create_callbacks(config, f"{model.name}_ft")
        
        # Train with fine-tuning
        ft_epochs = min(20, epochs)  # Fewer epochs for fine-tuning
        history_ft = model.fit(
            train_gen,
            epochs=ft_epochs,
            validation_data=val_gen,
            callbacks=ft_callbacks,
            verbose=1
        )
        
        # Combine histories
        for key in history.history:
            history.history[key].extend(history_ft.history[key])
    
    return history.history


def create_ensemble_model(config: Dict[str, Any], model_paths: Dict[str, str], num_classes: int) -> Model:
    """
    Create an ensemble model from multiple trained models.
    
    Args:
        config: Configuration dictionary
        model_paths: Dictionary of model name to model path
        num_classes: Number of output classes
        
    Returns:
        Compiled ensemble model
    """
    input_shape = (
        config["dataset"]["img_height"], 
        config["dataset"]["img_width"], 
        3
    )
    inputs = Input(shape=input_shape)
    
    models = {}
    outputs = []
    
    # Load each model and get its predictions
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"Warning: Model {name} not found at {path}. Skipping.")
            continue
            
        try:
            print(f"Loading model: {name} from {path}")
            models[name] = load_model(path)
            model_output = models[name](inputs)
            outputs.append(model_output)
        except Exception as e:
            print(f"Error loading model {name}: {e}")
    
    if not outputs:
        raise ValueError("No models could be loaded for the ensemble.")
    
    # Average the predictions
    ensemble_output = Average()(outputs)
    
    # Create the ensemble model
    ensemble = Model(inputs=inputs, outputs=ensemble_output, name='Ensemble')
    
    # Compile
    optimizer = Adam(learning_rate=config["training"]["learning_rate"])
    ensemble.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return ensemble