"""
Plant Disease Classification - Phase 2 Fine-Tuning
Using pre-trained Phase 1 model (best_model.h5)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 2       # Phase 2 epochs
LEARNING_RATE = 0.0001  # lower learning rate for fine-tuning
DATASET_PATH = 'PlantVillage'  # Update to your dataset path
TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
VAL_DIR = os.path.join(DATASET_PATH, 'val')
TEST_DIR = os.path.join(DATASET_PATH, 'test')
BEST_MODEL_PATH = 'best_model.h5'  # Phase 1 saved model

# ------------------ Data Generators ------------------ #
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    test_gen = val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_gen, val_gen, test_gen

# ------------------ Plotting ------------------ #
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('phase2_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Training history saved as 'phase2_training_history.png'")

# ------------------ Phase 2 Training ------------------ #
def fine_tune_phase2():
    print("Loading Phase 1 model...")
    model = keras.models.load_model(BEST_MODEL_PATH)

    # Load data generators
    train_gen, val_gen, test_gen = create_data_generators()

    # Unfreeze base layers for fine-tuning
    base_model = model.layers[1]  # EfficientNetB0 is the second layer in our previous model
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False  # freeze first 100 layers

    # Recompile with smaller learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint('final_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    ]

    # Train Phase 2
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Plot training history
    plot_training_history(history)

    print("Phase 2 fine-tuning completed! Final model saved as 'final_model.h5'")
    return model, history

# ------------------ Run ------------------ #
if __name__ == "__main__":
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    fine_tune_phase2()