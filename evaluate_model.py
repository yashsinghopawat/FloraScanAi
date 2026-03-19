"""
Model Evaluation Script
Generates detailed performance metrics and visualizations
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
TEST_DIR = 'PlantVillage/test'

def load_test_data():
    """Load test dataset"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Important for confusion matrix
    )
    
    return test_generator

def evaluate_model(model_path='best_model.h5'):
    """
    Comprehensive model evaluation
    """
    print("="*60)
    print("MODEL EVALUATION REPORT")
    print("="*60)
    
    # Load model
    print("\n[1/5] Loading model...")
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded: {model_path}")
    
    # Load test data
    print("\n[2/5] Loading test data...")
    test_gen = load_test_data()
    print(f"✓ Test samples: {test_gen.samples}")
    print(f"✓ Number of classes: {len(test_gen.class_indices)}")
    
    # Get class names
    class_names = {v: k for k, v in test_gen.class_indices.items()}
    
    # Evaluate
    print("\n[3/5] Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {test_accuracy*100:.2f}%")
    print(f"TEST LOSS: {test_loss:.4f}")
    print(f"{'='*60}")
    
    # Predictions
    print("\n[4/5] Generating predictions...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Classification Report
    print("\n[5/5] Generating detailed metrics...")
    print("\nCLASSIFICATION REPORT:")
    print("="*60)
    
    # Get short class names for display
    short_names = [class_names[i].split('___')[-1].replace('_', ' ') for i in range(len(class_names))]
    
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=short_names,
        digits=3
    )
    print(report)
    
    # Per-class accuracy
    print("\nPER-CLASS ACCURACY:")
    print("="*60)
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    # Sort by accuracy
    sorted_indices = np.argsort(class_accuracies)[::-1]
    
    print("\nTop 10 Best Performing Classes:")
    for i, idx in enumerate(sorted_indices[:10], 1):
        print(f"{i:2d}. {short_names[idx]:40s} {class_accuracies[idx]*100:6.2f}%")
    
    print("\nBottom 10 Classes (Need Improvement):")
    for i, idx in enumerate(sorted_indices[-10:], 1):
        print(f"{i:2d}. {short_names[idx]:40s} {class_accuracies[idx]*100:6.2f}%")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, short_names)
    
    # Plot per-class accuracy
    plot_class_accuracies(class_accuracies, short_names)
    
    # Sample predictions
    show_sample_predictions(test_gen, model, class_names)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - class_accuracies.png")
    print("  - sample_predictions.png")

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(20, 18))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=False,  # Too many classes for annotations
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Accuracy'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Confusion matrix saved: confusion_matrix.png")

def plot_class_accuracies(accuracies, class_names):
    """Plot per-class accuracy bar chart"""
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    sorted_accuracies = accuracies[sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]
    
    # Color based on accuracy
    colors = ['#d32f2f' if acc < 0.8 else '#ff9800' if acc < 0.9 else '#4caf50' 
              for acc in sorted_accuracies]
    
    bars = ax.barh(range(len(sorted_names)), sorted_accuracies * 100, color=colors)
    
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add accuracy values
    for i, (bar, acc) in enumerate(zip(bars, sorted_accuracies)):
        ax.text(acc * 100 + 1, i, f'{acc*100:.1f}%', 
                va='center', fontsize=7)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4caf50', label='Good (≥90%)'),
        Patch(facecolor='#ff9800', label='Fair (80-90%)'),
        Patch(facecolor='#d32f2f', label='Poor (<80%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('class_accuracies.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Class accuracies plot saved: class_accuracies.png")

def show_sample_predictions(test_gen, model, class_names):
    """Show sample predictions with images"""
    # Get a batch of images
    images, labels = next(test_gen)
    predictions = model.predict(images, verbose=0)
    
    # Select 9 random samples
    indices = np.random.choice(len(images), size=min(9, len(images)), replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Get prediction
        pred_idx = np.argmax(predictions[idx])
        true_idx = np.argmax(labels[idx])
        confidence = predictions[idx][pred_idx] * 100
        
        # Get class names (short version)
        pred_name = class_names[pred_idx].split('___')[-1].replace('_', ' ')
        true_name = class_names[true_idx].split('___')[-1].replace('_', ' ')
        
        # Determine if correct
        is_correct = pred_idx == true_idx
        color = 'green' if is_correct else 'red'
        
        # Plot image
        axes[i].imshow(images[idx])
        axes[i].axis('off')
        
        # Title with prediction
        title = f"True: {true_name}\n"
        title += f"Pred: {pred_name}\n"
        title += f"Conf: {confidence:.1f}%"
        
        axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
    
    plt.suptitle('Sample Predictions\n(Green = Correct, Red = Incorrect)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Sample predictions saved: sample_predictions.png")

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists('best_model.h5'):
        print("Error: Model file 'best_model.h5' not found!")
        print("Please train the model first using: python train_model.py")
    else:
        evaluate_model('best_model.h5')
