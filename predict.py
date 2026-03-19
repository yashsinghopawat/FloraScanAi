"""
Prediction script for Plant Disease Classification
Loads trained model and predicts disease from plant leaf image
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import json

# Disease treatment recommendations
TREATMENT_RECOMMENDATIONS = {
    'Apple___Apple_scab': {
        'disease': 'Apple Scab',
        'severity': 'Moderate',
        'treatment': [
            'Remove and destroy infected leaves',
            'Apply fungicide (Captan or Sulfur)',
            'Prune to improve air circulation',
            'Avoid overhead watering'
        ],
        'prevention': 'Plant resistant varieties, ensure proper spacing'
    },
    'Apple___Black_rot': {
        'disease': 'Apple Black Rot',
        'severity': 'Severe',
        'treatment': [
            'Prune infected branches immediately',
            'Apply copper-based fungicide',
            'Remove mummified fruits',
            'Improve drainage'
        ],
        'prevention': 'Regular pruning, remove dead wood'
    },
    'Apple___Cedar_apple_rust': {
        'disease': 'Cedar Apple Rust',
        'severity': 'Moderate',
        'treatment': [
            'Apply fungicide in early spring',
            'Remove nearby cedar trees if possible',
            'Destroy infected leaves',
            'Use resistant varieties'
        ],
        'prevention': 'Plant resistant apple varieties'
    },
    'Apple___healthy': {
        'disease': 'Healthy Plant',
        'severity': 'None',
        'treatment': ['No treatment needed - plant is healthy!'],
        'prevention': 'Continue regular care and monitoring'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'disease': 'Cercospora Leaf Spot',
        'severity': 'Moderate',
        'treatment': [
            'Apply fungicide (Azoxystrobin)',
            'Practice crop rotation',
            'Remove infected debris',
            'Ensure adequate spacing'
        ],
        'prevention': 'Use resistant hybrids, crop rotation'
    },
    'Corn_(maize)___Common_rust_': {
        'disease': 'Common Rust',
        'severity': 'Moderate',
        'treatment': [
            'Apply fungicide if severe',
            'Remove infected leaves',
            'Improve air circulation',
            'Monitor regularly'
        ],
        'prevention': 'Plant resistant varieties'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'disease': 'Northern Leaf Blight',
        'severity': 'Severe',
        'treatment': [
            'Apply fungicide immediately',
            'Remove infected plants',
            'Practice 2-year crop rotation',
            'Till infected debris'
        ],
        'prevention': 'Resistant hybrids, crop rotation'
    },
    'Corn_(maize)___healthy': {
        'disease': 'Healthy Plant',
        'severity': 'None',
        'treatment': ['No treatment needed - plant is healthy!'],
        'prevention': 'Continue regular care and monitoring'
    },
    'Tomato___Bacterial_spot': {
        'disease': 'Bacterial Spot',
        'severity': 'Severe',
        'treatment': [
            'Apply copper-based bactericide',
            'Remove infected plants',
            'Avoid overhead watering',
            'Disinfect tools'
        ],
        'prevention': 'Use disease-free seeds, drip irrigation'
    },
    'Tomato___Early_blight': {
        'disease': 'Early Blight',
        'severity': 'Moderate',
        'treatment': [
            'Apply fungicide (Chlorothalonil)',
            'Remove lower infected leaves',
            'Mulch around plants',
            'Water at base only'
        ],
        'prevention': 'Crop rotation, proper spacing'
    },
    'Tomato___Late_blight': {
        'disease': 'Late Blight',
        'severity': 'Critical',
        'treatment': [
            'Apply fungicide immediately (Mancozeb)',
            'Remove and destroy infected plants',
            'Improve drainage',
            'Avoid evening watering'
        ],
        'prevention': 'Plant resistant varieties, monitor weather'
    },
    'Tomato___healthy': {
        'disease': 'Healthy Plant',
        'severity': 'None',
        'treatment': ['No treatment needed - plant is healthy!'],
        'prevention': 'Continue regular care and monitoring'
    },
    # Add more diseases as needed...
    'default': {
        'disease': 'Unknown Disease',
        'severity': 'Unknown',
        'treatment': [
            'Consult agricultural expert',
            'Send sample to plant pathology lab',
            'Isolate affected plants',
            'Monitor for symptoms'
        ],
        'prevention': 'Regular monitoring and good agricultural practices'
    }
}

class PlantDiseasePredictor:
    def __init__(self, model_path='best_model.h5', class_names_path='class_names.npy'):
        """
        Initialize predictor with trained model and class names
        """
        print("Loading model...")
        self.model = keras.models.load_model(model_path)
        self.class_names = np.load(class_names_path, allow_pickle=True).item()
        self.img_size = self.model.input_shape[1]  # automatically 160 for your model
        print(f"Model loaded successfully! ({len(self.class_names)} classes)")
    
    def preprocess_image(self, img_path):
        """
        Load and preprocess image for prediction
        """
        img = image.load_img(img_path, target_size=(self.img_size, self.img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        return img_array
    
    def predict(self, img_path, top_k=3):
        """
        Predict disease from image
        Returns: predicted class, confidence, and top-k predictions
        """
        # Preprocess image
        img_array = self.preprocess_image(img_path)
        
        # Get predictions
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            class_name = self.class_names[idx]
            confidence = predictions[idx] * 100
            top_predictions.append({
                'class': class_name,
                'confidence': confidence
            })
        
        # Best prediction
        best_idx = np.argmax(predictions)
        predicted_class = self.class_names[best_idx]
        confidence = predictions[best_idx] * 100
        
        return predicted_class, confidence, top_predictions
    
    def get_treatment(self, predicted_class):
        """
        Get treatment recommendations for predicted disease
        """
        if predicted_class in TREATMENT_RECOMMENDATIONS:
            return TREATMENT_RECOMMENDATIONS[predicted_class]
        else:
            return TREATMENT_RECOMMENDATIONS['default']
    
    def predict_and_recommend(self, img_path):
        """
        Complete prediction with treatment recommendations
        """
        print("\n" + "="*60)
        print("PLANT DISEASE DETECTION RESULTS")
        print("="*60)
        
        # Predict
        predicted_class, confidence, top_predictions = self.predict(img_path)
        
        # Get treatment
        treatment_info = self.get_treatment(predicted_class)
        
        # Display results
        print(f"\nImage: {img_path}")
        print(f"\n🔍 DIAGNOSIS:")
        print(f"   Disease: {treatment_info['disease']}")
        print(f"   Severity: {treatment_info['severity']}")
        print(f"   Confidence: {confidence:.2f}%")
        
        print(f"\n💊 TREATMENT RECOMMENDATIONS:")
        for i, step in enumerate(treatment_info['treatment'], 1):
            print(f"   {i}. {step}")
        
        print(f"\n🛡️ PREVENTION:")
        print(f"   {treatment_info['prevention']}")
        
        print(f"\n📊 TOP PREDICTIONS:")
        for i, pred in enumerate(top_predictions, 1):
            disease_name = pred['class'].replace('___', ' - ').replace('_', ' ')
            print(f"   {i}. {disease_name}: {pred['confidence']:.2f}%")
        
        print("\n" + "="*60)
        
        return {
            'predicted_class': predicted_class,
            'disease_name': treatment_info['disease'],
            'confidence': confidence,
            'severity': treatment_info['severity'],
            'treatment': treatment_info['treatment'],
            'prevention': treatment_info['prevention'],
            'top_predictions': top_predictions
        }

def main():
    """
    Example usage
    """
    # Initialize predictor
    predictor = PlantDiseasePredictor(
        model_path='best_model.h5',
        class_names_path='class_names.npy'
    )
    
    # Example prediction
    test_image = 'test_leaf.jpg'  # Replace with your image path
    
    if os.path.exists(test_image):
        result = predictor.predict_and_recommend(test_image)
    else:
        print(f"\nError: Image '{test_image}' not found!")
        print("Please provide a valid image path.")
        print("\nExample usage:")
        print("  python predict.py path/to/your/leaf_image.jpg")

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1:
        # Image path provided as command line argument
        img_path = sys.argv[1]
        
        if os.path.exists(img_path):
            predictor = PlantDiseasePredictor()
            predictor.predict_and_recommend(img_path)
        else:
            print(f"Error: Image '{img_path}' not found!")
    else:
        # No argument provided, run example
        main()
