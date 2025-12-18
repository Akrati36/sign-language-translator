"""
Sign Classifier - Deep learning model for sign language classification
Classifies hand landmarks into sign language letters/words
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os

class SignClassifier:
    """Classify hand landmarks into sign language"""
    
    def __init__(self, model_path='models/sign_classifier.h5'):
        """
        Initialize classifier
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model"""
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                print(f"✓ Loaded model from {self.model_path}")
                
                # Load class names
                config_path = self.model_path.replace('.h5', '_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        self.class_names = config['class_names']
                else:
                    # Default: A-Z alphabet
                    self.class_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            except Exception as e:
                print(f"⚠ Error loading model: {e}")
                print("Using dummy model for demo")
                self.model = self._create_dummy_model()
                self.class_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        else:
            print(f"⚠ Model not found at {self.model_path}")
            print("Using dummy model for demo")
            self.model = self._create_dummy_model()
            self.class_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    def _create_dummy_model(self):
        """Create dummy model for demo purposes"""
        model = keras.Sequential([
            keras.layers.Input(shape=(21, 3)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(26, activation='softmax')
        ])
        return model
    
    def preprocess_landmarks(self, landmarks):
        """
        Preprocess landmarks for model input
        
        Args:
            landmarks: NumPy array (21, 3)
            
        Returns:
            Preprocessed landmarks ready for model
        """
        # Normalize landmarks
        # Center around wrist
        wrist = landmarks[0]
        centered = landmarks - wrist
        
        # Scale by hand size
        middle_finger_tip = landmarks[12]
        hand_size = np.linalg.norm(middle_finger_tip - wrist)
        
        if hand_size > 0:
            normalized = centered / hand_size
        else:
            normalized = centered
        
        # Reshape for model input
        processed = normalized.reshape(1, 21, 3)
        
        return processed
    
    def predict(self, landmarks):
        """
        Predict sign from landmarks
        
        Args:
            landmarks: NumPy array (21, 3) or list of arrays
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {
                'sign': 'Model not loaded',
                'confidence': 0.0,
                'alternatives': []
            }
        
        # Handle single or multiple hands
        if isinstance(landmarks, list):
            # Use first hand for now
            landmarks = landmarks[0]
        
        # Preprocess
        processed = self.preprocess_landmarks(landmarks)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)
        
        # Get top prediction
        top_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][top_idx])
        sign = self.class_names[top_idx]
        
        # Get top 3 alternatives
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        alternatives = [
            {
                'sign': self.class_names[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_idx[1:]  # Skip top prediction
        ]
        
        return {
            'sign': sign,
            'confidence': confidence,
            'alternatives': alternatives,
            'all_predictions': predictions[0].tolist()
        }
    
    def predict_sequence(self, landmarks_sequence):
        """
        Predict sign from sequence of landmarks (for dynamic signs)
        
        Args:
            landmarks_sequence: List of landmark arrays
            
        Returns:
            Prediction results
        """
        # For now, just use the last frame
        # TODO: Implement LSTM for sequence prediction
        return self.predict(landmarks_sequence[-1])
    
    @staticmethod
    def build_model(input_shape=(21, 3), num_classes=26):
        """
        Build CNN model for sign classification
        
        Args:
            input_shape: Shape of input landmarks
            num_classes: Number of sign classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            keras.layers.Input(shape=input_shape),
            
            # Flatten landmarks
            keras.layers.Flatten(),
            
            # Dense layers
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            
            # Output layer
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def train_model(X_train, y_train, X_val, y_val, 
                   epochs=50, batch_size=32):
        """
        Train sign classification model
        
        Args:
            X_train: Training landmarks
            y_train: Training labels
            X_val: Validation landmarks
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Trained model and history
        """
        # Build model
        num_classes = y_train.shape[1]
        model = SignClassifier.build_model(num_classes=num_classes)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history