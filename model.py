"""
model.py

Implements Stage 1 (XGBoost) and Stage 2 (LSTM/Hybrid) modeling logic.
Since the specified tech stack is XGBoost and Scikit-Learn, the XGBoost 
classifier is fully implemented. The LSTM hybrid structure is scaffolded 
ready for integration if a deep learning library (e.g., TensorFlow/PyTorch) is added.
"""

import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import config

logger = logging.getLogger(__name__)

class Stage1Model:
    """Stage 1 XGBoost classification model."""
    
    def __init__(self):
        # We use objective multi:softprob to output probabilities for 3 classes
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05
        )
        self.is_trained = False
        self.features = []

    def train(self, df: pd.DataFrame, target_col: str = 'target'):
        """
        Trains the XGBoost model.
        
        Args:
            df (pd.DataFrame): Training data containing features and target.
            target_col (str): Column name containing the target labels.
        """
        try:
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
                
            X = df.drop(columns=[target_col])
            # Drop unnecessary or non-numeric columns like time string if present
            X = X.select_dtypes(include=[np.number])
            y = df[target_col]
            
            self.features = list(X.columns)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            logger.info("Starting XGBoost training...")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            self.is_trained = True
            
            # Log metrics
            preds = self.model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            logger.info(f"XGBoost training completed. Validation Accuracy: {acc:.4f}")
            
        except Exception as e:
            logger.error(f"Error during Stage1Model training: {str(e)}")

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predicts trading signals for the latest row(s) based on confidence thresholds.
        
        Args:
            df (pd.DataFrame): Feature dataframe.
            
        Returns:
            dict: Dictionary containing predicted 'signal' (1=Buy, 2=Sell, 0=None)
                  and the 'confidence' level.
        """
        try:
            if not self.is_trained:
                logger.error("Model is not trained. Please load or train first.")
                return {'signal': 0, 'confidence': 0.0}
                
            # Use only the specified features
            X = df[self.features].copy()
            # Predict probabilities for the last row (current situation)
            latest_features = X.iloc[-1:]
            
            probs = self.model.predict_proba(latest_features)[0]
            
            # probs[0] = Hold/Loss, probs[1] = Buy, probs[2] = Sell
            prob_buy = probs[1]
            prob_sell = probs[2]
            
            if prob_buy >= config.CONFIDENCE_THRESHOLD and prob_buy > prob_sell:
                return {'signal': 1, 'confidence': float(prob_buy)}
            elif prob_sell >= config.CONFIDENCE_THRESHOLD and prob_sell > prob_buy:
                return {'signal': 2, 'confidence': float(prob_sell)}
                
            return {'signal': 0, 'confidence': float(max(prob_buy, prob_sell))}
            
        except Exception as e:
            logger.error(f"Error during Stage1Model prediction: {str(e)}")
            return {'signal': 0, 'confidence': 0.0}

    def save(self, filepath: str):
        if not self.is_trained:
            logger.error("Cannot save untaught model.")
            return
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump({'model': self.model, 'features': self.features}, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load(self, filepath: str):
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.features = data['features']
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")


class Stage2HybridModel:
    """
    Stage 2 Hybrid Model (LSTM for temporal sequence extraction -> XGBoost).
    This acts as a scaffold for when a deep learning library is integrated.
    """
    def __init__(self):
        self.xgb_model = Stage1Model()
        self.is_trained = False
        logger.warning("Stage2HybridModel LSTM components require TensorFlow/PyTorch. XGBoost logic isolated.")
        
    def train(self, df: pd.DataFrame, target_col: str = 'target'):
        # TODO: Implement LSTM sequence extraction feature mapping once TF/PyTorch is available
        logger.info("Training hybrid sequence. Bypassing to XGBoost...")
        self.xgb_model.train(df, target_col)
        self.is_trained = True
        
    def predict(self, df: pd.DataFrame) -> dict:
        return self.xgb_model.predict(df)
