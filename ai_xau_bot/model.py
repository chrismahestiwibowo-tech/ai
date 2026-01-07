"""
Model Training Module - XGBoost Regression with optimization
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("darkgrid")


class XGBoostPricePredictor:
    """XGBoost model for price prediction"""
    
    def __init__(self, params=None):
        """
        Initialize XGBoost predictor
        
        Args:
            params: Dictionary of XGBoost parameters
        """
        self.params = params or config.XGBOOST_PARAMS
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.train_score = {}
        self.test_score = {}
        
    def prepare_data(self, df, feature_columns, target='Close', future_steps=1):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
            target: Target column name
            future_steps: Number of steps ahead to predict
            
        Returns:
            X, y: Features and target arrays
        """
        # Create target variable (future price)
        df = df.copy()
        df['Target'] = df[target].shift(-future_steps)
        
        # Remove rows with NaN target
        df = df.dropna(subset=['Target'])
        
        # Extract features and target
        X = df[feature_columns].values
        y = df['Target'].values
        
        self.feature_names = feature_columns
        logger.info(f"Prepared data - X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y, df.index
    
    def split_data(self, X, y, indices, train_size=None):
        """
        Split data into train and test sets (time-series aware)
        
        Args:
            X: Features
            y: Target
            indices: Time indices
            train_size: Proportion of training data
            
        Returns:
            X_train, X_test, y_train, y_test, train_idx, test_idx
        """
        train_size = train_size or config.TRAIN_SIZE
        split_point = int(len(X) * train_size)
        
        X_train = X[:split_point]
        X_test = X[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]
        train_idx = indices[:split_point]
        test_idx = indices[split_point:]
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Train period: {train_idx[0]} to {train_idx[-1]}")
        logger.info(f"Test period: {test_idx[0]} to {test_idx[-1]}")
        
        return X_train, X_test, y_train, y_test, train_idx, test_idx
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            X_train_scaled, X_test_scaled
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Features scaled using StandardScaler")
        return X_train_scaled, X_test_scaled
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        logger.info("Training XGBoost model...")
        
        # Create XGBoost regressor
        self.model = xgb.XGBRegressor(**self.params)
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train model
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            verbose=100
        )
        
        logger.info("Model training completed!")
        
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
        """
        # Train predictions
        y_train_pred = self.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        self.train_score = {
            'MSE': train_mse,
            'RMSE': train_rmse,
            'MAE': train_mae,
            'R2': train_r2
        }
        
        # Test predictions
        y_test_pred = self.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        self.test_score = {
            'MSE': test_mse,
            'RMSE': test_rmse,
            'MAE': test_mae,
            'R2': test_r2
        }
        
        # Log results
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*60)
        logger.info("\nTRAIN SET:")
        logger.info(f"  MSE:  {train_mse:.6f}")
        logger.info(f"  RMSE: {train_rmse:.6f}")
        logger.info(f"  MAE:  {train_mae:.6f}")
        logger.info(f"  R²:   {train_r2:.6f}")
        logger.info("\nTEST SET:")
        logger.info(f"  MSE:  {test_mse:.6f}")
        logger.info(f"  RMSE: {test_rmse:.6f}")
        logger.info(f"  MAE:  {test_mae:.6f}")
        logger.info(f"  R²:   {test_r2:.6f}")
        logger.info("="*60 + "\n")
        
        return y_train_pred, y_test_pred
    
    def plot_predictions(self, y_train, y_train_pred, y_test, y_test_pred, 
                        train_idx, test_idx, save_path=None):
        """
        Plot actual vs predicted prices
        
        Args:
            y_train: Actual train prices
            y_train_pred: Predicted train prices
            y_test: Actual test prices
            y_test_pred: Predicted test prices
            train_idx: Train time indices
            test_idx: Test time indices
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Train predictions over time
        axes[0, 0].plot(train_idx, y_train, label='Actual', alpha=0.7)
        axes[0, 0].plot(train_idx, y_train_pred, label='Predicted', alpha=0.7)
        axes[0, 0].set_title('Train Set: Actual vs Predicted Prices', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test predictions over time
        axes[0, 1].plot(test_idx, y_test, label='Actual', alpha=0.7)
        axes[0, 1].plot(test_idx, y_test_pred, label='Predicted', alpha=0.7)
        axes[0, 1].set_title('Test Set: Actual vs Predicted Prices', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Price')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Train scatter plot
        axes[1, 0].scatter(y_train, y_train_pred, alpha=0.5, s=10)
        axes[1, 0].plot([y_train.min(), y_train.max()], 
                       [y_train.min(), y_train.max()], 'r--', lw=2)
        axes[1, 0].set_title(f'Train Set Scatter (R² = {self.train_score["R2"]:.4f})', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Actual Price')
        axes[1, 0].set_ylabel('Predicted Price')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Test scatter plot
        axes[1, 1].scatter(y_test, y_test_pred, alpha=0.5, s=10, color='orange')
        axes[1, 1].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 1].set_title(f'Test Set Scatter (R² = {self.test_score["R2"]:.4f})', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Actual Price')
        axes[1, 1].set_ylabel('Predicted Price')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save plot
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
        return feature_importance
    
    def save_model(self, model_path=None, scaler_path=None):
        """Save trained model and scaler"""
        model_path = model_path or config.MODEL_PATH
        scaler_path = scaler_path or config.SCALER_PATH
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path=None, scaler_path=None):
        """Load trained model and scaler"""
        model_path = model_path or config.MODEL_PATH
        scaler_path = scaler_path or config.SCALER_PATH
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Scaler loaded from {scaler_path}")


if __name__ == "__main__":
    # Test model training
    from data_loader import load_data
    from indicators import add_indicators
    
    logger.info("Loading and preparing data...")
    df = load_data()
    
    if df is not None:
        df_with_indicators, features = add_indicators(df)
        
        # Initialize predictor
        predictor = XGBoostPricePredictor()
        
        # Prepare data
        X, y, indices = predictor.prepare_data(df_with_indicators, features)
        
        # Split data
        X_train, X_test, y_train, y_test, train_idx, test_idx = predictor.split_data(X, y, indices)
        
        # Scale features
        X_train_scaled, X_test_scaled = predictor.scale_features(X_train, X_test)
        
        # Train model
        predictor.train(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Evaluate
        y_train_pred, y_test_pred = predictor.evaluate(X_train_scaled, y_train, 
                                                        X_test_scaled, y_test)
        
        # Plot results
        predictor.plot_predictions(y_train, y_train_pred, y_test, y_test_pred,
                                   train_idx, test_idx, 
                                   save_path='plots/predictions.png')
        
        # Feature importance
        predictor.plot_feature_importance(save_path='plots/feature_importance.png')
        
        # Save model
        predictor.save_model()
