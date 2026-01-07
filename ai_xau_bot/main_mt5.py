"""
Main Application - XAU/USD Price Prediction System
Combines MetaTrader5 data, technical indicators, XGBoost model, and Monte Carlo simulation
"""
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import modules
from data_loader import load_data
from indicators import add_indicators
from model import XGBoostPricePredictor
from monte_carlo import run_monte_carlo
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xauusd_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class XAUUSDPredictionSystem:
    """Complete price prediction system for XAU/USD"""
    
    def __init__(self):
        """Initialize the prediction system"""
        self.df = None
        self.df_with_indicators = None
        self.feature_columns = None
        self.predictor = None
        self.mc_simulator = None
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        logger.info("="*70)
        logger.info("XAU/USD PRICE PREDICTION SYSTEM")
        logger.info("="*70)
        logger.info(f"Symbol: {config.SYMBOL}")
        logger.info(f"Timeframe: {config.TIMEFRAME}")
        logger.info(f"Historical data: {config.YEARS_BACK} years")
        logger.info(f"Train/Test split: {config.TRAIN_SIZE}/{config.TEST_SIZE}")
        logger.info(f"Monte Carlo simulations: {config.MC_SIMULATIONS}")
        logger.info("="*70 + "\n")
    
    def load_and_prepare_data(self):
        """Load historical data and add technical indicators"""
        logger.info("STEP 1: Loading historical data from MetaTrader5...")
        self.df = load_data()
        
        if self.df is None:
            raise Exception("Failed to load data from MetaTrader5")
        
        logger.info("\nSTEP 2: Calculating technical indicators...")
        self.df_with_indicators, self.feature_columns = add_indicators(self.df)
        
        logger.info(f"\nData preparation completed!")
        logger.info(f"Total samples: {len(self.df_with_indicators)}")
        logger.info(f"Total features: {len(self.feature_columns)}")
        
        return self.df_with_indicators
    
    def train_xgboost_model(self):
        """Train XGBoost regression model"""
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Training XGBoost Model")
        logger.info("="*70 + "\n")
        
        # Initialize predictor
        self.predictor = XGBoostPricePredictor()
        
        # Prepare data
        X, y, indices = self.predictor.prepare_data(
            self.df_with_indicators, 
            self.feature_columns
        )
        
        # Split data
        X_train, X_test, y_train, y_test, train_idx, test_idx = self.predictor.split_data(
            X, y, indices
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.predictor.scale_features(X_train, X_test)
        
        # Train model
        self.predictor.train(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Evaluate model
        y_train_pred, y_test_pred = self.predictor.evaluate(
            X_train_scaled, y_train, 
            X_test_scaled, y_test
        )
        
        # Save results
        results_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
            'Train': [
                self.predictor.train_score['MSE'],
                self.predictor.train_score['RMSE'],
                self.predictor.train_score['MAE'],
                self.predictor.train_score['R2']
            ],
            'Test': [
                self.predictor.test_score['MSE'],
                self.predictor.test_score['RMSE'],
                self.predictor.test_score['MAE'],
                self.predictor.test_score['R2']
            ]
        })
        results_df.to_csv('results/model_metrics.csv', index=False)
        logger.info("Model metrics saved to results/model_metrics.csv")
        
        # Plot predictions
        self.predictor.plot_predictions(
            y_train, y_train_pred, y_test, y_test_pred,
            train_idx, test_idx,
            save_path='plots/predictions.png'
        )
        
        # Plot feature importance
        feature_importance = self.predictor.plot_feature_importance(
            save_path='plots/feature_importance.png'
        )
        feature_importance.to_csv('results/feature_importance.csv', index=False)
        logger.info("Feature importance saved to results/feature_importance.csv")
        
        # Save model
        self.predictor.save_model()
        
        return self.predictor
    
    def run_monte_carlo_simulation(self):
        """Run Monte Carlo simulation for price forecasting"""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Running Monte Carlo Simulation")
        logger.info("="*70 + "\n")
        
        price_data = self.df_with_indicators['Close']
        
        self.mc_simulator, forecast_stats = run_monte_carlo(
            price_data,
            n_simulations=config.MC_SIMULATIONS,
            days_ahead=config.MC_DAYS_AHEAD
        )
        
        # Save forecast statistics
        stats_df = pd.DataFrame([forecast_stats])
        stats_df.to_csv('results/monte_carlo_forecast.csv', index=False)
        logger.info("Monte Carlo forecast saved to results/monte_carlo_forecast.csv")
        
        # Calculate probability ranges
        current_price = price_data.iloc[-1]
        targets = [
            current_price * 0.90,  # 10% down
            current_price * 0.95,  # 5% down
            current_price * 0.98,  # 2% down
            current_price * 1.02,  # 2% up
            current_price * 1.05,  # 5% up
            current_price * 1.10   # 10% up
        ]
        
        probabilities = self.mc_simulator.calculate_probability_ranges(targets)
        
        # Save probabilities
        prob_df = pd.DataFrame([
            {'Target': f'${target:.2f}', 'Probability': f'{prob:.2f}%'}
            for target, prob in probabilities.items()
        ])
        prob_df.to_csv('results/price_probabilities.csv', index=False)
        logger.info("Price probabilities saved to results/price_probabilities.csv")
        
        return self.mc_simulator
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        logger.info("\n" + "="*70)
        logger.info("GENERATING SUMMARY REPORT")
        logger.info("="*70 + "\n")
        
        current_price = self.df_with_indicators['Close'].iloc[-1]
        
        report = f"""
{'='*70}
XAU/USD PRICE PREDICTION SYSTEM - SUMMARY REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA INFORMATION
{'-'*70}
Symbol:                 {config.SYMBOL}
Timeframe:              {config.TIMEFRAME}
Historical Period:      {config.YEARS_BACK} years
Total Samples:          {len(self.df_with_indicators)}
Date Range:             {self.df_with_indicators.index[0]} to {self.df_with_indicators.index[-1]}
Current Price:          ${current_price:.2f}

FEATURE ENGINEERING
{'-'*70}
Total Features:         {len(self.feature_columns)}
Technical Indicators:   11 core indicators + derived features
Main Indicators:        RSI, MACD, Bollinger Bands, ATR, SMA, EMA,
                        Stochastic, CCI, ADX, Williams %R, OBV

XGBOOST MODEL PERFORMANCE
{'-'*70}
Train/Test Split:       {config.TRAIN_SIZE:.0%} / {config.TEST_SIZE:.0%}

Training Set:
  MSE:                  {self.predictor.train_score['MSE']:.6f}
  RMSE:                 {self.predictor.train_score['RMSE']:.6f}
  MAE:                  {self.predictor.train_score['MAE']:.6f}
  R²:                   {self.predictor.train_score['R2']:.6f}

Test Set:
  MSE:                  {self.predictor.test_score['MSE']:.6f}
  RMSE:                 {self.predictor.test_score['RMSE']:.6f}
  MAE:                  {self.predictor.test_score['MAE']:.6f}
  R²:                   {self.predictor.test_score['R2']:.6f}

MONTE CARLO SIMULATION
{'-'*70}
Number of Simulations:  {config.MC_SIMULATIONS:,}
Forecast Horizon:       {config.MC_DAYS_AHEAD} days

Price Forecast:
  Mean:                 ${self.mc_simulator.get_forecast_statistics()['mean']:.2f}
  Median:               ${self.mc_simulator.get_forecast_statistics()['median']:.2f}
  5th Percentile:       ${self.mc_simulator.get_forecast_statistics()['percentile_5']:.2f}
  95th Percentile:      ${self.mc_simulator.get_forecast_statistics()['percentile_95']:.2f}

Risk Metrics:
  VaR (95%):            ${self.mc_simulator.get_forecast_statistics()['var_95']:.2f}
  CVaR (95%):           ${self.mc_simulator.get_forecast_statistics()['cvar_95']:.2f}

FILES GENERATED
{'-'*70}
Models:
  - models/xauusd_xgboost_model.pkl
  - models/scaler.pkl

Plots:
  - plots/predictions.png
  - plots/feature_importance.png
  - plots/monte_carlo_paths.png
  - plots/monte_carlo_distribution.png

Results:
  - results/model_metrics.csv
  - results/feature_importance.csv
  - results/monte_carlo_forecast.csv
  - results/price_probabilities.csv

{'='*70}
END OF REPORT
{'='*70}
"""
        
        # Save report
        with open('results/summary_report.txt', 'w') as f:
            f.write(report)
        
        logger.info(report)
        logger.info("\nSummary report saved to results/summary_report.txt")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            start_time = datetime.now()
            logger.info(f"Analysis started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Step 1 & 2: Load data and add indicators
            self.load_and_prepare_data()
            
            # Step 3: Train XGBoost model
            self.train_xgboost_model()
            
            # Step 4: Run Monte Carlo simulation
            self.run_monte_carlo_simulation()
            
            # Step 5: Generate summary report
            self.generate_summary_report()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("\n" + "="*70)
            logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
            logger.info("="*70)
            logger.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            logger.info("="*70 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"\nERROR: Analysis failed - {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("XAU/USD PRICE PREDICTION SYSTEM")
    print("Powered by MetaTrader5, XGBoost & Monte Carlo Simulation")
    print("="*70 + "\n")
    
    # Create and run prediction system
    system = XAUUSDPredictionSystem()
    success = system.run_complete_analysis()
    
    if success:
        print("\n✓ Analysis completed successfully!")
        print("\nCheck the following directories for results:")
        print("  - plots/      : Visualization charts")
        print("  - results/    : CSV files and reports")
        print("  - models/     : Trained models")
    else:
        print("\n✗ Analysis failed. Check logs for details.")
    
    return system


if __name__ == "__main__":
    system = main()
