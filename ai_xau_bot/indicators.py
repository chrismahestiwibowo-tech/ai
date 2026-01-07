"""
Feature Engineering Module - Calculates 11 technical indicators
"""
import pandas as pd
import numpy as np
import ta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Class to calculate technical indicators for trading"""
    
    def __init__(self, df):
        """
        Initialize with price data
        
        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df.copy()
        
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        self.df['RSI'] = ta.momentum.RSIIndicator(
            close=self.df['Close'], 
            window=period
        ).rsi()
        logger.info("Calculated RSI")
        
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD and Signal Line"""
        macd = ta.trend.MACD(
            close=self.df['Close'],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal
        )
        self.df['MACD'] = macd.macd()
        self.df['MACD_Signal'] = macd.macd_signal()
        self.df['MACD_Hist'] = macd.macd_diff()
        logger.info("Calculated MACD")
        
    def calculate_bollinger_bands(self, period=20, std=2):
        """Calculate Bollinger Bands"""
        bb = ta.volatility.BollingerBands(
            close=self.df['Close'],
            window=period,
            window_dev=std
        )
        self.df['BB_Upper'] = bb.bollinger_hband()
        self.df['BB_Middle'] = bb.bollinger_mavg()
        self.df['BB_Lower'] = bb.bollinger_lband()
        self.df['BB_Width'] = bb.bollinger_wband()
        logger.info("Calculated Bollinger Bands")
        
    def calculate_atr(self, period=14):
        """Calculate Average True Range"""
        self.df['ATR'] = ta.volatility.AverageTrueRange(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            window=period
        ).average_true_range()
        logger.info("Calculated ATR")
        
    def calculate_moving_averages(self):
        """Calculate Simple and Exponential Moving Averages"""
        # SMA
        self.df['SMA_20'] = ta.trend.SMAIndicator(
            close=self.df['Close'],
            window=20
        ).sma_indicator()
        
        self.df['SMA_50'] = ta.trend.SMAIndicator(
            close=self.df['Close'],
            window=50
        ).sma_indicator()
        
        # EMA
        self.df['EMA_20'] = ta.trend.EMAIndicator(
            close=self.df['Close'],
            window=20
        ).ema_indicator()
        
        self.df['EMA_50'] = ta.trend.EMAIndicator(
            close=self.df['Close'],
            window=50
        ).ema_indicator()
        
        logger.info("Calculated Moving Averages (SMA, EMA)")
        
    def calculate_stochastic(self, period=14, smooth=3):
        """Calculate Stochastic Oscillator"""
        stoch = ta.momentum.StochasticOscillator(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            window=period,
            smooth_window=smooth
        )
        self.df['Stochastic'] = stoch.stoch()
        self.df['Stochastic_Signal'] = stoch.stoch_signal()
        logger.info("Calculated Stochastic Oscillator")
        
    def calculate_cci(self, period=20):
        """Calculate Commodity Channel Index"""
        self.df['CCI'] = ta.trend.CCIIndicator(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            window=period
        ).cci()
        logger.info("Calculated CCI")
        
    def calculate_adx(self, period=14):
        """Calculate Average Directional Index"""
        adx = ta.trend.ADXIndicator(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            window=period
        )
        self.df['ADX'] = adx.adx()
        self.df['ADX_Pos'] = adx.adx_pos()
        self.df['ADX_Neg'] = adx.adx_neg()
        logger.info("Calculated ADX")
        
    def calculate_williams_r(self, period=14):
        """Calculate Williams %R"""
        self.df['Williams_R'] = ta.momentum.WilliamsRIndicator(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            lbp=period
        ).williams_r()
        logger.info("Calculated Williams %R")
        
    def calculate_obv(self):
        """Calculate On-Balance Volume"""
        self.df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            close=self.df['Close'],
            volume=self.df['tick_volume']
        ).on_balance_volume()
        logger.info("Calculated OBV")
        
    def calculate_roc(self, period=12):
        """Calculate Rate of Change"""
        self.df['ROC'] = ta.momentum.ROCIndicator(
            close=self.df['Close'],
            window=period
        ).roc()
        logger.info("Calculated ROC")
        
    def add_price_features(self):
        """Add additional price-based features"""
        # Price changes
        self.df['Price_Change'] = self.df['Close'].pct_change()
        self.df['Price_Range'] = self.df['High'] - self.df['Low']
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            self.df[f'Close_Lag_{lag}'] = self.df['Close'].shift(lag)
            self.df[f'Return_Lag_{lag}'] = self.df['Price_Change'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            self.df[f'Close_Rolling_Mean_{window}'] = self.df['Close'].rolling(window=window).mean()
            self.df[f'Close_Rolling_Std_{window}'] = self.df['Close'].rolling(window=window).std()
            self.df[f'Volume_Rolling_Mean_{window}'] = self.df['tick_volume'].rolling(window=window).mean()
        
        logger.info("Added price-based features")
        
    def calculate_all_indicators(self):
        """Calculate all 11 main indicators plus additional features"""
        logger.info("Starting calculation of all technical indicators...")
        
        # Core 11 indicators
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_atr()
        self.calculate_moving_averages()
        self.calculate_stochastic()
        self.calculate_cci()
        self.calculate_adx()
        self.calculate_williams_r()
        self.calculate_obv()
        self.calculate_roc()
        
        # Additional features
        self.add_price_features()
        
        # Remove NaN values
        initial_shape = self.df.shape
        self.df.dropna(inplace=True)
        logger.info(f"Removed NaN values: {initial_shape[0] - self.df.shape[0]} rows dropped")
        
        logger.info(f"Final dataset shape: {self.df.shape}")
        logger.info("All indicators calculated successfully!")
        
        return self.df
    
    def get_feature_columns(self):
        """Return list of feature columns (excluding OHLCV)"""
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'tick_volume', 'spread', 'real_volume']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        return feature_cols


def add_indicators(df):
    """
    Main function to add all technical indicators to DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all indicators added
    """
    ti = TechnicalIndicators(df)
    df_with_indicators = ti.calculate_all_indicators()
    return df_with_indicators, ti.get_feature_columns()


if __name__ == "__main__":
    # Test with sample data
    from data_loader import load_data
    
    logger.info("Loading data...")
    df = load_data()
    
    if df is not None:
        logger.info("Adding indicators...")
        df_indicators, features = add_indicators(df)
        
        print("\n" + "="*50)
        print("INDICATORS SUMMARY")
        print("="*50)
        print(f"\nDataset shape: {df_indicators.shape}")
        print(f"\nNumber of features: {len(features)}")
        print(f"\nFeature columns:\n{features}")
        print(f"\nFirst 5 rows:\n{df_indicators.head()}")
        print(f"\nStatistics:\n{df_indicators.describe()}")
