"""
Data Loader Module - Fetches historical data from MetaTrader5
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MT5DataLoader:
    """Class to handle MetaTrader5 data fetching operations"""
    
    def __init__(self):
        self.initialized = False
        
    def initialize(self):
        """Initialize connection to MetaTrader5"""
        try:
            # Initialize MT5 connection
            if config.MT5_CONFIG['path']:
                if not mt5.initialize(path=config.MT5_CONFIG['path']):
                    logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
                    return False
            else:
                if not mt5.initialize():
                    logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
                    return False
            
            # Login if credentials are provided
            if config.MT5_CONFIG['login'] and config.MT5_CONFIG['password']:
                authorized = mt5.login(
                    login=config.MT5_CONFIG['login'],
                    password=config.MT5_CONFIG['password'],
                    server=config.MT5_CONFIG['server']
                )
                
                if not authorized:
                    logger.error(f"Failed to login to MT5, error code = {mt5.last_error()}")
                    return False
                    
                logger.info(f"Connected to MetaTrader5 account #{config.MT5_CONFIG['login']}")
            else:
                logger.info("Connected to MetaTrader5 (no login credentials provided)")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing MT5: {str(e)}")
            return False
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        mt5.shutdown()
        self.initialized = False
        logger.info("MT5 connection closed")
    
    def get_timeframe(self, timeframe_str):
        """Convert timeframe string to MT5 constant"""
        timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        return timeframes.get(timeframe_str, mt5.TIMEFRAME_H1)
    
    def fetch_historical_data(self, symbol=None, timeframe=None, years_back=None):
        """
        Fetch historical data from MetaTrader5
        
        Args:
            symbol: Trading symbol (default: from config)
            timeframe: Timeframe string (default: from config)
            years_back: Number of years to fetch (default: from config)
            
        Returns:
            pandas.DataFrame: Historical price data
        """
        if not self.initialized:
            if not self.initialize():
                raise Exception("Failed to initialize MT5")
        
        symbol = symbol or config.SYMBOL
        timeframe = timeframe or config.TIMEFRAME
        years_back = years_back or config.YEARS_BACK
        
        try:
            # Calculate date range
            utc_to = datetime.now()
            utc_from = utc_to - timedelta(days=365 * years_back)
            
            # Get MT5 timeframe constant
            mt5_timeframe = self.get_timeframe(timeframe)
            
            # Fetch rates
            logger.info(f"Fetching {years_back} years of {symbol} data from {utc_from} to {utc_to}")
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, utc_from, utc_to)
            
            if rates is None or len(rates) == 0:
                logger.error(f"Failed to fetch data, error code = {mt5.last_error()}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Set time as index
            df.set_index('time', inplace=True)
            
            # Rename columns for clarity
            df.columns = ['Open', 'High', 'Low', 'Close', 'tick_volume', 'spread', 'real_volume']
            
            logger.info(f"Successfully fetched {len(df)} candles")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"Data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None
    
    def get_symbol_info(self, symbol=None):
        """Get symbol information"""
        symbol = symbol or config.SYMBOL
        
        if not self.initialized:
            if not self.initialize():
                return None
        
        symbol_info = mt5.symbol_info(symbol)
        
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return None
        
        return symbol_info._asdict()


def load_data():
    """
    Main function to load and return historical data
    
    Returns:
        pandas.DataFrame: Historical price data
    """
    loader = MT5DataLoader()
    
    try:
        data = loader.fetch_historical_data()
        return data
    finally:
        loader.shutdown()


if __name__ == "__main__":
    # Test the data loader
    df = load_data()
    
    if df is not None:
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"\nShape: {df.shape}")
        print(f"\nFirst 5 rows:\n{df.head()}")
        print(f"\nLast 5 rows:\n{df.tail()}")
        print(f"\nData Info:\n{df.info()}")
        print(f"\nStatistics:\n{df.describe()}")
