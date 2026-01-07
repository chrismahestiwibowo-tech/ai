"""
Monte Carlo Simulation Module - Price forecasting using Monte Carlo methods
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import os
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style("darkgrid")


class MonteCarloSimulator:
    """Monte Carlo simulation for price forecasting"""
    
    def __init__(self, price_data, n_simulations=None, days_ahead=None):
        """
        Initialize Monte Carlo simulator
        
        Args:
            price_data: Historical price series
            n_simulations: Number of simulations to run
            days_ahead: Number of days to forecast
        """
        self.price_data = price_data
        self.n_simulations = n_simulations or config.MC_SIMULATIONS
        self.days_ahead = days_ahead or config.MC_DAYS_AHEAD
        self.simulations = None
        self.last_price = price_data.iloc[-1] if isinstance(price_data, pd.Series) else price_data[-1]
        
        # Calculate historical statistics
        self.returns = self._calculate_returns()
        self.mean_return = np.mean(self.returns)
        self.std_return = np.std(self.returns)
        self.drift = self.mean_return - (0.5 * self.std_return ** 2)
        
        logger.info(f"Monte Carlo Simulator initialized")
        logger.info(f"Last price: {self.last_price:.2f}")
        logger.info(f"Mean return: {self.mean_return:.6f}")
        logger.info(f"Std return: {self.std_return:.6f}")
        logger.info(f"Drift: {self.drift:.6f}")
    
    def _calculate_returns(self):
        """Calculate log returns from price data"""
        if isinstance(self.price_data, pd.Series):
            returns = np.log(self.price_data / self.price_data.shift(1)).dropna()
        else:
            returns = np.log(self.price_data[1:] / self.price_data[:-1])
        return returns
    
    def run_simulation(self, use_gbm=True):
        """
        Run Monte Carlo simulation
        
        Args:
            use_gbm: Use Geometric Brownian Motion (True) or simple random walk (False)
            
        Returns:
            numpy.ndarray: Array of simulated price paths
        """
        logger.info(f"Running {self.n_simulations} Monte Carlo simulations for {self.days_ahead} days ahead...")
        
        # Initialize simulation matrix
        self.simulations = np.zeros((self.days_ahead, self.n_simulations))
        self.simulations[0] = self.last_price
        
        if use_gbm:
            # Geometric Brownian Motion
            for t in range(1, self.days_ahead):
                # Generate random shocks
                Z = np.random.standard_normal(self.n_simulations)
                
                # Calculate daily returns
                daily_returns = np.exp(self.drift + self.std_return * Z)
                
                # Update prices
                self.simulations[t] = self.simulations[t-1] * daily_returns
        else:
            # Simple random walk
            for t in range(1, self.days_ahead):
                # Generate random returns
                random_returns = np.random.normal(self.mean_return, self.std_return, self.n_simulations)
                
                # Update prices
                self.simulations[t] = self.simulations[t-1] * (1 + random_returns)
        
        logger.info("Simulation completed!")
        return self.simulations
    
    def get_forecast_statistics(self):
        """
        Calculate forecast statistics from simulations
        
        Returns:
            dict: Forecast statistics
        """
        if self.simulations is None:
            raise ValueError("Run simulation first!")
        
        final_prices = self.simulations[-1]
        
        stats_dict = {
            'mean': np.mean(final_prices),
            'median': np.median(final_prices),
            'std': np.std(final_prices),
            'min': np.min(final_prices),
            'max': np.max(final_prices),
            'percentile_5': np.percentile(final_prices, 5),
            'percentile_25': np.percentile(final_prices, 25),
            'percentile_75': np.percentile(final_prices, 75),
            'percentile_95': np.percentile(final_prices, 95),
            'var_95': self.last_price - np.percentile(final_prices, 5),  # Value at Risk
            'cvar_95': self.last_price - np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)])  # Conditional VaR
        }
        
        logger.info("\n" + "="*60)
        logger.info("MONTE CARLO FORECAST STATISTICS")
        logger.info("="*60)
        logger.info(f"Current Price:        {self.last_price:.2f}")
        logger.info(f"Forecast ({self.days_ahead} days ahead):")
        logger.info(f"  Mean:               {stats_dict['mean']:.2f}")
        logger.info(f"  Median:             {stats_dict['median']:.2f}")
        logger.info(f"  Std Dev:            {stats_dict['std']:.2f}")
        logger.info(f"  Min:                {stats_dict['min']:.2f}")
        logger.info(f"  Max:                {stats_dict['max']:.2f}")
        logger.info(f"  5th Percentile:     {stats_dict['percentile_5']:.2f}")
        logger.info(f"  95th Percentile:    {stats_dict['percentile_95']:.2f}")
        logger.info(f"  VaR (95%):          {stats_dict['var_95']:.2f}")
        logger.info(f"  CVaR (95%):         {stats_dict['cvar_95']:.2f}")
        logger.info("="*60 + "\n")
        
        return stats_dict
    
    def plot_simulation_paths(self, n_paths=100, save_path=None):
        """
        Plot simulation paths
        
        Args:
            n_paths: Number of paths to display
            save_path: Path to save plot
        """
        if self.simulations is None:
            raise ValueError("Run simulation first!")
        
        plt.figure(figsize=(14, 8))
        
        # Plot sample paths
        time_steps = np.arange(self.days_ahead)
        for i in range(min(n_paths, self.n_simulations)):
            plt.plot(time_steps, self.simulations[:, i], alpha=0.1, color='blue', linewidth=0.5)
        
        # Plot statistics
        mean_path = np.mean(self.simulations, axis=1)
        median_path = np.median(self.simulations, axis=1)
        percentile_5 = np.percentile(self.simulations, 5, axis=1)
        percentile_95 = np.percentile(self.simulations, 95, axis=1)
        
        plt.plot(time_steps, mean_path, color='red', linewidth=2, label='Mean')
        plt.plot(time_steps, median_path, color='green', linewidth=2, label='Median')
        plt.plot(time_steps, percentile_5, color='orange', linewidth=2, linestyle='--', label='5th Percentile')
        plt.plot(time_steps, percentile_95, color='orange', linewidth=2, linestyle='--', label='95th Percentile')
        
        plt.fill_between(time_steps, percentile_5, percentile_95, alpha=0.2, color='orange')
        
        plt.axhline(y=self.last_price, color='black', linestyle=':', linewidth=2, label=f'Current Price: {self.last_price:.2f}')
        
        plt.xlabel('Days Ahead', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.title(f'Monte Carlo Simulation - {self.n_simulations} Paths ({self.days_ahead} Days Ahead)', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Simulation paths plot saved to {save_path}")
        
        plt.show()
    
    def plot_price_distribution(self, save_path=None):
        """
        Plot final price distribution
        
        Args:
            save_path: Path to save plot
        """
        if self.simulations is None:
            raise ValueError("Run simulation first!")
        
        final_prices = self.simulations[-1]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(final_prices, bins=100, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(x=self.last_price, color='red', linestyle='--', linewidth=2, label=f'Current: {self.last_price:.2f}')
        axes[0].axvline(x=np.mean(final_prices), color='green', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(final_prices):.2f}')
        axes[0].axvline(x=np.median(final_prices), color='orange', linestyle='--', linewidth=2, 
                       label=f'Median: {np.median(final_prices):.2f}')
        axes[0].set_xlabel('Price', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Price Distribution ({self.days_ahead} Days Ahead)', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        box = axes[1].boxplot([final_prices], vert=True, patch_artist=True, 
                             labels=[f'{self.days_ahead} Days'])
        box['boxes'][0].set_facecolor('lightblue')
        axes[1].axhline(y=self.last_price, color='red', linestyle='--', linewidth=2, 
                       label=f'Current: {self.last_price:.2f}')
        axes[1].set_ylabel('Price', fontsize=12)
        axes[1].set_title('Price Distribution Box Plot', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Price distribution plot saved to {save_path}")
        
        plt.show()
    
    def calculate_probability_ranges(self, target_prices):
        """
        Calculate probability of reaching target prices
        
        Args:
            target_prices: List of target prices
            
        Returns:
            dict: Probabilities for each target
        """
        if self.simulations is None:
            raise ValueError("Run simulation first!")
        
        final_prices = self.simulations[-1]
        probabilities = {}
        
        logger.info("\n" + "="*60)
        logger.info("PROBABILITY ANALYSIS")
        logger.info("="*60)
        
        for target in target_prices:
            if target > self.last_price:
                prob = np.sum(final_prices >= target) / self.n_simulations * 100
                logger.info(f"Probability of reaching ${target:.2f} or higher: {prob:.2f}%")
            else:
                prob = np.sum(final_prices <= target) / self.n_simulations * 100
                logger.info(f"Probability of falling to ${target:.2f} or lower: {prob:.2f}%")
            
            probabilities[target] = prob
        
        logger.info("="*60 + "\n")
        
        return probabilities


def run_monte_carlo(price_data, n_simulations=None, days_ahead=None):
    """
    Main function to run Monte Carlo simulation
    
    Args:
        price_data: Historical price series
        n_simulations: Number of simulations
        days_ahead: Forecast horizon
        
    Returns:
        MonteCarloSimulator: Simulator object with results
    """
    simulator = MonteCarloSimulator(price_data, n_simulations, days_ahead)
    simulator.run_simulation(use_gbm=True)
    
    # Get statistics
    forecast_stats = simulator.get_forecast_statistics()
    
    # Plot results
    simulator.plot_simulation_paths(save_path='plots/monte_carlo_paths.png')
    simulator.plot_price_distribution(save_path='plots/monte_carlo_distribution.png')
    
    return simulator, forecast_stats


if __name__ == "__main__":
    # Test Monte Carlo simulation
    from data_loader import load_data
    
    logger.info("Loading data...")
    df = load_data()
    
    if df is not None:
        # Use Close prices
        price_data = df['Close']
        
        # Run simulation
        simulator, stats = run_monte_carlo(price_data)
        
        # Calculate probabilities for target prices
        current_price = price_data.iloc[-1]
        targets = [
            current_price * 0.95,  # 5% down
            current_price * 0.98,  # 2% down
            current_price * 1.02,  # 2% up
            current_price * 1.05   # 5% up
        ]
        
        probabilities = simulator.calculate_probability_ranges(targets)
