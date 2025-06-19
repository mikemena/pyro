import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockDataFetcher:
    def __init__(self, symbol='AAPL', period='2y'):
        """
        Initialize stock data fetcher

        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
            period (str): Time period ('1y', '2y', '5y', 'max')
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.processed_data = None

    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        print(f"Fetching data for {self.symbol}...")

        try:
            # Download stock data
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)

            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")

            print(f"Successfully fetched {len(self.data)} days of data")
            print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")

            return self.data

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def add_technical_indicators(self):
        """Add technical indicators as features"""
        if self.data is None:
            print("No data available. Please fetch data first.")
            return

        df = self.data.copy()

        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        # Price change indicators
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']

        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=10).std()

        self.processed_data = df
        print("Technical indicators added successfully!")
        return df

    def plot_stock_data(self, show_indicators=True):
        """Plot stock price and technical indicators"""
        if self.processed_data is None:
            print("No processed data available. Please add technical indicators first.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Price and moving averages
        axes[0].plot(self.processed_data.index, self.processed_data['Close'],
                    label='Close Price', linewidth=2)

        if show_indicators:
            axes[0].plot(self.processed_data.index, self.processed_data['MA_5'],
                        label='MA 5', alpha=0.7)
            axes[0].plot(self.processed_data.index, self.processed_data['MA_20'],
                        label='MA 20', alpha=0.7)
            axes[0].plot(self.processed_data.index, self.processed_data['MA_50'],
                        label='MA 50', alpha=0.7)

            # Bollinger Bands
            axes[0].fill_between(self.processed_data.index,
                               self.processed_data['BB_Upper'],
                               self.processed_data['BB_Lower'],
                               alpha=0.2, label='Bollinger Bands')

        axes[0].set_title(f'{self.symbol} Stock Price')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # RSI
        if show_indicators:
            axes[1].plot(self.processed_data.index, self.processed_data['RSI'])
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
            axes[1].set_title('RSI (Relative Strength Index)')
            axes[1].set_ylabel('RSI')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # Volume
        axes[2].bar(self.processed_data.index, self.processed_data['Volume'],
                   alpha=0.7, width=1)
        if show_indicators:
            axes[2].plot(self.processed_data.index, self.processed_data['Volume_MA'],
                        color='red', label='Volume MA')
            axes[2].legend()

        axes[2].set_title('Trading Volume')
        axes[2].set_ylabel('Volume')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_feature_columns(self):
        """Get list of feature columns for model training"""
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'BB_Position', 'Volume_Ratio',
            'Price_Change', 'High_Low_Ratio', 'Open_Close_Ratio',
            'Volatility'
        ]
        return feature_cols

    def save_data(self, filename=None):
        """Save processed data to CSV"""
        if self.processed_data is None:
            print("No processed data to save.")
            return

        if filename is None:
            filename = f"{self.symbol}_stock_data.csv"

        self.processed_data.to_csv(filename)
        print(f"Data saved to {filename}")

# Example usage
if __name__ == "__main__":
    # Create fetcher for Apple stock
    fetcher = StockDataFetcher(symbol='AAPL', period='2y')

    # Fetch and process data
    fetcher.fetch_data()
    fetcher.add_technical_indicators()

    # Plot the data
    fetcher.plot_stock_data()

    # Show basic statistics
    print("\nBasic Statistics:")
    print(fetcher.processed_data[['Close', 'Volume', 'RSI']].describe())

    # Save data
    fetcher.save_data()
