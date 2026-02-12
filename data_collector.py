"""
Data Collector Module for Binance Futures
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import logging
from tqdm import tqdm

load_dotenv()

class DataCollector:
    def __init__(self, config, gpu_config):
        self.config = config
        self.gpu_config = gpu_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Binance client
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if api_key and api_secret:
            self.client = Client(api_key, api_secret)
        else:
            self.client = Client()
            self.logger.warning("Using public API (no API keys)")
            
        self.interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '3m': Client.KLINE_INTERVAL_3MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR
        }
        
    def collect_historical_data(self, pairs, interval, lookback_days):
        """Collect historical data for multiple pairs"""
        self.logger.info(f"Collecting historical data for {len(pairs)} pairs")
        
        all_data = {}
        
        for pair in tqdm(pairs, desc="Collecting data"):
            try:
                data = self._fetch_historical_klines(pair, interval, lookback_days)
                all_data[pair] = data
                
                # Save raw data
                filename = f"data/historical/{pair}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
                data.to_csv(filename)
                self.logger.info(f"Saved {pair} data to {filename}")
                
            except Exception as e:
                self.logger.error(f"Error collecting data for {pair}: {e}")
                
        return all_data
    
    def _fetch_historical_klines(self, symbol, interval, lookback_days):
        """Fetch historical klines from Binance"""
        start_str = (datetime.now() - timedelta(days=lookback_days)).strftime("%d %b %Y %H:%M:%S")
        
        klines = self.client.get_historical_klines(
            symbol,
            self.interval_map[interval],
            start_str
        )
        
        # Convert to DataFrame
        data = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
        
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col])
            
        # Convert timestamps
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        
        return data
    
    def collect_order_book(self, symbol, limit=20):
        """Collect order book data"""
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            
            bids = pd.DataFrame(depth['bids'], columns=['price', 'quantity'], dtype=float)
            asks = pd.DataFrame(depth['asks'], columns=['price', 'quantity'], dtype=float)
            
            return {
                'bids': bids,
                'asks': asks,
                'timestamp': pd.Timestamp.now()
            }
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            return None
            
    def collect_recent_trades(self, symbol, limit=100):
        """Collect recent trades"""
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            
            df = pd.DataFrame(trades)
            df['price'] = df['price'].astype(float)
            df['qty'] = df['qty'].astype(float)
            df['quoteQty'] = df['quoteQty'].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            
            return df
        except BinanceAPIException as e:
            self.logger.error(f"Error fetching recent trades for {symbol}: {e}")
            return None