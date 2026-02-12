"""
Data Preprocessor Module for Feature Engineering
"""

import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import joblib
import logging
from pathlib import Path

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        
    def prepare_features(self, data_dict):
        """Prepare features for all pairs"""
        self.logger.info("Preparing features for all pairs")
        
        processed_data = {}
        
        for pair, data in data_dict.items():
            df = data.copy()
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Add order flow features
            df = self._add_order_flow_features(df)
            
            # Add market microstructure features
            df = self._add_microstructure_features(df)
            
            # Add price action features
            df = self._add_price_action_features(df)
            
            # Create sequences
            sequences = self._create_sequences(df)
            
            # Create labels
            labels = self._create_labels(df)
            
            processed_data[pair] = {
                'features': sequences,
                'labels': labels,
                'df': df
            }
            
        return processed_data
    
    def _add_technical_indicators(self, df):
        """Add technical indicators"""
        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']
        
        # ATR
        df['atr'] = AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=14
        ).average_true_range()
        
        # Moving Averages
        df['ema_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema_21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # VWAP
        df['vwap'] = (df['quote_volume'] / df['volume']).rolling(window=20).mean()
        
        return df
    
    def _add_order_flow_features(self, df):
        """Add order flow features"""
        # Bid-Ask imbalance (simulated from volume)
        df['bid_ask_imbalance'] = (df['taker_buy_base'] - (df['volume'] - df['taker_buy_base'])) / df['volume']
        
        # Tick volume intensity
        df['tick_intensity'] = df['volume'] / df['trades']
        
        # Cumulative delta
        df['cumulative_delta'] = (df['taker_buy_base'] - df['taker_buy_base'].shift(1)).cumsum()
        
        # Trade velocity
        df['trade_velocity'] = df['volume'] / df['volume'].rolling(window=5).mean()
        
        return df
    
    def _add_microstructure_features(self, df):
        """Add market microstructure features"""
        # Spread estimation (using high-low as proxy)
        df['spread'] = (df['high'] - df['low']) / df['close']
        
        # Liquidity score (volume relative to range)
        df['liquidity_score'] = df['volume'] / (df['high'] - df['low'])
        
        # Volatility regime
        df['volatility_regime'] = df['atr'] / df['atr'].rolling(window=50).mean()
        
        # Microstructure noise
        df['microstructure_noise'] = abs(df['close'] - df['vwap']) / df['close']
        
        return df
    
    def _add_price_action_features(self, df):
        """Add price action features"""
        # Candlestick patterns
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['body_ratio'] = df['body'] / (df['high'] - df['low'])
        
        # Price position within range
        df['position_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        return df
    
    def _create_sequences(self, df, sequence_length=None):
        """Create sequences for LSTM/Transformer"""
        if sequence_length is None:
            sequence_length = self.config['features']['sequence_length']
            
        # Select features
        feature_columns = []
        
        # Price features
        for feat in self.config['features']['price_features']:
            if feat in df.columns:
                feature_columns.append(feat)
                
        # Technical indicators
        for feat in self.config['features']['technical_indicators']:
            if feat in df.columns:
                feature_columns.append(feat)
                
        # Order flow features
        for feat in self.config['features']['order_flow_features']:
            if feat in df.columns:
                feature_columns.append(feat)
                
        # Microstructure features
        for feat in self.config['features']['microstructure_features']:
            if feat in df.columns:
                feature_columns.append(feat)
                
        # Get feature matrix
        feature_matrix = df[feature_columns].values
        
        # Normalize
        scaler = RobustScaler()
        normalized_features = scaler.fit_transform(feature_matrix)
        
        # Store scaler
        self.scalers['features'] = scaler
        
        # Create sequences
        sequences = []
        for i in range(len(normalized_features) - sequence_length):
            sequences.append(normalized_features[i:i + sequence_length])
            
        return np.array(sequences)
    
    def _create_labels(self, df, horizon=None):
        """Create labels for training"""
        if horizon is None:
            horizon = self.config['features']['target_horizon']
            
        # Future returns
        future_returns = df['close'].pct_change(horizon).shift(-horizon)
        
        # Create 3 classes: short (0), neutral (1), long (2)
        labels = pd.cut(
            future_returns,
            bins=[-np.inf, -0.0005, 0.0005, np.inf],
            labels=[0, 1, 2]
        )
        
        return labels.dropna().astype(int)
    
    def save_processed_data(self, processed_data):
        """Save processed data"""
        save_path = Path('data/processed')
        save_path.mkdir(exist_ok=True)
        
        # Save features and labels
        for pair, data in processed_data.items():
            np.save(f'data/processed/{pair}_features.npy', data['features'])
            np.save(f'data/processed/{pair}_labels.npy', data['labels'])
            
        # Save scalers
        joblib.dump(self.scalers, 'data/processed/scalers.pkl')
        
        self.logger.info(f"Saved processed data for {len(processed_data)} pairs")
        
    def load_processed_data(self):
        """Load processed data"""
        processed_data = {}
        
        for pair in self.config['trading']['pairs']:
            try:
                features = np.load(f'data/processed/{pair}_features.npy')
                labels = np.load(f'data/processed/{pair}_labels.npy')
                
                processed_data[pair] = {
                    'features': features,
                    'labels': labels
                }
            except FileNotFoundError:
                self.logger.warning(f"No processed data found for {pair}")
                
        return processed_data