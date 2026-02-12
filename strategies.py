"""
Scalping Strategies Module
Three popular scalping strategies: Order Flow, Volume Profile, Market Microstructure
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging

class OrderFlowStrategy:
    """Order Flow Imbalance Strategy"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_signals(self, df, model_predictions=None):
        """Generate trading signals based on order flow"""
        
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['confidence'] = 0
        
        # Order flow imbalance
        df['bid_ask_ratio'] = df['taker_buy_base'] / df['volume']
        df['flow_imbalance'] = (df['taker_buy_base'] - df['taker_buy_base'].shift(20)) / df['volume']
        
        # Volume delta
        df['volume_delta'] = df['taker_buy_base'] - (df['volume'] - df['taker_buy_base'])
        df['cumulative_delta'] = df['volume_delta'].rolling(window=20).sum()
        
        # Absorption patterns
        df['absorption'] = (df['volume'] / df['volume'].rolling(window=50).mean()) * \
                          (df['spread'] / df['spread'].rolling(window=50).mean())
        
        # Generate signals
        # Strong buying pressure
        long_condition = (
            (df['flow_imbalance'] > df['flow_imbalance'].rolling(50).mean() + df['flow_imbalance'].rolling(50).std()) &
            (df['cumulative_delta'] > 0) &
            (df['absorption'] > 1.5)
        )
        
        # Strong selling pressure
        short_condition = (
            (df['flow_imbalance'] < df['flow_imbalance'].rolling(50).mean() - df['flow_imbalance'].rolling(50).std()) &
            (df['cumulative_delta'] < 0) &
            (df['absorption'] > 1.5)
        )
        
        signals.loc[long_condition, 'signal'] = 2  # Long
        signals.loc[short_condition, 'signal'] = 0  # Short
        
        # Calculate confidence
        signals.loc[long_condition, 'confidence'] = np.abs(df['flow_imbalance']).clip(0, 1)
        signals.loc[short_condition, 'confidence'] = np.abs(df['flow_imbalance']).clip(0, 1)
        
        # Combine with model predictions if available
        if model_predictions is not None:
            signals['signal'] = self._combine_signals(signals['signal'], model_predictions)
            
        return signals
    
    def _combine_signals(self, flow_signals, model_signals):
        """Combine order flow signals with model predictions"""
        # Weighted average: 60% order flow, 40% model
        combined = (flow_signals * 0.6 + model_signals * 0.4)
        return combined.round()

class VolumeProfileStrategy:
    """Volume Profile Analysis Strategy"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_signals(self, df, model_predictions=None):
        """Generate signals based on volume profile"""
        
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['confidence'] = 0
        
        # Calculate volume profile
        df['volume_profile'] = self._calculate_volume_profile(df)
        df['value_area'] = self._calculate_value_area(df)
        
        # POC (Point of Control) analysis
        df['poc'] = df['volume_profile'].rolling(window=50).apply(lambda x: x.idxmax() if not x.empty else np.nan)
        df['distance_from_poc'] = (df['close'] - df['poc']) / df['poc']
        
        # HVN (High Volume Nodes) and LVN (Low Volume Nodes)
        df['hvn'] = df['volume_profile'] > df['volume_profile'].rolling(50).mean() + df['volume_profile'].rolling(50).std()
        df['lvn'] = df['volume_profile'] < df['volume_profile'].rolling(50).mean() - df['volume_profile'].rolling(50).std()
        
        # Generate signals
        # Long when price is below value area and showing strength
        long_condition = (
            (df['close'] < df['value_area']) &
            (df['distance_from_poc'] < -0.001) &
            (df['lvn']) &
            (df['volume'] > df['volume'].rolling(20).mean())
        )
        
        # Short when price is above value area and showing weakness
        short_condition = (
            (df['close'] > df['value_area']) &
            (df['distance_from_poc'] > 0.001) &
            (df['lvn']) &
            (df['volume'] > df['volume'].rolling(20).mean())
        )
        
        signals.loc[long_condition, 'signal'] = 2
        signals.loc[short_condition, 'signal'] = 0
        
        # Calculate confidence
        signals.loc[long_condition, 'confidence'] = np.abs(df['distance_from_poc']).clip(0, 1) * 2
        signals.loc[short_condition, 'confidence'] = np.abs(df['distance_from_poc']).clip(0, 1) * 2
        
        # Combine with model predictions
        if model_predictions is not None:
            signals['signal'] = self._combine_signals(signals['signal'], model_predictions)
            
        return signals
    
    def _calculate_volume_profile(self, df, bins=50):
        """Calculate volume profile for the day"""
        price_range = np.linspace(df['low'].min(), df['high'].max(), bins)
        volume_profile = np.zeros(bins)
        
        for i in range(len(df)):
            price_idx = np.digitize(df['close'].iloc[i], price_range) - 1
            if 0 <= price_idx < bins:
                volume_profile[price_idx] += df['volume'].iloc[i]
                
        return pd.Series(volume_profile, index=price_range[:-1])
    
    def _calculate_value_area(self, df, volume_percentage=0.7):
        """Calculate value area (70% of volume)"""
        vp = self._calculate_volume_profile(df)
        total_volume = vp.sum()
        target_volume = total_volume * volume_percentage
        
        # Find POC
        poc_price = vp.idxmax()
        poc_idx = vp.index.get_loc(poc_price)
        
        # Expand outward from POC
        cumulative_volume = vp.iloc[poc_idx]
        lower_idx = poc_idx - 1
        upper_idx = poc_idx + 1
        
        while cumulative_volume < target_volume and (lower_idx >= 0 or upper_idx < len(vp)):
            if lower_idx >= 0 and (upper_idx >= len(vp) or vp.iloc[lower_idx] >= vp.iloc[upper_idx]):
                cumulative_volume += vp.iloc[lower_idx]
                lower_idx -= 1
            elif upper_idx < len(vp):
                cumulative_volume += vp.iloc[upper_idx]
                upper_idx += 1
            else:
                break
                
        return (vp.index[lower_idx + 1] + vp.index[upper_idx - 1]) / 2
    
    def _combine_signals(self, vp_signals, model_signals):
        """Combine volume profile signals with model predictions"""
        # Weighted average: 50% volume profile, 50% model
        combined = (vp_signals * 0.5 + model_signals * 0.5)
        return combined.round()

class MarketMicrostructureStrategy:
    """Market Microstructure Strategy"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_signals(self, df, model_predictions=None):
        """Generate signals based on market microstructure"""
        
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['confidence'] = 0
        
        # Calculate microstructure metrics
        df['effective_spread'] = 2 * np.sqrt(df['volatility'] * df['spread'])
        df['price_impact'] = df['returns'].rolling(window=5).mean() / df['volume']
        df['adverse_selection'] = df['spread'].rolling(window=20).corr(df['volatility'])
        
        # Order flow toxicity (VPIN approximation)
        df['vpin'] = self._calculate_vpin(df)
        
        # Market efficiency ratio
        df['efficiency_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Generate signals
        # Long when toxicity is low and efficiency is high
        long_condition = (
            (df['vpin'] < df['vpin'].rolling(50).quantile(0.3)) &
            (df['efficiency_ratio'] > 0.6) &
            (df['adverse_selection'] < 0)
        )
        
        # Short when toxicity is high and efficiency is low
        short_condition = (
            (df['vpin'] > df['vpin'].rolling(50).quantile(0.7)) &
            (df['efficiency_ratio'] < 0.4) &
            (df['adverse_selection'] > 0)
        )
        
        signals.loc[long_condition, 'signal'] = 2
        signals.loc[short_condition, 'signal'] = 0
        
        # Calculate confidence
        signals.loc[long_condition, 'confidence'] = (1 - df['vpin']).clip(0, 1)
        signals.loc[short_condition, 'confidence'] = df['vpin'].clip(0, 1)
        
        # Combine with model predictions
        if model_predictions is not None:
            signals['signal'] = self._combine_signals(signals['signal'], model_predictions)
            
        return signals
    
    def _calculate_vpin(self, df, window=50):
        """Calculate Volume-Synchronized Probability of Informed Trading"""
        volume_buckets = 50
        total_volume = df['volume'].rolling(window).sum()
        
        # Bucket volume
        df['cumulative_volume'] = df['volume'].cumsum()
        df['bucket'] = (df['cumulative_volume'] // (total_volume / volume_buckets)).astype(int)
        
        # Calculate imbalance per bucket
        vpin_values = []
        
        for bucket in df['bucket'].unique():
            bucket_data = df[df['bucket'] == bucket]
            if len(bucket_data) > 0:
                buy_volume = bucket_data['taker_buy_base'].sum()
                sell_volume = bucket_data['volume'].sum() - buy_volume
                imbalance = abs(buy_volume - sell_volume) / (buy_volume + sell_volume)
                vpin_values.append(imbalance)
                
        if len(vpin_values) > 0:
            return np.mean(vpin_values[-window:])
        else:
            return 0.5
    
    def _combine_signals(self, mm_signals, model_signals):
        """Combine microstructure signals with model predictions"""
        # Weighted average: 55% microstructure, 45% model
        combined = (mm_signals * 0.55 + model_signals * 0.45)
        return combined.round()

class SignalAggregator:
    """Aggregate signals from multiple strategies"""
    
    def __init__(self, config):
        self.config = config
        self.strategies = {
            'order_flow': OrderFlowStrategy(config),
            'volume_profile': VolumeProfileStrategy(config),
            'market_microstructure': MarketMicrostructureStrategy(config)
        }
        
    def get_combined_signal(self, df, model_predictions=None):
        """Get combined signal from all strategies"""
        
        all_signals = []
        confidences = []
        
        for strategy_name in self.config['trading']['strategies']:
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                signals = strategy.generate_signals(df, model_predictions)
                all_signals.append(signals['signal'])
                confidences.append(signals['confidence'])
                
        if not all_signals:
            return pd.DataFrame({'signal': 1, 'confidence': 0}, index=df.index)
            
        # Weighted average of signals
        weights = [0.4, 0.35, 0.25]  # Order flow, Volume profile, Microstructure
        weighted_signals = np.average(
            [s.values for s in all_signals],
            weights=weights[:len(all_signals)],
            axis=0
        )
        
        # Average confidence
        avg_confidence = np.mean([c.values for c in confidences], axis=0)
        
        # Round to nearest integer (0,1,2)
        final_signals = np.round(weighted_signals)
        
        result = pd.DataFrame({
            'signal': final_signals,
            'confidence': avg_confidence
        }, index=df.index)
        
        return result