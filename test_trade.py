"""
Live Trading Module
"""

import os
import time
import threading
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import logging
from .model import create_model
from .strategies import SignalAggregator
from .risk_manager import RiskManager
from .retrain import Retrainer

load_dotenv()

class LiveTrader:
    def __init__(self, config, trade_config, device):
        self.config = config
        self.trade_config = trade_config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize Binance client
        self.client = Client(
            os.getenv('BINANCE_API_KEY'),
            os.getenv('BINANCE_API_SECRET')
        )
        
        # Use testnet if paper trading
        if self.trade_config['live_trading']['paper_trading']:
            self.client = Client(
                os.getenv('BINANCE_API_KEY'),
                os.getenv('BINANCE_API_SECRET'),
                testnet=True
            )
            self.logger.info("Using Binance Testnet")
        
        # Initialize components
        self.signal_aggregator = SignalAggregator(self.config)
        self.risk_manager = RiskManager(self.trade_config)
        self.retrainer = Retrainer(config, trade_config, device)
        
        # State
        self.positions = {}
        self.running = False
        self.model = None
        self.data_cache = {}
        
    def start(self, model_path):
        """Start live trading"""
        self.logger.info("Starting live trading...")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Start main loop
        self.running = True
        self._main_loop()
        
    def stop(self):
        """Stop live trading"""
        self.logger.info("Stopping live trading...")
        self.running = False
        
        # Close all positions
        self._close_all_positions()
        
    def _main_loop(self):
        """Main trading loop"""
        retrain_counter = 0
        failed_trades_counter = 0
        peak_balance = self._get_account_balance()
        
        while self.running:
            try:
                # Check each trading pair
                for pair in self.config['trading']['pairs']:
                    signal = self._analyze_pair(pair)
                    
                    if signal and signal['confidence'] >= self.trade_config['signal_thresholds']['long_threshold']:
                        self._execute_trade(pair, signal)
                
                # Check open positions
                self._manage_positions()
                
                # Check retraining conditions
                retrain_counter += 1
                if retrain_counter >= self.trade_config['retrain_conditions']['check_interval']:
                    if self._should_retrain(failed_trades_counter, peak_balance):
                        self._retrain_model()
                        retrain_counter = 0
                
                # Wait for next iteration
                time.sleep(self.trade_config['live_trading']['update_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(5)
    
    def _analyze_pair(self, pair):
        """Analyze a single pair for trading signals"""
        try:
            # Get latest data
            klines = self.client.get_klines(
                symbol=pair,
                interval=self.config['data']['interval'],
                limit=self.config['features']['sequence_length'] + 20
            )
            
            # Convert to DataFrame
            df = self._klines_to_dataframe(klines)
            
            # Add features
            df = self._add_features(df)
            
            # Get model prediction
            features = self._prepare_features(df)
            model_signal = self._get_model_prediction(features)
            
            # Get strategy signals
            strategy_signals = self.signal_aggregator.get_combined_signal(
                df, 
                model_predictions=model_signal['prediction']
            )
            
            # Combine signals
            confidence = (model_signal['confidence'] * 0.6 + 
                         strategy_signals.iloc[-1]['confidence'] * 0.4)
            final_signal = int(round(
                model_signal['prediction'] * 0.6 + 
                strategy_signals.iloc[-1]['signal'] * 0.4
            ))
            
            # Get current price
            current_price = float(klines[-1][4])
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                current_price,
                self._get_account_balance()
            )
            
            return {
                'signal': final_signal,
                'confidence': confidence,
                'price': current_price,
                'position_size': position_size,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {pair}: {e}")
            return None
    
    def _execute_trade(self, pair, signal):
        """Execute a trade based on signal"""
        if pair in self.positions:
            return  # Already have position
            
        if len(self.positions) >= self.trade_config['risk_management']['max_open_positions']:
            return  # Max positions reached
            
        try:
            side = 'BUY' if signal['signal'] == 2 else 'SELL'
            
            # Calculate quantity
            quantity = self._calculate_quantity(pair, signal['position_size'])
            
            if self.trade_config['live_trading']['paper_trading']:
                # Paper trading - simulate order
                order = self._simulate_order(pair, side, quantity, signal['price'])
            else:
                # Real trading
                order = self.client.futures_create_order(
                    symbol=pair,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
            
            # Record position
            self.positions[pair] = {
                'side': 'long' if side == 'BUY' else 'short',
                'entry_price': float(order['price']) if not self.trade_config['live_trading']['paper_trading'] else signal['price'],
                'entry_time': datetime.now(),
                'quantity': quantity,
                'stop_loss': self._calculate_stop_loss(pair, signal['price'], signal['signal']),
                'take_profit': self._calculate_take_profit(pair, signal['price'], signal['signal'])
            }
            
            self.logger.info(f"Opened {side} position for {pair}: {quantity} @ {signal['price']}")
            
        except BinanceAPIException as e:
            self.logger.error(f"Error executing trade for {pair}: {e}")
    
    def _manage_positions(self):
        """Manage open positions"""
        for pair, position in list(self.positions.items()):
            try:
                # Get current price
                ticker = self.client.get_symbol_ticker(symbol=pair)
                current_price = float(ticker['price'])
                
                # Check stop loss and take profit
                should_close = False
                close_reason = None
                
                if position['side'] == 'long':
                    if current_price <= position['stop_loss']:
                        should_close = True
                        close_reason = 'stop_loss'
                    elif current_price >= position['take_profit']:
                        should_close = True
                        close_reason = 'take_profit'
                else:
                    if current_price >= position['stop_loss']:
                        should_close = True
                        close_reason = 'stop_loss'
                    elif current_price <= position['take_profit']:
                        should_close = True
                        close_reason = 'take_profit'
                
                # Update trailing stop
                if self.trade_config['risk_management']['trailing_stop']:
                    self._update_trailing_stop(pair, position, current_price)
                
                if should_close:
                    self._close_position(pair, position, current_price, close_reason)
                    
            except Exception as e:
                self.logger.error(f"Error managing position for {pair}: {e}")
    
    def _close_position(self, pair, position, current_price, reason):
        """Close an open position"""
        try:
            side = 'SELL' if position['side'] == 'long' else 'BUY'
            
            if not self.trade_config['live_trading']['paper_trading']:
                order = self.client.futures_create_order(
                    symbol=pair,
                    side=side,
                    type='MARKET',
                    quantity=position['quantity']
                )
            
            # Calculate PnL
            if position['side'] == 'long':
                pnl = (current_price - position['entry_price']) / position['entry_price'] * 100
            else:
                pnl = (position['entry_price'] - current_price) / position['entry_price'] * 100
            
            self.logger.info(f"Closed {pair} position: {reason}, PnL: {pnl:.2f}%")
            
            # Remove position
            del self.positions[pair]
            
        except Exception as e:
            self.logger.error(f"Error closing position for {pair}: {e}")
    
    def _should_retrain(self, failed_trades_counter, peak_balance):
        """Check if model should be retrained"""
        current_balance = self._get_account_balance()
        
        # Check retrain conditions
        if failed_trades_counter >= self.trade_config['retrain_conditions']['after_failed_trades']:
            self.logger.info("Retraining due to failed trades threshold")
            return True
            
        drawdown = (peak_balance - current_balance) / peak_balance * 100
        if drawdown >= self.trade_config['retrain_conditions']['after_drawdown']:
            self.logger.info(f"Retraining due to drawdown: {drawdown:.2f}%")
            return True
            
        # Check time-based retraining
        last_retrain_time = getattr(self, '_last_retrain_time', None)
        if last_retrain_time:
            hours_since_retrain = (datetime.now() - last_retrain_time).total_seconds() / 3600
            retrain_schedule = self.config.get('retraining', {}).get('schedule', '24h')
            retrain_hours = int(retrain_schedule.replace('h', ''))
            
            if hours_since_retrain >= retrain_hours:
                self.logger.info("Retraining due to schedule")
                return True
                
        return False
    
    def _retrain_model(self):
        """Retrain the model with recent data"""
        self.logger.info("Starting model retraining...")
        
        try:
            # Collect recent data
            collector = DataCollector(self.config, {})
            recent_data = collector.collect_historical_data(
                self.config['trading']['pairs'],
                self.config['data']['interval'],
                7  # Last 7 days
            )
            
            # Preprocess data
            preprocessor = DataPreprocessor(self.config)
            processed_data = preprocessor.prepare_features(recent_data)
            
            # Retrain model
            self.model, metrics = self.retrainer.retrain(self.model, processed_data)
            
            self._last_retrain_time = datetime.now()
            self.logger.info(f"Model retrained successfully. Accuracy: {metrics.get('accuracy', 0):.4f}")
            
        except Exception as e:
            self.logger.error(f"Error retraining model: {e}")
    
    def _get_account_balance(self):
        """Get current account balance"""
        try:
            if self.trade_config['live_trading']['paper_trading']:
                return 10000  # Mock balance for paper trading
            else:
                account = self.client.futures_account()
                for asset in account['assets']:
                    if asset['asset'] == 'USDT':
                        return float(asset['walletBalance'])
                return 0
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return 0
    
    def _klines_to_dataframe(self, klines):
        """Convert Binance klines to DataFrame"""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _add_features(self, df):
        """Add technical indicators to DataFrame"""
        from ta.momentum import RSIIndicator
        from ta.trend import MACD, EMAIndicator
        from ta.volatility import BollingerBands, AverageTrueRange
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        
        # ATR
        df['atr'] = AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=14
        ).average_true_range()
        
        return df
    
    def _prepare_features(self, df):
        """Prepare features for model input"""
        # Select features (simplified for live trading)
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'rsi', 'macd', 'bb_high', 'bb_low', 'atr']
        
        # Get last sequence_length rows
        sequence = df[feature_columns].values[-self.config['features']['sequence_length']:]
        
        # Normalize
        mean = sequence.mean(axis=0)
        std = sequence.std(axis=0)
        std[std == 0] = 1
        normalized = (sequence - mean) / std
        
        return torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
    
    def _get_model_prediction(self, features):
        """Get model prediction"""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(features)
            probabilities = torch.softmax(output, dim=1)
            confidence = probabilities.max().item()
            prediction = probabilities.argmax().item()
            
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()
        }
    
    def _calculate_quantity(self, pair, position_size):
        """Calculate order quantity"""
        try:
            info = self.client.futures_exchange_info()
            for symbol_info in info['symbols']:
                if symbol_info['symbol'] == pair:
                    for filter in symbol_info['filters']:
                        if filter['filterType'] == 'LOT_SIZE':
                            step_size = float(filter['stepSize'])
                            quantity = position_size
                            quantity = round(quantity - (quantity % step_size), 8)
                            return quantity
            return position_size
        except:
            return position_size
    
    def _calculate_stop_loss(self, pair, price, signal):
        """Calculate stop loss price"""
        if signal == 2:  # Long
            return price * (1 - self.trade_config['risk_management']['stop_loss_atr'] / 100)
        else:  # Short
            return price * (1 + self.trade_config['risk_management']['stop_loss_atr'] / 100)
    
    def _calculate_take_profit(self, pair, price, signal):
        """Calculate take profit price"""
        if signal == 2:  # Long
            return price * (1 + self.trade_config['risk_management']['take_profit_atr'] / 100)
        else:  # Short
            return price * (1 - self.trade_config['risk_management']['take_profit_atr'] / 100)
    
    def _update_trailing_stop(self, pair, position, current_price):
        """Update trailing stop"""
        if position['side'] == 'long':
            if current_price > position['entry_price']:
                new_stop = current_price * (1 - self.trade_config['risk_management']['trailing_stop_atr'] / 100)
                if new_stop > position['stop_loss']:
                    position['stop_loss'] = new_stop
        else:
            if current_price < position['entry_price']:
                new_stop = current_price * (1 + self.trade_config['risk_management']['trailing_stop_atr'] / 100)
                if new_stop < position['stop_loss']:
                    position['stop_loss'] = new_stop
    
    def _simulate_order(self, pair, side, quantity, price):
        """Simulate order for paper trading"""
        return {
            'symbol': pair,
            'orderId': int(time.time() * 1000),
            'side': side,
            'price': str(price),
            'executedQty': str(quantity),
            'status': 'FILLED'
        }
    
    def _close_all_positions(self):
        """Close all open positions"""
        for pair in list(self.positions.keys()):
            try:
                ticker = self.client.get_symbol_ticker(symbol=pair)
                current_price = float(ticker['price'])
                self._close_position(pair, self.positions[pair], current_price, 'manual_shutdown')
            except Exception as e:
                self.logger.error(f"Error closing position for {pair}: {e}")
    
    def _load_model(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        input_size = checkpoint.get('input_size', 50)
        
        model = create_model(checkpoint.get('model_config', self.config), input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model