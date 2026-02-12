"""
Backtesting Module
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
from tqdm import tqdm
import torch
from .model import create_model
from .strategies import SignalAggregator
from .risk_manager import RiskManager

class Backtester:
    def __init__(self, config, trade_config, device):
        self.config = config
        self.trade_config = trade_config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def run(self, model_path):
        """Run backtest"""
        self.logger.info(f"Starting backtest with model: {model_path}")
        
        # Load model
        model, input_size = self._load_model(model_path)
        
        # Load data
        from .data_preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor(self.config)
        data = preprocessor.load_processed_data()
        
        # Initialize components
        signal_aggregator = SignalAggregator(self.config)
        risk_manager = RiskManager(self.trade_config)
        
        # Run backtest for each pair
        all_trades = []
        pair_results = {}
        
        for pair, pair_data in data.items():
            self.logger.info(f"Backtesting {pair}...")
            
            features = pair_data['features']
            df = pair_data['df']
            
            trades = self._backtest_pair(
                model, 
                features, 
                df, 
                signal_aggregator, 
                risk_manager,
                pair
            )
            
            all_trades.extend(trades)
            pair_results[pair] = self._calculate_metrics(trades)
            
        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(all_trades)
        
        results = {
            'overall': overall_metrics,
            'per_pair': pair_results,
            'trades': all_trades
        }
        
        return results
    
    def _backtest_pair(self, model, features, df, signal_aggregator, risk_manager, pair):
        """Run backtest for a single pair"""
        
        model.eval()
        trades = []
        capital = self.trade_config['backtest']['initial_capital']
        position = None
        
        # Process each sequence
        for i in tqdm(range(len(features)), desc=f"Backtesting {pair}"):
            # Get model prediction
            sequence = torch.FloatTensor(features[i:i+1]).to(self.device)
            
            with torch.no_grad():
                output = model(sequence)
                probabilities = torch.softmax(output, dim=1)
                confidence = probabilities.max().item()
                prediction = probabilities.argmax().item()
            
            # Get strategy signals
            current_idx = i + self.config['features']['sequence_length']
            if current_idx >= len(df):
                break
                
            current_df = df.iloc[:current_idx]
            strategy_signals = signal_aggregator.get_combined_signal(
                current_df, 
                model_predictions=prediction
            )
            
            signal = strategy_signals.iloc[-1]['signal']
            signal_confidence = strategy_signals.iloc[-1]['confidence']
            
            # Combine with model confidence
            final_confidence = (confidence * 0.6 + signal_confidence * 0.4)
            
            current_price = df.iloc[current_idx]['close']
            timestamp = df.index[current_idx]
            
            # Check for entry signals
            if position is None:
                if signal == 2 and final_confidence >= self.trade_config['signal_thresholds']['long_threshold']:
                    # Long entry
                    position = self._open_position(
                        'long', 
                        current_price, 
                        timestamp, 
                        capital,
                        df.iloc[current_idx]['atr']
                    )
                    
                elif signal == 0 and final_confidence >= self.trade_config['signal_thresholds']['short_threshold']:
                    # Short entry
                    position = self._open_position(
                        'short', 
                        current_price, 
                        timestamp, 
                        capital,
                        df.iloc[current_idx]['atr']
                    )
            
            # Check for exit signals
            elif position is not None:
                exit_signal = False
                exit_price = current_price
                
                # Stop loss
                if position['side'] == 'long':
                    if current_price <= position['stop_loss']:
                        exit_signal = True
                else:
                    if current_price >= position['stop_loss']:
                        exit_signal = True
                
                # Take profit
                if position['side'] == 'long':
                    if current_price >= position['take_profit']:
                        exit_signal = True
                else:
                    if current_price <= position['take_profit']:
                        exit_signal = True
                
                # Trailing stop
                if self.trade_config['risk_management']['trailing_stop']:
                    if position['side'] == 'long':
                        if current_price > position['entry_price']:
                            new_stop = current_price - (position['atr'] * 
                                self.trade_config['risk_management']['trailing_stop_atr'])
                            if new_stop > position['stop_loss']:
                                position['stop_loss'] = new_stop
                    else:
                        if current_price < position['entry_price']:
                            new_stop = current_price + (position['atr'] * 
                                self.trade_config['risk_management']['trailing_stop_atr'])
                            if new_stop < position['stop_loss']:
                                position['stop_loss'] = new_stop
                
                # Opposite signal exit
                if signal != position['side_int'] and signal != 1:
                    exit_signal = True
                
                if exit_signal:
                    trade = self._close_position(position, exit_price, timestamp)
                    trades.append(trade)
                    
                    # Update capital
                    capital += trade['pnl']
                    position = None
        
        return trades
    
    def _open_position(self, side, price, timestamp, capital, atr):
        """Open a new position"""
        risk_per_trade = self.trade_config['risk_management']['risk_per_trade'] / 100
        position_size = capital * risk_per_trade
        
        if side == 'long':
            stop_loss = price - (atr * self.trade_config['risk_management']['stop_loss_atr'])
            take_profit = price + (atr * self.trade_config['risk_management']['take_profit_atr'])
            side_int = 2
        else:
            stop_loss = price + (atr * self.trade_config['risk_management']['stop_loss_atr'])
            take_profit = price - (atr * self.trade_config['risk_management']['take_profit_atr'])
            side_int = 0
            
        return {
            'side': side,
            'side_int': side_int,
            'entry_price': price,
            'entry_time': timestamp,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'atr': atr
        }
    
    def _close_position(self, position, exit_price, exit_time):
        """Close a position and calculate PnL"""
        if position['side'] == 'long':
            pnl = (exit_price - position['entry_price']) / position['entry_price'] * position['position_size']
            pnl_percent = (exit_price - position['entry_price']) / position['entry_price'] * 100
        else:
            pnl = (position['entry_price'] - exit_price) / position['entry_price'] * position['position_size']
            pnl_percent = (position['entry_price'] - exit_price) / position['entry_price'] * 100
            
        # Apply commission
        commission = position['position_size'] * self.trade_config['backtest']['commission']
        pnl -= commission
        
        return {
            'pair': position.get('pair', 'UNKNOWN'),
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'position_size': position['position_size']
        }
    
    def _calculate_metrics(self, trades):
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
            
        df_trades = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = df_trades['pnl'].sum()
        
        # Returns for Sharpe ratio
        returns = df_trades['pnl_percent'].values / 100
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 else 0
        
        # Drawdown
        cumulative_pnl = df_trades['pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = (cumulative_pnl - running
        def _calculate_metrics(self, trades):
    """Calculate performance metrics"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0
        }
        
    df_trades = pd.DataFrame(trades)
    
    # Basic metrics
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['pnl'] > 0])
    losing_trades = len(df_trades[df_trades['pnl'] < 0])
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_pnl = df_trades['pnl'].sum()
    
    # Returns for Sharpe ratio
    returns = df_trades['pnl_percent'].values / 100
    sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 else 0
    
    # Drawdown
    cumulative_pnl = df_trades['pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = (cumulative_pnl - running_max) / (self.trade_config['backtest']['initial_capital'] / 100)
    max_drawdown = abs(drawdown.min())
    
    # Average win/loss
    avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
    
    # Profit factor
    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_pnl_percent': (total_pnl / self.trade_config['backtest']['initial_capital']) * 100
    }

def _load_model(self, model_path):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=self.device)
    
    # Get input size from checkpoint or use default
    input_size = checkpoint.get('input_size', 50)
    
    # Create model
    from .model import create_model
    model = create_model(checkpoint.get('model_config', self.model_config), input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(self.device)
    model.eval()
    
    return model, input_size

def display_results(self, results):
    """Display backtest results in a formatted way"""
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS")
    print("="*60)
    
    overall = results['overall']
    
    print(f"\n💰 Overall Performance:")
    print(f"   Total PnL: ${overall['total_pnl']:.2f} ({overall['total_pnl_percent']:.2f}%)")
    print(f"   Total Trades: {overall['total_trades']}")
    print(f"   Win Rate: {overall['win_rate']*100:.2f}%")
    print(f"   Sharpe Ratio: {overall['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {overall['max_drawdown']:.2f}%")
    print(f"   Profit Factor: {overall['profit_factor']:.2f}")
    print(f"   Avg Win: ${overall['avg_win']:.2f}")
    print(f"   Avg Loss: ${overall['avg_loss']:.2f}")
    
    print(f"\n📈 Per Pair Performance:")
    for pair, metrics in results['per_pair'].items():
        print(f"   {pair}: {metrics['total_trades']} trades, "
              f"PnL: ${metrics['total_pnl']:.2f}, "
              f"WR: {metrics['win_rate']*100:.1f}%")
    
    # Save results to file
    self._save_results(results)
    
def _save_results(self, results):
    """Save backtest results to file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"logs/backtest_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("BACKTEST RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Initial Capital: ${self.trade_config['backtest']['initial_capital']}\n\n")
        
        f.write("Overall Performance:\n")
        for key, value in results['overall'].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nPer Pair Performance:\n")
        for pair, metrics in results['per_pair'].items():
            f.write(f"\n  {pair}:\n")
            for key, value in metrics.items():
                f.write(f"    {key}: {value}\n")
    
    self.logger.info(f"Results saved to {filename}")