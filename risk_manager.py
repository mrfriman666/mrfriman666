"""
Risk Management Module
"""

import numpy as np
from datetime import datetime, timedelta
import logging

class RiskManager:
    def __init__(self, trade_config):
        self.trade_config = trade_config
        self.logger = logging.getLogger(__name__)
        self.daily_trades = []
        self.daily_pnl = 0
        self.last_reset = datetime.now()
        
    def calculate_position_size(self, price, account_balance):
        """Calculate position size based on risk per trade"""
        risk_per_trade = self.trade_config['risk_management']['risk_per_trade'] / 100
        position_size = account_balance * risk_per_trade
        
        # Adjust for leverage
        if 'leverage' in self.trade_config:
            position_size *= self.trade_config['leverage']
        
        return position_size
    
    def check_daily_limits(self, pnl):
        """Check if daily limits are hit"""
        self._reset_daily_if_needed()
        
        self.daily_pnl += pnl
        self.daily_trades.append(pnl)
        
        # Check daily loss limit
        initial_capital = self.trade_config['backtest']['initial_capital']
        daily_loss_pct = abs(self.daily_pnl) / initial_capital * 100
        
        if daily_loss_pct >= self.trade_config['risk_management']['daily_loss_limit']:
            self.logger.warning(f"Daily loss limit reached: {daily_loss_pct:.2f}%")
            return False
            
        return True
    
    def check_max_drawdown(self, current_balance, peak_balance):
        """Check if max drawdown is exceeded"""
        drawdown = (peak_balance - current_balance) / peak_balance * 100
        
        if drawdown >= self.trade_config['risk_management']['max_drawdown']:
            self.logger.warning(f"Max drawdown exceeded: {drawdown:.2f}%")
            return False
            
        return True
    
    def calculate_stop_loss(self, entry_price, atr, side):
        """Calculate dynamic stop loss based on ATR"""
        if side == 'long':
            stop_loss = entry_price - (atr * self.trade_config['risk_management']['stop_loss_atr'])
        else:
            stop_loss = entry_price + (atr * self.trade_config['risk_management']['stop_loss_atr'])
            
        return stop_loss
    
    def calculate_take_profit(self, entry_price, atr, side):
        """Calculate take profit based on ATR"""
        if side == 'long':
            take_profit = entry_price + (atr * self.trade_config['risk_management']['take_profit_atr'])
        else:
            take_profit = entry_price - (atr * self.trade_config['risk_management']['take_profit_atr'])
            
        return take_profit
    
    def calculate_risk_reward_ratio(self, entry, stop, target):
        """Calculate risk/reward ratio"""
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        if risk == 0:
            return float('inf')
            
        return reward / risk
    
    def validate_trade(self, signal, risk_reward_ratio):
        """Validate if trade meets minimum risk/reward criteria"""
        min_rr = 1.5  # Minimum 1.5:1 risk/reward
        
        if risk_reward_ratio < min_rr:
            return False
            
        if signal['confidence'] < self.trade_config['retrain_conditions']['min_confidence']:
            return False
            
        return True
    
    def _reset_daily_if_needed(self):
        """Reset daily counters if new day"""
        now = datetime.now()
        if now.date() > self.last_reset.date():
            self.daily_trades = []
            self.daily_pnl = 0
            self.last_reset = now