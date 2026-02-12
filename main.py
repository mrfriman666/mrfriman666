#!/usr/bin/env python3
"""
Binance Scalping Trading Bot with AI
Main entry point and menu system
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collector import DataCollector
from data_preprocessor import DataPreprocessor
from train import Trainer
from backtest import Backtester
from test_trade import LiveTrader
from optimizer import HyperparameterOptimizer
from utils import setup_logging, check_gpu_cpu, print_color

# Load environment variables
load_dotenv()

class ScalpingBot:
    def __init__(self):
        self.load_configs()
        self.setup_directories()
        self.setup_logging()
        self.device = self.setup_device()
        
    def load_configs(self):
        """Load all configuration files"""
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        with open('config_models.yaml', 'r') as f:
            self.model_config = yaml.safe_load(f)
        with open('config_trade.yaml', 'r') as f:
            self.trade_config = yaml.safe_load(f)
        with open('config_gpu_cpu.yaml', 'r') as f:
            self.gpu_config = yaml.safe_load(f)
            
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ['models', 'data', 'logs', 'data/historical', 'data/processed']
        for dir_name in dirs:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = f"logs/bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging(log_file)
        self.logger = logging.getLogger(__name__)
        
    def setup_device(self):
        """Setup GPU/CPU based on configuration"""
        if torch.cuda.is_available() and self.gpu_config['gpu']['enabled']:
            device = torch.device('cuda')
            # Limit GPU usage
            if self.gpu_config['gpu']['memory_limit'] > 0:
                torch.cuda.set_per_process_memory_fraction(
                    self.gpu_config['gpu']['memory_limit'] / 100
                )
        else:
            device = torch.device('cpu')
            # Limit CPU threads
            if self.gpu_config['cpu']['thread_limit'] > 0:
                torch.set_num_threads(self.gpu_config['cpu']['thread_limit'])
        
        self.logger.info(f"Using device: {device}")
        return device
    
    def show_menu(self):
        """Display main menu"""
        print_color("""
╔══════════════════════════════════════════════════════════╗
║     Binance AI Scalping Trading Bot - Main Menu         ║
╠══════════════════════════════════════════════════════════╣
║  1. 📊 Collect & Prepare Data                          ║
║  2. 🤖 Train New Model                                 ║
║  3. 📈 Run Backtest                                    ║
║  4. 💹 Start Live Trading                              ║
║  5. ⚙️  Auto-optimize Parameters                       ║
║  6. 🔄 Retrain Model                                   ║
║  7. 📋 View Logs                                       ║
║  8. ⚡ System Settings                                 ║
║  9. 🚪 Exit                                           ║
╚══════════════════════════════════════════════════════════╝
        """, 'cyan')
        
    def menu_collect_data(self):
        """Menu option 1: Collect and prepare data"""
        print_color("\n📊 Data Collection & Preparation", 'yellow')
        
        collector = DataCollector(self.config, self.gpu_config)
        preprocessor = DataPreprocessor(self.config)
        
        # Get pairs from config
        pairs = self.config['trading']['pairs']
        interval = self.config['data']['interval']
        lookback_days = self.config['data']['lookback_days']
        
        print_color(f"Collecting data for {len(pairs)} pairs: {pairs}", 'white')
        print_color(f"Interval: {interval}, Lookback: {lookback_days} days", 'white')
        
        # Collect data
        data = collector.collect_historical_data(pairs, interval, lookback_days)
        
        # Prepare features
        features = preprocessor.prepare_features(data)
        
        # Save processed data
        preprocessor.save_processed_data(features)
        
        print_color("✓ Data collection and preparation completed!", 'green')
        
    def menu_train_model(self):
        """Menu option 2: Train new model"""
        print_color("\n🤖 Training New Model", 'yellow')
        
        trainer = Trainer(self.model_config, self.device)
        
        # Load preprocessed data
        preprocessor = DataPreprocessor(self.config)
        data = preprocessor.load_processed_data()
        
        # Train model
        model, metrics = trainer.train(data)
        
        # Save model
        model_path = f"models/scalping_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
        trainer.save_model(model, model_path)
        
        print_color(f"✓ Model training completed! Saved to {model_path}", 'green')
        print_color(f"  Metrics: {metrics}", 'white')
        
    def menu_backtest(self):
        """Menu option 3: Run backtest"""
        print_color("\n📈 Running Backtest", 'yellow')
        
        backtester = Backtester(self.config, self.trade_config, self.device)
        
        # Load model
        model_path = self.select_model()
        if not model_path:
            return
            
        # Run backtest
        results = backtester.run(model_path)
        
        # Display results
        backtester.display_results(results)
        
        # Auto-optimize if results are poor
        if results['sharpe_ratio'] < self.trade_config['backtest']['min_sharpe_ratio']:
            print_color("\n⚠️  Poor backtest results. Starting auto-optimization...", 'yellow')
            self.menu_optimize()
            
    def menu_live_trading(self):
        """Menu option 4: Start live trading"""
        print_color("\n💹 Starting Live Trading", 'yellow')
        
        # Check API keys
        if not os.getenv('BINANCE_API_KEY') or not os.getenv('BINANCE_API_SECRET'):
            print_color("❌ Binance API keys not found in .env file!", 'red')
            return
            
        trader = LiveTrader(self.config, self.trade_config, self.device)
        
        # Load model
        model_path = self.select_model()
        if not model_path:
            return
            
        # Start trading
        trader.start(model_path)
        
    def menu_optimize(self):
        """Menu option 5: Auto-optimize parameters"""
        print_color("\n⚙️  Auto-optimizing Parameters", 'yellow')
        
        optimizer = HyperparameterOptimizer(self.config, self.model_config, self.device)
        
        # Load data
        preprocessor = DataPreprocessor(self.config)
        data = preprocessor.load_processed_data()
        
        # Run optimization
        best_params = optimizer.optimize(data)
        
        # Save optimized config
        optimizer.save_optimized_config(best_params)
        
        print_color("✓ Optimization completed!", 'green')
        print_color(f"  Best parameters: {best_params}", 'white')
        
    def menu_retrain(self):
        """Menu option 6: Retrain model"""
        print_color("\n🔄 Retraining Model", 'yellow')
        
        trainer = Trainer(self.model_config, self.device)
        
        # Load existing model
        model_path = self.select_model()
        if not model_path:
            return
            
        # Load new data
        preprocessor = DataPreprocessor(self.config)
        new_data = preprocessor.load_recent_data()
        
        # Retrain
        model, metrics = trainer.retrain(model_path, new_data)
        
        # Save updated model
        new_model_path = f"models/retrained_{Path(model_path).name}"
        trainer.save_model(model, new_model_path)
        
        print_color(f"✓ Model retraining completed! Saved to {new_model_path}", 'green')
        
    def menu_view_logs(self):
        """Menu option 7: View logs"""
        print_color("\n📋 Recent Logs", 'yellow')
        
        log_files = sorted(Path('logs').glob('*.log'), key=os.path.getmtime, reverse=True)
        
        if not log_files:
            print_color("No log files found.", 'red')
            return
            
        print_color("Recent log files:", 'white')
        for i, log_file in enumerate(log_files[:5], 1):
            size = os.path.getsize(log_file) / 1024  # KB
            modified = datetime.fromtimestamp(os.path.getmtime(log_file))
            print(f"  {i}. {log_file.name} - {size:.1f}KB - {modified.strftime('%Y-%m-%d %H:%M')}")
            
        choice = input("\nSelect log file number to view (or 0 to cancel): ")
        try:
            choice = int(choice)
            if 1 <= choice <= len(log_files):
                with open(log_files[choice-1], 'r') as f:
                    lines = f.readlines()[-50:]  # Last 50 lines
                    print_color(f"\n=== Last 50 lines of {log_files[choice-1].name} ===", 'cyan')
                    for line in lines:
                        print(line.strip())
        except:
            pass
            
    def menu_settings(self):
        """Menu option 8: System settings"""
        print_color("\n⚡ System Settings", 'yellow')
        
        while True:
            print_color("""
  1. 🔧 GPU Settings
  2. 💻 CPU Settings
  3. 📊 Trading Pairs
  4. ⏱️  Timeframes
  5. 💰 Risk Parameters
  6. 🔙 Back to Main Menu
            """, 'white')
            
            choice = input("Select setting to configure: ")
            
            if choice == '1':
                self.configure_gpu()
            elif choice == '2':
                self.configure_cpu()
            elif choice == '3':
                self.configure_pairs()
            elif choice == '4':
                self.configure_timeframes()
            elif choice == '5':
                self.configure_risk()
            elif choice == '6':
                break
                
    def select_model(self):
        """Select a model file"""
        model_files = list(Path('models').glob('*.pth'))
        
        if not model_files:
            print_color("❌ No trained models found in models/ directory!", 'red')
            return None
            
        print_color("\nAvailable models:", 'white')
        for i, model_file in enumerate(model_files, 1):
            modified = datetime.fromtimestamp(os.path.getmtime(model_file))
            size = os.path.getsize(model_file) / 1024 / 1024  # MB
            print(f"  {i}. {model_file.name} - {size:.1f}MB - {modified.strftime('%Y-%m-%d %H:%M')}")
            
        choice = input("\nSelect model number: ")
        try:
            choice = int(choice)
            if 1 <= choice <= len(model_files):
                return str(model_files[choice-1])
        except:
            pass
        
        return None
        
    def configure_gpu(self):
        """Configure GPU settings"""
        print_color("\n🔧 GPU Configuration", 'yellow')
        
        current_limit = self.gpu_config['gpu']['memory_limit']
        print(f"Current GPU memory limit: {current_limit}%")
        
        new_limit = input("Enter new GPU memory limit (0-75, 0 for no limit): ")
        try:
            new_limit = int(new_limit)
            if 0 <= new_limit <= 75:
                self.gpu_config['gpu']['memory_limit'] = new_limit
                with open('config_gpu_cpu.yaml', 'w') as f:
                    yaml.dump(self.gpu_config, f)
                print_color("✓ GPU configuration updated!", 'green')
        except:
            print_color("Invalid input.", 'red')
            
    def configure_cpu(self):
        """Configure CPU settings"""
        print_color("\n💻 CPU Configuration", 'yellow')
        
        import multiprocessing
        max_threads = multiprocessing.cpu_count()
        max_allowed = int(max_threads * 0.75)
        
        current_threads = self.gpu_config['cpu']['thread_limit']
        print(f"Current CPU thread limit: {current_threads} (Max available: {max_threads})")
        
        new_threads = input(f"Enter new CPU thread limit (1-{max_allowed}): ")
        try:
            new_threads = int(new_threads)
            if 1 <= new_threads <= max_allowed:
                self.gpu_config['cpu']['thread_limit'] = new_threads
                with open('config_gpu_cpu.yaml', 'w') as f:
                    yaml.dump(self.gpu_config, f)
                print_color("✓ CPU configuration updated!", 'green')
        except:
            print_color("Invalid input.", 'red')
            
    def configure_pairs(self):
        """Configure trading pairs"""
        print_color("\n📊 Trading Pairs Configuration", 'yellow')
        
        current_pairs = self.config['trading']['pairs']
        print(f"Current pairs: {current_pairs}")
        
        print("Available top pairs: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, ADAUSDT, XRPUSDT, DOTUSDT, DOGEUSDT, LINKUSDT, MATICUSDT")
        
        new_pairs = input("Enter new pairs (comma-separated, e.g., BTCUSDT,ETHUSDT,SOLUSDT): ")
        pairs = [p.strip().upper() for p in new_pairs.split(',') if p.strip()]
        
        if pairs:
            self.config['trading']['pairs'] = pairs
            with open('config.yaml', 'w') as f:
                yaml.dump(self.config, f)
            print_color("✓ Trading pairs updated!", 'green')
            
    def configure_timeframes(self):
        """Configure timeframes"""
        print_color("\n⏱️  Timeframe Configuration", 'yellow')
        
        current_interval = self.config['data']['interval']
        print(f"Current interval: {current_interval}")
        
        print("Available intervals: 1m, 3m, 5m, 15m, 30m, 1h")
        new_interval = input("Enter new interval: ")
        
        if new_interval in ['1m', '3m', '5m', '15m', '30m', '1h']:
            self.config['data']['interval'] = new_interval
            with open('config.yaml', 'w') as f:
                yaml.dump(self.config, f)
            print_color("✓ Timeframe updated!", 'green')
            
    def configure_risk(self):
        """Configure risk parameters"""
        print_color("\n💰 Risk Parameters Configuration", 'yellow')
        
        current_risk = self.trade_config['risk_management']
        print(f"Current risk per trade: {current_risk['risk_per_trade']}%")
        print(f"Current max drawdown: {current_risk['max_drawdown']}%")
        print(f"Current daily loss limit: {current_risk['daily_loss_limit']}%")
        
        new_risk = input("Enter new risk per trade % (default 0.5-2.0): ")
        new_drawdown = input("Enter new max drawdown % (default 5-10): ")
        
        try:
            if new_risk:
                current_risk['risk_per_trade'] = float(new_risk)
            if new_drawdown:
                current_risk['max_drawdown'] = float(new_drawdown)
                
            with open('config_trade.yaml', 'w') as f:
                yaml.dump(self.trade_config, f)
            print_color("✓ Risk parameters updated!", 'green')
        except:
            print_color("Invalid input.", 'red')
            
    def run(self):
        """Main bot loop"""
        print_color("""
╔══════════════════════════════════════════════════════════╗
║     🚀 Binance AI Scalping Trading Bot v1.0             ║
║         Advanced Multi-Currency Scalping System          ║
╚══════════════════════════════════════════════════════════╝
        """, 'cyan')
        
        print_color(f"Device: {self.device}", 'green')
        print_color(f"Trading pairs: {self.config['trading']['pairs']}", 'white')
        print_color(f"Timeframe: {self.config['data']['interval']}", 'white')
        print_color(f"Strategies: {self.config['trading']['strategies']}", 'white')
        
        while True:
            self.show_menu()
            choice = input("Select option (1-9): ")
            
            if choice == '1':
                self.menu_collect_data()
            elif choice == '2':
                self.menu_train_model()
            elif choice == '3':
                self.menu_backtest()
            elif choice == '4':
                self.menu_live_trading()
            elif choice == '5':
                self.menu_optimize()
            elif choice == '6':
                self.menu_retrain()
            elif choice == '7':
                self.menu_view_logs()
            elif choice == '8':
                self.menu_settings()
            elif choice == '9':
                print_color("\n👋 Goodbye!", 'yellow')
                sys.exit(0)
            else:
                print_color("❌ Invalid option. Please try again.", 'red')
                
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    bot = ScalpingBot()
    bot.run()