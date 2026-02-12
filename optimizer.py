"""
Hyperparameter Optimization Module
"""

import optuna
import numpy as np
import torch
from .train import Trainer
from .backtest import Backtester
import logging
import yaml
from datetime import datetime

class HyperparameterOptimizer:
    def __init__(self, config, model_config, device):
        self.config = config
        self.model_config = model_config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def optimize(self, data):
        """Run hyperparameter optimization"""
        self.logger.info("Starting hyperparameter optimization...")
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
                'num_layers': trial.suggest_int('num_layers', 2, 4),
                'dropout': trial.suggest_float('dropout', 0.1, 0.4),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            }
            
            # Update model config
            self.model_config['model']['hidden_size'] = params['hidden_size']
            self.model_config['model']['num_layers'] = params['num_layers']
            self.model_config['model']['dropout'] = params['dropout']
            self.model_config['training']['learning_rate'] = params['learning_rate']
            self.model_config['training']['batch_size'] = params['batch_size']
            self.model_config['training']['weight_decay'] = params['weight_decay']
            
            try:
                # Train model
                trainer = Trainer(self.model_config, self.device)
                model, train_metrics = trainer.train(data)
                
                # Run quick backtest
                backtester = Backtester(self.config, self.trade_config, self.device)
                
                # Save temp model
                temp_path = f"models/temp_model_{trial.number}.pth"
                trainer.save_model(model, temp_path)
                
                # Run backtest
                results = backtester.run(temp_path)
                
                # Objective: maximize Sharpe ratio
                score = results['overall']['sharpe_ratio']
                
                # Penalize models with low win rate
                if results['overall']['win_rate'] < 0.5:
                    score *= 0.5
                    
                return score
                
            except Exception as e:
                self.logger.error(f"Trial {trial.number} failed: {e}")
                return -1000
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Run optimization
        study.optimize(
            objective, 
            n_trials=self.model_config['optimization']['n_trials'],
            n_jobs=self.model_config['optimization']['n_jobs']
        )
        
        self.logger.info(f"Best trial: {study.best_trial.number}")
        self.logger.info(f"Best value: {study.best_value}")
        self.logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def save_optimized_config(self, best_params):
        """Save optimized parameters to config file"""
        # Update model config
        self.model_config['model']['hidden_size'] = best_params['hidden_size']
        self.model_config['model']['num_layers'] = best_params['num_layers']
        self.model_config['model']['dropout'] = best_params['dropout']
        self.model_config['training']['learning_rate'] = best_params['learning_rate']
        self.model_config['training']['batch_size'] = best_params['batch_size']
        self.model_config['training']['weight_decay'] = best_params['weight_decay']
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"configs/config_models_optimized_{timestamp}.yaml"
        
        with open(filename, 'w') as f:
            yaml.dump(self.model_config, f, default_flow_style=False)
            
        self.logger.info(f"Optimized config saved to {filename}")