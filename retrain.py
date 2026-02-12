"""
Model Retraining Module
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .train import Trainer
import logging
from copy import deepcopy

class Retrainer:
    def __init__(self, config, trade_config, device):
        self.config = config
        self.trade_config = trade_config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def retrain(self, model, new_data):
        """Retrain model with new data"""
        self.logger.info("Starting incremental retraining...")
        
        # Prepare data
        train_loader, val_loader = self._prepare_data(new_data)
        
        # Setup for fine-tuning
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['training']['learning_rate'] * 0.1,  # Lower learning rate
            weight_decay=self.config['training']['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3
        )
        
        # Fine-tune
        best_val_loss = float('inf')
        best_model_state = deepcopy(model.state_dict())
        
        for epoch in range(self.config['retraining']['fine_tune_epochs']):
            # Train
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss, metrics = self._validate(model, val_loader, criterion)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
            
            if (epoch + 1) % 5 == 0:
                self.logger.info(f"Retrain Epoch [{epoch+1}/{self.config['retraining']['fine_tune_epochs']}] "
                               f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                               f"Acc: {metrics['accuracy']:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return model, metrics
    
    def _train_epoch(self, model, train_loader, criterion, optimizer):
        """Train one epoch"""
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def _validate(self, model, val_loader, criterion):
        """Validate the model"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        
        return total_loss / len(val_loader), metrics
    
    def _prepare_data(self, data):
        """Prepare data loaders for retraining"""
        all_features = []
        all_labels = []
        
        for pair, pair_data in data.items():
            features = pair_data['features']
            labels = pair_data['labels']
            
            # Take most recent 30% of data for retraining
            n = len(features)
            features = features[-int(n * 0.3):]
            labels = labels[-int(n * 0.3):]
            
            all_features.append(features)
            all_labels.append(labels)
        
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Split data
        n = len(X_tensor)
        train_split = int(n * 0.8)
        
        train_dataset = TensorDataset(X_tensor[:train_split], y_tensor[:train_split])
        val_dataset = TensorDataset(X_tensor[train_split:], y_tensor[train_split:])
        
        batch_size = self.config['training']['batch_size']
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader