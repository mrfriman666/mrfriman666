"""
Training Module for AI Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
from datetime import datetime

from .model import create_model

class Trainer:
    def __init__(self, model_config, device):
        self.model_config = model_config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def train(self, data):
        """Train the model"""
        self.logger.info("Starting training...")
        
        # Prepare data
        train_loader, val_loader = self._prepare_data(data)
        
        # Get input size
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[2]
        
        # Create model
        model = create_model(self.model_config, input_size).to(self.device)
        
        # Setup loss function
        criterion = self._get_loss_function()
        
        # Setup optimizer
        optimizer = self._get_optimizer(model)
        
        # Setup scheduler
        scheduler = self._get_scheduler(optimizer)
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.model_config['training']['epochs']):
            # Train
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_metrics = self._validate(model, val_loader, criterion)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch [{epoch+1}/{self.model_config['training']['epochs']}] "
                               f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                               f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Scheduler step
            if scheduler:
                scheduler.step(val_loss)
                
            # Early stopping
            if val_loss < best_val_loss - self.model_config['training']['early_stopping']['min_delta']:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if (self.model_config['training']['early_stopping']['enabled'] and 
                patience_counter >= self.model_config['training']['early_stopping']['patience']):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
            
        # Final validation
        _, metrics = self._validate(model, val_loader, criterion)
        
        return model, metrics
    
    def retrain(self, model_path, new_data):
        """Retrain an existing model with new data"""
        self.logger.info(f"Retraining model from {model_path}")
        
        # Load existing model
        checkpoint = torch.load(model_path, map_location=self.device)
        input_size = checkpoint['input_size']
        
        model = create_model(self.model_config, input_size).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Prepare new data
        train_loader, val_loader = self._prepare_data(new_data)
        
        # Setup for fine-tuning
        optimizer = self._get_optimizer(model)
        criterion = self._get_loss_function()
        
        # Fine-tune for fewer epochs
        for epoch in range(self.model_config['retraining']['fine_tune_epochs']):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            
            if (epoch + 1) % 5 == 0:
                val_loss, metrics = self._validate(model, val_loader, criterion)
                self.logger.info(f"Retrain Epoch [{epoch+1}/{self.model_config['retraining']['fine_tune_epochs']}] "
                               f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
        return model, metrics
    
    def _prepare_data(self, data):
        """Prepare data loaders"""
        all_features = []
        all_labels = []
        
        # Combine data from all pairs
        for pair, pair_data in data.items():
            features = pair_data['features']
            labels = pair_data['labels']
            
            # Balance classes
            features, labels = self._balance_classes(features, labels)
            
            all_features.append(features)
            all_labels.append(labels)
            
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Split data
        n = len(X_tensor)
        train_split = int(n * self.model_config['data_split']['train'])
        val_split = int(n * (self.model_config['data_split']['train'] + 
                           self.model_config['data_split']['val']))
        
        train_dataset = TensorDataset(X_tensor[:train_split], y_tensor[:train_split])
        val_dataset = TensorDataset(X_tensor[train_split:val_split], y_tensor[train_split:val_split])
        
        # Create data loaders
        batch_size = self.model_config['training']['batch_size']
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        return train_loader, val_loader
    
    def _balance_classes(self, features, labels):
        """Balance classes by oversampling minority classes"""
        unique, counts = np.unique(labels, return_counts=True)
        max_count = counts.max()
        
        balanced_features = []
        balanced_labels = []
        
        for cls in unique:
            cls_mask = labels == cls
            cls_features = features[cls_mask]
            cls_labels = labels[cls_mask]
            
            # Oversample
            n_repeats = max_count // len(cls_features)
            remainder = max_count % len(cls_features)
            
            for _ in range(n_repeats):
                balanced_features.append(cls_features)
                balanced_labels.append(cls_labels)
                
            if remainder > 0:
                balanced_features.append(cls_features[:remainder])
                balanced_labels.append(cls_labels[:remainder])
                
        return np.concatenate(balanced_features), np.concatenate(balanced_labels)
    
    def _train_epoch(self, model, train_loader, criterion, optimizer):
        """Train one epoch"""
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Data augmentation
            if self.model_config['training']['augmentation']['enabled']:
                noise = torch.randn_like(batch_X) * self.model_config['training']['augmentation']['noise_factor']
                batch_X = batch_X + noise
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
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
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        
        return total_loss / len(val_loader), metrics
    
    def _get_loss_function(self):
        """Get loss function based on configuration"""
        loss_type = self.model_config['training']['loss_function']
        
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'huber':
            return nn.SmoothL1Loss()
        elif loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss()
    
    def _get_optimizer(self, model):
        """Get optimizer based on configuration"""
        optimizer_type = self.model_config['training']['optimizer']
        lr = self.model_config['training']['learning_rate']
        weight_decay = self.model_config['training']['weight_decay']
        
        if optimizer_type == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _get_scheduler(self, optimizer):
        """Get learning rate scheduler"""
        if not self.model_config['training']['scheduler']['enabled']:
            return None
            
        scheduler_type = self.model_config['training']['scheduler']['type']
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.model_config['training']['epochs']
            )
        elif scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            return None
    
    def save_model(self, model, path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': self.model_config,
            'input_size': model.input_projection.in_features if hasattr(model, 'input_projection') else 50,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")