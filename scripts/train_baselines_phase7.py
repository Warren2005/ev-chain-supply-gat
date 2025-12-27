"""
Phase 7 - Step 1: Train Baseline Models

Trains all baseline models on same data as ST-GAT:
1. GARCH(1,1) - per stock
2. VAR(5) - vector autoregression
3. Simple LSTM - no graph structure
4. Persistence - naive baseline

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from datetime import datetime
import json
import time

from models.baseline_models import GARCHBaseline, VARBaseline, SimpleLSTM, PersistenceBaseline
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader


def extract_targets_from_dataset(dataset):
    """Extract all targets from dataset"""
    targets = []
    for _, _, target in dataset:
        targets.append(target.numpy())
    return np.array(targets)


def train_simple_lstm(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=5e-5,
    device='cpu'
):
    """Train Simple LSTM baseline"""
    print("\nTraining Simple LSTM...")
    
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25
    
    history = {'train_losses': [], 'val_losses': []}
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_losses = []
        
        for features, _, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(features).squeeze(-1)
            loss = criterion(predictions, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validate
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for features, _, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                predictions = model(features).squeeze(-1)
                loss = criterion(predictions, targets)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'checkpoints/baselines/simple_lstm_best.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('checkpoints/baselines/simple_lstm_best.pt', weights_only=True))
    
    print(f"✓ Simple LSTM trained: Best val loss = {best_val_loss:.6f}")
    
    return model, history


def main():
    """Main baseline training"""
    print("="*70)
    print("PHASE 7 - STEP 1: TRAIN BASELINE MODELS")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup
    stock_names = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    edge_index = torch.tensor([
        [0, 0, 0, 4, 4, 4, 1, 1, 1, 0, 0, 4, 5, 5, 5],
        [6, 2, 3, 6, 2, 3, 2, 3, 6, 4, 1, 1, 6, 2, 3]
    ], dtype=torch.long)
    
    # Create checkpoint directory
    Path("checkpoints/baselines").mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = TemporalGraphDataset(
        data_path="data/processed/train_features_fixed.parquet",
        edge_index=edge_index,
        window_size=20
    )
    
    val_dataset = TemporalGraphDataset(
        data_path="data/processed/val_features_fixed.parquet",
        edge_index=edge_index,
        window_size=20
    )
    
    test_dataset = TemporalGraphDataset(
        data_path="data/processed/test_features_fixed.parquet",
        edge_index=edge_index,
        window_size=20
    )
    
    print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print()
    
    # Extract targets
    print("Extracting targets...")
    train_targets = extract_targets_from_dataset(train_dataset)
    val_targets = extract_targets_from_dataset(val_dataset)
    test_targets = extract_targets_from_dataset(test_dataset)
    print(f"✓ Targets extracted")
    print()
    
    # 1. Train GARCH
    print("="*70)
    print("BASELINE 1: GARCH(1,1)")
    print("="*70)
    
    garch = GARCHBaseline(stock_names)
    start = time.time()
    garch.fit(train_targets)
    garch_time = time.time() - start
    print(f"✓ GARCH training time: {garch_time:.2f}s")
    print()
    
    # 2. Train VAR
    print("="*70)
    print("BASELINE 2: VAR(5)")
    print("="*70)
    
    var = VARBaseline(stock_names, lag_order=5)
    start = time.time()
    var.fit(train_targets)
    var_time = time.time() - start
    print(f"✓ VAR training time: {var_time:.2f}s")
    print()
    
    # 3. Train Simple LSTM
    print("="*70)
    print("BASELINE 3: Simple LSTM (No Graph)")
    print("="*70)
    
    train_loader, val_loader, test_loader = TemporalGraphDataLoader.create_train_val_test_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=32, num_workers=0
    )
    
    lstm = SimpleLSTM(
        num_stocks=7,
        input_features=15,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3
    )
    
    start = time.time()
    lstm, lstm_history = train_simple_lstm(
        lstm, train_loader, val_loader,
        num_epochs=100, lr=5e-5, device='cpu'
    )
    lstm_time = time.time() - start
    print(f"✓ Simple LSTM training time: {lstm_time:.2f}s")
    print()
    
    # 4. Persistence baseline (no training)
    print("="*70)
    print("BASELINE 4: Persistence")
    print("="*70)
    
    persistence = PersistenceBaseline(stock_names)
    persistence.fit(train_targets)
    print()
    
    # Save training summary
    results_dir = Path("results/phase7_baselines")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'models': {
            'GARCH': {'training_time_s': garch_time},
            'VAR': {'training_time_s': var_time, 'lag_order': 5},
            'SimpleLSTM': {
                'training_time_s': lstm_time,
                'best_val_loss': min(lstm_history['val_losses']),
                'epochs_trained': len(lstm_history['train_losses'])
            },
            'Persistence': {'training_time_s': 0.0}
        }
    }
    
    with open(results_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("="*70)
    print("✓ ALL BASELINES TRAINED!")
    print("="*70)
    print()
    print(f"Training times:")
    print(f"  GARCH:      {garch_time:.2f}s")
    print(f"  VAR:        {var_time:.2f}s")
    print(f"  SimpleLSTM: {lstm_time:.2f}s")
    print(f"  Persistence: 0.00s")
    print()
    print("Next: Run evaluation script to compare all models")
    print("  python scripts/evaluate_baselines_phase7.py")
    print()


if __name__ == "__main__":
    main()