"""
Re-evaluate the retrained model on test set

Check if the fixed model produces varied predictions (not collapsed).
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
from models.st_gat import STGAT
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader
from utils.evaluator import STGATEvaluator


def main():
    """Re-evaluate retrained model"""
    print("="*70)
    print("RE-EVALUATING RETRAINED MODEL")
    print("="*70)
    print()
    
    # Stock names and edge index (15 edges with SQM)
    stock_names = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    edge_index = torch.tensor([
        [0, 0, 0, 4, 4, 4, 1, 1, 1, 0, 0, 4, 5, 5, 5],
        [6, 2, 3, 6, 2, 3, 2, 3, 6, 4, 1, 1, 6, 2, 3]
    ], dtype=torch.long)
    
    # Load test dataset (with FIXED data)
    print("Loading test dataset...")
    data_dir = Path("data/processed")
    
    test_dataset = TemporalGraphDataset(
        data_path=str(data_dir / "test_features_fixed.parquet"),
        edge_index=edge_index,
        window_size=20
    )
    
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    print()
    
    # Create test loader
    _, _, test_loader = TemporalGraphDataLoader.create_train_val_test_loaders(
        test_dataset, test_dataset, test_dataset,
        batch_size=32, num_workers=0
    )
    
    # Load retrained model
    print("Loading retrained model...")
    model = STGAT(
        num_nodes=7,
        num_edges=15,
        input_features=15,
        gat_hidden_dim=128,
        gat_heads=8,
        gat_layers=2,
        lstm_hidden_dim=128,
        lstm_layers=2,
        temporal_window=20,
        device='cpu'
    )
    
    checkpoint = torch.load(
        'checkpoints/retrained/best_model.pt',
        map_location='cpu',
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✓ Training val loss: {checkpoint['val_loss']:.6f}")
    print()
    
    # Evaluate
    evaluator = STGATEvaluator(
        model=model,
        test_loader=test_loader,
        stock_names=stock_names,
        device='cpu'
    )
    
    print("="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    print()
    
    metrics = evaluator.evaluate()
    
    # Per-stock metrics
    per_stock = evaluator.compute_per_stock_metrics()
    
    print("\nPer-Stock Performance:")
    print(per_stock.to_string(index=False))
    print()
    
    # Check for constant predictions
    print("="*70)
    print("PREDICTION VARIABILITY CHECK")
    print("="*70)
    print()
    
    for stock_idx, stock in enumerate(stock_names):
        preds = evaluator.predictions[:, stock_idx].numpy()
        
        pred_std = preds.std()
        pred_unique = len(np.unique(np.round(preds, 4)))
        
        print(f"{stock}:")
        print(f"  Pred Mean: {preds.mean():.6f}")
        print(f"  Pred Std:  {pred_std:.6f}")
        print(f"  Pred Range: [{preds.min():.6f}, {preds.max():.6f}]")
        print(f"  Unique values: {pred_unique}")
        
        if pred_std < 0.01:
            print(f"  ⚠️  WARNING: Near-constant predictions!")
        elif pred_std < 0.05:
            print(f"  ⚠️  CAUTION: Low variability")
        else:
            print(f"  ✓ Good variability")
        print()
    
    # Save results
    results_dir = Path("results/retrained_evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator.save_results(results_dir, metrics, per_stock)
    
    # Compare with old model
    print("="*70)
    print("COMPARISON WITH COLLAPSED MODEL")
    print("="*70)
    print()
    
    old_metrics = pd.read_csv('results/evaluation/test_metrics.csv')
    
    print("Metric Improvements:")
    print(f"  R² Score:    {old_metrics['r2'].values[0]:.4f} → {metrics['r2']:.4f}")
    print(f"  RMSE:        {old_metrics['rmse'].values[0]:.4f} → {metrics['rmse']:.4f}")
    print(f"  MAE:         {old_metrics['mae'].values[0]:.4f} → {metrics['mae']:.4f}")
    print(f"  Dir Acc:     {old_metrics['directional_accuracy'].values[0]*100:.1f}% → {metrics['directional_accuracy']*100:.1f}%")
    print(f"  Correlation: {old_metrics['correlation'].values[0]:.4f} → {metrics['correlation']:.4f}")
    print()
    
    if metrics['r2'] > 0:
        print("✓ R² is now POSITIVE! Model beats baseline!")
    else:
        print("✗ R² still negative - needs more improvement")
    
    if metrics['directional_accuracy'] > 0.5:
        print("✓ Directional accuracy beats coin flip!")
    else:
        print("✗ Directional accuracy still below 50%")
    
    print()
    print("="*70)
    print("✓ RE-EVALUATION COMPLETE")
    print("="*70)
    print()
    print(f"Results saved to: {results_dir}")
    print()


if __name__ == "__main__":
    main()