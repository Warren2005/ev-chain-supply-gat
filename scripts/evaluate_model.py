"""
Model Evaluation Script

Evaluates the trained ST-GAT model on test set with comprehensive
metrics and 3D visualizations.

Usage:
    python scripts/evaluate_model.py --checkpoint checkpoints/production/best_model.pt

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models.st_gat import STGAT
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader
from utils.evaluator import STGATEvaluator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate ST-GAT Model')
    
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/production/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--results_dir', type=str,
                        default='results/evaluation',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run evaluation on')
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()
    
    print("="*70)
    print("ST-GAT MODEL EVALUATION")
    print("="*70)
    print()
    
    # Stock names (7 stocks, alphabetical order)
    stock_names = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    
    # Edge index for 7 stocks
    edge_index = torch.tensor([
        [0, 0, 0, 0, 0, 1, 1, 1, 4, 4, 4, 0],
        [6, 2, 3, 4, 1, 2, 3, 6, 6, 2, 3, 1]
    ], dtype=torch.long)
    
    # Load test dataset
    print("Loading test dataset...")
    data_dir = project_root / "data" / "processed"
    
    test_dataset = TemporalGraphDataset(
        data_path=str(data_dir / "test_features_filtered.parquet"),
        edge_index=edge_index,
        window_size=20
    )
    
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    print()
    
    # Create test loader
    _, _, test_loader = TemporalGraphDataLoader.create_train_val_test_loaders(
        test_dataset,  # Using test as train (won't be used)
        test_dataset,  # Using test as val (won't be used)
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    print(f"✓ Test loader: {len(test_loader)} batches")
    print()
    
    # Initialize model
    print("Initializing model...")
    model = STGAT(
        num_nodes=7,
        num_edges=12,
        input_features=15,
        gat_hidden_dim=128,
        gat_heads=8,
        gat_layers=2,
        lstm_hidden_dim=128,
        lstm_layers=2,
        temporal_window=20,
        device=args.device
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✓ Training val loss: {checkpoint['val_loss']:.6f}")
    print()
    
    # Create evaluator
    evaluator = STGATEvaluator(
        model=model,
        test_loader=test_loader,
        stock_names=stock_names,
        device=args.device
    )
    
    # Run evaluation
    print("="*70)
    print("RUNNING TEST SET EVALUATION")
    print("="*70)
    print()
    
    metrics = evaluator.evaluate()
    
    # Compute per-stock metrics
    per_stock_metrics = evaluator.compute_per_stock_metrics()
    
    print("\nPer-Stock Performance:")
    print(per_stock_metrics.to_string(index=False))
    print()
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    evaluator.save_results(results_dir, metrics, per_stock_metrics)
    
    # Generate 3D visualizations
    print("="*70)
    print("GENERATING 3D VISUALIZATIONS")
    print("="*70)
    print()
    
    print("Creating 3D embedding visualization (t-SNE)...")
    evaluator.visualize_embeddings_3d(results_dir, method='tsne')
    
    print("Creating 3D embedding visualization (PCA)...")
    evaluator.visualize_embeddings_3d(results_dir, method='pca')
    
    print("Creating 3D prediction vs actual plot...")
    evaluator.plot_prediction_vs_actual_3d(results_dir)
    
    print()
    print("="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")
    print(f"\nGenerated files:")
    print(f"  • test_metrics.csv")
    print(f"  • per_stock_metrics.csv")
    print(f"  • predictions_detailed.csv")
    print(f"  • embeddings_3d_tsne.html (interactive 3D)")
    print(f"  • embeddings_3d_pca.html (interactive 3D)")
    print(f"  • predictions_3d.html (interactive 3D)")
    print()


if __name__ == "__main__":
    main()