"""
Create All Advanced 3D Visualizations

Orchestrates creation of comprehensive 3D visualization suite:
1. Supply chain graph (interactive 3D)
2. Temporal evolution (predictions over time)
3. Embedding clusters (t-SNE and PCA)
4. Loss landscape (training trajectory)
5. Performance space (stock comparisons)

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
from datetime import datetime

from models.simplified_st_gat import SimplifiedSTGAT
from utils.temporal_dataset import TemporalGraphDataset
from utils.graph_dataloader import TemporalGraphDataLoader
from utils.advanced_visualizer import AdvancedVisualizer


def main():
    """Create all 3D visualizations"""
    print("="*70)
    print("ADVANCED 3D VISUALIZATION SUITE")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup
    stock_names = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    edge_index = torch.tensor([
        [0, 0, 0, 4, 4, 4, 1, 1, 1, 0, 0, 4, 5, 5, 5],
        [6, 2, 3, 6, 2, 3, 2, 3, 6, 4, 1, 1, 6, 2, 3]
    ], dtype=torch.long)
    
    # Create output directory
    viz_dir = Path("results/visualizations_3d")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {viz_dir}")
    print()
    
    # Load test dataset
    print("Step 1/6: Loading test dataset...")
    test_dataset = TemporalGraphDataset(
        data_path="data/processed/test_features_fixed.parquet",
        edge_index=edge_index,
        window_size=20
    )
    
    _, _, test_loader = TemporalGraphDataLoader.create_train_val_test_loaders(
        test_dataset, test_dataset, test_dataset,
        batch_size=32, num_workers=0
    )
    print(f"✓ Test dataset: {len(test_dataset)} samples")
    print()
    
    # Load model
    print("Step 2/6: Loading Simplified ST-GAT model...")
    model = SimplifiedSTGAT(
        num_nodes=7,
        num_edges=15,
        input_features=15,
        gat_hidden_dim=64,
        gat_heads=4,
        lstm_hidden_dim=64,
        temporal_window=20,
        output_dim=1,
        dropout=0.3,
        device='cpu'
    )
    
    checkpoint = torch.load(
        'checkpoints/final_retrain/best_model.pt',
        map_location='cpu',
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print()
    
    # Initialize visualizer
    print("Step 3/6: Initializing Advanced Visualizer...")
    visualizer = AdvancedVisualizer(
        model=model,
        test_loader=test_loader,
        stock_names=stock_names,
        edge_index=edge_index,
        device='cpu'
    )
    print("✓ Visualizer ready")
    print()
    
    # Extract embeddings once (used by multiple visualizations)
    print("Step 4/6: Extracting predictions and embeddings...")
    visualizer.extract_predictions_and_embeddings()
    print("✓ Embeddings extracted")
    print()
    
    # Create visualizations
    print("Step 5/6: Creating 3D visualizations...")
    print("-"*70)
    
    # Visualization 1: Supply Chain Graph
    print("\n[1/5] Creating 3D Supply Chain Graph...")
    visualizer.create_3d_supply_chain_graph(
        save_path=viz_dir / "supply_chain_graph_3d.html"
    )
    
    # Visualization 2: Temporal Evolution
    print("\n[2/5] Creating 3D Temporal Evolution...")
    visualizer.create_3d_temporal_evolution(
        save_path=viz_dir / "temporal_evolution_3d.html",
        num_samples=100
    )
    
    # Visualization 3: Embedding Clusters (t-SNE)
    print("\n[3/5] Creating 3D Embedding Clusters (t-SNE)...")
    visualizer.create_3d_embedding_clusters(
        save_path=viz_dir / "embeddings_tsne_3d.html",
        method='tsne'
    )
    
    # Visualization 4: Embedding Clusters (PCA)
    print("\n[4/5] Creating 3D Embedding Clusters (PCA)...")
    visualizer.create_3d_embedding_clusters(
        save_path=viz_dir / "embeddings_pca_3d.html",
        method='pca'
    )
    
    # Visualization 5: Loss Landscape
    print("\n[5/5] Creating 3D Loss Landscape...")
    with open('results/final_retrain/train_history.json', 'r') as f:
        train_history = json.load(f)
    
    visualizer.create_3d_loss_landscape(
        train_history=train_history,
        save_path=viz_dir / "loss_landscape_3d.html"
    )
    
    print()
    print("-"*70)
    print()
    
    # Visualization 6: Performance Space
    print("Step 6/6: Creating 3D Performance Space...")
    visualizer.create_performance_3d_scatter(
        save_path=viz_dir / "performance_space_3d.html"
    )
    print()
    
    # Summary
    print("="*70)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print()
    print("Generated files:")
    print(f"  1. {viz_dir / 'supply_chain_graph_3d.html'}")
    print(f"  2. {viz_dir / 'temporal_evolution_3d.html'}")
    print(f"  3. {viz_dir / 'embeddings_tsne_3d.html'}")
    print(f"  4. {viz_dir / 'embeddings_pca_3d.html'}")
    print(f"  5. {viz_dir / 'loss_landscape_3d.html'}")
    print(f"  6. {viz_dir / 'performance_space_3d.html'}")
    print()
    print("All visualizations are interactive HTML files.")
    print("Open them in a web browser to explore in 3D!")
    print()
    print("Next: Analyze visualizations to inform optimization strategy")
    print()


if __name__ == "__main__":
    main()