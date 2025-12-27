"""
Create edge index tensor from supply chain relationships

Converts the CSV relationships into PyTorch tensor for ST-GAT
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import torch


def main():
    """Create edge index from relationships CSV"""
    print("="*70)
    print("CREATING EDGE INDEX FOR ST-GAT")
    print("="*70)
    print()
    
    # Load relationships
    relationships_path = project_root / "data" / "raw" / "sec_filings" / "supply_chain_relationships.csv"
    
    if not relationships_path.exists():
        print(f"✗ Relationships file not found: {relationships_path}")
        print("\nRun this first:")
        print("  python scripts/create_sample_relationships.py")
        return False
    
    df = pd.read_csv(relationships_path)
    print(f"Loaded {len(df)} relationships")
    print()
    
    # Stock ordering (alphabetical for consistency)
    stock_order = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'RIVN', 'SQM', 'TSLA']
    stock_to_idx = {stock: idx for idx, stock in enumerate(stock_order)}
    
    print("Stock to Node Index mapping:")
    for stock, idx in stock_to_idx.items():
        print(f"  {idx}: {stock}")
    print()
    
    # Convert relationships to edge indices
    source_nodes = []
    target_nodes = []
    
    for _, row in df.iterrows():
        source = row['source_ticker']
        target = row['target_ticker']
        
        # Only include if both stocks are in our dataset
        if source in stock_to_idx and target in stock_to_idx:
            source_nodes.append(stock_to_idx[source])
            target_nodes.append(stock_to_idx[target])
    
    # Create edge_index tensor
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    print(f"Created edge_index with {edge_index.shape[1]} edges")
    print(f"Shape: {edge_index.shape}")
    print()
    
    # Show edge index
    print("Edge Index (first 10 edges):")
    print("  [Source nodes]:", edge_index[0, :10].tolist())
    print("  [Target nodes]:", edge_index[1, :10].tolist())
    print()
    
    # Analyze graph structure
    print("Graph Structure Analysis:")
    print(f"  Nodes: {len(stock_order)}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Density: {edge_index.shape[1] / (len(stock_order) * (len(stock_order) - 1)) * 100:.1f}%")
    print()
    
    # Show degree (how many edges each node has)
    print("Node Degrees:")
    for idx, stock in enumerate(stock_order):
        out_degree = (edge_index[0] == idx).sum().item()
        in_degree = (edge_index[1] == idx).sum().item()
        total = out_degree + in_degree
        print(f"  {stock} (node {idx}): {in_degree} suppliers, {out_degree} customers, total {total}")
    print()
    
    # Save edge index
    output_dir = project_root / "data" / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    edge_index_path = output_dir / "edge_index.pt"
    torch.save(edge_index, edge_index_path)
    
    print(f"✓ Edge index saved: {edge_index_path}")
    print()
    
    # Also save as readable format
    edge_list_path = output_dir / "edge_list.csv"
    edge_df = pd.DataFrame({
        'source_idx': edge_index[0].tolist(),
        'target_idx': edge_index[1].tolist(),
        'source_ticker': [stock_order[i] for i in edge_index[0].tolist()],
        'target_ticker': [stock_order[i] for i in edge_index[1].tolist()]
    })
    edge_df.to_csv(edge_list_path, index=False)
    
    print(f"✓ Edge list saved: {edge_list_path}")
    print()
    
    # Generate Python code for easy copy-paste
    print("="*70)
    print("COPY-PASTE CODE FOR VALIDATION SCRIPTS:")
    print("="*70)
    print()
    print("# Edge index tensor (8 nodes, alphabetical order)")
    print(f"edge_index = torch.tensor({edge_index.tolist()}, dtype=torch.long)")
    print()
    print("# Stock order:")
    print(f"# {', '.join([f'{i}:{s}' for i, s in enumerate(stock_order)])}")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)