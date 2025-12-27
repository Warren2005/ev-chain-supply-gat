"""
Create edge index for 7 stocks (excluding RIVN)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import torch


def main():
    """Create edge index for 7-stock graph"""
    print("="*70)
    print("CREATING EDGE INDEX (7 STOCKS)")
    print("="*70)
    print()
    
    # Load relationships
    relationships_path = project_root / "data" / "raw" / "sec_filings" / "supply_chain_relationships.csv"
    df = pd.read_csv(relationships_path)
    
    # Stock ordering (7 stocks, alphabetical, excluding RIVN)
    stock_order = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    stock_to_idx = {stock: idx for idx, stock in enumerate(stock_order)}
    
    print("Stock to Node Index mapping:")
    for stock, idx in stock_to_idx.items():
        print(f"  {idx}: {stock}")
    print()
    
    # Convert relationships to edge indices (exclude RIVN)
    source_nodes = []
    target_nodes = []
    
    for _, row in df.iterrows():
        source = row['source_ticker']
        target = row['target_ticker']
        
        # Skip relationships involving RIVN
        if source == 'RIVN' or target == 'RIVN':
            continue
        
        # Only include if both stocks are in our 7-stock dataset
        if source in stock_to_idx and target in stock_to_idx:
            source_nodes.append(stock_to_idx[source])
            target_nodes.append(stock_to_idx[target])
    
    # Create edge_index tensor
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    print(f"Created edge_index with {edge_index.shape[1]} edges")
    print(f"Shape: {edge_index.shape}")
    print()
    
    # Analyze graph structure
    print("Graph Structure Analysis:")
    print(f"  Nodes: {len(stock_order)}")
    print(f"  Edges: {edge_index.shape[1]}")
    print()
    
    # Show degree
    print("Node Degrees:")
    for idx, stock in enumerate(stock_order):
        out_degree = (edge_index[0] == idx).sum().item()
        in_degree = (edge_index[1] == idx).sum().item()
        total = out_degree + in_degree
        print(f"  {stock} (node {idx}): {in_degree} suppliers, {out_degree} customers, total {total}")
    print()
    
    # Generate code for validation scripts
    print("="*70)
    print("COPY-PASTE CODE:")
    print("="*70)
    print()
    print("# Edge index tensor (7 nodes, alphabetical order, excluding RIVN)")
    print(f"edge_index = torch.tensor({edge_index.tolist()}, dtype=torch.long)")
    print()
    print(f"# Stock order: {', '.join([f'{i}:{s}' for i, s in enumerate(stock_order)])}")
    print()
    
    # Save
    output_dir = project_root / "data" / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(edge_index, output_dir / "edge_index_7stocks.pt")
    print(f"✓ Saved: {output_dir / 'edge_index_7stocks.pt'}")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)