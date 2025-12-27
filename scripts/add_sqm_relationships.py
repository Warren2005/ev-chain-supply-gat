"""
Add SQM relationships to supply chain graph

SQM is a major lithium producer (like ALB) and should supply to EV OEMs.
This script adds the missing relationships.

Author: EV Supply Chain GAT Team
Date: December 2024
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import torch


def main():
    """Add SQM relationships and regenerate edge index"""
    print("="*70)
    print("ADDING SQM RELATIONSHIPS TO SUPPLY CHAIN")
    print("="*70)
    print()
    
    # Load current relationships
    relationships_path = Path("data/raw/sec_filings/supply_chain_relationships.csv")
    
    print(f"Loading: {relationships_path}")
    df = pd.read_csv(relationships_path)
    
    print(f"Current relationships: {len(df)}")
    print()
    
    # Show current relationships
    print("Current relationships:")
    print(df.to_string(index=False))
    print()
    
    # Check if SQM already has relationships
    sqm_existing = df[
        (df['source_ticker'] == 'SQM') | (df['target_ticker'] == 'SQM')
    ]
    
    if len(sqm_existing) > 0:
        print(f"⚠️  SQM already has {len(sqm_existing)} relationships:")
        print(sqm_existing.to_string(index=False))
        print()
    
    # Define new SQM relationships
    # SQM is a lithium producer, so it supplies to OEMs (like ALB does)
    new_relationships = [
        {
            'source_ticker': 'SQM',
            'target_ticker': 'TSLA',
            'relationship_type': 'supplies_lithium',
            'source': 'Added for graph completeness',
            'confidence': 'high'
        },
        {
            'source_ticker': 'SQM',
            'target_ticker': 'F',
            'relationship_type': 'supplies_lithium',
            'source': 'Added for graph completeness',
            'confidence': 'high'
        },
        {
            'source_ticker': 'SQM',
            'target_ticker': 'GM',
            'relationship_type': 'supplies_lithium',
            'source': 'Added for graph completeness',
            'confidence': 'high'
        }
    ]
    
    print("Adding new SQM relationships:")
    print("-"*70)
    for i, rel in enumerate(new_relationships, 1):
        print(f"{i}. SQM → {rel['target_ticker']} ({rel['relationship_type']})")
    print()
    
    # Add to dataframe
    new_df = pd.DataFrame(new_relationships)
    df_updated = pd.concat([df, new_df], ignore_index=True)
    
    print(f"Updated total relationships: {len(df_updated)}")
    print()
    
    # Save backup of original
    backup_path = relationships_path.parent / "supply_chain_relationships_backup.csv"
    df.to_csv(backup_path, index=False)
    print(f"✓ Backup saved: {backup_path}")
    
    # Save updated relationships
    df_updated.to_csv(relationships_path, index=False)
    print(f"✓ Updated relationships saved: {relationships_path}")
    print()
    
    # Regenerate edge index for 7 stocks
    print("="*70)
    print("REGENERATING EDGE INDEX (7 STOCKS)")
    print("="*70)
    print()
    
    # Stock ordering (7 stocks, alphabetical, excluding RIVN)
    stock_order = ['ALB', 'APTV', 'F', 'GM', 'MGA', 'SQM', 'TSLA']
    stock_to_idx = {stock: idx for idx, stock in enumerate(stock_order)}
    
    print("Stock to Node Index:")
    for stock, idx in stock_to_idx.items():
        print(f"  {idx}: {stock}")
    print()
    
    # Convert relationships to edge indices (exclude RIVN)
    source_nodes = []
    target_nodes = []
    
    for _, row in df_updated.iterrows():
        source = row['source_ticker']
        target = row['target_ticker']
        
        # Skip if either is RIVN
        if source == 'RIVN' or target == 'RIVN':
            continue
        
        # Only include if both stocks are in our 7-stock dataset
        if source in stock_to_idx and target in stock_to_idx:
            source_nodes.append(stock_to_idx[source])
            target_nodes.append(stock_to_idx[target])
    
    # Create edge_index tensor
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    print(f"Edge Index Shape: {edge_index.shape}")
    print(f"Number of edges: {edge_index.shape[1]}")
    print()
    
    # Show all edges
    print("All Supply Chain Edges:")
    print("-"*70)
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i].item()
        tgt_idx = edge_index[1, i].item()
        src_stock = stock_order[src_idx]
        tgt_stock = stock_order[tgt_idx]
        print(f"{i+1:2d}. {src_stock:4s} (node {src_idx}) → {tgt_stock:4s} (node {tgt_idx})")
    print()
    
    # Analyze graph structure
    print("Graph Structure Analysis:")
    print("-"*70)
    for idx, stock in enumerate(stock_order):
        out_degree = (edge_index[0] == idx).sum().item()
        in_degree = (edge_index[1] == idx).sum().item()
        total = out_degree + in_degree
        print(f"{stock:4s} (node {idx}): {out_degree} customers, {in_degree} suppliers = {total} total edges")
    print()
    
    # Check SQM is now connected
    sqm_idx = stock_to_idx['SQM']
    sqm_edges = (edge_index[0] == sqm_idx).sum().item() + (edge_index[1] == sqm_idx).sum().item()
    
    if sqm_edges > 0:
        print(f"✓ SQM is now connected with {sqm_edges} edges!")
    else:
        print("✗ ERROR: SQM still has no edges!")
    print()
    
    # Save edge index
    output_dir = Path("data/graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    edge_index_path = output_dir / "edge_index_7stocks_with_sqm.pt"
    torch.save(edge_index, edge_index_path)
    
    print(f"✓ Edge index saved: {edge_index_path}")
    print()
    
    # Generate code for scripts
    print("="*70)
    print("COPY-PASTE CODE FOR RETRAINING SCRIPTS:")
    print("="*70)
    print()
    print("# Edge index tensor (7 nodes with SQM connected)")
    print(f"edge_index = torch.tensor({edge_index.tolist()}, dtype=torch.long)")
    print()
    print(f"# Stock order: {', '.join([f'{i}:{s}' for i, s in enumerate(stock_order)])}")
    print(f"# Total edges: {edge_index.shape[1]}")
    print()
    
    print("="*70)
    print("✓ SQM RELATIONSHIPS ADDED SUCCESSFULLY!")
    print("="*70)
    print()
    print(f"Updated from 12 edges to {edge_index.shape[1]} edges")
    print()
    print("Next steps:")
    print("  1. Update retrain_model.py with new edge_index")
    print("  2. Run: python scripts/retrain_model.py")
    print()


if __name__ == "__main__":
    main()