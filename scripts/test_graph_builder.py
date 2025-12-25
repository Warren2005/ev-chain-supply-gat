"""
Test graph builder

Validates graph construction pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.graph_builder import GraphBuilder


def main():
    """Test graph building"""
    print("="*70)
    print("GRAPH BUILDER - VALIDATION TEST")
    print("="*70)
    print()
    
    # Initialize builder
    builder = GraphBuilder(
        relationships_dir="data/raw/sec_filings",
        output_dir="data/graphs"
    )
    
    # Build graph
    G = builder.build_complete_graph(save=True)
    
    if G is not None and G.number_of_nodes() > 0:
        print("="*70)
        print("TEST RESULTS")
        print("="*70)
        print("✓ Graph construction successful!")
        print(f"✓ Graph saved to: data/graphs/supply_chain_graph.pkl")
        print()
        print("Next: Phase 2.2.2 - PyTorch Geometric Conversion")
        return True
    else:
        print("✗ Graph construction failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)