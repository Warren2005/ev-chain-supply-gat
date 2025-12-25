"""
Graph Builder for EV Supply Chain GAT Project

This module constructs the knowledge graph from supplier-customer relationships
and prepares it for the GAT model.

Features:
- NetworkX directed graph construction
- Node metadata (company info, tier)
- Edge attributes (confidence, relationship type)
- PyTorch Geometric conversion
- Graph visualization

Author: EV Supply Chain GAT Team
Date: December 2025
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import warnings

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle


class GraphBuilder:
    """
    Constructs and manages the supply chain knowledge graph.
    
    This class handles:
    - Loading relationships from CSV
    - Building NetworkX directed graph
    - Adding node/edge attributes
    - Graph validation and statistics
    - Saving/loading graphs
    
    Attributes:
        relationships_dir (Path): Directory with relationship CSVs
        output_dir (Path): Directory for graph outputs
        logger (logging.Logger): Logger instance
    """
    
    # Company tier mapping (supply chain hierarchy)
    COMPANY_TIERS = {
        # Tier 0: OEMs (Original Equipment Manufacturers)
        'TSLA': {'tier': 0, 'name': 'Tesla', 'category': 'OEM'},
        'F': {'tier': 0, 'name': 'Ford', 'category': 'OEM'},
        'GM': {'tier': 0, 'name': 'General Motors', 'category': 'OEM'},
        'RIVN': {'tier': 0, 'name': 'Rivian', 'category': 'OEM'},
        
        # Tier 1: Battery Manufacturers
        'PCRFY': {'tier': 1, 'name': 'Panasonic', 'category': 'Battery'},
        
        # Tier 2: Component Suppliers
        'MGA': {'tier': 2, 'name': 'Magna International', 'category': 'Components'},
        'APTV': {'tier': 2, 'name': 'Aptiv', 'category': 'Components'},
        
        # Tier 3: Raw Materials
        'ALB': {'tier': 3, 'name': 'Albemarle', 'category': 'Raw Materials'},
        'SQM': {'tier': 3, 'name': 'SQM', 'category': 'Raw Materials'},
        'LTHM': {'tier': 3, 'name': 'Livent', 'category': 'Raw Materials'},
        'LAC': {'tier': 3, 'name': 'Lithium Americas', 'category': 'Raw Materials'},
        'MP': {'tier': 3, 'name': 'MP Materials', 'category': 'Raw Materials'},
    }
    
    def __init__(
        self,
        relationships_dir: str = "data/raw/sec_filings",
        output_dir: str = "data/graphs"
    ):
        """
        Initialize the GraphBuilder.
        
        Args:
            relationships_dir: Directory with relationship CSV files
            output_dir: Directory to save graph outputs
        """
        self.relationships_dir = Path(relationships_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Graph will be stored here
        self.graph = None
        
        # Track statistics
        self.build_stats = {
            "nodes": 0,
            "edges": 0,
            "tiers": {},
            "isolated_nodes": []
        }
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configure logging for this class.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / "graph_builder.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_relationships(
        self,
        filename: str = "supply_chain_relationships.csv",
        min_confidence: float = 0.3
    ) -> pd.DataFrame:
        """
        Load relationships from CSV file.
        
        Args:
            filename: CSV filename
            min_confidence: Minimum confidence threshold (default: 0.3)
        
        Returns:
            DataFrame with filtered relationships
        """
        try:
            filepath = self.relationships_dir / filename
            
            if not filepath.exists():
                self.logger.error(f"Relationships file not found: {filepath}")
                self.logger.error("Please run: python scripts/create_sample_relationships.py")
                return pd.DataFrame()
            
            df = pd.read_csv(filepath)
            
            self.logger.info(f"Loaded {len(df)} relationships from {filename}")
            
            # Filter by confidence
            df_filtered = df[df['confidence'] >= min_confidence].copy()
            
            if len(df_filtered) < len(df):
                self.logger.info(
                    f"Filtered to {len(df_filtered)} relationships "
                    f"(confidence >= {min_confidence})"
                )
            
            return df_filtered
            
        except Exception as e:
            self.logger.error(f"Error loading relationships: {str(e)}")
            return pd.DataFrame()
    
    def build_graph(
        self,
        relationships_df: pd.DataFrame
    ) -> nx.DiGraph:
        """
        Build NetworkX directed graph from relationships.
        
        Args:
            relationships_df: DataFrame with supplier-customer relationships
        
        Returns:
            NetworkX DiGraph
        """
        self.logger.info("Building directed graph from relationships")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add edges from relationships
        for _, row in relationships_df.iterrows():
            source = row['source_ticker']
            target = row['target_ticker']
            
            # Add edge with attributes
            G.add_edge(
                source,
                target,
                confidence=row.get('confidence', 1.0),
                relationship_type=row.get('relationship_type', 'supplies_to'),
                num_mentions=row.get('num_mentions', 1)
            )
        
        self.logger.info(
            f"Created graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        
        self.graph = G
        return G
    
    def add_node_attributes(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Add node attributes (tier, name, category).
        
        Args:
            G: NetworkX graph
        
        Returns:
            Graph with node attributes added
        """
        self.logger.info("Adding node attributes")
        
        for node in G.nodes():
            if node in self.COMPANY_TIERS:
                # Add known attributes
                attrs = self.COMPANY_TIERS[node]
                nx.set_node_attributes(G, {node: attrs})
            else:
                # Unknown company - add default attributes
                self.logger.warning(f"Unknown company: {node}. Adding default attributes.")
                nx.set_node_attributes(G, {
                    node: {
                        'tier': -1,
                        'name': node,
                        'category': 'Unknown'
                    }
                })
        
        self.logger.info(f"Added attributes to {G.number_of_nodes()} nodes")
        
        return G
    
    def validate_graph(self, G: nx.DiGraph) -> Tuple[bool, List[str]]:
        """
        Validate graph structure and properties.
        
        Args:
            G: NetworkX graph
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check 1: Graph not empty
        if G.number_of_nodes() == 0:
            issues.append("Graph has no nodes")
            return False, issues
        
        # Check 2: Graph has edges
        if G.number_of_edges() == 0:
            issues.append("Graph has no edges")
        
        # Check 3: Check for isolated nodes
        isolated = list(nx.isolates(G))
        if isolated:
            issues.append(f"Graph has {len(isolated)} isolated nodes: {isolated}")
            self.build_stats["isolated_nodes"] = isolated
        
        # Check 4: Check for self-loops
        self_loops = list(nx.selfloop_edges(G))
        if self_loops:
            issues.append(f"Graph has {len(self_loops)} self-loops")
        
        # Check 5: Check if graph is weakly connected
        if not nx.is_weakly_connected(G):
            num_components = nx.number_weakly_connected_components(G)
            issues.append(
                f"Graph is not weakly connected ({num_components} components)"
            )
        
        # Check 6: Verify node attributes
        for node in G.nodes():
            if 'tier' not in G.nodes[node]:
                issues.append(f"Node {node} missing 'tier' attribute")
        
        is_valid = len([i for i in issues if not i.startswith("Graph has")]) == 0
        
        if issues:
            self.logger.warning(f"Validation issues: {issues}")
        else:
            self.logger.info("Graph validation passed")
        
        return is_valid, issues
    
    def compute_statistics(self, G: nx.DiGraph) -> Dict:
        """
        Compute graph statistics.
        
        Args:
            G: NetworkX graph
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        }
        
        # Count nodes by tier
        tier_counts = {}
        for node in G.nodes():
            tier = G.nodes[node].get('tier', -1)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        stats["nodes_by_tier"] = tier_counts
        
        # Degree statistics
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        
        stats["avg_in_degree"] = np.mean(list(in_degrees.values()))
        stats["avg_out_degree"] = np.mean(list(out_degrees.values()))
        stats["max_in_degree"] = max(in_degrees.values())
        stats["max_out_degree"] = max(out_degrees.values())
        
        # Identify hub nodes (high in-degree = many suppliers)
        hubs = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        stats["top_hubs"] = hubs
        
        self.build_stats = stats
        
        return stats
    
    def save_graph(
        self,
        G: nx.DiGraph,
        filename: str = "supply_chain_graph.pkl"
    ) -> bool:
        """
        Save NetworkX graph to pickle file.
        
        Args:
            G: NetworkX graph
            filename: Output filename
        
        Returns:
            True if save successful
        """
        try:
            filepath = self.output_dir / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(G, f)
            
            self.logger.info(f"Saved graph to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving graph: {str(e)}")
            return False
    
    def load_graph(
        self,
        filename: str = "supply_chain_graph.pkl"
    ) -> Optional[nx.DiGraph]:
        """
        Load NetworkX graph from pickle file.
        
        Args:
            filename: Graph filename
        
        Returns:
            Loaded NetworkX graph, or None if loading fails
        """
        try:
            filepath = self.output_dir / filename
            
            if not filepath.exists():
                self.logger.error(f"Graph file not found: {filepath}")
                return None
            
            with open(filepath, 'rb') as f:
                G = pickle.load(f)
            
            self.logger.info(
                f"Loaded graph from {filepath}: "
                f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
            )
            
            self.graph = G
            return G
            
        except Exception as e:
            self.logger.error(f"Error loading graph: {str(e)}")
            return None
    
    def build_complete_graph(
        self,
        relationships_file: str = "supply_chain_relationships.csv",
        min_confidence: float = 0.3,
        save: bool = True
    ) -> nx.DiGraph:
        """
        Complete pipeline: load, build, validate, and save graph.
        
        Args:
            relationships_file: CSV filename
            min_confidence: Minimum edge confidence
            save: Whether to save graph to file
        
        Returns:
            Complete NetworkX graph
        """
        self.logger.info("=" * 60)
        self.logger.info("BUILDING SUPPLY CHAIN GRAPH")
        self.logger.info("=" * 60)
        
        # Step 1: Load relationships
        df = self.load_relationships(relationships_file, min_confidence)
        
        if df.empty:
            self.logger.error("No relationships loaded. Aborting.")
            return None
        
        # Step 2: Build graph
        G = self.build_graph(df)
        
        # Step 3: Add node attributes
        G = self.add_node_attributes(G)
        
        # Step 4: Validate
        is_valid, issues = self.validate_graph(G)
        
        # Step 5: Compute statistics
        stats = self.compute_statistics(G)
        
        # Step 6: Save NetworkX graph
        if save:
            self.save_graph(G)
        
        # Step 7: Convert to PyTorch Geometric
        self.logger.info("\nConverting to PyTorch Geometric format...")
        pyg_data = self.to_pytorch_geometric(G, node_feature_dim=13)
        
        if pyg_data is not None:
            self.save_pyg_data(pyg_data)
        
        # Step 8: Visualize
        self.logger.info("\nCreating visualization...")
        self.visualize_graph(G, save_path="supply_chain_graph.png", show=False)
        
        # Print summary
        self._print_summary(stats, issues)
        
        return G
    
    def to_pytorch_geometric(
        self,
        G: nx.DiGraph,
        node_feature_dim: int = 13
    ):
        """
        Convert NetworkX graph to PyTorch Geometric Data format.
        
        Args:
            G: NetworkX directed graph
            node_feature_dim: Dimension of node features (default: 13 for our features)
        
        Returns:
            PyTorch Geometric Data object
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            self.logger.error(
                "PyTorch Geometric not installed. "
                "Install with: pip install torch-geometric"
            )
            return None
        
        # Create node mapping (ticker -> index)
        nodes = sorted(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        self.logger.info(f"Converting graph with {len(nodes)} nodes to PyG format")
        
        # Create edge index (2 x num_edges tensor)
        edge_list = []
        edge_attr_list = []
        
        for source, target, data in G.edges(data=True):
            source_idx = node_to_idx[source]
            target_idx = node_to_idx[target]
            
            edge_list.append([source_idx, target_idx])
            
            # Edge attributes: [confidence, relationship_type_encoded]
            confidence = data.get('confidence', 1.0)
            edge_attr_list.append([confidence])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        
        # Node features (placeholder - will be filled with actual features later)
        # For now, create zero features of the right dimension
        x = torch.zeros((len(nodes), node_feature_dim), dtype=torch.float)
        
        # Node attributes as additional info
        tier_list = []
        for node in nodes:
            tier = G.nodes[node].get('tier', -1)
            tier_list.append(tier)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(nodes)
        )
        
        # Store additional metadata
        data.node_names = nodes
        data.node_to_idx = node_to_idx
        data.tiers = torch.tensor(tier_list, dtype=torch.long)
        
        self.logger.info(
            f"Created PyG Data: {data.num_nodes} nodes, "
            f"{data.edge_index.shape[1]} edges, "
            f"feature dim: {node_feature_dim}"
        )
        
        return data
    
    def save_pyg_data(
        self,
        data,
        filename: str = "supply_chain_pyg.pt"
    ) -> bool:
        """
        Save PyTorch Geometric Data object.
        
        Args:
            data: PyG Data object
            filename: Output filename
        
        Returns:
            True if save successful
        """
        try:
            import torch
            
            filepath = self.output_dir / filename
            torch.save(data, filepath)
            
            self.logger.info(f"Saved PyG data to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving PyG data: {str(e)}")
            return False
    
    def visualize_graph(
        self,
        G: nx.DiGraph,
        save_path: str = None,
        show: bool = True
    ) -> bool:
        """
        Visualize the supply chain graph with hierarchical layout.
        
        Args:
            G: NetworkX graph
            save_path: Path to save figure (optional)
            show: Whether to display the plot
        
        Returns:
            True if visualization successful
        """
        try:
            import matplotlib.pyplot as plt
            
            self.logger.info("Creating graph visualization")
            
            # Create hierarchical layout based on tiers
            pos = {}
            tier_nodes = {0: [], 1: [], 2: [], 3: []}
            
            for node in G.nodes():
                tier = G.nodes[node].get('tier', -1)
                if tier in tier_nodes:
                    tier_nodes[tier].append(node)
            
            # Position nodes by tier (top to bottom)
            y_positions = {0: 3, 1: 2, 2: 1, 3: 0}  # OEMs at top, raw materials at bottom
            
            for tier, nodes in tier_nodes.items():
                n_nodes = len(nodes)
                y = y_positions.get(tier, 0)
                
                for i, node in enumerate(sorted(nodes)):
                    # Spread nodes horizontally
                    x = (i - n_nodes/2) * 2
                    pos[node] = (x, y)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Define colors by tier
            tier_colors = {
                0: '#FF6B6B',  # OEMs - Red
                1: '#4ECDC4',  # Battery - Teal
                2: '#45B7D1',  # Components - Blue
                3: '#96CEB4'   # Raw Materials - Green
            }
            
            node_colors = [tier_colors.get(G.nodes[node].get('tier', -1), '#CCCCCC') 
                          for node in G.nodes()]
            
            # Draw edges with varying width based on confidence
            edge_widths = [G[u][v].get('confidence', 0.5) * 3 for u, v in G.edges()]
            
            nx.draw_networkx_edges(
                G, pos,
                width=edge_widths,
                alpha=0.4,
                edge_color='gray',
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos,
                node_color=node_colors,
                node_size=3000,
                alpha=0.9,
                ax=ax
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                G, pos,
                font_size=10,
                font_weight='bold',
                ax=ax
            )
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=tier_colors[0], label='Tier 0: OEMs'),
                Patch(facecolor=tier_colors[2], label='Tier 2: Components'),
                Patch(facecolor=tier_colors[3], label='Tier 3: Raw Materials')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
            
            # Title
            ax.set_title(
                'EV Supply Chain Knowledge Graph\n'
                f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges',
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            
            ax.axis('off')
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                save_filepath = self.output_dir / save_path
                plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved visualization to {save_filepath}")
            
            # Show plot
            if show:
                plt.show()
            else:
                plt.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error visualizing graph: {str(e)}")
            return False
    
    def _print_summary(self, stats: Dict, issues: List[str]) -> None:
        """Print graph building summary."""
        print("\n" + "="*60)
        print("GRAPH BUILDING SUMMARY")
        print("="*60)
        print(f"Nodes: {stats['nodes']}")
        print(f"Edges: {stats['edges']}")
        print(f"Density: {stats['density']:.3f}")
        print(f"Average degree: {stats['avg_degree']:.2f}")
        
        print(f"\nNodes by tier:")
        for tier, count in sorted(stats['nodes_by_tier'].items()):
            tier_name = {0: 'OEM', 1: 'Battery', 2: 'Components', 3: 'Raw Materials'}.get(tier, 'Unknown')
            print(f"  Tier {tier} ({tier_name}): {count} nodes")
        
        print(f"\nTop hubs (by in-degree):")
        for node, degree in stats['top_hubs']:
            print(f"  {node}: {degree} suppliers")
        
        if issues:
            print(f"\nValidation issues: {len(issues)}")
            for issue in issues[:3]:  # Show first 3
                print(f"  - {issue}")
        else:
            print("\n✓ Validation passed")
        
        print("="*60 + "\n")