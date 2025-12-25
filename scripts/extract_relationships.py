"""
Extract supplier-customer relationships from SEC filings

Uses the RelationshipExtractor to mine relationships from downloaded 10-Ks.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.relationship_extractor import RelationshipExtractor


def main():
    """Extract relationships from SEC filings"""
    print("="*70)
    print("EXTRACTING SUPPLY CHAIN RELATIONSHIPS FROM SEC FILINGS")
    print("="*70)
    print()
    
    # Initialize extractor
    extractor = RelationshipExtractor(
        output_dir="data/raw/sec_filings"
    )
    
    # All tickers we have (some may not have filings downloaded)
    ticker_list = [
        "TSLA", "F", "GM", "RIVN",      # OEMs
        "MGA", "APTV",                   # Component suppliers
        "ALB", "SQM", "LTHM", "LAC", "MP"  # Raw materials
    ]
    
    print(f"Scanning filings for {len(ticker_list)} companies")
    print(f"Looking for supplier-customer relationships...")
    print()
    print("This will take 3-5 minutes to process all filings")
    print()
    
    # Extract relationships
    relationships_df = extractor.extract_all_relationships(
        filings_dir="data/raw/sec_filings",
        ticker_list=ticker_list
    )
    
    # Check if we got any results
    if relationships_df is None or len(relationships_df) == 0:
        print("\n" + "="*70)
        print("⚠️  NO RELATIONSHIPS EXTRACTED")
        print("="*70)
        print()
        print("Why this happened:")
        print("  • Companies typically describe suppliers generically in 10-Ks")
        print("  • Explicit supplier names are rarely mentioned")
        print("  • Keyword matching requires exact company name mentions")
        print()
        print("This is NORMAL and expected for keyword-based extraction!")
        print()
        print("✓ RECOMMENDED: Use sample relationships for MVP")
        print("  Sample relationships are based on publicly known supply chains")
        print("  (e.g., Panasonic supplies Tesla, Magna supplies Ford/GM)")
        print()
        print("To proceed with sample data:")
        print("  python scripts/create_sample_relationships.py")
        print("  python scripts/test_graph_builder.py")
        return False
    
    # Filter by confidence threshold
    min_confidence = 0.3
    relationships_df = relationships_df[relationships_df['confidence'] >= min_confidence]
    
    if len(relationships_df) > 0:
        print("\n" + "="*70)
        print("EXTRACTION COMPLETE")
        print("="*70)
        print(f"✓ Found {len(relationships_df)} relationships (confidence >= {min_confidence})")
        
        # Show confidence distribution
        print(f"\nConfidence distribution:")
        print(f"  High confidence (>0.7): {len(relationships_df[relationships_df['confidence'] > 0.7])}")
        print(f"  Medium confidence (0.5-0.7): {len(relationships_df[(relationships_df['confidence'] > 0.5) & (relationships_df['confidence'] <= 0.7)])}")
        print(f"  Low confidence (0.3-0.5): {len(relationships_df[(relationships_df['confidence'] >= 0.3) & (relationships_df['confidence'] <= 0.5)])}")
        
        # Show top relationships
        print(f"\nTop 10 relationships by confidence:")
        top_10 = relationships_df.nlargest(10, 'confidence')[
            ['source_ticker', 'target_ticker', 'confidence', 'relationship_type']
        ]
        print(top_10.to_string(index=False))
        
        # Save all relationships
        save_success = extractor.save_relationships(
            relationships_df,
            "supply_chain_relationships.csv"
        )
        
        if save_success:
            print(f"\n✓ Relationships saved to: data/raw/sec_filings/supply_chain_relationships.csv")
        
        # Generate validation file (top 20 for manual review)
        validation_success = extractor.generate_validation_file(
            relationships_df,
            top_n=min(20, len(relationships_df))
        )
        
        if validation_success:
            print(f"✓ Validation file created: data/raw/sec_filings/relationships_for_validation.csv")
        
        print()
        print("📋 OPTIONAL: Review relationships_for_validation.csv to verify quality")
        print("   (You can manually mark relationships as valid/invalid)")
        print()
        print("Next step: Build knowledge graph")
        print("  python scripts/test_graph_builder.py")
        return True
    else:
        print("\n" + "="*70)
        print("⚠️  NO RELATIONSHIPS EXTRACTED")
        print("="*70)
        print()
        print("This could mean:")
        print("  1. No SEC filings were downloaded successfully")
        print("  2. The keyword matching didn't find supplier mentions")
        print("  3. Companies don't mention each other in their 10-Ks")
        print()
        
        # Check if we have any filings at all
        filings_dir = Path("data/raw/sec_filings")
        total_filings = sum(1 for _ in filings_dir.rglob("*.html"))
        
        print(f"Debug info:")
        print(f"  Total HTML files found: {total_filings}")
        print(f"  Extraction stats: {extractor.extraction_stats}")
        print()
        print("Recommendations:")
        print("  1. Check if filings were downloaded: ls data/raw/sec_filings/*/")
        print("  2. Try lowering min_confidence to 0.2 (edit line 45 in this script)")
        print("  3. Use sample relationships for MVP: python scripts/create_sample_relationships.py")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)