"""
Quick validation script for Relationship Extractor

This script tests the relationship extractor with sample data.

Run this script to verify the Relationship Extractor works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.relationship_extractor import RelationshipExtractor


def create_test_filing(filepath: Path, content: str):
    """Create a test HTML filing"""
    html = f"""
    <html>
    <head><title>10-K Filing</title></head>
    <body>
    {content}
    </body>
    </html>
    """
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(html)


def main():
    """Run quick validation test"""
    print("="*70)
    print("RELATIONSHIP EXTRACTOR - QUICK VALIDATION TEST")
    print("="*70)
    print()
    
    # Initialize extractor
    test_dir = "data/raw/relationships_test"
    extractor = RelationshipExtractor(output_dir=test_dir)
    
    print("Testing relationship extraction functionality:")
    print("  1. Company name matching")
    print("  2. Context analysis")
    print("  3. Confidence scoring")
    print("  4. Relationship extraction")
    print()
    
    # Test 1: Company name matching
    print("Test 1: Company Name Matching")
    test_text = "We purchase batteries from Panasonic Corporation and lithium from Albemarle."
    
    panasonic_matches = extractor.find_company_mentions(test_text, 'PCRFY')
    albemarle_matches = extractor.find_company_mentions(test_text, 'ALB')
    
    print(f"  ✓ Found {len(panasonic_matches)} mentions of Panasonic")
    print(f"  ✓ Found {len(albemarle_matches)} mentions of Albemarle")
    print()
    
    # Test 2: Confidence scoring
    print("Test 2: Confidence Scoring")
    
    high_conf_context = "Panasonic is our key supplier and strategic partner under a long-term agreement."
    low_conf_context = "Companies such as Panasonic may be potential suppliers."
    
    high_score = extractor.score_relationship(high_conf_context, 'PCRFY')
    low_score = extractor.score_relationship(low_conf_context, 'PCRFY')
    
    print(f"  ✓ High confidence context: {high_score:.2f}")
    print(f"  ✓ Low confidence context: {low_score:.2f}")
    print(f"  ✓ Scoring differential: {(high_score - low_score):.2f}")
    print()
    
    # Test 3: Extract from sample filing
    print("Test 3: Relationship Extraction from Sample Filing")
    
    # Create test filing directory
    filings_dir = Path("data/raw/sec_filings_test")
    tesla_dir = filings_dir / "TSLA"
    tesla_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample filing
    sample_content = """
    <h2>ITEM 1. BUSINESS</h2>
    <p>
    We are an electric vehicle manufacturer. Our battery cells are supplied by
    Panasonic Corporation under a strategic partnership agreement. We have a
    long-term contract with Albemarle Corporation for lithium supply.
    </p>
    <p>
    Magna International provides various automotive components including
    structural parts and assemblies. We also source copper and other materials
    from various suppliers.
    </p>
    <h2>ITEM 1A. RISK FACTORS</h2>
    <p>
    Our dependence on key suppliers such as Panasonic and Albemarle creates
    supply chain concentration risk.
    </p>
    """
    
    create_test_filing(tesla_dir / "10-K_2023.html", sample_content)
    
    # Extract relationships
    ticker_list = ['TSLA', 'PCRFY', 'ALB', 'MGA', 'F']
    
    relationships_df = extractor.extract_all_relationships(
        str(filings_dir),
        ticker_list
    )
    
    print()
    print("="*70)
    print("TEST RESULTS")
    print("="*70)
    
    if len(relationships_df) > 0:
        print("✓ RELATIONSHIP EXTRACTION TEST PASSED")
        print(f"  ✓ Extracted {len(relationships_df)} relationships")
        print()
        print("Top Relationships Found:")
        print("-" * 70)
        
        # Show top relationships
        top_rels = relationships_df.nlargest(5, 'confidence')
        for idx, row in top_rels.iterrows():
            print(f"  {row['source_ticker']} → {row['target_ticker']}")
            print(f"    Confidence: {row['confidence']:.2f}")
            print(f"    Mentions: {row['num_mentions']}")
            print()
        
        # Save results
        extractor.save_relationships(relationships_df)
        extractor.generate_validation_file(relationships_df, top_n=10)
        
        print("✓ Files saved to:", test_dir)
        print("  - supply_chain_edges.csv (all relationships)")
        print("  - relationships_for_validation.csv (top 10 for review)")
        
        return True
    else:
        print("✗ TEST FAILED - No relationships extracted")
        print("  Check that test filing was created correctly")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)