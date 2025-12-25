"""
Download SEC 10-K filings for relationship extraction

This downloads filings for the stocks we have market data for.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.sec_downloader import SECFilingDownloader


def main():
    """Download SEC filings"""
    print("="*70)
    print("DOWNLOADING SEC 10-K FILINGS")
    print("="*70)
    print()
    
    # Initialize downloader
    downloader = SECFilingDownloader(
        output_dir="data/raw/sec_filings",
        db_name="data/metadata/sec_filings.db"
    )
    
    # Your 10 stocks that successfully downloaded market data
    # Only US companies have SEC filings
    us_tickers = [
        "TSLA",   # Tesla
        "F",      # Ford
        "GM",     # General Motors
        "RIVN",   # Rivian
        "MGA",    # Magna International
        "APTV",   # Aptiv
        "ALB",    # Albemarle
        "SQM",    # SQM (Chilean, but files with SEC)
        "LTHM",   # Livent
        "LAC",    # Lithium Americas
        "MP"      # MP Materials
    ]
    
    # Download 10-Ks for multiple years to get more relationship mentions
    fiscal_years = [2023, 2022, 2021]
    
    print(f"Downloading 10-K filings for {len(us_tickers)} companies")
    print(f"Fiscal years: {fiscal_years}")
    print()
    print("⚠️  This will take 10-15 minutes due to SEC rate limiting (0.1s per request)")
    print("⚠️  Some downloads may fail - this is expected for certain companies")
    print()
    
    successful = []
    failed = []
    total_downloads = 0
    
    for ticker in us_tickers:
        print(f"\n{'='*60}")
        print(f"Downloading filings for {ticker}")
        print('='*60)
        
        ticker_success = False
        
        for year in fiscal_years:
            print(f"  Year {year}...", end=" ")
            filepath = downloader.download_filing(ticker, year)
            
            if filepath:
                print(f"✓")
                ticker_success = True
                total_downloads += 1
            else:
                print(f"✗")
        
        if ticker_success:
            successful.append(ticker)
        else:
            failed.append(ticker)
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Companies with at least 1 filing: {len(successful)}/{len(us_tickers)}")
    print(f"Total filings downloaded: {total_downloads}")
    
    if successful:
        print(f"\nSuccessful companies: {', '.join(successful)}")
    
    if failed:
        print(f"\nFailed companies: {', '.join(failed)}")
        print("(Some failures are expected - international companies, recent IPOs, etc.)")
    
    if len(successful) >= 5:
        print("\n✓ Sufficient filings for relationship extraction!")
        print(f"\nFilings saved to: data/raw/sec_filings/")
        print("\nNext step: Extract relationships")
        print("  python scripts/extract_relationships.py")
        return True
    else:
        print("\n⚠️  Too few filings downloaded (<5 companies)")
        print("The relationship extraction may not find enough connections.")
        print("\nYou can still try extraction, or use sample relationships.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)