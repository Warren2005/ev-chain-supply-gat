"""
Relationship Extractor for EV Supply Chain GAT Project

Extracts supplier-customer relationships from SEC 10-K filings using
keyword matching and context analysis.

Author: EV Supply Chain GAT Team
Date: December 2025
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict

import pandas as pd
from bs4 import BeautifulSoup


class RelationshipExtractor:
    """
    Extracts supplier-customer relationships from SEC filings.
    
    Uses keyword matching and context analysis to identify mentions
    of supplier companies in customer filings.
    
    Attributes:
        output_dir (Path): Directory where edge lists will be saved
        logger (logging.Logger): Logger instance
    """
    
    # Keywords that indicate a supplier relationship
    SUPPLIER_KEYWORDS = [
        'supplier', 'supply', 'supplies', 'vendor', 'partner',
        'source', 'sources', 'procure', 'purchase', 'contract',
        'agreement', 'provide', 'provides', 'manufacturer',
        'key supplier', 'strategic supplier', 'primary supplier',
        'major supplier', 'critical supplier'
    ]
    
    # Keywords that increase confidence
    HIGH_CONFIDENCE_KEYWORDS = [
        'key supplier', 'strategic supplier', 'primary supplier',
        'exclusive', 'long-term agreement', 'multi-year contract',
        'sole source', 'strategic partnership', 'critical supplier'
    ]
    
    # Keywords that decrease confidence (potential false positives)
    LOW_CONFIDENCE_KEYWORDS = [
        'potential', 'possible', 'may', 'could', 'example',
        'such as', 'including', 'among others'
    ]
    
    def __init__(self, output_dir: str = "data/raw/relationships"):
        """
        Initialize the RelationshipExtractor.
        
        Args:
            output_dir: Directory to save extracted relationships
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Company name mapping (ticker -> full company names)
        self.company_names = self._init_company_names()
        
        # Extraction statistics
        self.extraction_stats = {
            "files_processed": 0,
            "relationships_found": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging for this class."""
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
        log_file = self.output_dir / "relationship_extractor.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _init_company_names(self) -> Dict[str, List[str]]:
        """
        Initialize mapping of tickers to company names and variations.
        
        Returns:
            Dictionary mapping tickers to lists of name variations
        """
        # For each company, list name variations that might appear in filings
        return {
            # Tier 0: OEMs
            'TSLA': ['Tesla', 'Tesla Inc', 'Tesla Motors'],
            'F': ['Ford', 'Ford Motor', 'Ford Motor Company'],
            'GM': ['General Motors', 'GM', 'General Motors Company'],
            'RIVN': ['Rivian', 'Rivian Automotive'],
            
            # Tier 1: Battery
            'PCRFY': ['Panasonic', 'Panasonic Corporation'],
            'LG': ['LG Energy Solution', 'LG Chem', 'LG'],
            'CATL': ['CATL', 'Contemporary Amperex Technology'],
            
            # Tier 2: Components  
            'MGA': ['Magna', 'Magna International'],
            'APTV': ['Aptiv', 'Aptiv PLC'],
            
            # Tier 3: Raw Materials
            'ALB': ['Albemarle', 'Albemarle Corporation'],
            'SQM': ['SQM', 'Sociedad Quimica', 'Sociedad Química y Minera'],
            'LTHM': ['Livent', 'Livent Corporation'],
            'LAC': ['Lithium Americas', 'Lithium Americas Corp'],
            'MP': ['MP Materials', 'MP Materials Corp'],
            'PLS.AX': ['Pilbara', 'Pilbara Minerals']
        }
    
    def parse_filing_text(self, filepath: str) -> str:
        """
        Parse HTML filing to extract plain text.
        
        Args:
            filepath: Path to HTML filing
        
        Returns:
            Extracted plain text
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'head']):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error parsing {filepath}: {str(e)}")
            return ""
    
    def find_company_mentions(
        self,
        text: str,
        supplier_ticker: str
    ) -> List[Dict[str, any]]:
        """
        Find mentions of a supplier company in text.
        
        Args:
            text: Text to search
            supplier_ticker: Ticker of supplier to search for
        
        Returns:
            List of matches with context
        """
        matches = []
        
        # Get name variations for this supplier
        if supplier_ticker not in self.company_names:
            return matches
        
        company_names = self.company_names[supplier_ticker]
        
        # Search for each name variation
        for name in company_names:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(name) + r'\b'
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Extract context window (200 chars before and after)
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 200)
                context = text[start:end]
                
                matches.append({
                    'supplier_ticker': supplier_ticker,
                    'matched_name': name,
                    'position': match.start(),
                    'context': context
                })
        
        return matches
    
    def score_relationship(
        self,
        context: str,
        supplier_ticker: str
    ) -> float:
        """
        Score the confidence of a supplier relationship based on context.
        
        Scoring rules:
        - 1.0: High confidence (strategic partner, key supplier, contract value)
        - 0.8: Clear supplier mention with strong keywords
        - 0.6: Multiple mentions or moderate keywords
        - 0.4: Single mention with weak keywords
        - 0.2: Uncertain or ambiguous mention
        
        Args:
            context: Text context around the company mention
            supplier_ticker: Ticker of the supplier
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        context_lower = context.lower()
        score = 0.5  # Base score
        
        # Check for high confidence keywords
        high_conf_count = sum(
            1 for keyword in self.HIGH_CONFIDENCE_KEYWORDS
            if keyword in context_lower
        )
        
        if high_conf_count > 0:
            score = 0.9  # Very high confidence
            return min(score, 1.0)
        
        # Check for supplier keywords
        supplier_keyword_count = sum(
            1 for keyword in self.SUPPLIER_KEYWORDS
            if keyword in context_lower
        )
        
        if supplier_keyword_count >= 3:
            score = 0.8  # High confidence
        elif supplier_keyword_count >= 2:
            score = 0.7  # Good confidence
        elif supplier_keyword_count >= 1:
            score = 0.6  # Moderate confidence
        else:
            score = 0.3  # Low confidence
        
        # Check for low confidence indicators
        low_conf_count = sum(
            1 for keyword in self.LOW_CONFIDENCE_KEYWORDS
            if keyword in context_lower
        )
        
        if low_conf_count > 0:
            score -= 0.2 * low_conf_count  # Reduce score
        
        # Check for dollar amounts (contracts/values)
        if re.search(r'\$[\d,]+(?:\.\d+)?\s*(?:million|billion)?', context):
            score += 0.1  # Boost for specific contract values
        
        # Ensure score is in valid range
        return max(0.0, min(score, 1.0))
    
    def extract_relationships_from_filing(
        self,
        filepath: str,
        customer_ticker: str,
        supplier_tickers: List[str]
    ) -> List[Dict[str, any]]:
        """
        Extract supplier relationships from a single filing.
        
        Args:
            filepath: Path to the filing
            customer_ticker: Ticker of the company that filed (customer)
            supplier_tickers: List of potential supplier tickers to search for
        
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Parse filing
        text = self.parse_filing_text(filepath)
        
        if not text:
            self.logger.warning(f"No text extracted from {filepath}")
            return relationships
        
        # Extract year from filepath (assumes format: 10-K_YYYY.html)
        try:
            year = int(Path(filepath).stem.split('_')[1])
        except:
            year = None
        
        # Search for each supplier
        for supplier_ticker in supplier_tickers:
            # Don't search for self-references
            if supplier_ticker == customer_ticker:
                continue
            
            # Find mentions
            mentions = self.find_company_mentions(text, supplier_ticker)
            
            if not mentions:
                continue
            
            # Aggregate mentions and calculate best score
            best_score = 0.0
            best_context = ""
            
            for mention in mentions:
                score = self.score_relationship(
                    mention['context'],
                    supplier_ticker
                )
                
                if score > best_score:
                    best_score = score
                    best_context = mention['context']
            
            # Only include if confidence is above threshold
            if best_score >= 0.3:  # Minimum threshold
                relationships.append({
                    'source_ticker': supplier_ticker,
                    'target_ticker': customer_ticker,
                    'relationship_type': 'supplies_to',
                    'evidence_source': Path(filepath).name,
                    'evidence_text': best_context[:500],  # Limit length
                    'confidence': round(best_score, 2),
                    'fiscal_year': year,
                    'num_mentions': len(mentions),
                    'validated': False
                })
                
                self.logger.debug(
                    f"Found relationship: {supplier_ticker} -> {customer_ticker} "
                    f"(confidence: {best_score:.2f}, mentions: {len(mentions)})"
                )
        
        return relationships
    
    def extract_all_relationships(
        self,
        filings_dir: str,
        ticker_list: List[str]
    ) -> pd.DataFrame:
        """
        Extract relationships from all filings in a directory.
        
        Args:
            filings_dir: Directory containing SEC filings (organized by ticker)
            ticker_list: List of all tickers to search for
        
        Returns:
            DataFrame with all extracted relationships
        """
        all_relationships = []
        filings_dir = Path(filings_dir)
        
        self.logger.info(
            f"Extracting relationships from filings in {filings_dir}"
        )
        
        # Reset stats
        self.extraction_stats = {
            "files_processed": 0,
            "relationships_found": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0
        }
        
        # Process each customer's filings
        for customer_ticker in ticker_list:
            ticker_dir = filings_dir / customer_ticker
            
            if not ticker_dir.exists():
                self.logger.debug(f"No filings directory for {customer_ticker}")
                continue
            
            # Process each filing
            for filepath in ticker_dir.glob("*.html"):
                self.logger.info(f"Processing {filepath.name} for {customer_ticker}")
                
                relationships = self.extract_relationships_from_filing(
                    str(filepath),
                    customer_ticker,
                    ticker_list
                )
                
                all_relationships.extend(relationships)
                self.extraction_stats["files_processed"] += 1
        
        # Convert to DataFrame
        if all_relationships:
            df = pd.DataFrame(all_relationships)
            
            # Update statistics
            self.extraction_stats["relationships_found"] = len(df)
            self.extraction_stats["high_confidence"] = len(df[df['confidence'] >= 0.7])
            self.extraction_stats["medium_confidence"] = len(df[(df['confidence'] >= 0.5) & (df['confidence'] < 0.7)])
            self.extraction_stats["low_confidence"] = len(df[df['confidence'] < 0.5])
            
            # Remove duplicates (keep highest confidence)
            df = df.sort_values('confidence', ascending=False)
            df = df.drop_duplicates(
                subset=['source_ticker', 'target_ticker'],
                keep='first'
            )
            
            self.logger.info(
                f"Extracted {len(df)} unique relationships from "
                f"{self.extraction_stats['files_processed']} filings"
            )
        else:
            df = pd.DataFrame()
            self.logger.warning("No relationships found")
        
        # Print summary
        self._print_summary()
        
        return df
    
    def save_relationships(
        self,
        df: pd.DataFrame,
        filename: str = "supply_chain_edges.csv"
    ) -> bool:
        """
        Save relationships to CSV file.
        
        Args:
            df: DataFrame with relationships
            filename: Output filename
        
        Returns:
            True if save successful
        """
        try:
            filepath = self.output_dir / filename
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved {len(df)} relationships to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving relationships: {str(e)}")
            return False
    
    def generate_validation_file(
        self,
        df: pd.DataFrame,
        top_n: int = 30,
        filename: str = "relationships_for_validation.csv"
    ) -> bool:
        """
        Generate a file with top relationships for manual validation.
        
        Args:
            df: DataFrame with relationships
            top_n: Number of top relationships to include
            filename: Output filename
        
        Returns:
            True if save successful
        """
        try:
            # Sort by confidence and take top N
            validation_df = df.nlargest(top_n, 'confidence').copy()
            
            # Add validation column
            validation_df['manually_validated'] = ''
            validation_df['validator_notes'] = ''
            
            # Reorder columns for easier review
            cols = [
                'source_ticker', 'target_ticker', 'confidence',
                'relationship_type', 'num_mentions', 'fiscal_year',
                'evidence_text', 'manually_validated', 'validator_notes'
            ]
            validation_df = validation_df[cols]
            
            filepath = self.output_dir / filename
            validation_df.to_csv(filepath, index=False)
            
            self.logger.info(
                f"Generated validation file with top {len(validation_df)} "
                f"relationships: {filepath}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating validation file: {str(e)}")
            return False
    
    def _print_summary(self) -> None:
        """Print extraction summary statistics."""
        print("\n" + "="*60)
        print("RELATIONSHIP EXTRACTION SUMMARY")
        print("="*60)
        print(f"Files processed: {self.extraction_stats['files_processed']}")
        print(f"Total relationships found: {self.extraction_stats['relationships_found']}")
        print(f"  High confidence (≥0.7): {self.extraction_stats['high_confidence']}")
        print(f"  Medium confidence (0.5-0.7): {self.extraction_stats['medium_confidence']}")
        print(f"  Low confidence (<0.5): {self.extraction_stats['low_confidence']}")
        print("="*60 + "\n")