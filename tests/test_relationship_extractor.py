"""
Unit tests for RelationshipExtractor

Tests all functionality of the relationship extraction module.

Run with: pytest tests/test_relationship_extractor.py -v
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from utils.relationship_extractor import RelationshipExtractor


class TestRelationshipExtractor:
    """Test suite for RelationshipExtractor class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def extractor(self, temp_dir):
        """Create a RelationshipExtractor instance with temp directory"""
        return RelationshipExtractor(output_dir=temp_dir)
    
    @pytest.fixture
    def sample_filing_text(self):
        """Generate sample filing text for testing"""
        return """
        ITEM 1. BUSINESS
        
        We are a leading electric vehicle manufacturer. Our key suppliers include
        Panasonic Corporation, which provides battery cells under a long-term
        strategic partnership agreement. We have a multi-year contract with
        Albemarle Corporation for lithium supply. 
        
        Additionally, Magna International supplies various automotive components.
        We may also procure materials from other suppliers such as LG Chem.
        
        ITEM 1A. RISK FACTORS
        
        Our dependence on key suppliers like Panasonic and Albemarle creates
        supply chain risk.
        """
    
    def test_initialization(self, temp_dir):
        """Test that RelationshipExtractor initializes correctly"""
        extractor = RelationshipExtractor(output_dir=temp_dir)
        
        assert extractor.output_dir.exists()
        assert extractor.logger is not None
        assert len(extractor.company_names) > 0
        assert 'TSLA' in extractor.company_names
        assert len(extractor.SUPPLIER_KEYWORDS) > 0
    
    def test_company_names_mapping(self, extractor):
        """Test that company name variations are defined"""
        # Check a few key companies
        assert 'TSLA' in extractor.company_names
        assert 'Tesla' in extractor.company_names['TSLA']
        
        assert 'ALB' in extractor.company_names
        assert 'Albemarle' in extractor.company_names['ALB']
    
    def test_parse_filing_text(self, extractor, tmp_path):
        """Test parsing HTML to plain text"""
        # Create test HTML file
        html_content = """
        <html>
        <head><title>10-K</title></head>
        <body>
            <h1>Item 1. Business</h1>
            <p>We purchase components from Panasonic.</p>
            <script>console.log('remove this');</script>
        </body>
        </html>
        """
        
        test_file = tmp_path / "test.html"
        test_file.write_text(html_content)
        
        text = extractor.parse_filing_text(str(test_file))
        
        assert "Item 1. Business" in text
        assert "Panasonic" in text
        assert "console.log" not in text  # Script removed
    
    def test_find_company_mentions(self, extractor, sample_filing_text):
        """Test finding company mentions in text"""
        # Search for Panasonic
        matches = extractor.find_company_mentions(
            sample_filing_text,
            'PCRFY'  # Panasonic ticker
        )
        
        assert len(matches) > 0
        assert any('Panasonic' in m['matched_name'] for m in matches)
        assert all('context' in m for m in matches)
    
    def test_find_company_mentions_case_insensitive(self, extractor):
        """Test that company search is case insensitive"""
        text = "We work with PANASONIC and panasonic Corporation."
        
        matches = extractor.find_company_mentions(text, 'PCRFY')
        
        # Should find both uppercase and lowercase
        assert len(matches) >= 2
    
    def test_find_company_mentions_word_boundaries(self, extractor):
        """Test that search respects word boundaries"""
        # "Tesla" should not match "Teslameter"
        text = "We use a Teslameter for measurements. Tesla Inc supplies batteries."
        
        matches = extractor.find_company_mentions(text, 'TSLA')
        
        # Should only find "Tesla Inc", not "Teslameter"
        assert len(matches) == 1
        assert 'Tesla Inc' in matches[0]['matched_name']
    
    def test_score_relationship_high_confidence(self, extractor):
        """Test scoring with high confidence keywords"""
        context = """
        Panasonic Corporation is our key supplier and strategic partner
        under a long-term agreement for battery cell supply.
        """
        
        score = extractor.score_relationship(context, 'PCRFY')
        
        # Should be high confidence
        assert score >= 0.8
    
    def test_score_relationship_with_contract_value(self, extractor):
        """Test scoring boost for contract values"""
        context = """
        We have a $500 million contract with Albemarle for lithium supply.
        """
        
        score = extractor.score_relationship(context, 'ALB')
        
        # Should have decent confidence
        assert score >= 0.6
    
    def test_score_relationship_low_confidence(self, extractor):
        """Test scoring with low confidence indicators"""
        context = """
        Potential suppliers may include companies such as Panasonic.
        """
        
        score = extractor.score_relationship(context, 'PCRFY')
        
        # Should be low confidence due to "potential" and "may"
        assert score < 0.6
    
    def test_score_relationship_multiple_supplier_keywords(self, extractor):
        """Test scoring with multiple supplier keywords"""
        context = """
        Panasonic is a key supplier and vendor that provides battery cells.
        """
        
        score = extractor.score_relationship(context, 'PCRFY')
        
        # Multiple keywords should boost confidence
        assert score >= 0.7
    
    def test_extract_relationships_from_filing(self, extractor, tmp_path):
        """Test extracting relationships from a filing"""
        # Create test filing
        html_content = """
        <html>
        <body>
        <p>We purchase battery cells from Panasonic Corporation under a
        strategic partnership. Albemarle Corporation supplies lithium
        under a long-term contract.</p>
        </body>
        </html>
        """
        
        test_file = tmp_path / "10-K_2020.html"
        test_file.write_text(html_content)
        
        # Extract relationships
        relationships = extractor.extract_relationships_from_filing(
            str(test_file),
            'TSLA',  # Customer
            ['PCRFY', 'ALB', 'MGA']  # Potential suppliers
        )
        
        # Should find Panasonic and Albemarle
        assert len(relationships) >= 2
        
        # Check structure
        for rel in relationships:
            assert 'source_ticker' in rel
            assert 'target_ticker' in rel
            assert 'confidence' in rel
            assert rel['target_ticker'] == 'TSLA'
    
    def test_extract_relationships_excludes_self_reference(self, extractor, tmp_path):
        """Test that companies don't appear as their own supplier"""
        html_content = """
        <html>
        <body>
        <p>Tesla manufactures vehicles. Tesla also purchases from Panasonic.</p>
        </body>
        </html>
        """
        
        test_file = tmp_path / "10-K_2020.html"
        test_file.write_text(html_content)
        
        relationships = extractor.extract_relationships_from_filing(
            str(test_file),
            'TSLA',
            ['TSLA', 'PCRFY']
        )
        
        # Should not find Tesla -> Tesla
        assert all(rel['source_ticker'] != 'TSLA' for rel in relationships)
    
    def test_extract_relationships_confidence_threshold(self, extractor, tmp_path):
        """Test that low confidence matches are filtered out"""
        html_content = """
        <html>
        <body>
        <p>Panasonic was mentioned in passing.</p>
        </body>
        </html>
        """
        
        test_file = tmp_path / "10-K_2020.html"
        test_file.write_text(html_content)
        
        relationships = extractor.extract_relationships_from_filing(
            str(test_file),
            'TSLA',
            ['PCRFY']
        )
        
        # Very weak mention should be filtered
        assert all(rel['confidence'] >= 0.3 for rel in relationships)
    
    def test_save_relationships(self, extractor):
        """Test saving relationships to CSV"""
        df = pd.DataFrame([
            {
                'source_ticker': 'ALB',
                'target_ticker': 'TSLA',
                'confidence': 0.85,
                'relationship_type': 'supplies_to'
            },
            {
                'source_ticker': 'PCRFY',
                'target_ticker': 'TSLA',
                'confidence': 0.92,
                'relationship_type': 'supplies_to'
            }
        ])
        
        result = extractor.save_relationships(df, "test_edges.csv")
        
        assert result is True
        
        # Check file exists
        filepath = extractor.output_dir / "test_edges.csv"
        assert filepath.exists()
        
        # Verify content
        loaded = pd.read_csv(filepath)
        assert len(loaded) == 2
        assert 'source_ticker' in loaded.columns
    
    def test_generate_validation_file(self, extractor):
        """Test generating validation file for manual review"""
        df = pd.DataFrame([
            {
                'source_ticker': 'ALB',
                'target_ticker': 'TSLA',
                'confidence': 0.85,
                'relationship_type': 'supplies_to',
                'num_mentions': 3,
                'fiscal_year': 2020,
                'evidence_text': 'Test context'
            },
            {
                'source_ticker': 'PCRFY',
                'target_ticker': 'TSLA',
                'confidence': 0.92,
                'relationship_type': 'supplies_to',
                'num_mentions': 5,
                'fiscal_year': 2020,
                'evidence_text': 'Test context 2'
            }
        ])
        
        result = extractor.generate_validation_file(df, top_n=2)
        
        assert result is True
        
        # Check file
        filepath = extractor.output_dir / "relationships_for_validation.csv"
        assert filepath.exists()
        
        # Verify it has validation columns
        loaded = pd.read_csv(filepath)
        assert 'manually_validated' in loaded.columns
        assert 'validator_notes' in loaded.columns
    
    def test_extraction_statistics(self, extractor):
        """Test that extraction statistics are tracked"""
        assert 'files_processed' in extractor.extraction_stats
        assert 'relationships_found' in extractor.extraction_stats
        assert 'high_confidence' in extractor.extraction_stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])