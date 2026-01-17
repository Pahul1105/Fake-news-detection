"""
Tests for text preprocessing module.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import TextCleaner, clean_text, tokenize


class TestTextCleaner:
    """Tests for TextCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = TextCleaner()
    
    def test_clean_text_removes_html(self):
        """Test HTML tag removal."""
        result = clean_text("<p>Hello <b>World</b></p>")
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Hello" in result
        assert "World" in result
    
    def test_clean_text_removes_urls(self):
        """Test URL removal."""
        result = clean_text("Check out https://example.com for more")
        assert "https://" not in result
        assert "example.com" not in result
    
    def test_clean_text_normalizes_punctuation(self):
        """Test multiple punctuation normalization."""
        result = clean_text("This is SHOCKING!!!")
        assert "!!!" not in result
        assert "!" in result
    
    def test_clean_text_handles_empty_input(self):
        """Test empty string handling."""
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_tokenize_returns_list(self):
        """Test tokenization returns list of tokens."""
        tokens = tokenize("Hello world, this is a test.")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "hello" in tokens
        assert "world" in tokens
    
    def test_tokenize_removes_punctuation_tokens(self):
        """Test punctuation-only tokens are removed."""
        tokens = tokenize("Hello, world!")
        assert "," not in tokens
        assert "!" not in tokens
    
    def test_preprocess_article(self):
        """Test full preprocessing pipeline."""
        text = "<p>Breaking news! The weather is nice today.</p>"
        result = self.cleaner.preprocess_article(text)
        
        assert result.original == text
        assert "<p>" not in result.cleaned
        assert result.word_count > 0
        assert result.sentence_count >= 1
    
    def test_extract_headline_body(self):
        """Test headline/body extraction."""
        article = "This is the headline\nThis is the body content."
        headline, body = self.cleaner.extract_headline_body(article)
        
        assert "headline" in headline.lower()
        assert "body" in body.lower()


class TestSensationalismPatterns:
    """Test detection of sensationalist language patterns."""
    
    def setup_method(self):
        self.cleaner = TextCleaner()
    
    def test_extreme_punctuation_cleaned(self):
        """Test extreme punctuation is normalized."""
        result = clean_text("SHOCKING!!! You WON'T BELIEVE THIS!!!")
        # Multiple ! should be reduced
        assert result.count("!") <= 2
    
    def test_preserves_content_words(self):
        """Test content words are preserved after cleaning."""
        text = "Scientists announce new breakthrough in research"
        result = clean_text(text)
        assert "scientists" in result.lower()
        assert "breakthrough" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
