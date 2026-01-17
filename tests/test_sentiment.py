"""
Tests for sentiment analysis module.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sentiment_analyzer import (
    SentimentAnalyzer, 
    SentimentResult,
    analyze_sentiment,
    get_polarity,
    has_manipulation_signals,
)


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_analyze_positive_text(self):
        """Test positive sentiment detection."""
        result = self.analyzer.analyze("This is wonderful news! Great success!")
        assert result.polarity > 0
        assert result.positive_score > result.negative_score
    
    def test_analyze_negative_text(self):
        """Test negative sentiment detection."""
        result = self.analyzer.analyze("This is terrible news. Very disappointing.")
        assert result.polarity < 0
        assert result.negative_score > result.positive_score
    
    def test_analyze_neutral_text(self):
        """Test neutral sentiment detection."""
        result = self.analyzer.analyze("The meeting is scheduled for 3pm.")
        assert abs(result.polarity) < 0.3
        assert result.neutral_score > 0.5
    
    def test_analyze_returns_sentiment_result(self):
        """Test that analyze returns SentimentResult dataclass."""
        result = self.analyzer.analyze("Sample text")
        assert isinstance(result, SentimentResult)
    
    def test_empty_text_handling(self):
        """Test empty string handling."""
        result = self.analyzer.analyze("")
        assert result.polarity == 0.0
        assert result.neutral_score == 1.0
    
    def test_none_handling(self):
        """Test None input handling."""
        result = self.analyzer.analyze(None)
        assert result.polarity == 0.0


class TestSensationalismDetection:
    """Tests for sensationalism/manipulation detection."""
    
    def setup_method(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_sensationalism_keywords_detected(self):
        """Test sensationalism keyword detection."""
        result = self.analyzer.analyze(
            "SHOCKING news! You won't believe this exclusive story!"
        )
        assert result.sensationalism_score > 0
        assert len(result.sensationalism_keywords_found) > 0
        assert "shocking" in [k.lower() for k in result.sensationalism_keywords_found]
    
    def test_fear_mongering_detected(self):
        """Test fear-mongering keyword detection."""
        result = self.analyzer.analyze(
            "Deadly threat poses dangerous crisis for everyone!"
        )
        assert result.fear_mongering_score > 0
        assert len(result.fear_keywords_found) > 0
    
    def test_exaggeration_detected(self):
        """Test exaggeration keyword detection."""
        result = self.analyzer.analyze(
            "Everyone always believes everything they read. Never trust anyone."
        )
        assert result.exaggeration_score > 0
        assert len(result.exaggeration_keywords_found) > 0
    
    def test_manipulation_signals_flag(self):
        """Test manipulation signals flag is set correctly."""
        # Neutral news should not have manipulation signals
        neutral = self.analyzer.analyze(
            "Government announces revised tax policy after review"
        )
        assert neutral.has_manipulation_signals is False
        
        # Sensationalist news should have manipulation signals
        sensational = self.analyzer.analyze(
            "SHOCKING scandal EXPOSED! You won't believe what happened!"
        )
        assert sensational.has_manipulation_signals is True
    
    def test_clean_news_no_keywords(self):
        """Test clean news has no manipulation keywords."""
        result = self.analyzer.analyze(
            "Scientists publish research findings on climate patterns"
        )
        assert result.sensationalism_score == 0
        assert result.fear_mongering_score == 0


class TestArticleAnalysis:
    """Tests for full article analysis."""
    
    def setup_method(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_analyze_article_body_only(self):
        """Test article analysis with body only."""
        result = self.analyzer.analyze_article(
            body="This is the article body with important information."
        )
        assert "body_sentiment" in result
        assert "overall_polarity" in result
    
    def test_analyze_article_with_headline(self):
        """Test article analysis with headline."""
        result = self.analyzer.analyze_article(
            body="The article provides balanced information on the topic.",
            headline="BREAKING: Major announcement!"
        )
        assert "headline_sentiment" in result
        assert "headline_body_sentiment_gap" in result
    
    def test_headline_body_gap(self):
        """Test headline-body sentiment gap calculation."""
        result = self.analyzer.analyze_article(
            body="The weather might be slightly warmer tomorrow.",  # Neutral
            headline="SHOCKING weather disaster imminent!"  # Negative/sensational
        )
        # Should show a gap between sensationalist headline and neutral body
        assert result["headline_body_sentiment_gap"] > 0


class TestSentimentFeatures:
    """Tests for feature extraction."""
    
    def setup_method(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_get_sentiment_features_returns_dict(self):
        """Test feature extraction returns dictionary."""
        features = self.analyzer.get_sentiment_features("Sample text")
        assert isinstance(features, dict)
    
    def test_features_contain_required_keys(self):
        """Test all required features are present."""
        features = self.analyzer.get_sentiment_features("Sample text")
        required_keys = [
            "polarity", "intensity", "subjectivity",
            "sensationalism_score", "fear_mongering_score",
            "has_manipulation_signals"
        ]
        for key in required_keys:
            assert key in features
    
    def test_features_are_numeric(self):
        """Test all features are numeric."""
        features = self.analyzer.get_sentiment_features("Sample text")
        for key, value in features.items():
            assert isinstance(value, (int, float))


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_analyze_sentiment_function(self):
        """Test standalone analyze_sentiment function."""
        result = analyze_sentiment("Great news!")
        assert isinstance(result, SentimentResult)
        assert result.polarity > 0
    
    def test_get_polarity_function(self):
        """Test standalone get_polarity function."""
        polarity = get_polarity("This is wonderful!")
        assert isinstance(polarity, float)
        assert polarity > 0
    
    def test_has_manipulation_signals_function(self):
        """Test standalone has_manipulation_signals function."""
        assert has_manipulation_signals("Normal news.") is False
        assert has_manipulation_signals("SHOCKING exclusive scandal!") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
