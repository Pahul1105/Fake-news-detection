"""
Sentiment Analysis Module for Fake News Detection.

Extracts emotional signals: polarity, subjectivity, intensity,
and detects sensationalism/fear-mongering patterns.
"""

from dataclasses import dataclass
from typing import List, Optional
import re

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    SentimentIntensityAnalyzer = None

# Import config for thresholds and keywords
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    EXTREME_POLARITY_THRESHOLD,
    HIGH_INTENSITY_THRESHOLD,
    HIGH_SUBJECTIVITY_THRESHOLD,
    SENSATIONALISM_KEYWORDS,
    FEAR_MONGERING_KEYWORDS,
    EXAGGERATION_KEYWORDS,
)


@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    
    # Core sentiment scores
    polarity: float          # -1 (negative) to 1 (positive)
    intensity: float         # 0 to 1 (emotional intensity/compound magnitude)
    subjectivity: float      # 0 (objective) to 1 (subjective)
    
    # Detailed scores
    positive_score: float    # 0 to 1
    negative_score: float    # 0 to 1
    neutral_score: float     # 0 to 1
    compound_score: float    # -1 to 1 (overall sentiment)
    
    # Manipulation indicators
    sensationalism_score: float   # 0 to 1
    fear_mongering_score: float   # 0 to 1
    exaggeration_score: float     # 0 to 1
    
    # Flags
    is_extreme_polarity: bool
    is_high_intensity: bool
    is_highly_subjective: bool
    has_manipulation_signals: bool
    
    # Raw counts
    sensationalism_keywords_found: List[str]
    fear_keywords_found: List[str]
    exaggeration_keywords_found: List[str]


class SentimentAnalyzer:
    """
    Sentiment analyzer for detecting emotional manipulation in news.
    
    Uses VADER for sentiment analysis and custom keyword detection
    for sensationalism, fear-mongering, and exaggeration patterns.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        if not VADER_AVAILABLE:
            raise ImportError(
                "vaderSentiment not installed. "
                "Run: pip3 install vaderSentiment"
            )
        
        self._analyzer = SentimentIntensityAnalyzer()
        
        # Precompile keyword patterns for efficiency
        self._sensationalism_patterns = self._compile_patterns(SENSATIONALISM_KEYWORDS)
        self._fear_patterns = self._compile_patterns(FEAR_MONGERING_KEYWORDS)
        self._exaggeration_patterns = self._compile_patterns(EXAGGERATION_KEYWORDS)
    
    def _compile_patterns(self, keywords: List[str]) -> List[re.Pattern]:
        """Compile keyword list into regex patterns for word boundary matching."""
        patterns = []
        for keyword in keywords:
            # Use word boundaries for single words, looser matching for phrases
            if ' ' in keyword:
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            else:
                pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
            patterns.append((keyword, pattern))
        return patterns
    
    def _find_keywords(
        self, 
        text: str, 
        patterns: List[tuple]
    ) -> List[str]:
        """Find all matching keywords in text."""
        found = []
        for keyword, pattern in patterns:
            if pattern.search(text):
                found.append(keyword)
        return found
    
    def _calculate_keyword_score(
        self, 
        keywords_found: List[str], 
        max_keywords: int = 5
    ) -> float:
        """
        Calculate a 0-1 score based on keyword count.
        
        Uses diminishing returns - first few keywords matter most.
        """
        count = len(keywords_found)
        if count == 0:
            return 0.0
        # Sigmoid-like scoring with cap
        score = min(count / max_keywords, 1.0)
        return score
    
    def _estimate_subjectivity(self, text: str, vader_scores: dict) -> float:
        """
        Estimate subjectivity from VADER scores and text features.
        
        Higher neutral score = more objective
        More emotional words = more subjective
        """
        # Base subjectivity from neutral score (inverse relationship)
        neutral = vader_scores['neu']
        base_subjectivity = 1.0 - neutral
        
        # Adjust based on emotional intensity
        compound_magnitude = abs(vader_scores['compound'])
        intensity_factor = compound_magnitude * 0.3
        
        # Combine factors
        subjectivity = min(base_subjectivity + intensity_factor, 1.0)
        return subjectivity
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Perform full sentiment analysis on text.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            SentimentResult with all sentiment features.
        """
        if not text or not isinstance(text, str):
            return self._empty_result()
        
        # Get VADER sentiment scores
        vader_scores = self._analyzer.polarity_scores(text)
        
        # Extract core metrics
        polarity = vader_scores['compound']  # -1 to 1
        intensity = abs(polarity)  # Magnitude of emotion
        subjectivity = self._estimate_subjectivity(text, vader_scores)
        
        # Find manipulation keywords
        sensationalism_found = self._find_keywords(text, self._sensationalism_patterns)
        fear_found = self._find_keywords(text, self._fear_patterns)
        exaggeration_found = self._find_keywords(text, self._exaggeration_patterns)
        
        # Calculate manipulation scores
        sensationalism_score = self._calculate_keyword_score(sensationalism_found)
        fear_score = self._calculate_keyword_score(fear_found)
        exaggeration_score = self._calculate_keyword_score(exaggeration_found)
        
        # Determine flags
        is_extreme_polarity = abs(polarity) > EXTREME_POLARITY_THRESHOLD
        is_high_intensity = intensity > HIGH_INTENSITY_THRESHOLD
        is_highly_subjective = subjectivity > HIGH_SUBJECTIVITY_THRESHOLD
        
        # Has manipulation signals if any score is significant
        has_manipulation = (
            sensationalism_score > 0.2 or 
            fear_score > 0.2 or 
            exaggeration_score > 0.4 or
            is_extreme_polarity
        )
        
        return SentimentResult(
            polarity=polarity,
            intensity=intensity,
            subjectivity=subjectivity,
            positive_score=vader_scores['pos'],
            negative_score=vader_scores['neg'],
            neutral_score=vader_scores['neu'],
            compound_score=vader_scores['compound'],
            sensationalism_score=sensationalism_score,
            fear_mongering_score=fear_score,
            exaggeration_score=exaggeration_score,
            is_extreme_polarity=is_extreme_polarity,
            is_high_intensity=is_high_intensity,
            is_highly_subjective=is_highly_subjective,
            has_manipulation_signals=has_manipulation,
            sensationalism_keywords_found=sensationalism_found,
            fear_keywords_found=fear_found,
            exaggeration_keywords_found=exaggeration_found,
        )
    
    def analyze_article(
        self, 
        body: str, 
        headline: Optional[str] = None
    ) -> dict:
        """
        Analyze a complete news article including headline.
        
        Args:
            body: Article body text.
            headline: Optional headline text.
            
        Returns:
            Dictionary with body, headline, and combined analysis.
        """
        body_result = self.analyze(body)
        
        result = {
            "body_sentiment": body_result,
            "overall_polarity": body_result.polarity,
            "overall_intensity": body_result.intensity,
            "has_manipulation_signals": body_result.has_manipulation_signals,
        }
        
        if headline:
            headline_result = self.analyze(headline)
            result["headline_sentiment"] = headline_result
            
            # Headlines often more sensationalist than body
            result["headline_body_sentiment_gap"] = abs(
                headline_result.polarity - body_result.polarity
            )
            
            # Update overall flags
            result["has_manipulation_signals"] = (
                body_result.has_manipulation_signals or 
                headline_result.has_manipulation_signals
            )
            
            # Average intensity considering headline weight
            result["overall_intensity"] = (
                headline_result.intensity * 0.4 + body_result.intensity * 0.6
            )
        
        return result
    
    def _empty_result(self) -> SentimentResult:
        """Return empty result for invalid input."""
        return SentimentResult(
            polarity=0.0,
            intensity=0.0,
            subjectivity=0.0,
            positive_score=0.0,
            negative_score=0.0,
            neutral_score=1.0,
            compound_score=0.0,
            sensationalism_score=0.0,
            fear_mongering_score=0.0,
            exaggeration_score=0.0,
            is_extreme_polarity=False,
            is_high_intensity=False,
            is_highly_subjective=False,
            has_manipulation_signals=False,
            sensationalism_keywords_found=[],
            fear_keywords_found=[],
            exaggeration_keywords_found=[],
        )
    
    def get_sentiment_features(self, text: str) -> dict:
        """
        Extract sentiment features as a dictionary for feature fusion.
        
        Args:
            text: Input text.
            
        Returns:
            Dictionary of numeric features suitable for ML models.
        """
        result = self.analyze(text)
        
        return {
            "polarity": result.polarity,
            "intensity": result.intensity,
            "subjectivity": result.subjectivity,
            "positive_score": result.positive_score,
            "negative_score": result.negative_score,
            "neutral_score": result.neutral_score,
            "sensationalism_score": result.sensationalism_score,
            "fear_mongering_score": result.fear_mongering_score,
            "exaggeration_score": result.exaggeration_score,
            "is_extreme_polarity": float(result.is_extreme_polarity),
            "is_high_intensity": float(result.is_high_intensity),
            "has_manipulation_signals": float(result.has_manipulation_signals),
        }


# Convenience functions
def analyze_sentiment(text: str) -> SentimentResult:
    """Quick function to analyze sentiment of text."""
    analyzer = SentimentAnalyzer()
    return analyzer.analyze(text)


def get_polarity(text: str) -> float:
    """Quick function to get polarity score (-1 to 1)."""
    analyzer = SentimentAnalyzer()
    return analyzer.analyze(text).polarity


def has_manipulation_signals(text: str) -> bool:
    """Quick function to check for manipulation signals."""
    analyzer = SentimentAnalyzer()
    return analyzer.analyze(text).has_manipulation_signals
