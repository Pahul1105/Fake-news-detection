"""
Text Preprocessing Module for Fake News Detection.

Handles text cleaning, tokenization, and normalization.
"""

import re
import string
from dataclasses import dataclass
from typing import List, Optional

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (run once)
def ensure_nltk_data():
    """Download required NLTK data if not present."""
    required = ['punkt', 'stopwords', 'punkt_tab']
    for item in required:
        try:
            nltk.data.find(f'tokenizers/{item}' if 'punkt' in item else f'corpora/{item}')
        except LookupError:
            nltk.download(item, quiet=True)


@dataclass
class PreprocessedText:
    """Container for preprocessed text data."""
    original: str
    cleaned: str
    tokens: List[str]
    sentences: List[str]
    word_count: int
    sentence_count: int
    headline: Optional[str] = None
    body: Optional[str] = None


class TextCleaner:
    """
    Text preprocessing pipeline for news articles.
    
    Handles:
    - HTML tag removal
    - Special character normalization
    - Case normalization
    - Tokenization
    - Optional stopword removal
    """
    
    def __init__(self, remove_stopwords: bool = False):
        """
        Initialize the text cleaner.
        
        Args:
            remove_stopwords: Whether to remove stopwords during tokenization.
        """
        ensure_nltk_data()
        self.remove_stopwords = remove_stopwords
        self._stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw text by removing HTML, special characters, and normalizing whitespace.
        
        Args:
            text: Raw input text.
            
        Returns:
            Cleaned text string.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Keep only letters, numbers, and basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
        
        # Normalize multiple punctuation (e.g., !!! -> !)
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_case(self, text: str) -> str:
        """
        Convert text to lowercase while preserving sentence structure.
        
        Args:
            text: Input text.
            
        Returns:
            Lowercase text.
        """
        return text.lower()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text.
            
        Returns:
            List of tokens.
        """
        tokens = word_tokenize(text.lower())
        
        # Remove pure punctuation tokens
        tokens = [t for t in tokens if t not in string.punctuation]
        
        # Remove stopwords if configured
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]
        
        return tokens
    
    def get_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text.
            
        Returns:
            List of sentences.
        """
        return sent_tokenize(text)
    
    def preprocess_article(
        self, 
        text: str, 
        headline: Optional[str] = None,
        normalize_case: bool = False
    ) -> PreprocessedText:
        """
        Full preprocessing pipeline for a news article.
        
        Args:
            text: Article body text.
            headline: Optional headline text.
            normalize_case: Whether to convert to lowercase.
            
        Returns:
            PreprocessedText object with all processed data.
        """
        # Clean the text
        cleaned = self.clean_text(text)
        
        if normalize_case:
            cleaned = self.normalize_case(cleaned)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        sentences = self.get_sentences(cleaned)
        
        # Process headline if provided
        cleaned_headline = None
        if headline:
            cleaned_headline = self.clean_text(headline)
            if normalize_case:
                cleaned_headline = self.normalize_case(cleaned_headline)
        
        return PreprocessedText(
            original=text,
            cleaned=cleaned,
            tokens=tokens,
            sentences=sentences,
            word_count=len(tokens),
            sentence_count=len(sentences),
            headline=cleaned_headline,
            body=cleaned
        )
    
    def extract_headline_body(self, article: str) -> tuple:
        """
        Extract headline and body from article text.
        
        Assumes first sentence/line is the headline.
        
        Args:
            article: Full article text.
            
        Returns:
            Tuple of (headline, body).
        """
        lines = article.strip().split('\n')
        
        if len(lines) > 1:
            headline = lines[0].strip()
            body = ' '.join(lines[1:]).strip()
        else:
            # Single block of text - use first sentence as headline
            sentences = self.get_sentences(article)
            if len(sentences) > 1:
                headline = sentences[0]
                body = ' '.join(sentences[1:])
            else:
                headline = article
                body = article
        
        return headline, body


# Convenience functions for quick usage
def clean_text(text: str) -> str:
    """Quick clean text function."""
    return TextCleaner().clean_text(text)


def tokenize(text: str) -> List[str]:
    """Quick tokenize function."""
    return TextCleaner().tokenize(text)


def preprocess_article(text: str, headline: Optional[str] = None) -> PreprocessedText:
    """Quick preprocess function."""
    return TextCleaner().preprocess_article(text, headline)
