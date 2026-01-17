"""
BERT Encoder Module for Semantic Analysis.

Generates semantic embeddings for news articles using sentence transformers.
"""

from typing import List, Optional, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class BERTEncoder:
    """
    BERT-based encoder for generating semantic embeddings.
    
    Uses sentence-transformers for efficient, high-quality embeddings.
    Supports Apple Silicon MPS acceleration when available.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the BERT encoder.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
                       Default is MiniLM (lightweight, 22M params).
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip3 install sentence-transformers"
            )
        
        self.model_name = model_name
        self._model = None  # Lazy loading
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def encode(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string.
            
        Returns:
            Numpy array of shape (embedding_dim,).
        """
        if not text or not isinstance(text, str):
            # Return zero vector for empty input
            return np.zeros(self.embedding_dim)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input text strings.
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim).
        """
        if not texts:
            return np.array([])
        
        # Filter out empty strings
        valid_texts = [t if t else "" for t in texts]
        embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
        return embeddings
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Cosine similarity score between -1 and 1.
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        return float(similarity)
    
    def get_headline_body_consistency(self, headline: str, body: str) -> float:
        """
        Measure semantic consistency between headline and body.
        
        This is crucial for fake news detection - misleading headlines
        often have low consistency with the article body.
        
        Args:
            headline: Article headline.
            body: Article body text.
            
        Returns:
            Consistency score between 0 and 1.
            Low scores indicate potential headline-body mismatch.
        """
        similarity = self.get_similarity(headline, body)
        # Normalize from [-1, 1] to [0, 1]
        consistency = (similarity + 1) / 2
        return consistency
    
    def analyze_semantic_features(
        self, 
        text: str, 
        headline: Optional[str] = None
    ) -> dict:
        """
        Extract semantic features for fake news detection.
        
        Args:
            text: Article body text.
            headline: Optional headline text.
            
        Returns:
            Dictionary with semantic analysis results.
        """
        embedding = self.encode(text)
        
        result = {
            "embedding": embedding,
            "embedding_dim": len(embedding),
            "embedding_norm": float(np.linalg.norm(embedding)),
        }
        
        if headline:
            headline_embedding = self.encode(headline)
            consistency = self.get_headline_body_consistency(headline, text)
            
            result.update({
                "headline_embedding": headline_embedding,
                "headline_body_consistency": consistency,
                "headline_body_mismatch": consistency < 0.5,  # Flag potential issues
            })
        
        return result


# Convenience functions
def get_embedding(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Quick function to get text embedding."""
    encoder = BERTEncoder(model_name)
    return encoder.encode(text)


def get_similarity(text1: str, text2: str) -> float:
    """Quick function to get similarity between two texts."""
    encoder = BERTEncoder()
    return encoder.get_similarity(text1, text2)
