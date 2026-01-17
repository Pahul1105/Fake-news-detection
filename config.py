"""
Configuration settings for Fake News Detection System.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ===========================================
# Project Paths
# ===========================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# ===========================================
# Model Configuration
# ===========================================
# BERT model for semantic embeddings
BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight, 22M params
# Alternative: "bert-base-uncased"  # Full BERT, 110M params

# Sentiment analysis settings
SENTIMENT_MODEL = "vader"  # Options: "vader", "textblob", "transformer"

# ===========================================
# Classification Thresholds
# ===========================================
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for prediction
FAKE_NEWS_THRESHOLD = 0.5   # Probability threshold for FAKE label

# Sentiment indicators
EXTREME_POLARITY_THRESHOLD = 0.7    # |polarity| > this = extreme
HIGH_INTENSITY_THRESHOLD = 0.8      # intensity > this = very high
HIGH_SUBJECTIVITY_THRESHOLD = 0.7   # subjectivity > this = highly subjective

# ===========================================
# LLM Configuration (for explanations)
# ===========================================
# API Keys (loaded from environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# LLM Provider: "openai", "google", or "none"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Model names
OPENAI_MODEL = "gpt-3.5-turbo"
GOOGLE_MODEL = "gemini-pro"

# ===========================================
# Sensationalism Detection Keywords
# ===========================================
SENSATIONALISM_KEYWORDS = [
    "shocking", "breaking", "urgent", "exclusive", "alert",
    "you won't believe", "must see", "incredible", "unbelievable",
    "exposed", "scandal", "secret", "conspiracy", "cover-up"
]

FEAR_MONGERING_KEYWORDS = [
    "dangerous", "deadly", "kill", "destroy", "threat",
    "warning", "crisis", "emergency", "disaster", "terrifying",
    "alarming", "devastating", "catastrophic"
]

EXAGGERATION_KEYWORDS = [
    "always", "never", "everyone", "nobody", "everything",
    "nothing", "definitely", "absolutely", "completely", "totally"
]

# ===========================================
# Logging Configuration
# ===========================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
SHOW_PIPELINE_STEPS = True  # Show step-by-step progress in CLI

# ===========================================
# Sample Data
# ===========================================
SAMPLE_ARTICLES_PATH = DATA_DIR / "sample_articles.json"
