"""
AI Engine Package - Multi-Model AI Integration

FEATURES:
- Multiple AI Provider Support (OpenAI, Groq, Gemini, Anthropic)
- Sync & Async Architecture
- Intelligent Model Selection
- Performance Optimization
- Error Handling & Fallbacks

VERWENDUNG:
from ai_engine.models import OpenAIClient, GroqClient, GeminiClient, AnthropicClient
from ai_engine.models import AIModelType, GiftRecommendationSchema
"""

try:
    # Only import the core types that we know exist
    from .models.base_client import (
        AIModelType,
        ModelCapability, 
        ResponseFormat,
        GiftRecommendationSchema
    )
    
    # Try to import main clients
    from .models.openai_client import OpenAIClient
    from .models.anthropic_client import AnthropicClient
    from .models.gemini_client import GeminiClient
    from .models.groq_client import GroqClient
    
    print("✅ AI Engine core imports successful")
    
except ImportError as e:
    print(f"⚠️  AI Engine import issue: {e}")
    # Continue without full imports for now


__version__ = "2.1.0"
__description__ = "Multi-Model AI Engine for Gift Recommendations"

# Minimal __all__ list
__all__ = [
    "AIModelType",
    "ModelCapability",
    "ResponseFormat", 
    "GiftRecommendationSchema"
]

# Add available clients to __all__
try:
    OpenAIClient
    __all__.append("OpenAIClient")
except NameError:
    pass

try:
    AnthropicClient
    __all__.append("AnthropicClient")
except NameError:
    pass

try:
    GeminiClient
    __all__.append("GeminiClient")
except NameError:
    pass

try:
    GroqClient  
    __all__.append("GroqClient")
except NameError:
    pass

print(f"🤖 AI Engine initialized - Available exports: {len(__all__)} items")