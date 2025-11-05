"""
AI Engine Models Package - Zentrale AI-Model Verwaltung (FIXED VERSION)

PACKAGE STRUKTUR:
- Base Classes: BaseAIClient, AsyncBaseAIClient
- Sync Clients: OpenAI, Groq, Gemini, Anthropic
- Async Clients: AsyncOpenAI, AsyncGroq, AsyncGemini, AsyncAnthropic
- Intelligence: ModelFactory, AIIntelligenceDirector

HAUPTFEATURES:
- Multi-Model Support (OpenAI, Groq, Gemini, Anthropic)
- Sync & Async API
- Intelligente Model-Selektion
- Performance Optimization
- Cost Management
- Error Handling & Fallbacks

VERWENDUNG:
from ai_engine.models import OpenAIClient, GroqClient, GeminiClient, AnthropicClient
from ai_engine.models import AsyncOpenAIClient, AsyncGroqClient, AsyncGeminiClient, AnthropicCliet
from ai_engine.models import AIModelFactory, get_ai_director
"""

# === CORE BASE CLASSES ===

# Base Client Infrastructure
from .base_client import (
    # Abstract Base Classes
    BaseAIClient,
    
    # Data Classes & Enums
    AIRequest,
    AIResponse, 
    AIModelType,
    ModelCapability,
    ResponseFormat,
    ModelMetrics,
    
    # Pydantic Schemas
    GiftRecommendationSchema,
    PersonalityInsightSchema,
    
    # Utility Functions (nur die existierenden!)
    get_available_models,
    get_best_model_for_task
)

# Async Base Client Infrastructure
try:
    from .async_base_client import (
        # Async Base Classes
        AsyncBaseAIClient,
        
        # Async Utilities
        AsyncRateLimiter,
        
        # Comparison Functions
        compare_sync_vs_async_performance
    )
    ASYNC_BASE_AVAILABLE = True
except ImportError:
    print("⚠️  Async base client not available")
    ASYNC_BASE_AVAILABLE = False


# === SYNC AI CLIENTS ===

# OpenAI GPT-4 Client (High Quality)
try:
    from .openai_client import (
        OpenAIClient,
        create_openai_client,
        test_openai_integration
    )
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️  OpenAI client not available")
    OPENAI_AVAILABLE = False

# Groq Client (Ultra Fast)
try:
    from .groq_client import (
        GroqClient,
        create_groq_client,
        create_speed_optimized_client,
        test_groq_integration
    )
    GROQ_AVAILABLE = True
except ImportError:
    print("⚠️  Groq client not available")
    GROQ_AVAILABLE = False

# Google Gemini Client (Complex Reasoning)
try:
    from .gemini_client import (
        GeminiClient,
        create_gemini_client,
        create_reasoning_optimized_client,
        test_gemini_integration
    )
    GEMINI_AVAILABLE = True
except ImportError:
    print("⚠️  Gemini client not available")
    GEMINI_AVAILABLE = False

# Anthropic Claude Client (Ethical Reasoning)
try:
    from .anthropic_client import (
        AnthropicClient,
        create_anthropic_client,
        create_ethical_reasoning_client,
        test_anthropic_integration
    )
    ANTHROPIC_AVAILABLE = True
except ImportError:
    print("⚠️  Anthropic client not available")
    ANTHROPIC_AVAILABLE = False


# === ASYNC AI CLIENTS ===

# Async OpenAI Client (Concurrent High Quality)
try:
    from .async_openai_client import (
        AsyncOpenAIClient,
        create_async_openai_client,
        test_async_openai_integration
    )
    ASYNC_OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️  Async OpenAI client not available")
    ASYNC_OPENAI_AVAILABLE = False

# Async Groq Client (Concurrent Ultra Fast)
try:
    from .async_groq_client import (
        AsyncGroqClient,
        create_async_groq_client,
        create_speed_optimized_async_client,
        test_async_groq_integration
    )
    ASYNC_GROQ_AVAILABLE = True
except ImportError:
    print("⚠️  Async Groq client not available")
    ASYNC_GROQ_AVAILABLE = False

# Async Gemini Client (Concurrent Complex Reasoning)  
try:
    from .async_gemini_client import (
        AsyncGeminiClient,
        create_async_gemini_client,
        create_reasoning_optimized_async_client,
        test_async_gemini_integration
    )
    ASYNC_GEMINI_AVAILABLE = True
except ImportError:
    print("⚠️  Async Gemini client not available")
    ASYNC_GEMINI_AVAILABLE = False

# Async Anthropic Claude Client (Concurrent Ethical Reasoning)
try:
    from .async_anthropic_client import (
        AsyncAnthropicClient,
        create_async_anthropic_client,
        create_ethical_reasoning_async_client,
        test_async_anthropic_integration
    )
    ASYNC_ANTHROPIC_AVAILABLE = True
except ImportError:
    print("⚠️  Async Anthropic client not available")
    ASYNC_ANTHROPIC_AVAILABLE = False


# === INTELLIGENCE & FACTORY ===

# AI Model Factory & Intelligence Director
try:
    from .model_factory import (
        # Factory Classes
        AIModelFactory,
        AIIntelligenceDirector,
        
        # Task Classification
        TaskType,
        TaskPriority,
        UserContext,
        TaskPerformance,
        
        # Selection Strategies
        ModelSelectionStrategy,
        SpeedFirstStrategy,
        QualityFirstStrategy,
        CostOptimizedStrategy,
        IntelligentStrategy,
        
        # Global Functions
        get_ai_director,
        smart_gift_recommendation
    )
    FACTORY_AVAILABLE = True
except ImportError:
    print("⚠️  Model factory not available")
    FACTORY_AVAILABLE = False


# === PACKAGE METADATA ===

__version__ = "2.1.0"
__author__ = "SensationGifts AI Team"
__description__ = "AI Models Package for Multi-Model Gift Recommendations"


# === DYNAMIC PUBLIC API ===

# Build __all__ dynamically based on what's available
__all__ = [
    # === BASE CLASSES (always available) ===
    "BaseAIClient",
    "AIRequest", 
    "AIResponse",
    "AIModelType",
    "ModelCapability",
    "ResponseFormat",
    "ModelMetrics",
    "GiftRecommendationSchema",
    "PersonalityInsightSchema",
    "get_available_models",
    "get_best_model_for_task"
]

# Add async base if available
if ASYNC_BASE_AVAILABLE:
    __all__.extend([
        "AsyncBaseAIClient",
        "AsyncRateLimiter",
        "compare_sync_vs_async_performance"
    ])

# Add sync clients if available
if OPENAI_AVAILABLE:
    __all__.extend([
        "OpenAIClient",
        "create_openai_client",
        "test_openai_integration"
    ])

if GROQ_AVAILABLE:
    __all__.extend([
        "GroqClient",
        "create_groq_client", 
        "create_speed_optimized_client",
        "test_groq_integration"
    ])

if GEMINI_AVAILABLE:
    __all__.extend([
        "GeminiClient",
        "create_gemini_client",
        "create_reasoning_optimized_client",
        "test_gemini_integration"
    ])

if ANTHROPIC_AVAILABLE:
    __all__.extend([
        "AnthropicClient",
        "create_anthropic_client",
        "create_ethical_reasoning_client",
        "test_anthropic_integration"
    ])

# Add async clients if available
if ASYNC_OPENAI_AVAILABLE:
    __all__.extend([
        "AsyncOpenAIClient",
        "create_async_openai_client",
        "test_async_openai_integration"
    ])

if ASYNC_GROQ_AVAILABLE:
    __all__.extend([
        "AsyncGroqClient",
        "create_async_groq_client",
        "create_speed_optimized_async_client",
        "test_async_groq_integration"
    ])

if ASYNC_GEMINI_AVAILABLE:
    __all__.extend([
        "AsyncGeminiClient",
        "create_async_gemini_client",
        "create_reasoning_optimized_async_client",
        "test_async_gemini_integration"
    ])

if ASYNC_ANTHROPIC_AVAILABLE:
    __all__.extend([
        "AsyncAnthropicClient",
        "create_async_anthropic_client",
        "create_ethical_reasoning_async_client",
        "test_async_anthropic_integration"
    ])

# Add factory if available
if FACTORY_AVAILABLE:
    __all__.extend([
        "AIModelFactory",
        "AIIntelligenceDirector",
        "TaskType",
        "TaskPriority",
        "UserContext", 
        "TaskPerformance",
        "ModelSelectionStrategy",
        "SpeedFirstStrategy",
        "QualityFirstStrategy",
        "CostOptimizedStrategy",
        "IntelligentStrategy",
        "get_ai_director",
        "smart_gift_recommendation"
    ])


# === SIMPLIFIED MODEL REGISTRY ===

# Dynamische Model Registry nur mit verfügbaren Models
MODEL_REGISTRY = {}

if OPENAI_AVAILABLE:
    MODEL_REGISTRY[AIModelType.OPENAI_GPT4] = {
        "client_class": OpenAIClient,
        "factory_function": create_openai_client,
        "capabilities": [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.HIGH_QUALITY,
            ModelCapability.JSON_OUTPUT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION
        ],
        "strengths": ["creativity", "detailed_reasoning", "natural_language"],
        "use_cases": ["detailed_recommendations", "creative_suggestions"],
        "cost_level": "high",
        "speed_level": "medium"
    }

if GROQ_AVAILABLE:
    MODEL_REGISTRY[AIModelType.GROQ_MIXTRAL] = {
        "client_class": GroqClient,
        "factory_function": create_groq_client,
        "capabilities": [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.FAST_RESPONSE,
            ModelCapability.COST_EFFICIENT,
            ModelCapability.JSON_OUTPUT
        ],
        "strengths": ["ultra_fast", "cost_efficient", "high_throughput"],
        "use_cases": ["quick_suggestions", "real_time_chat"],
        "cost_level": "low",
        "speed_level": "very_high"
    }

if GEMINI_AVAILABLE:
    MODEL_REGISTRY[AIModelType.GEMINI_PRO] = {
        "client_class": GeminiClient,
        "factory_function": create_gemini_client,
        "capabilities": [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.JSON_OUTPUT,
            ModelCapability.HIGH_QUALITY
        ],
        "strengths": ["complex_reasoning", "multi_constraint_optimization"],
        "use_cases": ["complex_analysis", "constraint_optimization"],
        "cost_level": "medium",
        "speed_level": "medium"
    }

if ANTHROPIC_AVAILABLE:
    MODEL_REGISTRY[AIModelType.ANTHROPIC_CLAUDE] = {
        "client_class": AnthropicClient,
        "factory_function": create_anthropic_client,
        "capabilities": [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.HIGH_QUALITY,
            ModelCapability.JSON_OUTPUT
        ],
        "strengths": ["ethical_reasoning", "cultural_sensitivity"],
        "use_cases": ["ethical_analysis", "cultural_sensitivity"],
        "cost_level": "medium",
        "speed_level": "medium"
    }


# === CONVENIENCE FUNCTIONS ===

def get_model_info(model_type: AIModelType) -> dict:
    """
    Get comprehensive information about a specific AI model
    
    Args:
        model_type: AIModelType enum value
        
    Returns:
        Dictionary with model information
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type {model_type} not available or not configured")
    
    info = MODEL_REGISTRY[model_type].copy()
    info["model_type"] = model_type.value
    info["display_name"] = model_type.value.replace("_", " ").title()
    
    return info


def get_available_model_types() -> list[AIModelType]:
    """Get list of actually available and configured model types"""
    return list(MODEL_REGISTRY.keys())


def print_available_models():
    """Print all available models"""
    print("🤖 Available AI Models:")
    print("=" * 40)
    
    if not MODEL_REGISTRY:
        print("❌ No models available! Check your AI client imports and API keys.")
        return
    
    for model_type, info in MODEL_REGISTRY.items():
        model_name = model_type.value.replace("_", " ").title()
        strengths = ", ".join(info["strengths"][:2])
        print(f"✅ {model_name}: {strengths}")
    
    print(f"\nTotal: {len(MODEL_REGISTRY)} models available")


# === PACKAGE INITIALIZATION ===

def _initialize_models_package():
    """Initialize models package with validation"""
    
    available_count = len(MODEL_REGISTRY)
    
    if available_count == 0:
        print("⚠️  WARNING: No AI models available!")
        print("💡 Check your API keys and client imports")
    else:
        print(f"✅ AI Models package initialized with {available_count} available models")
    
    # Show what's available
    available_clients = []
    if OPENAI_AVAILABLE:
        available_clients.append("OpenAI")
    if GROQ_AVAILABLE:
        available_clients.append("Groq") 
    if GEMINI_AVAILABLE:
        available_clients.append("Gemini")
    if ANTHROPIC_AVAILABLE:
        available_clients.append("Anthropic")
    
    if available_clients:
        print(f"🤖 Available providers: {', '.join(available_clients)}")
    
    if FACTORY_AVAILABLE:
        print("🏭 Model Factory available")
    
    if ASYNC_BASE_AVAILABLE:
        print("⚡ Async capabilities available")


# Initialize on import
_initialize_models_package()


if __name__ == "__main__":
    # Print info when run directly
    print_available_models()
    print(f"\n🔧 Models Package Info:")
    print(f"Version: {__version__}")
    print(f"Available Models: {len(MODEL_REGISTRY)}")