"""
AI Engine Configuration - SensationGifts
=======================================

Verbindet deine brillante AI-Engine mit Flask Settings
- Multi-AI-Provider Configuration
- Model Selection Strategies
- Performance & Cost Optimization
- Error Handling & Fallbacks
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AIProvider(Enum):
    """Verfügbare AI Provider"""
    OPENAI = "openai"
    GROQ = "groq" 
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"

class OptimizationGoal(Enum):
    """AI Optimization Goals"""
    SPEED = "speed"
    QUALITY = "quality"
    COST = "cost"
    RELIABILITY = "reliability"

@dataclass
class AIProviderConfig:
    """Konfiguration für einen AI Provider"""
    provider: AIProvider
    api_key: Optional[str]
    models: List[str]
    default_model: str
    rate_limit_per_minute: int = 60
    max_tokens_default: int = 1000
    temperature_default: float = 0.7
    enabled: bool = True

class AIEngineConfig:
    """
    Central AI Engine Configuration Manager
    
    Verbindet deine Settings mit AI-Engine Components:
    - Model Factory
    - Model Selector  
    - Response Parser
    - Performance Tracking
    """
    
    def __init__(self, settings):
        """
        Initialize AI Config von Flask Settings
        
        Args:
            settings: get_settings() Objekt
        """
        self.settings = settings
        self._providers = {}
        self._initialize_providers()
        
        logger.info("AI Engine Configuration initialized")
    
    def _initialize_providers(self):
        """Initialisiert alle verfügbaren AI Provider"""
        
        # OpenAI Configuration
        if self.settings.openai_api_key:
            self._providers[AIProvider.OPENAI] = AIProviderConfig(
                provider=AIProvider.OPENAI,
                api_key=self.settings.openai_api_key,
                models=["gpt-4", "gpt-3.5-turbo"],
                default_model="gpt-4",
                rate_limit_per_minute=60,
                max_tokens_default=1500,
                temperature_default=0.7
            )
        
        # Groq Configuration  
        if self.settings.groq_api_key:
            self._providers[AIProvider.GROQ] = AIProviderConfig(
                provider=AIProvider.GROQ,
                api_key=self.settings.groq_api_key,
                models=["mixtral-8x7b-32768", "llama2-70b-4096"],
                default_model="mixtral-8x7b-32768",
                rate_limit_per_minute=120,  # Groq ist schneller
                max_tokens_default=2000,
                temperature_default=0.6
            )
        
        # Gemini Configuration
        if self.settings.gemini_api_key:
            self._providers[AIProvider.GEMINI] = AIProviderConfig(
                provider=AIProvider.GEMINI,
                api_key=self.settings.gemini_api_key,
                models=["gemini-pro", "gemini-pro-vision"],
                default_model="gemini-pro",
                rate_limit_per_minute=60,
                max_tokens_default=1000,
                temperature_default=0.8
            )
        
        # Anthropic Configuration
        if self.settings.anthropic_api_key:
            self._providers[AIProvider.ANTHROPIC] = AIProviderConfig(
                provider=AIProvider.ANTHROPIC,
                api_key=self.settings.anthropic_api_key,
                models=["claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                default_model="claude-3-sonnet-20240229",
                rate_limit_per_minute=50,
                max_tokens_default=1200,
                temperature_default=0.7
            )
    
    def get_available_providers(self) -> List[AIProvider]:
        """Gibt verfügbare AI Provider zurück"""
        return list(self._providers.keys())
    
    def get_provider_config(self, provider: AIProvider) -> Optional[AIProviderConfig]:
        """Holt Konfiguration für einen Provider"""
        return self._providers.get(provider)
    
    def get_default_provider(self) -> Optional[AIProvider]:
        """Bestimmt Default AI Provider basierend auf Settings"""
        
        # Preference: Quality > Speed > Cost
        if AIProvider.OPENAI in self._providers:
            return AIProvider.OPENAI
        elif AIProvider.ANTHROPIC in self._providers:
            return AIProvider.ANTHROPIC
        elif AIProvider.GROQ in self._providers:
            return AIProvider.GROQ
        elif AIProvider.GEMINI in self._providers:
            return AIProvider.GEMINI
        
        return None
    
    def get_provider_for_goal(self, goal: OptimizationGoal) -> Optional[AIProvider]:
        """Wählt besten Provider für Optimization Goal"""
        
        if goal == OptimizationGoal.SPEED:
            # Groq ist am schnellsten
            if AIProvider.GROQ in self._providers:
                return AIProvider.GROQ
            elif AIProvider.GEMINI in self._providers:
                return AIProvider.GEMINI
                
        elif goal == OptimizationGoal.QUALITY:
            # OpenAI hat beste Qualität
            if AIProvider.OPENAI in self._providers:
                return AIProvider.OPENAI
            elif AIProvider.ANTHROPIC in self._providers:
                return AIProvider.ANTHROPIC
                
        elif goal == OptimizationGoal.COST:
            # Groq ist am günstigsten
            if AIProvider.GROQ in self._providers:
                return AIProvider.GROQ
            elif AIProvider.GEMINI in self._providers:
                return AIProvider.GEMINI
                
        elif goal == OptimizationGoal.RELIABILITY:
            # OpenAI ist am zuverlässigsten
            if AIProvider.OPENAI in self._providers:
                return AIProvider.OPENAI
            elif AIProvider.ANTHROPIC in self._providers:
                return AIProvider.ANTHROPIC
        
        # Fallback to default
        return self.get_default_provider()
    
    def get_model_factory_config(self) -> Dict[str, Any]:
        """
        Konfiguration für deine AI Model Factory
        
        Returns:
            Config Dictionary für Model Factory
        """
        config = {
            "available_providers": {},
            "default_provider": self.get_default_provider().value if self.get_default_provider() else None,
            "fallback_chain": self._get_fallback_chain(),
            "rate_limiting": {
                "enabled": True,
                "default_limit": 60
            },
            "performance_tracking": {
                "enabled": True,
                "track_costs": self.settings.enable_cost_tracking,
                "track_performance": True
            }
        }
        
        # Provider-spezifische Configs
        for provider, provider_config in self._providers.items():
            config["available_providers"][provider.value] = {
                "api_key": provider_config.api_key,
                "models": provider_config.models,
                "default_model": provider_config.default_model,
                "rate_limit": provider_config.rate_limit_per_minute,
                "enabled": provider_config.enabled
            }
        
        return config
    
    def get_model_selector_config(self) -> Dict[str, Any]:
        """
        Konfiguration für deinen Model Selector
        
        Returns:
            Config Dictionary für Model Selector
        """
        return {
            "optimization_strategies": {
                OptimizationGoal.SPEED.value: self.get_provider_for_goal(OptimizationGoal.SPEED).value if self.get_provider_for_goal(OptimizationGoal.SPEED) else None,
                OptimizationGoal.QUALITY.value: self.get_provider_for_goal(OptimizationGoal.QUALITY).value if self.get_provider_for_goal(OptimizationGoal.QUALITY) else None,
                OptimizationGoal.COST.value: self.get_provider_for_goal(OptimizationGoal.COST).value if self.get_provider_for_goal(OptimizationGoal.COST) else None,
                OptimizationGoal.RELIABILITY.value: self.get_provider_for_goal(OptimizationGoal.RELIABILITY).value if self.get_provider_for_goal(OptimizationGoal.RELIABILITY) else None
            },
            "fallback_enabled": True,
            "performance_learning": True,
            "cost_optimization": self.settings.enable_cost_tracking,
            "max_cost_per_session": self.settings.max_ai_cost_per_session
        }
    
    def _get_fallback_chain(self) -> List[str]:
        """Erstellt Fallback-Kette für Reliability"""
        
        chain = []
        
        # Primary: Quality providers
        if AIProvider.OPENAI in self._providers:
            chain.append(AIProvider.OPENAI.value)
        if AIProvider.ANTHROPIC in self._providers:
            chain.append(AIProvider.ANTHROPIC.value)
        
        # Secondary: Speed providers
        if AIProvider.GROQ in self._providers:
            chain.append(AIProvider.GROQ.value)
        if AIProvider.GEMINI in self._providers:
            chain.append(AIProvider.GEMINI.value)
        
        return chain
    
    def get_ai_engine_health(self) -> Dict[str, Any]:
        """AI Engine Health Check"""
        
        health = {
            "status": "healthy",
            "available_providers": len(self._providers),
            "providers": {},
            "default_provider": self.get_default_provider().value if self.get_default_provider() else None,
            "features": {
                "multi_provider": len(self._providers) > 1,
                "cost_tracking": self.settings.enable_cost_tracking,
                "multi_model_testing": self.settings.enable_multi_model_testing,
                "ai_recommendations": self.settings.enable_ai_recommendations
            }
        }
        
        # Provider Health
        for provider, config in self._providers.items():
            health["providers"][provider.value] = {
                "enabled": config.enabled,
                "api_key_configured": bool(config.api_key),
                "models_available": len(config.models),
                "default_model": config.default_model
            }
        
        # Overall Status
        if len(self._providers) == 0:
            health["status"] = "critical"
        elif not self.settings.enable_ai_recommendations:
            health["status"] = "disabled"
        
        return health
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validiert AI Configuration"""
        
        issues = []
        warnings = []
        
        # Check if any providers are available
        if len(self._providers) == 0:
            issues.append("No AI providers configured - add at least one API key")
        
        # Check for recommended setup
        if len(self._providers) == 1:
            warnings.append("Only one AI provider configured - consider adding backup provider")
        
        # Check AI features
        if not self.settings.enable_ai_recommendations:
            warnings.append("AI recommendations are disabled in settings")
        
        # Check cost tracking
        if self.settings.max_ai_cost_per_session < 0.10:
            warnings.append("Very low AI cost limit may impact recommendation quality")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "providers_configured": len(self._providers),
            "features_enabled": {
                "ai_recommendations": self.settings.enable_ai_recommendations,
                "cost_tracking": self.settings.enable_cost_tracking,
                "multi_model_testing": self.settings.enable_multi_model_testing
            }
        }


# Global AI Config Instance
_ai_config_instance = None

def get_ai_config(settings=None):
    """
    Singleton AI Config Instance
    
    Args:
        settings: Optional Settings override
        
    Returns:
        AIEngineConfig instance
    """
    global _ai_config_instance
    
    if _ai_config_instance is None or settings is not None:
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        
        _ai_config_instance = AIEngineConfig(settings)
    
    return _ai_config_instance

def init_ai_engine(app):
    """
    Initialisiert AI Engine für Flask App
    
    Args:
        app: Flask Application
    """
    try:
        # AI Config laden
        ai_config = get_ai_config()
        
        # Flask App Config erweitern
        app.config['AI_ENGINE'] = {
            'model_factory_config': ai_config.get_model_factory_config(),
            'model_selector_config': ai_config.get_model_selector_config(),
            'health': ai_config.get_ai_engine_health()
        }
        
        logger.info("AI Engine initialized for Flask app")
        
        # Validation
        validation = ai_config.validate_configuration()
        if not validation["valid"]:
            logger.warning(f"AI Config issues: {validation['issues']}")
        if validation["warnings"]:
            logger.info(f"AI Config warnings: {validation['warnings']}")
        
    except Exception as e:
        logger.error(f"AI Engine initialization failed: {e}")
        raise

# Export
__all__ = [
    'AIProvider',
    'OptimizationGoal', 
    'AIProviderConfig',
    'AIEngineConfig',
    'get_ai_config',
    'init_ai_engine'
]