"""
AI Model Factory & Intelligence Director

INNOVATION: Intelligent Model Selection basierend auf:
- Task Type (speed vs. quality vs. creativity)
- User Context (real-time vs. detailed analysis)
- Cost Constraints (budget-aware selection)
- Performance History (learning system)
- Fallback Chains (robust error handling)


ARCHITECTURE AND DESIGN PATTERN IMPLEMENTATION:

1. FACTORY PATTERN für Model Creation (AIModelFactory):
   - Zentrale Stelle für Client-Erstellung
   - Lazy Loading: Clients werden nur bei Bedarf erstellt
   - Kapselt komplexe Erstellungslogik (API Keys, Konfiguration)
   - Ermöglicht einfache Erweiterung neuer Model-Typen
   - Realisiert in: _create_client(), get_client()

2. STRATEGY PATTERN für Model Selection (ModelSelectionStrategy):
   - Verschiedene Auswahlstrategien für Model-Selection
   - SpeedFirstStrategy, QualityFirstStrategy, CostOptimizedStrategy, IntelligentStrategy
   - Austauschbare Algorithmen ohne Code-Änderung
   - Ermöglicht dynamische Anpassung basierend auf Prioritäten
   - Realisiert in: select_model() Methoden der verschiedenen Strategy-Klassen

3. OBSERVER PATTERN für Performance Tracking (Performance Tracking):
   - TaskPerformance trackt automatisch alle Model-Performance
   - AIIntelligenceDirector beobachtet und lernt aus Performance-Daten
   - Automatische Anpassung der Strategien basierend auf Erfolgsraten
   - Realisiert in: _track_performance(), performance_history Dict

4. COMMAND PATTERN für Request Routing (Request Routing):
   - Einheitliche Schnittstelle für verschiedene AI-Requests
   - Kapselt Request-Logik in wiederverwendbare Commands
   - Ermöglicht Queue-System und Retry-Mechanismen
   - Realisiert in: recommend_gift_intelligent(), _try_fallback_chain()

ZIEL: Robuste, skalierbare und intelligente AI-Model-Verwaltung mit automatischer Optimierung.
"""

import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

from config.settings import get_settings
from .base_client import (
    BaseAIClient, AIRequest, AIResponse,
    ModelCapability, GiftRecommendationSchema
)
from ai_engine.schemas.prompt_schemas import AIModelType
from .openai_client import OpenAIClient, create_openai_client
from .groq_client import GroqClient, create_groq_client


# Claude Integration
try:
    from .anthropic_client import AnthropicClient, create_anthropic_client
except ImportError:
    pass


# === TASK CLASSIFICATION ===

class TaskType(Enum):
    """Types of AI tasks with different requirements"""
    QUICK_SUGGESTION = "quick_suggestion"        # Speed priority
    DETAILED_ANALYSIS = "detailed_analysis"     # Quality priority  
    CREATIVE_GENERATION = "creative_generation" # Creativity priority
    STRUCTURED_OUTPUT = "structured_output"     # Reliability priority
    REAL_TIME_CHAT = "real_time_chat"          # Speed + interaction
    BULK_PROCESSING = "bulk_processing"        # Cost efficiency
    VALIDATION = "validation"                  # Accuracy priority


class TaskPriority(Enum):
    """What's most important for this task?"""
    SPEED = "speed"           # Sub-second response needed
    QUALITY = "quality"       # Best possible output
    COST = "cost"            # Minimize expenses
    RELIABILITY = "reliability" # Must work every time
    CREATIVITY = "creativity"  # Novel, original ideas


class UserContext(Enum):
    """User interaction context"""
    INTERACTIVE = "interactive"     # User waiting actively
    BACKGROUND = "background"       # Background processing
    BULK_OPERATION = "bulk"        # Many requests at once
    CRITICAL_PATH = "critical"     # Must not fail


# === PERFORMANCE TRACKING ===

@dataclass
class TaskPerformance:
    """Tracks performance of models for specific tasks"""
    task_type: TaskType
    model_type: AIModelType
    success_count: int = 0
    failure_count: int = 0
    total_response_time: float = 0.0
    total_cost: float = 0.0
    last_used: Optional[datetime] = None
    user_satisfaction_scores: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def avg_response_time(self) -> float:
        return self.total_response_time / self.success_count if self.success_count > 0 else 0.0
    
    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.success_count if self.success_count > 0 else 0.0
    
    @property
    def avg_satisfaction(self) -> float:
        return sum(self.user_satisfaction_scores) / len(self.user_satisfaction_scores) if self.user_satisfaction_scores else 0.0
    
    @property
    def overall_score(self) -> float:
        """Combined performance score (0-1)"""
        return (
            self.success_rate * 0.4 +
            min(self.avg_satisfaction, 1.0) * 0.3 +
            (1.0 - min(self.avg_response_time / 10.0, 1.0)) * 0.2 +  # Faster = better
            (1.0 - min(self.avg_cost * 1000, 1.0)) * 0.1  # Cheaper = better
        )


# === MODEL SELECTION STRATEGIES ===

class ModelSelectionStrategy(ABC):
    """Abstract strategy for model selection"""
    
    @abstractmethod
    def select_model(self, 
                    task_type: TaskType,
                    priority: TaskPriority,
                    context: UserContext,
                    available_models: List[AIModelType],
                    performance_history: Dict[Tuple[TaskType, AIModelType], TaskPerformance]) -> AIModelType:
        pass


class SpeedFirstStrategy(ModelSelectionStrategy):
    """Always prioritizes fastest model"""
    
    def select_model(self, task_type, priority, context, available_models, performance_history):
        # Groq models are fastest
        if AIModelType.GROQ_LLAMA in available_models:
            return AIModelType.GROQ_LLAMA
        elif AIModelType.GROQ_MIXTRAL in available_models:
            return AIModelType.GROQ_MIXTRAL
        elif available_models:
            return available_models[0]
        raise ValueError("No models available")


class QualityFirstStrategy(ModelSelectionStrategy):
    """Always prioritizes highest quality model"""
    
    def select_model(self, task_type, priority, context, available_models, performance_history):
        # OpenAI GPT-4 is highest quality
        if AIModelType.OPENAI_GPT4 in available_models:
            return AIModelType.OPENAI_GPT4
        elif AIModelType.ANTHROPIC_CLAUDE in available_models:
            return AIModelType.ANTHROPIC_CLAUDE
        elif available_models:
            return available_models[0]
        raise ValueError("No models available")


class CostOptimizedStrategy(ModelSelectionStrategy):
    """Prioritizes cost efficiency"""
    
    def select_model(self, task_type, priority, context, available_models, performance_history):
        # Groq models are cheapest
        if AIModelType.GROQ_MIXTRAL in available_models:
            return AIModelType.GROQ_MIXTRAL
        elif AIModelType.GROQ_LLAMA in available_models:
            return AIModelType.GROQ_LLAMA
        elif AIModelType.GEMINI_FLASH in available_models:
            return AIModelType.GEMINI_FLASH
        elif available_models:
            return available_models[0]
        raise ValueError("No models available")


class IntelligentStrategy(ModelSelectionStrategy):
    """Intelligent selection based on performance history and context"""
    
    def select_model(self, task_type, priority, context, available_models, performance_history):
        # Build scores for each available model
        model_scores = {}
        
        for model_type in available_models:
            # Get performance history
            perf_key = (task_type, model_type)
            perf = performance_history.get(perf_key)
            
            if perf:
                # Use historical performance
                base_score = perf.overall_score
            else:
                # Use heuristic for new models
                base_score = self._get_heuristic_score(model_type, task_type, priority)
            
            # Adjust based on context
            context_multiplier = self._get_context_multiplier(model_type, context, priority)
            
            model_scores[model_type] = base_score * context_multiplier
        
        # Select best scoring model
        if model_scores:
            return max(model_scores.keys(), key=lambda m: model_scores[m])
        
        raise ValueError("No models available")
    
    def _get_heuristic_score(self, model_type: AIModelType, task_type: TaskType, priority: TaskPriority) -> float:
        """Heuristic scoring for models without history"""
        # Model capabilities matrix
        model_strengths = {
            AIModelType.OPENAI_GPT4: {"quality": 0.95, "speed": 0.3, "cost": 0.2, "reliability": 0.9},
            AIModelType.OPENAI_GPT35: {"quality": 0.8, "speed": 0.6, "cost": 0.4, "reliability": 0.85},
            AIModelType.GROQ_MIXTRAL: {"quality": 0.75, "speed": 0.95, "cost": 0.9, "reliability": 0.8},
            AIModelType.GROQ_LLAMA: {"quality": 0.7, "speed": 0.98, "cost": 0.95, "reliability": 0.75},
            AIModelType.GEMINI_PRO: {"quality": 0.85, "speed": 0.7, "cost": 0.6, "reliability": 0.8},
            AIModelType.GEMINI_FLASH: {"quality": 0.7, "speed": 0.85, "cost": 0.8, "reliability": 0.75}
        }
        
        strengths = model_strengths.get(model_type, {"quality": 0.5, "speed": 0.5, "cost": 0.5, "reliability": 0.5})
        
        # Weight based on priority
        if priority == TaskPriority.SPEED:
            return strengths["speed"]
        elif priority == TaskPriority.QUALITY:
            return strengths["quality"]
        elif priority == TaskPriority.COST:
            return strengths["cost"]
        elif priority == TaskPriority.RELIABILITY:
            return strengths["reliability"]
        else:
            # Balanced scoring
            return (strengths["quality"] * 0.3 + 
                   strengths["speed"] * 0.3 + 
                   strengths["cost"] * 0.2 + 
                   strengths["reliability"] * 0.2)
    
    def _get_context_multiplier(self, model_type: AIModelType, context: UserContext, priority: TaskPriority) -> float:
        """Adjust score based on user context"""
        if context == UserContext.INTERACTIVE:
            # Interactive users need speed
            if model_type in [AIModelType.GROQ_LLAMA, AIModelType.GROQ_MIXTRAL]:
                return 1.2
            return 0.8
        elif context == UserContext.BULK_OPERATION:
            # Bulk operations need cost efficiency
            if model_type in [AIModelType.GROQ_MIXTRAL, AIModelType.GROQ_LLAMA, AIModelType.GEMINI_FLASH]:
                return 1.3
            return 0.7
        elif context == UserContext.CRITICAL_PATH:
            # Critical operations need reliability
            if model_type in [AIModelType.OPENAI_GPT4, AIModelType.OPENAI_GPT35]:
                return 1.2
            return 0.9
        
        return 1.0  # No adjustment


# === MODEL FACTORY ===

class AIModelFactory:
    """
    Factory for creating and managing AI model clients
    
    Features:
    - Lazy loading of clients (only create when needed)
    - API key management
    - Health checking
    - Performance tracking
    - Intelligent fallbacks
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._clients: Dict[AIModelType, BaseAIClient] = {}
        self._client_health: Dict[AIModelType, bool] = {}
        self._last_health_check: Dict[AIModelType, datetime] = {}
        self.health_check_interval = timedelta(minutes=5)
        
    def get_client(self, model_type: AIModelType) -> BaseAIClient:
        """
        Get or create client for specific model type
        """
        if model_type not in self._clients:
            self._clients[model_type] = self._create_client(model_type)
        
        return self._clients[model_type]
    
    def _create_client(self, model_type: AIModelType) -> BaseAIClient:
        """Create specific client based on model type"""
        
        if model_type in [AIModelType.OPENAI_GPT4, AIModelType.OPENAI_GPT3_5_TURBO]:
            if not self.settings.openai_api_key:
                raise ValueError(f"OpenAI API key required for {model_type}")
            return create_openai_client(self.settings.openai_api_key, model_type)
        
        elif model_type in [AIModelType.GROQ_MIXTRAL, AIModelType.GROQ_LLAMA2]:
            if not self.settings.groq_api_key:
                raise ValueError(f"Groq API key required for {model_type}")
            return create_groq_client(self.settings.groq_api_key, model_type)
        
        # Add other providers here (Gemini, Claude, etc.)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def get_available_models(self) -> List[AIModelType]:
        """Get list of models that have API keys configured"""
        available = []
        
        if self.settings.openai_api_key:
            available.extend([AIModelType.OPENAI_GPT4, AIModelType.OPENAI_GPT3_5_TURBO])
        
        if self.settings.groq_api_key:
            available.extend([AIModelType.GROQ_MIXTRAL, AIModelType.GROQ_LLAMA2])
        
        if self.settings.gemini_api_key:
            available.extend([AIModelType.GOOGLE_GEMINI, AIModelType.GOOGLE_GEMINI_PRO])
        
        if self.settings.anthropic_api_key:
            available.append(AIModelType.ANTHROPIC_CLAUDE)
        
        return available
    
    def get_healthy_models(self) -> List[AIModelType]:
        """Get list of models that are currently healthy"""
        healthy = []
        
        for model_type in self.get_available_models():
            if self.is_model_healthy(model_type):
                healthy.append(model_type)
        
        return healthy
    
    def is_model_healthy(self, model_type: AIModelType) -> bool:
        """Check if model is healthy (cached with periodic refresh)"""
        now = datetime.now()
        last_check = self._last_health_check.get(model_type)
        
        # Use cached result if recent
        if (last_check and 
            now - last_check < self.health_check_interval and
            model_type in self._client_health):
            return self._client_health[model_type]
        
        # Perform health check
        try:
            client = self.get_client(model_type)
            # Quick test request
            response = client.generate_text("Test", max_tokens=5)
            is_healthy = response.success
        except Exception:
            is_healthy = False
        
        # Cache result
        self._client_health[model_type] = is_healthy
        self._last_health_check[model_type] = now
        
        return is_healthy


# === INTELLIGENCE DIRECTOR ===

class AIIntelligenceDirector:
    """
    Central intelligence for routing AI requests to optimal models
    
    Features:
    - Intelligent model selection
    - Performance tracking and learning
    - Fallback chains for reliability
    - Cost optimization
    - A/B testing support
    """
    
    def __init__(self, factory: AIModelFactory = None):
        self.factory = factory or AIModelFactory()
        self.performance_history: Dict[Tuple[TaskType, AIModelType], TaskPerformance] = {}
        
        # Selection strategies
        self.strategies = {
            TaskPriority.SPEED: SpeedFirstStrategy(),
            TaskPriority.QUALITY: QualityFirstStrategy(),
            TaskPriority.COST: CostOptimizedStrategy(),
            TaskPriority.RELIABILITY: IntelligentStrategy()
        }
        
        self.default_strategy = IntelligentStrategy()
        
        # Fallback chains for reliability
        self.fallback_chains = {
            TaskType.QUICK_SUGGESTION: [
                AIModelType.GROQ_LLAMA,
                AIModelType.GROQ_MIXTRAL,
                AIModelType.OPENAI_GPT35
            ],
            TaskType.DETAILED_ANALYSIS: [
                AIModelType.OPENAI_GPT4,
                AIModelType.GROQ_MIXTRAL,
                AIModelType.OPENAI_GPT35
            ],
            TaskType.STRUCTURED_OUTPUT: [
                AIModelType.OPENAI_GPT4,
                AIModelType.GROQ_MIXTRAL
            ]
        }
    
    def recommend_gift_intelligent(self,
                                 personality_profile: Dict,
                                 occasion: str,
                                 budget_range: str,
                                 relationship: str,
                                 priority: TaskPriority = TaskPriority.QUALITY,
                                 context: UserContext = UserContext.INTERACTIVE) -> Tuple[GiftRecommendationSchema, Dict[str, Any]]:
        """
        Intelligent gift recommendation with automatic model selection
        
        Returns:
            Tuple of (recommendation, metadata about the process)
        """
        task_type = TaskType.DETAILED_ANALYSIS if priority == TaskPriority.QUALITY else TaskType.QUICK_SUGGESTION
        
        # Select optimal model
        selected_model = self._select_optimal_model(task_type, priority, context)
        
        metadata = {
            "selected_model": selected_model.value,
            "task_type": task_type.value,
            "priority": priority.value,
            "context": context.value,
            "fallbacks_used": []
        }
        
        # Try primary model
        try:
            start_time = time.time()
            client = self.factory.get_client(selected_model)
            
            if hasattr(client, 'recommend_gift_with_reasoning'):
                # Use OpenAI's detailed method
                recommendation = client.recommend_gift_with_reasoning(
                    personality_profile, occasion, budget_range, relationship
                )
            else:
                # Use fast method
                recommendation = client.fast_gift_recommendation(
                    personality_profile, occasion, budget_range, relationship
                )
            
            response_time = time.time() - start_time
            
            # Track performance
            self._track_performance(task_type, selected_model, True, response_time, 0.02)  # Assume small cost
            
            metadata.update({
                "success": True,
                "response_time": response_time,
                "recommendation_confidence": recommendation.confidence
            })
            
            return recommendation, metadata
            
        except Exception as e:
            # Try fallback chain
            metadata["primary_error"] = str(e)
            return self._try_fallback_chain(task_type, personality_profile, occasion, budget_range, relationship, metadata)
    
    def _select_optimal_model(self, 
                            task_type: TaskType,
                            priority: TaskPriority,
                            context: UserContext) -> AIModelType:
        """Select the optimal model for this request"""
        
        available_models = self.factory.get_healthy_models()
        if not available_models:
            raise RuntimeError("No healthy AI models available")
        
        # Use appropriate strategy
        strategy = self.strategies.get(priority, self.default_strategy)
        
        return strategy.select_model(
            task_type, priority, context, available_models, self.performance_history
        )
    
    def _try_fallback_chain(self, 
                          task_type: TaskType,
                          personality_profile: Dict,
                          occasion: str,
                          budget_range: str,
                          relationship: str,
                          metadata: Dict) -> Tuple[GiftRecommendationSchema, Dict]:
        """Try fallback models in sequence"""
        
        fallback_chain = self.fallback_chains.get(task_type, [])
        available_models = self.factory.get_healthy_models()
        
        for model_type in fallback_chain:
            if model_type in available_models:
                try:
                    client = self.factory.get_client(model_type)
                    
                    if hasattr(client, 'fast_gift_recommendation'):
                        recommendation = client.fast_gift_recommendation(
                            personality_profile, occasion, budget_range, relationship
                        )
                    else:
                        # Create basic fallback recommendation
                        recommendation = GiftRecommendationSchema(
                            gift_name="Personalized Gift Card",
                            reasoning="Fallback recommendation when primary models failed",
                            match_score=0.6,
                            emotional_appeal="choice",
                            personalization_ideas=["Custom amount", "Favorite store"],
                            price_range=budget_range,
                            alternative_gifts=["Cash", "Experience voucher"],
                            confidence=0.4
                        )
                    
                    metadata["fallbacks_used"].append(model_type.value)
                    metadata["success"] = True
                    metadata["used_fallback"] = True
                    
                    return recommendation, metadata
                    
                except Exception as e:
                    metadata["fallbacks_used"].append(f"{model_type.value} (failed: {str(e)})")
                    continue
        
        # All fallbacks failed - return emergency recommendation
        emergency_recommendation = GiftRecommendationSchema(
            gift_name="Gift Card",
            reasoning="Emergency fallback - all AI models unavailable",
            match_score=0.5,
            emotional_appeal="flexibility",
            personalization_ideas=["Choose amount based on budget"],
            price_range=budget_range,
            alternative_gifts=["Cash", "Ask for wishlist"],
            confidence=0.2
        )
        
        metadata["success"] = False
        metadata["used_emergency_fallback"] = True
        
        return emergency_recommendation, metadata
    
    def _track_performance(self, 
                         task_type: TaskType,
                         model_type: AIModelType,
                         success: bool,
                         response_time: float,
                         cost: float):
        """Track performance for learning"""
        
        key = (task_type, model_type)
        
        if key not in self.performance_history:
            self.performance_history[key] = TaskPerformance(task_type, model_type)
        
        perf = self.performance_history[key]
        
        if success:
            perf.success_count += 1
            perf.total_response_time += response_time
            perf.total_cost += cost
        else:
            perf.failure_count += 1
        
        perf.last_used = datetime.now()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            "available_models": [m.value for m in self.factory.get_available_models()],
            "healthy_models": [m.value for m in self.factory.get_healthy_models()],
            "task_performance": {},
            "model_rankings": {}
        }
        
        # Performance by task type
        for (task_type, model_type), perf in self.performance_history.items():
            task_key = task_type.value
            if task_key not in report["task_performance"]:
                report["task_performance"][task_key] = {}
            
            report["task_performance"][task_key][model_type.value] = {
                "success_rate": perf.success_rate,
                "avg_response_time": perf.avg_response_time,
                "avg_cost": perf.avg_cost,
                "overall_score": perf.overall_score,
                "total_requests": perf.success_count + perf.failure_count
            }
        
        return report


# === GLOBAL INSTANCE ===

# Global director instance for the application
_director_instance = None

def get_ai_director() -> AIIntelligenceDirector:
    """Get global AI director instance (singleton)"""
    global _director_instance
    if _director_instance is None:
        _director_instance = AIIntelligenceDirector()
    return _director_instance


def get_async_client(model_type: AIModelType) -> BaseAIClient:
    """
    Legacy function for getting async client.
    This is a simplified version for backward compatibility.
    """
    factory = AIModelFactory()
    return factory.get_client(model_type)


# === CONVENIENCE FUNCTIONS ===

def smart_gift_recommendation(personality_profile: Dict,
                            occasion: str,
                            budget_range: str,
                            relationship: str,
                            priority: str = "quality") -> Tuple[GiftRecommendationSchema, Dict]:
    """
    Convenience function for smart gift recommendations
    
    Args:
        personality_profile: User's personality data
        occasion: Gift occasion
        budget_range: Budget as string "min-max"
        relationship: Relationship type
        priority: "speed", "quality", "cost", or "reliability"
        
    Returns:
        Tuple of (recommendation, metadata)
    """
    director = get_ai_director()
    
    priority_map = {
        "speed": TaskPriority.SPEED,
        "quality": TaskPriority.QUALITY,
        "cost": TaskPriority.COST,
        "reliability": TaskPriority.RELIABILITY
    }
    
    task_priority = priority_map.get(priority, TaskPriority.QUALITY)
    
    return director.recommend_gift_intelligent(
        personality_profile=personality_profile,
        occasion=occasion,
        budget_range=budget_range,
        relationship=relationship,
        priority=task_priority,
        context=UserContext.INTERACTIVE
    )