"""
Model Selector - Intelligent AI-Model Selection Engine
=====================================================

Intelligente Orchestrierung und Auswahl von AI-Models für optimale Performance.
Berücksichtigt Performance, Kosten, Verfügbarkeit und Request-Kontext.

Core Features:
- Performance-based model selection
- Cost-performance optimization
- Dynamic load balancing
- Real-time health monitoring
- Context-aware model matching
- Predictive fallback strategies
- A/B testing support
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass, field
import logging
import statistics
from collections import defaultdict, deque

from ai_engine.schemas import (
    # Core Schemas
    GiftRecommendationRequest,
    AIModelType,
    PromptOptimizationGoal,
    
    # Performance Schemas
    AIModelPerformanceMetrics,
)


# =============================================================================
# MODEL SELECTION STRATEGY TYPES
# =============================================================================

class SelectionStrategy(str, Enum):
    """Strategien für Model-Auswahl"""
    PERFORMANCE_OPTIMIZED = "performance_optimized"     # Beste Performance
    COST_OPTIMIZED = "cost_optimized"                   # Niedrigste Kosten
    SPEED_OPTIMIZED = "speed_optimized"                 # Schnellste Antwort
    QUALITY_OPTIMIZED = "quality_optimized"             # Beste Qualität
    BALANCED = "balanced"                               # Ausgewogene Optimierung
    CONTEXT_ADAPTIVE = "context_adaptive"               # Kontext-basierte Auswahl
    LOAD_BALANCED = "load_balanced"                     # Load-Distribution
    EXPERIMENTAL = "experimental"                        # A/B Testing


class ModelHealthStatus(str, Enum):
    """Gesundheitsstatus von AI-Models"""
    HEALTHY = "healthy"                 # Alles optimal
    DEGRADED = "degraded"              # Performance-Probleme
    UNSTABLE = "unstable"              # Häufige Fehler
    OVERLOADED = "overloaded"          # Zu viele Requests
    UNAVAILABLE = "unavailable"        # Nicht erreichbar
    MAINTENANCE = "maintenance"        # Wartungsmodus


class RequestComplexity(str, Enum):
    """Komplexität von Requests für Model-Matching"""
    SIMPLE = "simple"           # Einfache Empfehlungen
    MODERATE = "moderate"       # Standard-Komplexität
    COMPLEX = "complex"         # Multi-Factor Analysis
    EXPERT = "expert"          # Höchste Komplexität
    EMERGENCY = "emergency"     # Zeit-kritisch
    BULK = "bulk"              # Viele Requests


# =============================================================================
# MODEL PERFORMANCE TRACKING
# =============================================================================

@dataclass
class ModelPerformanceSnapshot:
    """Performance-Snapshot eines AI-Models zu einem Zeitpunkt"""
    
    model_type: AIModelType
    timestamp: datetime
    
    # Performance Metrics
    response_time_ms: int
    success_rate: float          # 0.0-1.0
    quality_score: float         # 0.0-1.0
    user_satisfaction: float     # 0.0-1.0
    
    # Cost Metrics
    cost_per_request: Decimal
    tokens_per_request: int
    
    # Load Metrics
    current_load: int           # Current requests
    max_capacity: int           # Maximum capacity
    queue_length: int           # Waiting requests
    
    # Error Metrics
    error_rate: float          # 0.0-1.0
    timeout_rate: float        # 0.0-1.0
    
    # Context
    request_type: str
    complexity_level: RequestComplexity


@dataclass
class ModelCapabilities:
    """Capabilities und Stärken eines AI-Models"""
    
    model_type: AIModelType
    
    # Performance Characteristics
    avg_response_time_ms: int
    max_quality_score: float
    reliability_score: float    # 0.0-1.0
    
    # Specializations
    best_for_speed: bool = False
    best_for_quality: bool = False
    best_for_cost: bool = False
    best_for_creativity: bool = False
    best_for_reasoning: bool = False
    best_for_cultural_sensitivity: bool = False
    
    # Capacity Limits
    max_concurrent_requests: int = 100
    rate_limit_per_minute: int = 60
    max_tokens_per_request: int = 4000
    
    # Cost Structure
    cost_per_1k_tokens: Decimal = Decimal('0.001')
    
    # Context Specializations
    optimal_complexities: List[RequestComplexity] = field(default_factory=list)
    supported_features: List[str] = field(default_factory=list)


# =============================================================================
# INTELLIGENT MODEL SELECTOR ENGINE
# =============================================================================

class ModelSelector:
    """
    Intelligent AI-Model Selection Engine
    
    Core Responsibilities:
    - Performance-based model selection
    - Real-time health monitoring
    - Cost-performance optimization
    - Dynamic load balancing
    - Predictive fallback strategies
    """
    
    def __init__(self):

        self.logger = logging.getLogger(__name__)

        # Initialize capabilities first, then derive real models from keys
        self.model_capabilities = self._initialize_model_capabilities()
        self._real_models: List[AIModelType] = list(self.model_capabilities.keys())
        
        # Health & tracking state only for supported models
        self.model_health_status = {model: ModelHealthStatus.HEALTHY for model in self._real_models}
        
        # Performance Tracking
        self.performance_history: Dict[AIModelType, deque] = {
            model: deque(maxlen=1000) for model in self._real_models
        }
        self.current_performance: Dict[AIModelType, ModelPerformanceSnapshot] = {}
        
        # Load Balancing
        self.current_loads: Dict[AIModelType, int] = defaultdict(int)
        self.request_queues: Dict[AIModelType, int] = defaultdict(int)
        
        # Selection Statistics
        self.selection_statistics = {
            "total_selections": 0,
            "strategy_usage": defaultdict(int),
            "model_usage": defaultdict(int),
            "fallback_triggered": 0,
            "selection_time_ms": deque(maxlen=100)
        }
        
        # A/B Testing
        self.ab_test_configs: Dict[str, Dict[str, Any]] = {}
        self.ab_test_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # 🚀 ENHANCED OPTIMIZATION FEATURES
        self.optimization_state = {
            "learning_enabled": True,
            "adaptive_strategies": True,
            "performance_baseline": 0.8,
            "cost_optimization": True,
            "quality_optimization": True
        }
        
        # Advanced Selection Features
        self.context_aware_selection = True
        self.predictive_selection = True
        self.ensemble_selection = True

    
    def select_optimal_model(
        self,
        request: GiftRecommendationRequest,
        optimization_goal: PromptOptimizationGoal,
        strategy: SelectionStrategy = None,
        fallback_models: Optional[List[AIModelType]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Haupt-Methode: Wählt optimales AI-Model für Request
        """
        # Strategie-Default defensiv setzen (keine neuen Funktionen)
        strategy = strategy or SelectionStrategy.BALANCED

        selection_start = datetime.now()
        
        try:
            # 1. REQUEST ANALYSIS
            request_analysis = self._analyze_request_complexity(request, context)
            
            # 2. MODEL HEALTH CHECK (supported models only)
            healthy_models = [m for m in self._get_healthy_models() if m in self.model_capabilities]
            
            if not healthy_models:
                return self._handle_no_healthy_models(request, optimization_goal)
            
            # 3. ENHANCED STRATEGY EXECUTION
            if self.optimization_state["adaptive_strategies"]:
                # Use adaptive strategy selection
                strategy = self._select_adaptive_strategy(request_analysis, optimization_goal, context)
            
            selection_result = self._execute_enhanced_selection_strategy(
                strategy, request_analysis, optimization_goal, healthy_models, context
            )
            
            # Handle AUTO_SELECT case specifically
            selected_model = selection_result.get("primary_model", AIModelType.AUTO_SELECT)
            
            # ——— Normalisierung: str → Enum (wenn möglich) ———
            if not isinstance(selected_model, AIModelType):
                try:
                    selected_model = AIModelType(str(selected_model))
                except Exception:
                    selected_model = AIModelType.AUTO_SELECT

            # Map unsupported variants to supported ones
            if selected_model == AIModelType.OPENAI_GPT4_TURBO:
                selected_model = AIModelType.OPENAI_GPT4
                selection_result["primary_model"] = selected_model

            # ——— AUTO_SELECT auflösen ———
            is_auto = (
                (isinstance(selected_model, AIModelType) and selected_model == AIModelType.AUTO_SELECT)
                or (isinstance(selected_model, str) and str(selected_model).lower() == "auto_select")
            )

            if is_auto:
                self.logger.info("🔄 Resolving AUTO_SELECT with goal: %s", optimization_goal)
                # Goal robust in String wandeln (Enum.value → Enum.name → str)
                goal_str = (
                    getattr(optimization_goal, "value", None)
                    or getattr(optimization_goal, "name", None)
                    or str(optimization_goal)
                )
                goal_str = str(goal_str).lower()

                if "speed" in goal_str:
                    selected_model = AIModelType.GROQ_MIXTRAL
                elif "quality" in goal_str:
                    selected_model = AIModelType.OPENAI_GPT4
                elif "cost" in goal_str:
                    selected_model = AIModelType.GROQ_MIXTRAL
                else:
                    selected_model = AIModelType.OPENAI_GPT4

                selection_result["primary_model"] = selected_model
                self.logger.info("✅ AUTO_SELECT resolved to %s", getattr(selected_model, "value", selected_model))

            # 4. CAPACITY CHECK
            if not self._check_model_capacity(selected_model):
                # Try alternative models
                for alt_model in selection_result.get("alternatives", []):
                    if alt_model != AIModelType.AUTO_SELECT and alt_model in self.model_capabilities and self._check_model_capacity(alt_model):
                        selection_result["primary_model"] = alt_model
                        selection_result["capacity_fallback_used"] = True
                        selected_model = alt_model
                        break
                else:
                    # All models at capacity - use least loaded (excluding AUTO_SELECT)
                    available_models = [m for m in healthy_models if m != AIModelType.AUTO_SELECT]
                    if available_models:
                        selected_model = self._get_least_loaded_model(available_models)
                        selection_result["primary_model"] = selected_model
                        selection_result["load_balancing_used"] = True
            
            # 5. FALLBACK PREPARATION (exclude AUTO_SELECT)
            fallback_chain = self._prepare_fallback_chain(
                selected_model, 
                [m for m in (fallback_models or selection_result.get("alternatives", [])) if m != AIModelType.AUTO_SELECT and m in self.model_capabilities]
            )
            
            # 6. PERFORMANCE PREDICTION
            predicted_performance = self._predict_performance(
                selected_model, request_analysis
            )
            
            # 7. A/B TEST CHECK
            ab_test_adjustment = self._check_ab_test_override(
                selected_model, request, context
            )
            if ab_test_adjustment:
                selection_result.update(ab_test_adjustment)
            
            # 8. FINAL RESULT ASSEMBLY
            selection_time = (datetime.now() - selection_start).total_seconds() * 1000
            
            result = {
                "selected_model": selected_model,  # ✅ Guaranteed supported model
                "selection_reasoning": selection_result.get("reasoning", ""),
                "alternatives": [m for m in selection_result.get("alternatives", []) if m != AIModelType.AUTO_SELECT and m in self.model_capabilities],
                "fallback_chain": fallback_chain,
                "predicted_performance": predicted_performance,
                "selection_metadata": {
                    "strategy_used": strategy,
                    "optimization_goal": optimization_goal,
                    "request_complexity": request_analysis["complexity"],
                    "selection_time_ms": int(selection_time),
                    "model_health_considered": len(healthy_models),
                    "capacity_checked": True,
                    "ab_test_active": ab_test_adjustment is not None,
                    "auto_select_resolved": selection_result.get("auto_select_resolved", False)
                }
            }
            
            # 9. UPDATE STATISTICS
            self._update_selection_statistics(result, strategy, selection_time)
            
            # 10. RESERVE CAPACITY
            self._reserve_model_capacity(selected_model)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            emergency_result = self._create_emergency_fallback(request, optimization_goal)
            # Ensure we return the emergency result correctly
            return emergency_result

    def _select_adaptive_strategy(self, request_analysis: Dict[str, Any], optimization_goal, context: Optional[Dict[str, Any]]) -> SelectionStrategy:
        """Select optimal strategy based on request analysis and optimization goal"""
        try:
            complexity = request_analysis.get("complexity", RequestComplexity.MODERATE)
            complexity_score = request_analysis.get("complexity_score", 0)
            
            # Goal-based strategy selection
            goal_str = str(optimization_goal).lower()
            
            if "speed" in goal_str or complexity == RequestComplexity.EMERGENCY:
                return SelectionStrategy.SPEED_OPTIMIZED
            elif "quality" in goal_str or complexity == RequestComplexity.EXPERT:
                return SelectionStrategy.QUALITY_OPTIMIZED
            elif "cost" in goal_str:
                return SelectionStrategy.COST_OPTIMIZED
            elif complexity_score >= 4:
                return SelectionStrategy.CONTEXT_ADAPTIVE
            elif context and context.get("load_balancing_needed"):
                return SelectionStrategy.LOAD_BALANCED
            else:
                return SelectionStrategy.BALANCED
                
        except Exception as e:
            self.logger.warning(f"Adaptive strategy selection failed: {e}")
            return SelectionStrategy.BALANCED

    def _execute_enhanced_selection_strategy(self, strategy: SelectionStrategy, request_analysis: Dict[str, Any], 
                                           optimization_goal, healthy_models: List[AIModelType], 
                                           context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced strategy execution with advanced optimization"""
        try:
            # Get base strategy result
            base_result = self._execute_selection_strategy(
                strategy, request_analysis, optimization_goal, healthy_models, context
            )
            
            # Apply enhancements based on optimization state
            if self.optimization_state["cost_optimization"]:
                base_result = self._apply_cost_optimization(base_result, request_analysis, healthy_models)
            
            if self.optimization_state["quality_optimization"]:
                base_result = self._apply_quality_optimization(base_result, request_analysis, healthy_models)
            
            if self.context_aware_selection:
                base_result = self._apply_context_aware_selection(base_result, context, request_analysis)
            
            if self.predictive_selection:
                base_result = self._apply_predictive_selection(base_result, request_analysis, healthy_models)
            
            return base_result
            
        except Exception as e:
            self.logger.warning(f"Enhanced strategy execution failed: {e}")
            # Fallback to standard strategy
            return self._execute_selection_strategy(
                strategy, request_analysis, optimization_goal, healthy_models, context
            )

    def _apply_cost_optimization(self, result: Dict[str, Any], request_analysis: Dict[str, Any], 
                               healthy_models: List[AIModelType]) -> Dict[str, Any]:
        """Apply cost optimization to model selection"""
        try:
            # Calculate cost efficiency for each model
            cost_scores = {}
            for model in healthy_models:
                if model in self.model_capabilities:
                    capabilities = self.model_capabilities[model]
                    # ✅ FIX: Safe Decimal to float conversion
                    cost_per_1k = capabilities.cost_per_1k_tokens
                    if hasattr(cost_per_1k, 'quantize'):  # Check if it's a Decimal
                        cost_per_1k_float = float(cost_per_1k)
                    else:
                        cost_per_1k_float = float(cost_per_1k) if cost_per_1k else 0.0
                    # Cost per quality point
                    cost_per_quality = cost_per_1k_float / float(capabilities.max_quality_score)
                    cost_scores[model] = 1.0 / (1.0 + cost_per_quality)  # Higher is better
            
            # Adjust selection if cost-optimized model is significantly better
            if cost_scores:
                best_cost_model = max(cost_scores.items(), key=lambda x: x[1])
                current_model = result.get("primary_model")
                
                if (current_model != best_cost_model[0] and 
                    best_cost_model[1] > cost_scores.get(current_model, 0) * 1.2):  # 20% better
                    result["primary_model"] = best_cost_model[0]
                    result["cost_optimization_applied"] = True
                    result["cost_savings_estimate"] = self._calculate_cost_savings(current_model, best_cost_model[0])
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Cost optimization failed: {e}")
            return result

    def _apply_quality_optimization(self, result: Dict[str, Any], request_analysis: Dict[str, Any], 
                                  healthy_models: List[AIModelType]) -> Dict[str, Any]:
        """Apply quality optimization to model selection"""
        try:
            # Calculate quality scores for each model
            quality_scores = {}
            for model in healthy_models:
                if model in self.model_capabilities:
                    capabilities = self.model_capabilities[model]
                    # Weighted quality score
                    quality_score = (
                        capabilities.max_quality_score * 0.4 +
                        capabilities.reliability_score * 0.3 +
                        (1.0 - capabilities.avg_response_time_ms / 10000.0) * 0.3  # Speed factor
                    )
                    quality_scores[model] = quality_score
            
            # Adjust selection if quality-optimized model is significantly better
            if quality_scores:
                best_quality_model = max(quality_scores.items(), key=lambda x: x[1])
                current_model = result.get("primary_model")
                
                if (current_model != best_quality_model[0] and 
                    best_quality_model[1] > quality_scores.get(current_model, 0) * 1.15):  # 15% better
                    result["primary_model"] = best_quality_model[0]
                    result["quality_optimization_applied"] = True
                    result["quality_improvement_estimate"] = best_quality_model[1] - quality_scores.get(current_model, 0)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Quality optimization failed: {e}")
            return result

    def _apply_context_aware_selection(self, result: Dict[str, Any], context: Optional[Dict[str, Any]], 
                                     request_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply context-aware model selection"""
        try:
            if not context:
                return result
            
            # Context-based model preferences
            context_preferences = {}
            
            # Time-sensitive requests
            if context.get("time_constraint") in ["emergency", "last_minute"]:
                context_preferences[AIModelType.GROQ_MIXTRAL] = 0.9  # Fast model
                context_preferences[AIModelType.OPENAI_GPT4] = 0.7
            
            # High-quality requests
            if request_analysis.get("complexity") == RequestComplexity.EXPERT:
                context_preferences[AIModelType.OPENAI_GPT4] = 0.9  # Best quality
                context_preferences[AIModelType.ANTHROPIC_CLAUDE] = 0.8
            
            # Cultural sensitivity
            if context.get("cultural_context"):
                context_preferences[AIModelType.ANTHROPIC_CLAUDE] = 0.9  # Good at cultural sensitivity
                context_preferences[AIModelType.OPENAI_GPT4] = 0.8
            
            # Apply context preferences
            if context_preferences:
                current_model = result.get("primary_model")
                best_context_model = max(context_preferences.items(), key=lambda x: x[1])
                
                if (current_model != best_context_model[0] and 
                    best_context_model[1] > 0.8):  # High confidence
                    result["primary_model"] = best_context_model[0]
                    result["context_aware_selection_applied"] = True
                    result["context_reasoning"] = f"Selected {best_context_model[0]} based on context analysis"
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Context-aware selection failed: {e}")
            return result

    def _apply_predictive_selection(self, result: Dict[str, Any], request_analysis: Dict[str, Any], 
                                  healthy_models: List[AIModelType]) -> Dict[str, Any]:
        """Apply predictive model selection based on historical performance"""
        try:
            # Predict performance for each model based on historical data
            predicted_performance = {}
            
            for model in healthy_models:
                if model in self.performance_history and len(self.performance_history[model]) > 0:
                    # Calculate predicted performance based on recent history
                    recent_performance = list(self.performance_history[model])[-10:]  # Last 10 requests
                    
                    if recent_performance:
                        avg_quality = sum(p.quality_score for p in recent_performance) / len(recent_performance)
                        avg_success = sum(p.success_rate for p in recent_performance) / len(recent_performance)
                        predicted_performance[model] = (avg_quality + avg_success) / 2
            
            # Adjust selection if predicted performance suggests a better model
            if predicted_performance:
                best_predicted_model = max(predicted_performance.items(), key=lambda x: x[1])
                current_model = result.get("primary_model")
                
                if (current_model != best_predicted_model[0] and 
                    best_predicted_model[1] > predicted_performance.get(current_model, 0) * 1.1):  # 10% better
                    result["primary_model"] = best_predicted_model[0]
                    result["predictive_selection_applied"] = True
                    result["predicted_performance"] = best_predicted_model[1]
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Predictive selection failed: {e}")
            return result

    def _calculate_cost_savings(self, current_model: AIModelType, optimized_model: AIModelType) -> float:
        """Calculate estimated cost savings from model optimization"""
        try:
            if (current_model in self.model_capabilities and 
                optimized_model in self.model_capabilities):
                
                current_cost = float(self.model_capabilities[current_model].cost_per_1k_tokens)
                optimized_cost = float(self.model_capabilities[optimized_model].cost_per_1k_tokens)
                
                if current_cost > 0:
                    savings_percent = (current_cost - optimized_cost) / current_cost
                    return max(0.0, savings_percent)
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Cost savings calculation failed: {e}")
            return 0.0

    def _analyze_request_complexity(
        self, 
        request: GiftRecommendationRequest, 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analysiert Request-Komplexität für optimale Model-Auswahl
        """
        
        complexity_factors = 0
        complexity_details = []
        
        # ✅ FIX: Sichere Personality Data Extraktion
        personality_data = request.personality_data
        
        # ✅ FIXED: Handle different personality_data formats
        if isinstance(personality_data, dict):
            # Dict format - extract scores safely
            if "personality_scores" in personality_data:
                # Nested format: {"personality_scores": {"openness": 4.2, ...}}
                scores_data = personality_data["personality_scores"]
            elif "big_five_scores" in personality_data:
                # Alternative nested format: {"big_five_scores": {"openness": 4.2, ...}}
                scores_data = personality_data["big_five_scores"]
            else:
                # Flat format: {"openness": 4.2, "conscientiousness": 3.8, ...}
                scores_data = personality_data
            
            # Extract Big Five scores safely
            big_five_scores = []
            big_five_traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
            for trait in big_five_traits:
                if trait in scores_data and scores_data[trait] is not None:
                    big_five_scores.append(float(scores_data[trait]))
            
            # Extract Limbic scores safely
            limbic_scores = []
            limbic_traits = ["stimulanz", "dominanz", "balance"]
            for trait in limbic_traits:
                if trait in scores_data and scores_data[trait] is not None:
                    limbic_scores.append(float(scores_data[trait]))
        
        else:
            # Object format - try to access attributes
            try:
                if hasattr(personality_data, 'personality_scores'):
                    scores_obj = personality_data.personality_scores
                    big_five_scores = [
                        getattr(scores_obj, trait, None) 
                        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
                        if getattr(scores_obj, trait, None) is not None
                    ]
                    limbic_scores = [
                        getattr(scores_obj, trait, None)
                        for trait in ["stimulanz", "dominanz", "balance"] 
                        if getattr(scores_obj, trait, None) is not None
                    ]
                else:
                    # Fallback to empty lists
                    big_five_scores = []
                    limbic_scores = []
            except Exception:
                # Safe fallback
                big_five_scores = []
                limbic_scores = []

        # High personality variance = complex analysis needed
        all_scores = big_five_scores + limbic_scores
        if all_scores and len(all_scores) > 1:
            import statistics
            if statistics.stdev(all_scores) > 0.3:  # Adjusted for 0-1 scale
                complexity_factors += 2
        
        # Multiple constraints
        constraints = 0
        
        # ✅ FIXED: Safe constraint checking
        try:
            # Budget constraints
            if hasattr(request, 'budget_min') and request.budget_min:
                constraints += 1
            if hasattr(request, 'budget_max') and request.budget_max:
                constraints += 1
            
            # Category constraints  
            if hasattr(request, 'exclude_categories') and request.exclude_categories:
                constraints += 1
            if hasattr(request, 'gift_categories') and request.gift_categories:
                constraints += 1
            
            # Context constraints
            if hasattr(request, 'cultural_context') and request.cultural_context:
                constraints += 1
            if hasattr(request, 'additional_context') and request.additional_context:
                constraints += 1
        except AttributeError:
            # If request doesn't have expected attributes, skip constraint counting
            pass
        
        if constraints >= 3:
            complexity_factors += 1
            complexity_details.append("multiple_constraints")
        
        # Number of recommendations
        try:
            num_recs = getattr(request, 'number_of_recommendations', 5)
            if num_recs > 5:
                complexity_factors += 1
                complexity_details.append("many_recommendations")
        except AttributeError:
            pass
        
        # Cultural context
        try:
            if hasattr(request, 'cultural_context') and request.cultural_context and len(str(request.cultural_context)) > 100:
                complexity_factors += 1
                complexity_details.append("cultural_context")
        except (AttributeError, TypeError):
            pass
        
        # Time sensitivity (from context)
        if context and context.get("time_constraint") in ["emergency", "last_minute"]:
            complexity_factors += 2
            complexity_details.append("time_sensitive")
        
        # Determine complexity level
        if complexity_factors >= 5:
            complexity = RequestComplexity.EXPERT
        elif complexity_factors >= 3:
            complexity = RequestComplexity.COMPLEX
        elif complexity_factors >= 1:
            complexity = RequestComplexity.MODERATE
        else:
            complexity = RequestComplexity.SIMPLE
        
        # Special cases
        if context and context.get("time_constraint") == "emergency":
            complexity = RequestComplexity.EMERGENCY
        
        return {
            "complexity": complexity,
            "complexity_score": complexity_factors,
            "complexity_details": complexity_details,
            "estimated_processing_time": self._estimate_processing_time(complexity),
            "recommended_models": self._get_models_for_complexity(complexity)
        }
    
    def _get_healthy_models(self) -> List[AIModelType]:
        """
        Gibt Liste gesunder Models zurück
        """
        healthy_models = []
        
        for model, status in self.model_health_status.items():
            if status in [ModelHealthStatus.HEALTHY, ModelHealthStatus.DEGRADED]:
                # Check recent performance
                if self._has_recent_good_performance(model):
                    healthy_models.append(model)
        
        # Always ensure at least one model is available
        if not healthy_models:
            # Emergency fallback - use best available model
            best_available = self._get_best_available_emergency_model()
            if best_available:
                healthy_models = [best_available]
        
        return healthy_models
    
    def _execute_selection_strategy(
        self,
        strategy: SelectionStrategy,
        request_analysis: Dict[str, Any],
        optimization_goal: PromptOptimizationGoal,
        healthy_models: List[AIModelType],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Führt gewählte Selection-Strategy aus
        """
        
        if strategy == SelectionStrategy.PERFORMANCE_OPTIMIZED:
            return self._select_by_performance(healthy_models, request_analysis)
            
        elif strategy == SelectionStrategy.COST_OPTIMIZED:
            return self._select_by_cost(healthy_models, request_analysis)
            
        elif strategy == SelectionStrategy.SPEED_OPTIMIZED:
            return self._select_by_speed(healthy_models, request_analysis)
            
        elif strategy == SelectionStrategy.QUALITY_OPTIMIZED:
            return self._select_by_quality(healthy_models, request_analysis)
            
        elif strategy == SelectionStrategy.BALANCED:
            return self._select_balanced(healthy_models, request_analysis, optimization_goal)
            
        elif strategy == SelectionStrategy.CONTEXT_ADAPTIVE:
            return self._select_context_adaptive(healthy_models, request_analysis, context)
            
        elif strategy == SelectionStrategy.LOAD_BALANCED:
            return self._select_load_balanced(healthy_models, request_analysis)
            
        elif strategy == SelectionStrategy.EXPERIMENTAL:
            return self._select_experimental(healthy_models, request_analysis)
            
        else:
            # Default to balanced
            return self._select_balanced(healthy_models, request_analysis, optimization_goal)
    
    # Strategy Implementation Methods
    def _select_by_performance(self, models: List[AIModelType], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Wählt Model mit bester Overall-Performance"""
        
        model_scores = {}
        
        for model in models:
            # Get recent performance metrics
            recent_metrics = self._get_recent_performance_metrics(model, hours=24)
            
            if recent_metrics:
                # Calculate composite performance score
                avg_quality = statistics.mean([m.quality_score for m in recent_metrics if m.quality_score])
                avg_satisfaction = statistics.mean([m.user_satisfaction for m in recent_metrics if m.user_satisfaction])
                avg_success_rate = statistics.mean([m.success_rate for m in recent_metrics])
                
                performance_score = (avg_quality * 0.4 + avg_satisfaction * 0.3 + avg_success_rate * 0.3)
                model_scores[model] = performance_score
            else:
                # Use baseline capabilities
                capabilities = self.model_capabilities[model]
                model_scores[model] = capabilities.reliability_score
        
        # Sort by performance score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "primary_model": sorted_models[0][0],
            "alternatives": [model for model, _ in sorted_models[1:3]],
            "reasoning": f"Selected {sorted_models[0][0].value} for highest performance score: {sorted_models[0][1]:.3f}",
            "performance_scores": model_scores
        }
    
    def _select_by_cost(self, models: List[AIModelType], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Wählt kostengünstigstes Model"""
        
        model_costs = {}
        
        for model in models:
            capabilities = self.model_capabilities[model]
            # Estimate cost based on complexity
            estimated_tokens = self._estimate_tokens_for_complexity(analysis["complexity"])
            # ✅ FIX: Safe Decimal to float conversion
            cost_per_1k = capabilities.cost_per_1k_tokens
            if hasattr(cost_per_1k, 'quantize'):  # Check if it's a Decimal
                cost_per_1k_float = float(cost_per_1k)
            else:
                cost_per_1k_float = float(cost_per_1k) if cost_per_1k else 0.0
            estimated_cost = cost_per_1k_float * (float(estimated_tokens) / 1000.0)
            model_costs[model] = estimated_cost
        
        # Sort by cost (lowest first)
        sorted_models = sorted(model_costs.items(), key=lambda x: x[1])
        
        return {
            "primary_model": sorted_models[0][0],
            "alternatives": [model for model, _ in sorted_models[1:3]],
            "reasoning": f"Selected {sorted_models[0][0].value} for lowest cost: €{sorted_models[0][1]:.4f}",
            "estimated_costs": model_costs
        }
    
    def _select_by_speed(self, models: List[AIModelType], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Wählt schnellstes Model"""
        
        model_speeds = {}
        
        for model in models:
            # Get recent response times
            recent_metrics = self._get_recent_performance_metrics(model, hours=6)
            
            if recent_metrics:
                avg_response_time = statistics.mean([m.response_time_ms for m in recent_metrics])
            else:
                capabilities = self.model_capabilities[model]
                avg_response_time = capabilities.avg_response_time_ms
            
            model_speeds[model] = avg_response_time
        
        # Sort by speed (fastest first)
        sorted_models = sorted(model_speeds.items(), key=lambda x: x[1])
        
        return {
            "primary_model": sorted_models[0][0],
            "alternatives": [model for model, _ in sorted_models[1:3]],
            "reasoning": f"Selected {sorted_models[0][0].value} for fastest response: {sorted_models[0][1]:.0f}ms",
            "response_times": model_speeds
        }
    
    def _select_by_quality(self, models: List[AIModelType], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Wählt Model mit höchster Qualität"""
        
        model_qualities = {}
        
        for model in models:
            # Get recent quality metrics
            recent_metrics = self._get_recent_performance_metrics(model, hours=24)
            
            if recent_metrics:
                quality_scores = [m.quality_score for m in recent_metrics if m.quality_score]
                if quality_scores:
                    avg_quality = statistics.mean(quality_scores)
                else:
                    avg_quality = self.model_capabilities[model].max_quality_score
            else:
                avg_quality = self.model_capabilities[model].max_quality_score
            
            model_qualities[model] = avg_quality
        
        # Sort by quality (highest first)
        sorted_models = sorted(model_qualities.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "primary_model": sorted_models[0][0],
            "alternatives": [model for model, _ in sorted_models[1:3]],
            "reasoning": f"Selected {sorted_models[0][0].value} for highest quality: {sorted_models[0][1]:.3f}",
            "quality_scores": model_qualities
        }
    
    def _select_balanced(
        self, 
        models: List[AIModelType], 
        analysis: Dict[str, Any], 
        optimization_goal: PromptOptimizationGoal
    ) -> Dict[str, Any]:
        """Balanced selection basierend auf Optimization Goal"""
        
        model_scores = {}
        
        # Weight factors based on optimization goal
        weights = self._get_optimization_weights(optimization_goal)
        
        for model in models:
            capabilities = self.model_capabilities[model]
            recent_metrics = self._get_recent_performance_metrics(model, hours=12)
            
            # Calculate weighted score
            if recent_metrics:
                avg_speed = statistics.mean([m.response_time_ms for m in recent_metrics])
                avg_quality = statistics.mean([m.quality_score for m in recent_metrics if m.quality_score])
                avg_cost = statistics.mean([float(m.cost_per_request) for m in recent_metrics if m.cost_per_request])
            else:
                avg_speed = capabilities.avg_response_time_ms
                avg_quality = capabilities.max_quality_score
                # ✅ FIX: Safe Decimal to float conversion
                cost_per_1k = capabilities.cost_per_1k_tokens
                if hasattr(cost_per_1k, 'quantize'):  # Check if it's a Decimal
                    cost_per_1k_float = float(cost_per_1k)
                else:
                    cost_per_1k_float = float(cost_per_1k) if cost_per_1k else 0.0
                avg_cost = cost_per_1k_float * 2.0  # Estimate
            
            # Normalize scores (lower is better for speed and cost)
            speed_score = 1.0 - min(avg_speed / 10000, 1.0)  # Normalize to 0-1
            quality_score = avg_quality
            cost_score = 1.0 - min(avg_cost / 0.1, 1.0)  # Normalize to 0-1
            
            # Weighted combination
            balanced_score = (
                speed_score * weights["speed"] +
                quality_score * weights["quality"] +
                cost_score * weights["cost"]
            )
            
            model_scores[model] = balanced_score
        
        # Sort by balanced score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "primary_model": sorted_models[0][0],
            "alternatives": [model for model, _ in sorted_models[1:3]],
            "reasoning": f"Selected {sorted_models[0][0].value} for balanced optimization (score: {sorted_models[0][1]:.3f})",
            "balanced_scores": model_scores,
            "optimization_weights": weights
        }
    
    def _select_context_adaptive(
        self, 
        models: List[AIModelType], 
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Context-adaptive selection basierend auf Request-Kontext"""
        
        complexity = analysis["complexity"]
        context = context or {}
        
        # Model preferences based on context
        model_preferences = {}
        
        for model in models:
            capabilities = self.model_capabilities[model]
            preference_score = 0.0
            
            # Complexity matching
            if complexity in capabilities.optimal_complexities:
                preference_score += 2.0
            
            # Special context matching
            if complexity == RequestComplexity.EMERGENCY and capabilities.best_for_speed:
                preference_score += 3.0
            elif complexity == RequestComplexity.EXPERT and capabilities.best_for_quality:
                preference_score += 2.0
            elif context.get("cultural_context") and capabilities.best_for_cultural_sensitivity:
                preference_score += 2.0
            elif context.get("creative_request") and capabilities.best_for_creativity:
                preference_score += 2.0
            
            # Load balancing bonus
            current_load_ratio = self.current_loads[model] / capabilities.max_concurrent_requests
            load_bonus = 1.0 - current_load_ratio
            preference_score += load_bonus
            
            model_preferences[model] = preference_score
        
        # Sort by preference score
        sorted_models = sorted(model_preferences.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "primary_model": sorted_models[0][0],
            "alternatives": [model for model, _ in sorted_models[1:3]],
            "reasoning": f"Selected {sorted_models[0][0].value} for context adaptation (score: {sorted_models[0][1]:.3f})",
            "context_scores": model_preferences,
            "context_factors": {
                "complexity": complexity.value,
                "special_context": list(context.keys()) if context else []
            }
        }
    
    def _select_load_balanced(self, models: List[AIModelType], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Load-balanced selection für optimale Resource-Verteilung"""
        
        model_load_scores = {}
        
        for model in models:
            capabilities = self.model_capabilities[model]
            current_load = self.current_loads[model]
            max_capacity = capabilities.max_concurrent_requests
            
            # Calculate load ratio
            load_ratio = current_load / max_capacity
            
            # Load score (lower load = higher score)
            load_score = 1.0 - load_ratio
            
            # Adjust for model capabilities
            if analysis["complexity"] in capabilities.optimal_complexities:
                load_score += 0.2
            
            model_load_scores[model] = load_score
        
        # Sort by load score (highest first)
        sorted_models = sorted(model_load_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "primary_model": sorted_models[0][0],
            "alternatives": [model for model, _ in sorted_models[1:3]],
            "reasoning": f"Selected {sorted_models[0][0].value} for optimal load distribution (load: {1-sorted_models[0][1]:.1%})",
            "load_scores": model_load_scores,
            "current_loads": dict(self.current_loads)
        }
    
    def _select_experimental(self, models: List[AIModelType], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Experimental selection für A/B Testing"""
        
        # Check if there are active A/B tests
        active_tests = [test for test in self.ab_test_configs.values() if test.get("active", False)]
        
        if active_tests:
            # Use A/B test configuration
            test_config = active_tests[0]  # Use first active test
            test_models = test_config.get("models", models)
            
            # Random selection for A/B testing
            import random
            selected_model = random.choice(test_models)
            
            return {
                "primary_model": selected_model,
                "alternatives": [m for m in test_models if m != selected_model][:2],
                "reasoning": f"Selected {selected_model.value} for A/B test: {test_config.get('name', 'unnamed')}",
                "ab_test_info": test_config
            }
        else:
            # Fall back to balanced selection
            return self._select_balanced(models, analysis, PromptOptimizationGoal.BALANCE)
    
    # Helper Methods
    def _check_model_capacity(self, model: AIModelType) -> bool:
        """Prüft ob Model Kapazität für neuen Request hat"""
        capabilities = self.model_capabilities[model]
        current_load = self.current_loads[model]
        
        return current_load < capabilities.max_concurrent_requests
    
    def _get_least_loaded_model(self, models: List[AIModelType]) -> AIModelType:
        """Gibt Model mit geringster Last zurück"""
        model_loads = {
            model: self.current_loads[model] / self.model_capabilities[model].max_concurrent_requests
            for model in models
        }
        
        return min(model_loads, key=model_loads.get)
    
    def _prepare_fallback_chain(
        self, 
        primary_model: AIModelType, 
        alternative_models: List[AIModelType]
    ) -> List[AIModelType]:
        """Erstellt Fallback-Chain für robuste Error-Recovery"""
        
        fallback_chain = [primary_model]
        
        # Add alternatives in order of preference
        for alt_model in alternative_models:
            if alt_model not in fallback_chain and alt_model in self.model_capabilities:
                fallback_chain.append(alt_model)
        
        # Ensure all major model types are represented
        major_models = [AIModelType.OPENAI_GPT4, AIModelType.GROQ_MIXTRAL, AIModelType.ANTHROPIC_CLAUDE]
        for major_model in major_models:
            if major_model not in fallback_chain and major_model in self.model_capabilities:
                fallback_chain.append(major_model)
        
        return fallback_chain[:4]  # Limit to 4 models
    
    def _predict_performance(
        self, 
        model: AIModelType, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Vorhersage der Performance für gewähltes Model"""
        
        capabilities = self.model_capabilities[model]
        recent_metrics = self._get_recent_performance_metrics(model, hours=6)
        
        if recent_metrics:
            # Use recent data for prediction
            avg_response_time = statistics.mean([m.response_time_ms for m in recent_metrics])
            avg_quality = statistics.mean([m.quality_score for m in recent_metrics if m.quality_score])
            avg_success_rate = statistics.mean([m.success_rate for m in recent_metrics])
        else:
            # Use baseline capabilities
            avg_response_time = capabilities.avg_response_time_ms
            avg_quality = capabilities.max_quality_score
            avg_success_rate = capabilities.reliability_score
        
        # Adjust for complexity
        complexity_multiplier = {
            RequestComplexity.SIMPLE: 0.8,
            RequestComplexity.MODERATE: 1.0,
            RequestComplexity.COMPLEX: 1.3,
            RequestComplexity.EXPERT: 1.6,
            RequestComplexity.EMERGENCY: 0.9,  # Speed optimized
            RequestComplexity.BULK: 1.2
        }
        
        multiplier = complexity_multiplier.get(analysis["complexity"], 1.0)
        predicted_response_time = int(avg_response_time * multiplier)
        
        return {
            "predicted_response_time_ms": predicted_response_time,
            "predicted_quality_score": avg_quality,
            "predicted_success_rate": avg_success_rate,
            "confidence_level": len(recent_metrics) / 100 if recent_metrics else 0.5,
            "based_on_samples": len(recent_metrics) if recent_metrics else 0
        }
    
    # Performance Tracking Methods
    def record_model_performance(self, metrics: AIModelPerformanceMetrics):
        """Zeichnet Model-Performance für zukünftige Auswahl auf"""
        
        model_type = AIModelType(metrics.model_name) if isinstance(metrics.model_name, str) else metrics.model_name
        
        # Create performance snapshot
        snapshot = ModelPerformanceSnapshot(
            model_type=model_type,
            timestamp=metrics.timestamp,
            response_time_ms=metrics.response_time_ms,
            success_rate=1.0 if not metrics.had_errors else 0.0,
            quality_score=metrics.output_quality_score or 0.0,
            user_satisfaction=metrics.user_satisfaction_predicted or 0.0,
            cost_per_request=metrics.cost_estimate or Decimal('0'),
            tokens_per_request=metrics.tokens_used or 0,
            current_load=self.current_loads[model_type],
            max_capacity=self.model_capabilities[model_type].max_concurrent_requests,
            queue_length=self.request_queues[model_type],
            error_rate=1.0 if metrics.had_errors else 0.0,
            timeout_rate=0.0,  # Would need timeout detection
            request_type=metrics.request_type,
            complexity_level=RequestComplexity.MODERATE  # Default
        )
        
        # Add to history
        self.performance_history[model_type].append(snapshot)
        self.current_performance[model_type] = snapshot
        
        # Update health status
        self._update_model_health_status(model_type, snapshot)
    
    def _update_model_health_status(self, model: AIModelType, snapshot: ModelPerformanceSnapshot):
        """Aktualisiert Gesundheitsstatus basierend auf Performance"""
        
        recent_snapshots = list(self.performance_history[model])[-10:]  # Last 10 snapshots
        
        if len(recent_snapshots) < 3:
            return  # Need more data
        
        # Calculate health metrics
        avg_success_rate = statistics.mean([s.success_rate for s in recent_snapshots])
        avg_error_rate = statistics.mean([s.error_rate for s in recent_snapshots])
        avg_response_time = statistics.mean([s.response_time_ms for s in recent_snapshots])
        
        # Determine health status
        if avg_success_rate >= 0.95 and avg_error_rate <= 0.05 and avg_response_time <= 5000:
            self.model_health_status[model] = ModelHealthStatus.HEALTHY
        elif avg_success_rate >= 0.85 and avg_error_rate <= 0.15:
            self.model_health_status[model] = ModelHealthStatus.DEGRADED
        elif avg_success_rate >= 0.70:
            self.model_health_status[model] = ModelHealthStatus.UNSTABLE
        else:
            self.model_health_status[model] = ModelHealthStatus.UNAVAILABLE
        
        # Check for overload
        capabilities = self.model_capabilities[model]
        if snapshot.current_load > capabilities.max_concurrent_requests * 0.9:
            self.model_health_status[model] = ModelHealthStatus.OVERLOADED
    
    # Utility Methods
    def _initialize_model_capabilities(self) -> Dict[AIModelType, ModelCapabilities]:
        """Initialisiert Model-Capabilities basierend auf bekannten Charakteristiken"""
        
        return {
            AIModelType.OPENAI_GPT4: ModelCapabilities(
                model_type=AIModelType.OPENAI_GPT4,
                avg_response_time_ms=3000,
                max_quality_score=0.95,
                reliability_score=0.92,
                best_for_quality=True,
                best_for_reasoning=True,
                max_concurrent_requests=50,
                rate_limit_per_minute=60,
                max_tokens_per_request=4000,
                cost_per_1k_tokens=Decimal('0.03'),
                optimal_complexities=[RequestComplexity.COMPLEX, RequestComplexity.EXPERT],
                supported_features=["reasoning", "creativity", "detailed_analysis"]
            ),
            
            AIModelType.GROQ_MIXTRAL: ModelCapabilities(
                model_type=AIModelType.GROQ_MIXTRAL,
                avg_response_time_ms=800,
                max_quality_score=0.85,
                reliability_score=0.88,
                best_for_speed=True,
                best_for_cost=True,
                max_concurrent_requests=200,
                rate_limit_per_minute=300,
                max_tokens_per_request=2000,
                cost_per_1k_tokens=Decimal('0.0005'),
                optimal_complexities=[RequestComplexity.SIMPLE, RequestComplexity.MODERATE, RequestComplexity.EMERGENCY],
                supported_features=["speed", "efficiency", "quick_responses"]
            ),
            
            AIModelType.ANTHROPIC_CLAUDE: ModelCapabilities(
                model_type=AIModelType.ANTHROPIC_CLAUDE,
                avg_response_time_ms=2500,
                max_quality_score=0.93,
                reliability_score=0.90,
                best_for_cultural_sensitivity=True,
                best_for_reasoning=True,
                max_concurrent_requests=75,
                rate_limit_per_minute=100,
                max_tokens_per_request=3500,
                cost_per_1k_tokens=Decimal('0.025'),
                optimal_complexities=[RequestComplexity.COMPLEX, RequestComplexity.EXPERT],
                supported_features=["ethical_reasoning", "cultural_sensitivity", "thoughtful_analysis"]
            ),
            
            AIModelType.GOOGLE_GEMINI: ModelCapabilities(
                model_type=AIModelType.GOOGLE_GEMINI,
                avg_response_time_ms=2200,
                max_quality_score=0.88,
                reliability_score=0.85,
                best_for_creativity=True,
                max_concurrent_requests=100,
                rate_limit_per_minute=120,
                max_tokens_per_request=3000,
                cost_per_1k_tokens=Decimal('0.02'),
                optimal_complexities=[RequestComplexity.MODERATE, RequestComplexity.COMPLEX],
                supported_features=["creativity", "multimodal", "innovation"]
            )
        }
    
    def _get_recent_performance_metrics(self, model: AIModelType, hours: int = 24) -> List[ModelPerformanceSnapshot]:
        """Gibt recent performance metrics für Model zurück"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            snapshot for snapshot in self.performance_history[model]
            if snapshot.timestamp >= cutoff_time
        ]
    
    def _has_recent_good_performance(self, model: AIModelType) -> bool:
        """Prüft ob Model recent good performance hat"""
        recent_metrics = self._get_recent_performance_metrics(model, hours=6)
        
        if not recent_metrics:
            return True  # No data = assume good
        
        avg_success_rate = statistics.mean([m.success_rate for m in recent_metrics])
        return avg_success_rate >= 0.7
    
    def _get_best_available_emergency_model(self) -> Optional[AIModelType]:
        """Gibt bestes verfügbares Model für Emergency zurück"""
        # Prefer Groq for speed in emergencies
        emergency_preference = [
            AIModelType.GROQ_MIXTRAL,
            AIModelType.OPENAI_GPT4,
            AIModelType.ANTHROPIC_CLAUDE,
            AIModelType.GOOGLE_GEMINI
        ]
        
        for model in emergency_preference:
            if self.model_health_status[model] != ModelHealthStatus.UNAVAILABLE:
                return model
        
        return None
    
    def _estimate_processing_time(self, complexity: RequestComplexity) -> int:
        """Schätzt Processing-Zeit basierend auf Komplexität"""
        base_times = {
            RequestComplexity.SIMPLE: 1000,
            RequestComplexity.MODERATE: 2500,
            RequestComplexity.COMPLEX: 4000,
            RequestComplexity.EXPERT: 6000,
            RequestComplexity.EMERGENCY: 800,
            RequestComplexity.BULK: 5000
        }
        
        return base_times.get(complexity, 2500)
    
    def _get_models_for_complexity(self, complexity: RequestComplexity) -> List[AIModelType]:
        """Gibt empfohlene Models für Komplexität zurück"""
        recommendations = {
            RequestComplexity.SIMPLE: [AIModelType.GROQ_MIXTRAL, AIModelType.GOOGLE_GEMINI],
            RequestComplexity.MODERATE: [AIModelType.OPENAI_GPT4, AIModelType.ANTHROPIC_CLAUDE, AIModelType.GROQ_MIXTRAL],
            RequestComplexity.COMPLEX: [AIModelType.OPENAI_GPT4, AIModelType.ANTHROPIC_CLAUDE],
            RequestComplexity.EXPERT: [AIModelType.OPENAI_GPT4, AIModelType.ANTHROPIC_CLAUDE],
            RequestComplexity.EMERGENCY: [AIModelType.GROQ_MIXTRAL, AIModelType.OPENAI_GPT4],
            RequestComplexity.BULK: [AIModelType.GROQ_MIXTRAL, AIModelType.GOOGLE_GEMINI]
        }
        
        return recommendations.get(complexity, list(AIModelType))
    
    def _estimate_tokens_for_complexity(self, complexity: RequestComplexity) -> int:
        """Schätzt Token-Anzahl basierend auf Komplexität"""
        token_estimates = {
            RequestComplexity.SIMPLE: 800,
            RequestComplexity.MODERATE: 1500,
            RequestComplexity.COMPLEX: 2500,
            RequestComplexity.EXPERT: 3500,
            RequestComplexity.EMERGENCY: 600,
            RequestComplexity.BULK: 2000
        }
        
        return token_estimates.get(complexity, 1500)
    
    def _get_optimization_weights(self, goal: PromptOptimizationGoal) -> Dict[str, float]:
        """Gibt Gewichtungen für Optimization Goals zurück"""
        weights = {
            PromptOptimizationGoal.SPEED: {"speed": 0.6, "quality": 0.2, "cost": 0.2},
            PromptOptimizationGoal.QUALITY: {"speed": 0.2, "quality": 0.6, "cost": 0.2},
            PromptOptimizationGoal.COST: {"speed": 0.2, "quality": 0.2, "cost": 0.6},
            PromptOptimizationGoal.CREATIVITY: {"speed": 0.3, "quality": 0.5, "cost": 0.2},
            PromptOptimizationGoal.ACCURACY: {"speed": 0.2, "quality": 0.6, "cost": 0.2},
            PromptOptimizationGoal.BALANCE: {"speed": 0.33, "quality": 0.34, "cost": 0.33}
        }
        
        return weights.get(goal, weights[PromptOptimizationGoal.BALANCE])
    
    # A/B Testing Methods
    def _check_ab_test_override(
        self, 
        selected_model: AIModelType, 
        request: GiftRecommendationRequest,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Prüft ob A/B Test Override nötig ist"""
        
        # Check for active A/B tests
        for test_name, test_config in self.ab_test_configs.items():
            if not test_config.get("active", False):
                continue
            
            # Check if request matches test criteria
            if self._request_matches_ab_test_criteria(request, test_config, context):
                # Determine test variant
                import random
                if random.random() < test_config.get("split_ratio", 0.5):
                    test_model = test_config.get("variant_model", selected_model)
                    
                    return {
                        "primary_model": test_model,
                        "ab_test_active": True,
                        "ab_test_name": test_name,
                        "ab_test_variant": "test" if test_model != selected_model else "control"
                    }
        
        return None
    
    def _request_matches_ab_test_criteria(
        self, 
        request: GiftRecommendationRequest, 
        test_config: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Prüft ob Request A/B Test Kriterien erfüllt"""
        
        criteria = test_config.get("criteria", {})
        
        # Check occasion criteria
        if "occasions" in criteria:
            if request.occasion not in criteria["occasions"]:
                return False
        
        # Check relationship criteria
        if "relationships" in criteria:
            if request.personality_data.relationship_to_giver not in criteria["relationships"]:
                return False
        
        # Check complexity criteria
        if "min_complexity" in criteria and context:
            request_complexity = context.get("complexity_score", 0)
            if request_complexity < criteria["min_complexity"]:
                return False
        
        return True
    
    # Management Methods
    def _reserve_model_capacity(self, model: AIModelType):
        """Reserviert Kapazität für ausgewähltes Model"""
        self.current_loads[model] += 1
    
    def release_model_capacity(self, model: AIModelType):
        """Gibt Kapazität nach Request-Completion frei"""
        if self.current_loads[model] > 0:
            self.current_loads[model] -= 1
    
    def _update_selection_statistics(self, result: Dict[str, Any], strategy: SelectionStrategy, selection_time: float):
        """Aktualisiert Selection-Statistiken"""
        self.selection_statistics["total_selections"] += 1
        self.selection_statistics["strategy_usage"][strategy.value] += 1
        self.selection_statistics["model_usage"][result["selected_model"].value] += 1
        self.selection_statistics["selection_time_ms"].append(selection_time)
        
        if result.get("capacity_fallback_used") or result.get("load_balancing_used"):
            self.selection_statistics["fallback_triggered"] += 1
    
    def _handle_no_healthy_models(self, request: GiftRecommendationRequest, optimization_goal: PromptOptimizationGoal) -> Dict[str, Any]:
        """Behandelt Fall wenn keine gesunden Models verfügbar sind"""
        
        # Emergency fallback to any available model
        emergency_model = self._get_best_available_emergency_model()
        
        if not emergency_model:
            emergency_model = AIModelType.GROQ_MIXTRAL  # Ultimate fallback
        
        return {
            "selected_model": emergency_model,
            "selection_reasoning": "Emergency fallback - no healthy models available",
            "alternatives": [],
            "fallback_chain": [emergency_model],
            "predicted_performance": {
                "predicted_response_time_ms": 5000,
                "predicted_quality_score": 0.5,
                "predicted_success_rate": 0.7,
                "confidence_level": 0.1
            },
            "selection_metadata": {
                "emergency_mode": True,
                "health_check_failed": True,
                "model_health_status": dict(self.model_health_status)
            }
        }
    
    # ✅ ADDITIONAL FIX: Update emergency fallback to avoid AUTO_SELECT
    def _create_emergency_fallback(self, request: GiftRecommendationRequest, optimization_goal: PromptOptimizationGoal) -> Dict[str, Any]:
        """Erstellt Emergency-Fallback bei kritischen Fehlern"""
        
        # ✅ FIX: Use concrete model, never AUTO_SELECT
        emergency_model = AIModelType.GROQ_MIXTRAL  # Fast and reliable fallback
        
        return {
            "selected_model": emergency_model,
            "selection_reasoning": "Critical error - using emergency fallback",
            "alternatives": [AIModelType.OPENAI_GPT4],
            "fallback_chain": [emergency_model, AIModelType.OPENAI_GPT4],
            "predicted_performance": {
                "predicted_response_time_ms": 1000,
                "predicted_quality_score": 0.6,
                "predicted_success_rate": 0.8,
                "confidence_level": 0.3
            },
            "selection_metadata": {
                "emergency_fallback": True,
                "selection_error": True,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    # Public API Methods
    def get_model_health_status(self) -> Dict[str, Any]:
        """Gibt aktuellen Gesundheitsstatus aller Models zurück"""
        return {
            "model_health": {model.value: status.value for model, status in self.model_health_status.items()},
            "current_loads": dict(self.current_loads),
            "model_capabilities": {
                model.value: {
                    "max_concurrent": cap.max_concurrent_requests,
                    "rate_limit": cap.rate_limit_per_minute,
                    "avg_response_time": cap.avg_response_time_ms
                }
                for model, cap in self.model_capabilities.items()
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Gibt Selection-Statistiken zurück"""
        stats = self.selection_statistics.copy()
        
        if stats["selection_time_ms"]:
            stats["avg_selection_time_ms"] = statistics.mean(stats["selection_time_ms"])
        
        return stats
    
    def configure_ab_test(self, test_name: str, config: Dict[str, Any]):
        """Konfiguriert A/B Test"""
        self.ab_test_configs[test_name] = config
    
    def get_ab_test_results(self, test_name: str) -> List[Dict[str, Any]]:
        """Gibt A/B Test Ergebnisse zurück"""
        return self.ab_test_results.get(test_name, [])


# =============================================================================
# LEGACY SUPPORT FUNCTIONS
# =============================================================================

def select_engine_for_limbic_type(limbic_type: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Legacy function for selecting AI engine based on limbic type.
    This is a simplified version for backward compatibility.

    Args:
        limbic_type: The limbic type (disciplined, traditionalist, etc.)
        context: Optional context information

    Returns:
        Selected AI model type as string
    """
    # Simple mapping based on limbic type characteristics
    limbic_model_mapping = {
        'disciplined': 'openai_gpt4',      # High quality for structured thinking
        'traditionalist': 'anthropic_claude',  # Conservative, reliable
        'performer': 'openai_gpt35',       # Fast for social interactions
        'adventurer': 'groq_mixtral',      # Fast for exploration
        'harmonizer': 'anthropic_claude',  # Empathetic responses
        'hedonist': 'openai_gpt35',        # Fast for pleasure-seeking
        'pioneer': 'openai_gpt4'           # High quality for innovation
    }

    # Default to GPT-4 if limbic type not found
    return limbic_model_mapping.get(limbic_type, 'openai_gpt4')


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'SelectionStrategy',
    'ModelHealthStatus', 
    'RequestComplexity',
    
    # Data Classes
    'ModelPerformanceSnapshot',
    'ModelCapabilities',
    
    # Main Class
    'ModelSelector',
    
    # Legacy Functions
    'select_engine_for_limbic_type'
]