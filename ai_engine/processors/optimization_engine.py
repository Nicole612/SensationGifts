"""
Optimization Engine - Performance & Cost Optimization Master Controller
======================================================================

Intelligente Orchestrierung und kontinuierliche Optimierung der gesamten AI-Pipeline.
Maximiert Performance, minimiert Kosten und lernt aus jedem Request.

Core Features:
- Real-time performance tuning and adaptation
- Cost-performance analytics and ROI optimization
- Adaptive learning from historical data
- Resource efficiency maximization
- Predictive optimization and capacity planning
- Multi-dimensional optimization (speed, quality, cost, user satisfaction)
- Advanced A/B testing and experimentation
- Dynamic prompt and model optimization
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass
import logging
import statistics
from collections import defaultdict, deque


from ai_engine.schemas import (
    # Core Schemas
    GiftRecommendationRequest,
    GiftRecommendationResponse,
    AIModelType,
  
    
    # Performance Schemas
    AIModelPerformanceMetrics,
   
)

from .prompt_builder import DynamicPromptBuilder, PromptBuildingStrategy
from .response_parser import ResponseParser
from .model_selector import ModelSelector, SelectionStrategy


# =============================================================================
# OPTIMIZATION STRATEGY TYPES
# =============================================================================

class OptimizationObjective(str, Enum):
    """Haupt-Optimierungsziele"""
    COST_EFFICIENCY = "cost_efficiency"               # Minimiere Kosten bei akzeptabler Qualität
    PERFORMANCE_MAXIMIZATION = "performance_maximization"  # Maximiere Performance unabhängig von Kosten
    BALANCED_ROI = "balanced_roi"                     # Optimiere ROI (Return on Investment)
    USER_SATISFACTION = "user_satisfaction"          # Maximiere User-Zufriedenheit
    RESOURCE_UTILIZATION = "resource_utilization"    # Optimiere Resource-Nutzung
    INNOVATION_FOCUSED = "innovation_focused"        # Fokus auf neue Techniken und Experimente


class OptimizationHorizon(str, Enum):
    """Zeithorizont für Optimierungen"""
    REAL_TIME = "real_time"        # Sofortige Anpassungen
    SHORT_TERM = "short_term"      # Stunden bis Tage
    MEDIUM_TERM = "medium_term"    # Wochen bis Monate
    LONG_TERM = "long_term"        # Monate bis Jahre


class PerformanceDimension(str, Enum):
    """Dimensionen der Performance-Optimierung"""
    RESPONSE_TIME = "response_time"
    OUTPUT_QUALITY = "output_quality"
    COST_PER_REQUEST = "cost_per_request"
    USER_SATISFACTION = "user_satisfaction"
    SUCCESS_RATE = "success_rate"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    CREATIVITY_SCORE = "creativity_score"
    ACCURACY_SCORE = "accuracy_score"


# =============================================================================
# OPTIMIZATION DATA STRUCTURES
# =============================================================================

@dataclass
class OptimizationTarget:
    """Definition eines Optimierungs-Ziels"""
    
    dimension: PerformanceDimension
    target_value: float
    weight: float = 1.0
    constraint_type: str = "minimize"  # minimize, maximize, target
    tolerance: float = 0.1
    priority: int = 1  # 1 = highest priority


@dataclass
class OptimizationResult:
    """Ergebnis einer Optimierung"""
    
    optimization_id: str
    timestamp: datetime
    objective: OptimizationObjective
    
    # Before/After Metrics
    baseline_metrics: Dict[str, float]
    optimized_metrics: Dict[str, float]
    improvement_percentages: Dict[str, float]
    
    # Optimization Details
    changes_made: List[Dict[str, Any]]
    confidence_score: float
    estimated_impact: Dict[str, float]
    
    # Validation
    validation_period_days: int = 7
    validation_status: str = "pending"  # pending, validated, failed
    actual_impact: Optional[Dict[str, float]] = None


@dataclass
class ResourceUtilizationSnapshot:
    """Snapshot der Resource-Nutzung"""
    
    timestamp: datetime
    
    # Model Utilization
    model_utilization: Dict[AIModelType, float]  # 0.0-1.0
    model_costs: Dict[AIModelType, Decimal]
    model_performance: Dict[AIModelType, Dict[str, float]]
    
    # System Metrics
    total_requests_per_hour: int
    average_response_time: float
    success_rate: float
    cost_per_hour: Decimal
    
    # Efficiency Metrics
    cost_efficiency_score: float  # Quality/Cost ratio
    resource_efficiency_score: float  # Throughput/Resources ratio
    user_satisfaction_score: float


# =============================================================================
# CORE OPTIMIZATION ENGINE
# =============================================================================

class OptimizationEngine:
    """
    Master Controller für AI-Pipeline-Optimization
    
    Responsibilities:
    - Kontinuierliche Performance-Überwachung
    - Intelligente Resource-Allocation
    - Cost-Performance-Optimierung
    - Adaptive Learning und Anpassung
    - Predictive Optimization
    """
    
    def __init__(self, prompt_builder: DynamicPromptBuilder = None, response_parser: ResponseParser = None, model_selector: ModelSelector = None):
        # Core Components (with safe initialization)
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.model_selector = model_selector
        
        # Optimization Configuration
        self.optimization_objectives = [
            OptimizationTarget(PerformanceDimension.COST_PER_REQUEST, 0.05, weight=0.3, constraint_type="minimize"),
            OptimizationTarget(PerformanceDimension.OUTPUT_QUALITY, 0.85, weight=0.4, constraint_type="maximize"),
            OptimizationTarget(PerformanceDimension.RESPONSE_TIME, 3000, weight=0.2, constraint_type="minimize"),
            OptimizationTarget(PerformanceDimension.USER_SATISFACTION, 0.9, weight=0.1, constraint_type="maximize")
        ]
        
        # Performance Tracking
        self.performance_history: deque = deque(maxlen=10000)
        self.resource_utilization_history: deque = deque(maxlen=1000)
        self.optimization_history: List[OptimizationResult] = []
        
        # Learning and Adaptation
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.05  # Minimum improvement to trigger changes
        self.optimization_weights = {
            "cost": 0.3,
            "quality": 0.4,
            "speed": 0.2,
            "satisfaction": 0.1
        }
        
        # Real-time Optimization State
        self.current_optimization_strategy = OptimizationObjective.BALANCED_ROI
        self.last_optimization_time = datetime.now()
        self.optimization_interval_minutes = 60  # Run major optimizations hourly
        
        # A/B Testing Framework
        self.active_experiments: Dict[str, Dict[str, Any]] = {}
        self.experiment_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Performance Prediction Models
        self.performance_predictors = {
            "response_time": self._build_response_time_predictor(),
            "quality_score": self._build_quality_predictor(),
            "cost_estimate": self._build_cost_predictor()
        }
        
        # Statistics and Analytics
        self.optimization_statistics = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "total_cost_savings": Decimal('0'),
            "total_performance_gains": 0.0,
            "optimization_frequency": defaultdict(int),
            "last_major_optimization": None
        }
        
        # 🚀 ENHANCED OPTIMIZATION FEATURES
        self.advanced_optimization_enabled = True
        self.adaptive_learning_enabled = True
        self.predictive_optimization_enabled = True
        self.real_time_optimization_enabled = True
        
        # Advanced Analytics
        self.performance_analytics = {
            "trend_analysis": {},
            "anomaly_detection": {},
            "optimization_effectiveness": {},
            "cost_benefit_analysis": {}
        }
        
        # Optimization State Management
        self.optimization_state = {
            "current_phase": "learning",
            "optimization_confidence": 0.7,
            "last_successful_optimization": None,
            "optimization_cycles": 0,
            "performance_baseline": 0.8
        }
    
    async def optimize_request_pipeline(
        self,
        request: GiftRecommendationRequest,
        context: Optional[Dict[str, Any]] = None,
        optimization_preference: OptimizationObjective = OptimizationObjective.BALANCED_ROI
    ) -> Dict[str, Any]:
        """
        Haupt-Methode: Optimiert komplette Request-Pipeline
        
        Returns:
            Optimized pipeline configuration und predicted performance
        """
        
        optimization_start = datetime.now()
        
        try:
            # 1. PERFORMANCE PREDICTION
            predicted_metrics = await self._predict_request_performance(request, context)
            
            # 2. OPTIMIZATION STRATEGY SELECTION
            optimization_strategy = self._select_optimization_strategy(
                request, predicted_metrics, optimization_preference
            )
            
            # 3. PROMPT OPTIMIZATION
            prompt_config = await self._optimize_prompt_configuration(
                request, optimization_strategy, context
            )
            
            # 4. MODEL SELECTION OPTIMIZATION
            model_config = await self._optimize_model_selection(
                request, optimization_strategy, prompt_config, context
            )
            
            # 5. RESOURCE ALLOCATION OPTIMIZATION
            resource_config = await self._optimize_resource_allocation(
                model_config, optimization_strategy
            )
            
            # 6. COST-PERFORMANCE OPTIMIZATION
            cost_config = await self._optimize_cost_performance(
                model_config, resource_config, optimization_strategy
            )
            
            # 7. PIPELINE CONFIGURATION ASSEMBLY
            pipeline_config = {
                "prompt_configuration": prompt_config,
                "model_configuration": model_config,
                "resource_configuration": resource_config,
                "cost_configuration": cost_config,
                "optimization_metadata": {
                    "strategy": optimization_strategy,
                    "predicted_metrics": predicted_metrics,
                    "optimization_time_ms": int((datetime.now() - optimization_start).total_seconds() * 1000),
                    "optimization_confidence": self._calculate_optimization_confidence(predicted_metrics)
                }
            }
            
            # 8. ENHANCED REAL-TIME LEARNING UPDATE
            if self.adaptive_learning_enabled:
                await self._update_enhanced_real_time_learning(request, pipeline_config, predicted_metrics)
            
            # 9. PREDICTIVE OPTIMIZATION
            if self.predictive_optimization_enabled:
                pipeline_config = await self._apply_predictive_optimization(pipeline_config, request, context)
            
            # 10. ADVANCED ANALYTICS UPDATE
            if self.advanced_optimization_enabled:
                await self._update_advanced_analytics(request, pipeline_config, predicted_metrics)
            
            return pipeline_config
            
        except Exception as e:
            logging.error(f"Pipeline optimization failed: {e}")
            return await self._create_fallback_pipeline_config(request, optimization_preference)

    async def _update_enhanced_real_time_learning(self, request: GiftRecommendationRequest, 
                                                pipeline_config: Dict[str, Any], 
                                                predicted_metrics: Dict[str, float]):
        """Enhanced real-time learning with advanced analytics"""
        try:
            # Update optimization weights based on performance
            if predicted_metrics.get("response_time", 0) > 5000:
                self.optimization_weights["speed"] = min(0.5, self.optimization_weights["speed"] + 0.05)
                self.optimization_weights["quality"] = max(0.2, self.optimization_weights["quality"] - 0.02)
            
            if predicted_metrics.get("cost_estimate", 0) > 0.1:
                self.optimization_weights["cost"] = min(0.5, self.optimization_weights["cost"] + 0.05)
                self.optimization_weights["quality"] = max(0.2, self.optimization_weights["quality"] - 0.02)
            
            # Update optimization state
            self.optimization_state["optimization_cycles"] += 1
            self.optimization_state["optimization_confidence"] = min(0.95, 
                self.optimization_state["optimization_confidence"] + 0.01)
            
            # Track performance trends
            self._update_performance_trends(predicted_metrics)
            
        except Exception as e:
            logging.warning(f"Enhanced real-time learning failed: {e}")

    async def _apply_predictive_optimization(self, pipeline_config: Dict[str, Any], 
                                           request: GiftRecommendationRequest, 
                                           context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply predictive optimization based on historical patterns"""
        try:
            # Predict optimal configuration based on similar requests
            similar_requests = self._find_similar_requests(request)
            
            if similar_requests:
                # Apply successful patterns from similar requests
                best_pattern = self._get_best_performing_pattern(similar_requests)
                
                if best_pattern:
                    # Apply predictive optimizations
                    pipeline_config["predictive_optimizations"] = {
                        "applied_pattern": best_pattern["pattern_id"],
                        "expected_improvement": best_pattern["improvement_estimate"],
                        "confidence": best_pattern["confidence"]
                    }
                    
                    # Adjust configuration based on pattern
                    if best_pattern.get("prompt_optimization"):
                        pipeline_config["prompt_configuration"].update(best_pattern["prompt_optimization"])
                    
                    if best_pattern.get("model_optimization"):
                        pipeline_config["model_configuration"].update(best_pattern["model_optimization"])
            
            return pipeline_config
            
        except Exception as e:
            logging.warning(f"Predictive optimization failed: {e}")
            return pipeline_config

    async def _update_advanced_analytics(self, request: GiftRecommendationRequest, 
                                       pipeline_config: Dict[str, Any], 
                                       predicted_metrics: Dict[str, float]):
        """Update advanced analytics for optimization insights"""
        try:
            # Trend analysis
            self.performance_analytics["trend_analysis"] = self._analyze_performance_trends()
            
            # Anomaly detection
            self.performance_analytics["anomaly_detection"] = self._detect_performance_anomalies(predicted_metrics)
            
            # Optimization effectiveness
            self.performance_analytics["optimization_effectiveness"] = self._calculate_optimization_effectiveness()
            
            # Cost-benefit analysis
            self.performance_analytics["cost_benefit_analysis"] = self._analyze_cost_benefit(pipeline_config)
            
        except Exception as e:
            logging.warning(f"Advanced analytics update failed: {e}")

    def _update_performance_trends(self, predicted_metrics: Dict[str, float]):
        """Update performance trend analysis"""
        try:
            current_time = datetime.now()
            
            # Add to trend data
            trend_data = {
                "timestamp": current_time,
                "metrics": predicted_metrics,
                "optimization_weights": self.optimization_weights.copy()
            }
            
            # Store in performance history
            self.performance_history.append(trend_data)
            
            # Analyze trends
            if len(self.performance_history) > 10:
                recent_trends = list(self.performance_history)[-10:]
                self._analyze_recent_trends(recent_trends)
                
        except Exception as e:
            logging.warning(f"Performance trend update failed: {e}")

    def _find_similar_requests(self, request: GiftRecommendationRequest) -> List[Dict[str, Any]]:
        """Find similar requests from history for pattern matching"""
        try:
            similar_requests = []
            
            # Extract request features
            request_features = self._extract_request_features(request, None)
            
            # Find similar requests in history
            for historical_request in self.performance_history:
                if isinstance(historical_request, dict) and "metrics" in historical_request:
                    similarity_score = self._calculate_request_similarity(request_features, historical_request)
                    
                    if similarity_score > 0.7:  # High similarity threshold
                        similar_requests.append({
                            "request": historical_request,
                            "similarity_score": similarity_score,
                            "performance": historical_request.get("metrics", {})
                        })
            
            # Sort by similarity and performance
            similar_requests.sort(key=lambda x: (x["similarity_score"], x["performance"].get("quality_score", 0)), reverse=True)
            
            return similar_requests[:5]  # Top 5 similar requests
            
        except Exception as e:
            logging.warning(f"Similar request finding failed: {e}")
            return []

    def _get_best_performing_pattern(self, similar_requests: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the best performing pattern from similar requests"""
        try:
            if not similar_requests:
                return None
            
            # Find the best performing similar request
            best_request = max(similar_requests, key=lambda x: x["performance"].get("quality_score", 0))
            
            # Extract pattern
            pattern = {
                "pattern_id": f"pattern_{len(self.optimization_history)}",
                "confidence": best_request["similarity_score"],
                "improvement_estimate": best_request["performance"].get("quality_score", 0.8),
                "prompt_optimization": best_request["request"].get("prompt_config", {}),
                "model_optimization": best_request["request"].get("model_config", {})
            }
            
            return pattern
            
        except Exception as e:
            logging.warning(f"Best pattern extraction failed: {e}")
            return None

    def _calculate_request_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between two request feature sets"""
        try:
            similarity_score = 0.0
            total_features = 0
            
            # Compare key features
            key_features = ["personality_complexity", "budget_range", "occasion_type", "relationship_type"]
            
            for feature in key_features:
                if feature in features1 and feature in features2:
                    val1 = features1[feature]
                    val2 = features2[feature]
                    
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        # Numeric similarity
                        similarity = 1.0 - abs(val1 - val2) / max(val1, val2, 1.0)
                        similarity_score += similarity
                        total_features += 1
                    elif isinstance(val1, str) and isinstance(val2, str):
                        # String similarity
                        similarity = 1.0 if val1 == val2 else 0.0
                        similarity_score += similarity
                        total_features += 1
            
            return similarity_score / total_features if total_features > 0 else 0.0
            
        except Exception as e:
            logging.warning(f"Request similarity calculation failed: {e}")
            return 0.0

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from historical data"""
        try:
            if len(self.performance_history) < 5:
                return {"trend": "insufficient_data"}
            
            recent_data = list(self.performance_history)[-20:]  # Last 20 requests
            
            # Calculate trend metrics
            response_times = [d.get("metrics", {}).get("response_time", 0) for d in recent_data if isinstance(d, dict)]
            quality_scores = [d.get("metrics", {}).get("quality_score", 0) for d in recent_data if isinstance(d, dict)]
            
            trends = {}
            
            if response_times:
                trends["response_time_trend"] = "improving" if response_times[-1] < response_times[0] else "degrading"
                trends["avg_response_time"] = sum(response_times) / len(response_times)
            
            if quality_scores:
                trends["quality_trend"] = "improving" if quality_scores[-1] > quality_scores[0] else "degrading"
                trends["avg_quality_score"] = sum(quality_scores) / len(quality_scores)
            
            return trends
            
        except Exception as e:
            logging.warning(f"Performance trend analysis failed: {e}")
            return {"trend": "analysis_failed"}

    def _detect_performance_anomalies(self, predicted_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect performance anomalies in predicted metrics"""
        try:
            anomalies = {}
            
            # Check for response time anomalies
            if predicted_metrics.get("response_time", 0) > 8000:  # Very high response time
                anomalies["response_time_anomaly"] = {
                    "type": "high_latency",
                    "severity": "high",
                    "value": predicted_metrics["response_time"]
                }
            
            # Check for cost anomalies
            if predicted_metrics.get("cost_estimate", 0) > 0.2:  # Very high cost
                anomalies["cost_anomaly"] = {
                    "type": "high_cost",
                    "severity": "medium",
                    "value": predicted_metrics["cost_estimate"]
                }
            
            # Check for quality anomalies
            if predicted_metrics.get("quality_score", 1.0) < 0.5:  # Very low quality
                anomalies["quality_anomaly"] = {
                    "type": "low_quality",
                    "severity": "high",
                    "value": predicted_metrics["quality_score"]
                }
            
            return anomalies
            
        except Exception as e:
            logging.warning(f"Anomaly detection failed: {e}")
            return {}

    def _calculate_optimization_effectiveness(self) -> Dict[str, Any]:
        """Calculate effectiveness of recent optimizations"""
        try:
            if len(self.optimization_history) < 2:
                return {"effectiveness": "insufficient_data"}
            
            recent_optimizations = self.optimization_history[-5:]  # Last 5 optimizations
            
            effectiveness = {
                "total_optimizations": len(recent_optimizations),
                "successful_optimizations": sum(1 for opt in recent_optimizations if opt.confidence_score > 0.7),
                "avg_improvement": sum(opt.improvement_percentages.get("overall", 0) for opt in recent_optimizations) / len(recent_optimizations),
                "optimization_frequency": len(recent_optimizations) / 7  # Per week
            }
            
            return effectiveness
            
        except Exception as e:
            logging.warning(f"Optimization effectiveness calculation failed: {e}")
            return {"effectiveness": "calculation_failed"}

    def _analyze_cost_benefit(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cost-benefit of current pipeline configuration"""
        try:
            # ✅ FIX: Safe Decimal to float conversion
            estimated_cost_raw = pipeline_config.get("cost_configuration", {}).get("estimated_cost", 0)
            if isinstance(estimated_cost_raw, Decimal):
                estimated_cost = float(estimated_cost_raw)
            else:
                estimated_cost = float(estimated_cost_raw) if estimated_cost_raw else 0.0
            
            cost_benefit = {
                "estimated_cost": estimated_cost,
                "estimated_quality": float(pipeline_config.get("model_configuration", {}).get("predicted_performance", {}).get("quality_score", 0.8)),
                "cost_efficiency": 0.0,
                "roi_estimate": 0.0
            }
            
            # Calculate cost efficiency (quality per cost unit)
            if cost_benefit["estimated_cost"] > 0:
                cost_benefit["cost_efficiency"] = cost_benefit["estimated_quality"] / cost_benefit["estimated_cost"]
            
            # Estimate ROI (simplified)
            cost_benefit["roi_estimate"] = cost_benefit["estimated_quality"] * 10.0 - cost_benefit["estimated_cost"]
            
            return cost_benefit
            
        except Exception as e:
            logging.warning(f"Cost-benefit analysis failed: {e}")
            return {"cost_benefit": "analysis_failed"}
    
    async def _predict_request_performance(
        self,
        request: GiftRecommendationRequest,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Vorhersage der Performance für Request basierend auf Historical Data
        """
        
        # Extract request features for prediction
        request_features = self._extract_request_features(request, context)
        
        predictions = {}
        
        for metric, predictor in self.performance_predictors.items():
            try:
                predicted_value = predictor(request_features)
                predictions[metric] = predicted_value
            except Exception as e:
                logging.warning(f"Prediction failed for {metric}: {e}")
                # Use historical averages as fallback
                predictions[metric] = self._get_historical_average(metric)
        
        # Add confidence intervals
        predictions["prediction_confidence"] = self._calculate_prediction_confidence(request_features)
        
        return predictions
    
    def _select_optimization_strategy(
        self,
        request: GiftRecommendationRequest,
        predicted_metrics: Dict[str, float],
        preference: OptimizationObjective
    ) -> Dict[str, Any]:
        """Wählt optimale Optimization-Strategy basierend auf Request und Predictions"""
        
        strategy = {
            "primary_objective": preference,
            "optimization_weights": self.optimization_weights.copy(),
            "constraints": [],
            "experimental_features": []
        }
        
        # Adjust strategy based on predictions
        if predicted_metrics.get("response_time", 0) > 5000:
            # High latency predicted - prioritize speed
            strategy["optimization_weights"]["speed"] += 0.2
            strategy["optimization_weights"]["quality"] -= 0.1
            strategy["constraints"].append("max_response_time_ms:3000")
        
        if predicted_metrics.get("cost_estimate", 0) > 0.1:
            # High cost predicted - prioritize cost efficiency
            strategy["optimization_weights"]["cost"] += 0.15
            strategy["constraints"].append("max_cost_per_request:0.05")
        
        # Context-specific adjustments
        # ✅ FIXED: personality_data → personality_data
        personality_complexity = self._calculate_personality_complexity(request.personality_data)
        if personality_complexity > 0.8:
            # Complex personality - prioritize quality
            strategy["optimization_weights"]["quality"] += 0.1
            strategy["experimental_features"].append("advanced_personality_analysis")
        
        # Time-sensitive adjustments
        if request.occasion_date and (request.occasion_date - datetime.now().date()).days <= 7:
            # Time-sensitive request - prioritize speed
            strategy["optimization_weights"]["speed"] += 0.15
            strategy["constraints"].append("emergency_processing")
        
        return strategy
    
    # ✅ MINIMALER FIX - Verwende sichere Fallback-Strategie:
    async def _optimize_prompt_configuration(
        self,
        request: GiftRecommendationRequest,
        optimization_strategy: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimiert Prompt-Konfiguration basierend auf Strategy und Historical Performance
        """
        
        # Analyze historical prompt performance
        try:
            # ✅ SAFE: Analyze historical prompt performance (non-async call)
            try:
                best_performing_prompts = self._analyze_prompt_performance_history(request, optimization_strategy)
            except Exception as e:
                # Fallback if performance history analysis fails
                logging.warning(f"Performance history analysis failed: {e}")
                best_performing_prompts = {
                    "best_prompts": [],
                    "performance_hints": ["Use clear, specific prompts"],
                    "confidence": 0.5
                }
            
            # ✅ FIXED: Use safe string values instead of non-existent enum values
            optimization_weights = optimization_strategy.get("optimization_weights", {})
            
            if optimization_weights.get("speed", 0) > 0.4:
                # Speed optimization
                prompt_strategy = "speed_optimized"
                complexity_level = "standard"
                advanced_techniques = ["template_based"]
            elif optimization_weights.get("quality", 0) > 0.4:
                # Quality optimization  
                prompt_strategy = "quality_focused"
                complexity_level = "enhanced"
                advanced_techniques = ["ensemble_prompting", "self_correction"]
            else:
                # Balanced approach
                prompt_strategy = "balanced_adaptive"
                complexity_level = "standard"
                advanced_techniques = ["contextual_adaptive"]
            
            # Advanced technique selection based on request context
            if hasattr(request, 'personalization_level') and getattr(request, 'personalization_level') == "maximum":
                advanced_techniques.append("deep_personalization")
            
            if hasattr(request, 'prioritize_emotional_impact') and getattr(request, 'prioritize_emotional_impact'):
                advanced_techniques.append("emotional_intelligence")
            
            # ✅ GIFT-FOCUSED: Add gift-specific optimizations
            occasion = getattr(request, 'occasion', 'birthday')
            relationship = getattr(request, 'relationship', 'friend')
            
            if occasion in ['valentinstag', 'jahrestag'] and relationship in ['partner', 'spouse']:
                advanced_techniques.append("romantic_optimization")
                prompt_strategy = "emotional_romantic"
            elif relationship in ['boss', 'colleague']:
                prompt_strategy = "professional_safe"
                complexity_level = "standard"
                advanced_techniques = ["professional_boundaries"]
            
            # ✅ SAFE: Build prompt configuration (ensure no await issues)
            prompt_config = {
                "building_strategy": prompt_strategy,
                "complexity_level": complexity_level,
                "advanced_techniques": advanced_techniques,
                "optimization_goal": self._map_strategy_to_prompt_goal(optimization_strategy),
                "performance_hints": best_performing_prompts.get("performance_hints", []),
                "estimated_token_usage": self._estimate_prompt_tokens(prompt_strategy, complexity_level),
                "estimated_processing_time": self._estimate_prompt_processing_time(prompt_strategy),
                
                # ✅ GIFT SHOP FOCUS: Add shop-specific configurations
                "gift_focus": {
                    "occasion": occasion,
                    "relationship": relationship,
                    "emotional_priority": getattr(request, 'prioritize_emotional_impact', True),
                    "personalization_level": getattr(request, 'personalization_level', 'medium')
                }
            }
            
            return prompt_config
        except Exception as e:
            # ✅ ROBUST ERROR HANDLING: Ensure we always return a valid dict
            logging.error(f"Prompt configuration optimization failed: {e}")
            
            # Return safe fallback configuration
            return {
                "building_strategy": "safe_fallback",
                "complexity_level": "standard", 
                "advanced_techniques": ["basic_prompting"],
                "optimization_goal": "quality",
                "performance_hints": ["Use simple, clear prompts"],
                "estimated_token_usage": 1500,
                "estimated_processing_time": 2000,
                "gift_focus": {
                    "occasion": getattr(request, 'occasion', 'birthday'),
                    "relationship": getattr(request, 'relationship', 'friend'),
                    "emotional_priority": True,
                    "personalization_level": 'medium'
                },
                "fallback_reason": str(e)
            }
    
    async def _optimize_model_selection(
        self,
        request: GiftRecommendationRequest,
        optimization_strategy: Dict[str, Any],
        prompt_config: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimiert Model-Selection basierend auf Strategy und Prompt-Config
        """
        
        # Map optimization strategy to model selection strategy
        if optimization_strategy["optimization_weights"]["speed"] > 0.4:
            selection_strategy = SelectionStrategy.SPEED_OPTIMIZED
        elif optimization_strategy["optimization_weights"]["cost"] > 0.4:
            selection_strategy = SelectionStrategy.COST_OPTIMIZED
        elif optimization_strategy["optimization_weights"]["quality"] > 0.4:
            selection_strategy = SelectionStrategy.QUALITY_OPTIMIZED
        else:
            selection_strategy = SelectionStrategy.BALANCED
        
        # Get model recommendation from selector
        model_selection = self.model_selector.select_optimal_model(
            request=request,
            optimization_goal=prompt_config["optimization_goal"],
            strategy=selection_strategy,
            context=context
        )
        
        # Enhance with optimization-specific configurations
        model_config = {
            "selected_model": model_selection["selected_model"],
            "selection_strategy": selection_strategy,
            "fallback_chain": model_selection["fallback_chain"],
            "predicted_performance": model_selection["predicted_performance"],
            "optimization_specific": {
                "load_balancing_enabled": optimization_strategy["optimization_weights"]["speed"] > 0.3,
                "cost_monitoring_enabled": optimization_strategy["optimization_weights"]["cost"] > 0.3,
                "quality_validation_enabled": optimization_strategy["optimization_weights"]["quality"] > 0.4,
                "adaptive_timeout": self._calculate_adaptive_timeout(optimization_strategy),
                "retry_strategy": self._determine_retry_strategy(optimization_strategy)
            }
        }
        
        return model_config
    
    async def _optimize_resource_allocation(
        self,
        model_config: Dict[str, Any],
        optimization_strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimiert Resource-Allocation für maximale Effizienz
        """
        
        selected_model = model_config["selected_model"]
        
        # Calculate optimal resource allocation
        current_utilization = self.model_selector.current_loads.get(selected_model, 0)
        model_capacity = self.model_selector.model_capabilities[selected_model].max_concurrent_requests
        
        # Determine priority level
        if "emergency_processing" in optimization_strategy.get("constraints", []):
            priority_level = "high"
            resource_allocation = 1.2  # Allow over-allocation for emergencies
        elif optimization_strategy["optimization_weights"]["speed"] > 0.4:
            priority_level = "medium"
            resource_allocation = 1.0
        else:
            priority_level = "normal"
            resource_allocation = 0.8  # Conservative allocation
        
        # Calculate queue position and expected wait time
        queue_position = self._calculate_queue_position(selected_model, priority_level)
        expected_wait_time = self._estimate_queue_wait_time(selected_model, queue_position)
        
        resource_config = {
            "priority_level": priority_level,
            "resource_allocation_factor": resource_allocation,
            "queue_position": queue_position,
            "expected_wait_time_ms": expected_wait_time,
            "load_balancing": {
                "enabled": optimization_strategy["optimization_weights"]["speed"] > 0.3,
                "alternative_models": model_config.get("fallback_chain", [])[:2],
                "load_threshold": 0.8
            },
            "resource_monitoring": {
                "enabled": True,
                "utilization_target": 0.85,
                "scaling_strategy": "horizontal" if resource_allocation > 1.0 else "none"
            }
        }
        
        return resource_config
    
    async def _optimize_cost_performance(
        self,
        model_config: Dict[str, Any],
        resource_config: Dict[str, Any],
        optimization_strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimiert Cost-Performance-Ratio
        """
        
        selected_model = model_config["selected_model"]
        model_capabilities = self.model_selector.model_capabilities[selected_model]
        
        # Calculate cost estimates
        estimated_tokens = model_config.get("estimated_token_usage", 2000)
        
        # ✅ FIX: Safe Decimal handling for cost calculations
        try:
            base_cost_per_1k = model_capabilities.cost_per_1k_tokens
            
            # Convert Decimal to float for calculations
            if hasattr(base_cost_per_1k, 'quantize'):  # Check if it's a Decimal
                base_cost_float = float(base_cost_per_1k)
            else:
                base_cost_float = float(base_cost_per_1k)
            
            # Safe calculation with float conversion
            base_cost = base_cost_float * (float(estimated_tokens) / 1000.0)
            
        except (TypeError, ValueError, AttributeError) as e:
            # Fallback to safe default cost
            logging.warning(f"Cost calculation failed: {e}, using default cost")
            base_cost = 0.02  # Default cost fallback
        
        # Apply resource allocation cost adjustments
        resource_factor = resource_config["resource_allocation_factor"]
        priority_factor = {"high": 1.5, "medium": 1.2, "normal": 1.0}[resource_config["priority_level"]]
        
        # ✅ FIX: Ensure all calculations use float
        estimated_total_cost = float(base_cost) * float(resource_factor) * float(priority_factor)
        
        # Cost optimization strategies
        cost_optimizations = []
        
        if optimization_strategy["optimization_weights"]["cost"] > 0.4:
            # Aggressive cost optimization
            cost_optimizations.extend([
                "token_usage_optimization",
                "response_caching",
                "batch_processing_eligible"
            ])
        
        # ✅ FIX: Safe cost comparison
        cost_threshold = 0.05
        if estimated_total_cost > cost_threshold:
            # High cost - apply additional optimizations
            cost_optimizations.extend([
                "alternative_model_suggestion",
                "prompt_compression",
                "quality_tolerance_adjustment"
            ])
        
        # ROI calculation
        estimated_quality = model_config.get("predicted_performance", {}).get("predicted_quality_score", 0.8)
        
        # ✅ FIX: Safe ROI calculation with zero division protection
        try:
            roi_score = float(estimated_quality) / float(estimated_total_cost) if estimated_total_cost > 0 else 0.0
        except (TypeError, ZeroDivisionError):
            roi_score = 0.0
        
        # ✅ FIX: Convert final cost back to Decimal for consistency
        from decimal import Decimal
        final_estimated_cost = Decimal(str(round(estimated_total_cost, 4)))
        final_base_cost = Decimal(str(round(base_cost, 4)))
        
        cost_config = {
            "estimated_cost": final_estimated_cost,
            "cost_breakdown": {
                "base_model_cost": final_base_cost,
                "resource_factor": float(resource_factor),
                "priority_factor": float(priority_factor)
            },
            "cost_optimizations": cost_optimizations,
            "roi_metrics": {
                "estimated_quality": float(estimated_quality),
                "estimated_cost": float(final_estimated_cost),
                "roi_score": float(roi_score),
                "cost_efficiency_ranking": self._calculate_cost_efficiency_ranking(roi_score)
            },
            "budget_compliance": {
                "within_budget": final_estimated_cost <= Decimal('0.10'),
                "budget_utilization": float(final_estimated_cost) / 0.10,
                "cost_alert_level": self._determine_cost_alert_level(final_estimated_cost)
            }
        }
        
        return cost_config

    
    # Learning and Adaptation Methods
    async def _update_real_time_learning(
        self,
        request: GiftRecommendationRequest,
        pipeline_config: Dict[str, Any],
        predicted_metrics: Dict[str, float]
    ):
        """
        Updates real-time learning basierend auf Pipeline-Performance
        """
        
        # Extract learning features
        learning_data = {
            "request_features": self._extract_request_features(request),
            "pipeline_config": pipeline_config,
            "predicted_metrics": predicted_metrics,
            "timestamp": datetime.now(),
            "optimization_context": pipeline_config.get("optimization_metadata", {})
        }
        
        # Add to learning history
        self.performance_history.append(learning_data)
        
        # Trigger learning updates if enough new data
        if len(self.performance_history) % 100 == 0:
            await self._update_performance_predictors()
        
        # Update optimization weights based on recent performance
        if len(self.performance_history) % 50 == 0:
            await self._update_optimization_weights()
    
    async def _update_performance_predictors(self):
        """
        Updates Performance-Prediction-Models basierend auf Historical Data
        """
        
        if len(self.performance_history) < 100:
            return  # Need more data
        
        recent_data = list(self.performance_history)[-1000:]  # Use recent 1000 samples
        
        # Extract features and targets for each predictor
        for metric, predictor in self.performance_predictors.items():
            try:
                features, targets = self._prepare_training_data(recent_data, metric)
                
                if len(features) > 50:  # Minimum samples for training
                    updated_predictor = self._retrain_predictor(predictor, features, targets)
                    self.performance_predictors[metric] = updated_predictor
                    
                    logging.info(f"Updated {metric} predictor with {len(features)} samples")
                    
            except Exception as e:
                logging.warning(f"Failed to update {metric} predictor: {e}")
    
    async def _update_optimization_weights(self):
        """
        Updates Optimization-Weights basierend auf Performance-Feedback
        """
        
        if len(self.performance_history) < 50:
            return
        
        recent_performance = list(self.performance_history)[-50:]
        
        # Calculate performance improvements for each dimension
        performance_improvements = {
            "cost": self._calculate_cost_improvements(recent_performance),
            "quality": self._calculate_quality_improvements(recent_performance),
            "speed": self._calculate_speed_improvements(recent_performance),
            "satisfaction": self._calculate_satisfaction_improvements(recent_performance)
        }
        
        # Adjust weights based on what's working
        total_improvement = sum(performance_improvements.values())
        
        if total_improvement > 0:
            for dimension, improvement in performance_improvements.items():
                # Increase weight for dimensions showing good improvement
                weight_adjustment = (improvement / total_improvement) * self.learning_rate
                old_weight = self.optimization_weights[dimension]
                self.optimization_weights[dimension] = min(0.6, old_weight + weight_adjustment)
            
            # Normalize weights to sum to 1.0
            total_weight = sum(self.optimization_weights.values())
            for dimension in self.optimization_weights:
                self.optimization_weights[dimension] /= total_weight
            
            logging.info(f"Updated optimization weights: {self.optimization_weights}")
    
    # Performance Analysis Methods
    def record_actual_performance(
        self,
        pipeline_config: Dict[str, Any],
        actual_response: GiftRecommendationResponse,
        actual_metrics: AIModelPerformanceMetrics
    ):
        """
        Records actual performance für Learning-Feedback
        """
        
        performance_record = {
            "timestamp": datetime.now(),
            "pipeline_config": pipeline_config,
            "predicted_metrics": pipeline_config.get("optimization_metadata", {}).get("predicted_metrics", {}),
            "actual_metrics": {
                "response_time": actual_metrics.response_time_ms,
                "quality_score": actual_metrics.output_quality_score or 0.0,
                "cost": float(actual_metrics.cost_estimate or 0),
                "success": not actual_metrics.had_errors,
                "user_satisfaction": actual_metrics.user_satisfaction_predicted or 0.0
            },
            "prediction_accuracy": self._calculate_prediction_accuracy(
                pipeline_config.get("optimization_metadata", {}).get("predicted_metrics", {}),
                {
                    "response_time": actual_metrics.response_time_ms,
                    "quality_score": actual_metrics.output_quality_score or 0.0,
                    "cost_estimate": float(actual_metrics.cost_estimate or 0)
                }
            )
        }
        
        self.performance_history.append(performance_record)
        
        # Update optimization statistics
        self._update_optimization_statistics(performance_record)
    
    def _calculate_prediction_accuracy(
        self,
        predicted_metrics: Dict[str, Any],
        actual_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate prediction accuracy between predicted and actual metrics
        
        Returns:
            Dictionary with accuracy scores for each metric (0.0-1.0, higher is better)
        """
        if not predicted_metrics or not actual_metrics:
            return {
                "overall_accuracy": 0.0,
                "response_time_accuracy": 0.0,
                "quality_accuracy": 0.0,
                "cost_accuracy": 0.0
            }
        
        accuracies = {}
        
        # Response time accuracy (lower is better, so we invert)
        if "response_time" in predicted_metrics and "response_time" in actual_metrics:
            predicted_time = float(predicted_metrics.get("response_time", predicted_metrics.get("predicted_response_time_ms", 0)))
            actual_time = float(actual_metrics.get("response_time", 0))
            
            if predicted_time > 0:
                # Calculate accuracy: how close predicted is to actual (inverted for time)
                time_diff = abs(predicted_time - actual_time)
                time_accuracy = max(0.0, 1.0 - (time_diff / predicted_time))
                accuracies["response_time_accuracy"] = min(1.0, time_accuracy)
            else:
                accuracies["response_time_accuracy"] = 0.0
        else:
            accuracies["response_time_accuracy"] = 0.0
        
        # Quality accuracy (higher is better)
        if "quality_score" in predicted_metrics and "quality_score" in actual_metrics:
            predicted_quality = float(predicted_metrics.get("quality_score", predicted_metrics.get("predicted_quality_score", 0)))
            actual_quality = float(actual_metrics.get("quality_score", 0))
            
            if predicted_quality > 0:
                quality_diff = abs(predicted_quality - actual_quality)
                quality_accuracy = max(0.0, 1.0 - (quality_diff / predicted_quality))
                accuracies["quality_accuracy"] = min(1.0, quality_accuracy)
            else:
                accuracies["quality_accuracy"] = 0.0
        else:
            accuracies["quality_accuracy"] = 0.0
        
        # Cost accuracy (lower is better, so we invert)
        if "cost_estimate" in predicted_metrics and "cost_estimate" in actual_metrics:
            predicted_cost = float(predicted_metrics.get("cost_estimate", 0))
            actual_cost = float(actual_metrics.get("cost_estimate", 0))
            
            if predicted_cost > 0:
                cost_diff = abs(predicted_cost - actual_cost)
                cost_accuracy = max(0.0, 1.0 - (cost_diff / predicted_cost))
                accuracies["cost_accuracy"] = min(1.0, cost_accuracy)
            else:
                accuracies["cost_accuracy"] = 0.0
        else:
            accuracies["cost_accuracy"] = 0.0
        
        # Overall accuracy (average of all metrics)
        accuracy_values = [v for v in accuracies.values() if v > 0]
        accuracies["overall_accuracy"] = (
            statistics.mean(accuracy_values) if accuracy_values else 0.0
        )
        
        return accuracies
    
    def analyze_optimization_effectiveness(self, days: int = 7) -> Dict[str, Any]:
        """
        Analysiert Effectiveness der Optimizations über Zeitraum
        """
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_performance = [
            record for record in self.performance_history
            if record["timestamp"] >= cutoff_date
        ]
        
        if not recent_performance:
            return {"status": "insufficient_data"}
        
        # Calculate aggregate metrics
        avg_response_time = statistics.mean([r["actual_metrics"]["response_time"] for r in recent_performance])
        avg_quality = statistics.mean([r["actual_metrics"]["quality_score"] for r in recent_performance])
        avg_cost = statistics.mean([r["actual_metrics"]["cost"] for r in recent_performance])
        success_rate = statistics.mean([r["actual_metrics"]["success"] for r in recent_performance])
        
        # Calculate improvements over time
        if len(recent_performance) > 10:
            early_half = recent_performance[:len(recent_performance)//2]
            later_half = recent_performance[len(recent_performance)//2:]
            
            improvements = {
                "response_time": self._calculate_improvement(
                    [r["actual_metrics"]["response_time"] for r in early_half],
                    [r["actual_metrics"]["response_time"] for r in later_half],
                    lower_is_better=True
                ),
                "quality": self._calculate_improvement(
                    [r["actual_metrics"]["quality_score"] for r in early_half],
                    [r["actual_metrics"]["quality_score"] for r in later_half]
                ),
                "cost": self._calculate_improvement(
                    [r["actual_metrics"]["cost"] for r in early_half],
                    [r["actual_metrics"]["cost"] for r in later_half],
                    lower_is_better=True
                )
            }
        else:
            improvements = {"insufficient_data": True}
        
        # Prediction accuracy analysis
        prediction_accuracies = [r["prediction_accuracy"] for r in recent_performance if "prediction_accuracy" in r]
        avg_prediction_accuracy = statistics.mean(prediction_accuracies) if prediction_accuracies else 0.0
        
        return {
            "analysis_period_days": days,
            "sample_size": len(recent_performance),
            "current_performance": {
                "avg_response_time_ms": avg_response_time,
                "avg_quality_score": avg_quality,
                "avg_cost_per_request": avg_cost,
                "success_rate": success_rate
            },
            "improvements": improvements,
            "prediction_accuracy": avg_prediction_accuracy,
            "optimization_statistics": self.optimization_statistics.copy(),
            "recommendations": self._generate_optimization_recommendations(recent_performance)
        }
    
    # Utility and Helper Methods
    def _analyze_prompt_performance_history(self, request, optimization_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ FIXED: Analysiert Prompt-Performance History für bessere Empfehlungen
        
        Args:
            request: GiftRecommendationRequest
            optimization_strategy: Optimization strategy dict
        
        Returns:
            Dict mit best-performing prompt insights
        """
        
        if not self.performance_history:
            # Fallback wenn keine History vorhanden
            return {
                "best_prompts": [],
                "performance_hints": ["Use clear, specific prompts", "Include personality context"],
                "confidence": 0.5
            }
        
        # Analysiere recent performance (last 50 entries)
        try:
            # Safe analysis of performance history
            recent_performance = list(self.performance_history)[-50:]
            
            # Extract context safely
            target_occasion = getattr(request, 'occasion', 'birthday')
            target_relationship = getattr(request, 'relationship', 'friend')
            
            # Find similar requests
            similar_requests = []
            for perf_record in recent_performance:
                try:
                    req_features = perf_record.get('request_features', {})
                    if (req_features.get('occasion') == target_occasion or 
                        req_features.get('relationship') == target_relationship):
                        similar_requests.append(perf_record)
                except (AttributeError, KeyError, TypeError):
                    continue
            
            if not similar_requests:
                similar_requests = recent_performance[-10:]
            
            # Generate performance hints
            performance_hints = [
                "Include detailed personality context",
                "Focus on emotional connection", 
                "Provide specific occasion details"
            ]
            
            if len(similar_requests) > 5:
                performance_hints.append("Use proven successful patterns")
            
            confidence = min(1.0, len(similar_requests) / 20)
            
            return {
                "best_prompts": ["personality_focused", "emotion_aware"],
                "performance_hints": performance_hints,
                "confidence": confidence,
                "sample_size": len(similar_requests)
            }
            
        except Exception as e:
            logging.warning(f"Performance history analysis error: {e}")
            return {
                "best_prompts": [],
                "performance_hints": ["Use balanced approach", "Include context"],
                "confidence": 0.3
            }

    def _extract_request_features(self, request: GiftRecommendationRequest, context=None):
        """
        ✅ VOLLSTÄNDIG: Nutzt ALLE Schema-Felder für komplette Feature-Extraktion
        """
        
        # Personality Features
        personality_data = request.personality_data
        big_five_scores = personality_data.get('big_five_scores', {})
        
        features = {
            # === BIG FIVE PERSONALITY ===
            "openness": big_five_scores.get('openness', 3.5),
            "conscientiousness": big_five_scores.get('conscientiousness', 3.5),
            "extraversion": big_five_scores.get('extraversion', 3.5),
            "agreeableness": big_five_scores.get('agreeableness', 3.5),
            "neuroticism": big_five_scores.get('neuroticism', 3.5),
            
            # === CORE REQUEST FEATURES ===
            "num_recommendations": request.number_of_recommendations,
            "occasion": request.occasion,
            "relationship": request.relationship,
            "optimization_goal": request.optimization_goal,
            
            # === BUDGET FEATURES ===
            "budget_min": request.budget_min,
            "budget_max": request.budget_max,
            "has_budget_constraints": bool(request.budget_min or request.budget_max),
            "budget_range": (request.budget_max - request.budget_min) if (request.budget_min and request.budget_max) else 0,
            
            # === TEMPORAL FEATURES ===
            "occasion_date": request.occasion_date,
            "urgency_level": request.urgency_level or "medium",
            "time_pressure": self._calculate_time_pressure(request.occasion_date) if request.occasion_date else 0.0,
            
            # === CONTEXT FEATURES ===
            "cultural_context": bool(request.cultural_context),
            "additional_context": bool(request.additional_context),
            "context_complexity": len(request.additional_context) if request.additional_context else 0,
            
            # === PREFERENCE FEATURES ===
            "personalization_level": request.personalization_level,
            "prioritize_emotional_impact": request.prioritize_emotional_impact,
            "include_explanation": request.include_explanation,
            "explanation_detail": request.explanation_detail,
            
            # === CATEGORY FEATURES ===
            "has_gift_categories": bool(request.gift_categories),
            "num_gift_categories": len(request.gift_categories) if request.gift_categories else 0,
            "has_exclude_categories": bool(request.exclude_categories),
            "num_exclude_categories": len(request.exclude_categories),
            
            # === AI MODEL FEATURES ===
            "target_ai_model": request.target_ai_model or "auto",
            "use_consensus_validation": request.use_consensus_validation,
            
            # === USER FEATURES ===
            "user_id": request.user_id or "anonymous",
            "is_anonymous_user": request.user_id is None or request.user_id == "anonymous_user",
            
            # === COMPLEXITY SCORES ===
            "request_complexity": self._calculate_request_complexity(request),
            "personality_complexity": self._calculate_personality_complexity(request.personality_data),
            
            # === EXTERNAL CONTEXT ===
            "external_context_complexity": len(context) if context else 0.0
        }
        
        # === DERIVED FEATURES ===
        features["total_constraints"] = (
            features["has_budget_constraints"] + 
            features["has_exclude_categories"] + 
            features["cultural_context"] +
            (1 if features["urgency_level"] in ["high", "urgent"] else 0)
        )
        
        features["personalization_intensity"] = self._calculate_personalization_intensity(
            features["personalization_level"],
            features["prioritize_emotional_impact"],
            features["personality_complexity"]
        )
        
        return features

    def _calculate_request_complexity(self, request: GiftRecommendationRequest) -> float:
        """Berechnet Gesamt-Komplexität des Requests"""
        complexity = 0.0
        
        # Personality complexity
        complexity += self._calculate_personality_complexity(request.personality_data) * 0.3
        
        # Constraint complexity
        if request.gift_categories:
            complexity += len(request.gift_categories) * 0.1
        if request.exclude_categories:
            complexity += len(request.exclude_categories) * 0.1
        
        # Context complexity
        if request.cultural_context:
            complexity += 0.2
        if request.additional_context:
            complexity += min(0.3, len(request.additional_context) / 1000)
        
        # Temporal complexity
        if request.urgency_level in ["high", "urgent"]:
            complexity += 0.2
        
        # Personalization complexity
        personalization_weights = {"low": 0.1, "medium": 0.2, "high": 0.3, "maximum": 0.4}
        complexity += personalization_weights.get(request.personalization_level, 0.2)
        
        return min(1.0, complexity)

    def _calculate_personalization_intensity(self, level: str, emotional_priority: bool, personality_complexity: float) -> float:
        """Berechnet Personalisierungs-Intensität"""
        base_intensity = {"low": 0.3, "medium": 0.6, "high": 0.8, "maximum": 1.0}
        intensity = base_intensity.get(level, 0.6)
        
        # Emotional impact boost
        if emotional_priority:
            intensity += 0.1
        
        # Personality complexity boost  
        intensity += personality_complexity * 0.2
        
        return min(1.0, intensity)
    
    def _calculate_optimization_confidence(self, predicted_metrics: Dict[str, float]) -> float:
        """Berechnet Confidence in Optimization-Decisions"""
        
        # Base confidence from prediction confidence
        base_confidence = predicted_metrics.get("prediction_confidence", 0.5)
        
        # Adjust based on historical optimization success
        if self.optimization_statistics["total_optimizations"] > 0:
            success_rate = self.optimization_statistics["successful_optimizations"] / self.optimization_statistics["total_optimizations"]
            base_confidence = (base_confidence + success_rate) / 2
        
        # Adjust based on data availability
        data_availability = min(1.0, len(self.performance_history) / 1000)
        
        return base_confidence * data_availability
    
    # Predictor Building Methods (Simplified implementations)
    def _build_response_time_predictor(self) -> Callable:
        """Builds response time predictor (simplified implementation)"""
        
        def predictor(features: Dict[str, float]) -> float:
            # Simple heuristic-based predictor
            base_time = 2000  # Base 2 seconds
            
            # Complexity adjustments
            complexity_factor = (
                features.get("num_recommendations", 5) * 100 +
                features.get("context_complexity", 0) * 50 +
                (1 if features.get("include_explanation", 0) else 0) * 500
            )
            
            # Personality complexity
            personality_complexity = sum([
                features.get("honesty_humility", 3),
                features.get("emotionality", 3),
                features.get("extraversion", 3),
                features.get("agreeableness", 3),
                features.get("conscientiousness", 3),
                features.get("openness", 3)
            ]) / 6
            
            personality_factor = abs(personality_complexity - 3) * 200  # Deviation from neutral
            
            return base_time + complexity_factor + personality_factor
        
        return predictor
    
    def _build_quality_predictor(self) -> Callable:
        """Builds quality score predictor"""
        
        def predictor(features: Dict[str, float]) -> float:
            # Base quality score
            base_quality = 0.75
            
            # Higher complexity can lead to better quality with more effort
            complexity_bonus = (
                features.get("context_complexity", 0) * 0.02 +
                (1 if features.get("has_additional_context", 0) else 0) * 0.05
            )
            
            # Personality variance can improve personalization
            personality_variance = statistics.stdev([
                features.get("honesty_humility", 3),
                features.get("emotionality", 3),
                features.get("extraversion", 3),
                features.get("agreeableness", 3),
                features.get("conscientiousness", 3),
                features.get("openness", 3)
            ])
            
            personalization_bonus = min(0.1, personality_variance * 0.05)
            
            return min(1.0, base_quality + complexity_bonus + personalization_bonus)
        
        return predictor
    
    def _build_cost_predictor(self) -> Callable:
        """Builds cost predictor"""
        
        def predictor(features: Dict[str, float]) -> float:
            # Base cost
            base_cost = 0.02
            
            # Token usage estimation
            token_factor = features.get("num_recommendations", 5) * 0.003
            explanation_factor = (1 if features.get("include_explanation", 0) else 0) * 0.01
            complexity_factor = features.get("context_complexity", 0) * 0.001
            
            return base_cost + token_factor + explanation_factor + complexity_factor
        
        return predictor
    
    # Additional helper methods would be implemented here...
    def _calculate_cost_efficiency_ranking(self, roi_score: float) -> str:
        """Calculate cost efficiency ranking safely"""
        try:
            roi_score = float(roi_score)
            if roi_score >= 20.0:
                return "excellent"
            elif roi_score >= 15.0:
                return "very_good"
            elif roi_score >= 10.0:
                return "good"
            elif roi_score >= 5.0:
                return "fair"
            else:
                return "poor"
        except (TypeError, ValueError):
            return "unknown"

    # ✅ ADDITIONAL FIX: Helper method for cost alert level
    def _determine_cost_alert_level(self, estimated_cost) -> str:
        """Determine cost alert level safely"""
        try:
            cost_value = float(estimated_cost)
            if cost_value <= 0.02:
                return "low"
            elif cost_value <= 0.05:
                return "medium"
            elif cost_value <= 0.10:
                return "high"
            else:
                return "critical"
        except (TypeError, ValueError):
            return "unknown"


    def _get_historical_average(self, metric: str) -> float:
        """Gets historical average for metric"""
        defaults = {
            "response_time": 2500.0,
            "quality_score": 0.8,
            "cost_estimate": 0.03
        }
        return defaults.get(metric, 0.0)
    
    def _calculate_prediction_confidence(self, features: Dict[str, float]) -> float:
        """Calculates confidence in predictions"""
        # Simple implementation - would be more sophisticated in practice
        return max(0.3, min(0.9, len(self.performance_history) / 1000))
    
    def _map_strategy_to_prompt_goal(self, strategy: Dict[str, Any]) -> str:
        """Maps optimization strategy to prompt optimization goal"""
        weights = strategy["optimization_weights"]
        
        if weights["speed"] > 0.4:
            return "speed"
        elif weights["quality"] > 0.4:
            return "quality"
        elif weights["cost"] > 0.4:
            return "cost"
        else:
            return "balance"
    
    def _estimate_prompt_tokens(self, strategy: str, complexity: str) -> int:
        """Estimates token usage for prompt configuration"""
        
        base_tokens = {
            "standard": 1500, 
            "enhanced": 2500, 
            "expert": 3500
        }
        
        strategy_multipliers = {
            "speed_optimized": 0.7,
            "quality_focused": 1.4,
            "emotional_romantic": 1.3,
            "professional_safe": 0.9,
            "balanced_adaptive": 1.0
        }
        
        base = base_tokens.get(complexity, 1500)
        multiplier = strategy_multipliers.get(strategy, 1.0)
        
        return int(base * multiplier)

    def _estimate_prompt_processing_time(self, strategy: str) -> int:
        """Estimates processing time for prompt strategy"""
        
        base_times = {
            "speed_optimized": 800,
            "quality_focused": 2500,
            "emotional_romantic": 2000,
            "professional_safe": 1200,
            "balanced_adaptive": 1500
        }
        
        return base_times.get(strategy, 1500)

        
    # Placeholder methods for complex calculations
    def _calculate_adaptive_timeout(self, strategy: Dict[str, Any]) -> int:
        """Calculates adaptive timeout based on strategy"""
        base_timeout = 30000  # 30 seconds
        if strategy["optimization_weights"]["speed"] > 0.4:
            return int(base_timeout * 0.7)
        elif strategy["optimization_weights"]["quality"] > 0.4:
            return int(base_timeout * 1.5)
        return base_timeout
    
    def _determine_retry_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Determines retry strategy based on optimization goals"""
        return {
            "max_retries": 3 if strategy["optimization_weights"]["quality"] > 0.4 else 1,
            "exponential_backoff": strategy["optimization_weights"]["speed"] <= 0.4,
            "fallback_models": True
        }
    
    def _calculate_queue_position(self, model: AIModelType, priority: str) -> int:
        """Calculates queue position for model"""
        base_queue = self.model_selector.request_queues.get(model, 0)
        priority_adjustment = {"high": -2, "medium": 0, "normal": 1}
        return max(0, base_queue + priority_adjustment.get(priority, 0))
    
    def _estimate_queue_wait_time(self, model: AIModelType, queue_position: int) -> int:
        """Estimates wait time based on queue position"""
        avg_processing_time = self.model_selector.model_capabilities[model].avg_response_time_ms
        return queue_position * avg_processing_time
    
    # Fallback and Error Handling
    async def _create_fallback_pipeline_config(
        self,
        request: GiftRecommendationRequest,
        optimization_preference: OptimizationObjective
    ) -> Dict[str, Any]:
        """Creates fallback configuration when optimization fails"""
        
        return {
            "prompt_configuration": {
                "building_strategy": "simple_template",
                "complexity_level": "standard",
                "advanced_techniques": [],
                "optimization_goal": "balance"
            },
            "model_configuration": {
                "selected_model": AIModelType.GROQ_MIXTRAL,  # Fast and reliable
                "selection_strategy": "speed_optimized",
                "fallback_chain": [AIModelType.GROQ_MIXTRAL, AIModelType.OPENAI_GPT4]
            },
            "resource_configuration": {
                "priority_level": "normal",
                "resource_allocation_factor": 1.0
            },
            "cost_configuration": {
                "estimated_cost": Decimal('0.02'),
                "cost_optimizations": ["basic_optimization"]
            },
            "optimization_metadata": {
                "fallback_mode": True,
                "optimization_preference": optimization_preference,
                "fallback_reason": "optimization_engine_error"
            }
        }
    
    # Statistics and Reporting
    def _update_optimization_statistics(self, performance_record: Dict[str, Any]):
        """Updates optimization statistics"""
        self.optimization_statistics["total_optimizations"] += 1
        
        # Check if optimization was successful (improved metrics)
        predicted = performance_record.get("predicted_metrics", {})
        actual = performance_record.get("actual_metrics", {})
        
        if self._optimization_was_successful(predicted, actual):
            self.optimization_statistics["successful_optimizations"] += 1
    
    def _optimization_was_successful(self, predicted: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Determines if optimization was successful"""
        # Simple success criteria - actual performance meets or exceeds predictions
        if not predicted or not actual:
            return False
        
        success_criteria = 0
        total_criteria = 0
        
        if "response_time" in predicted and "response_time" in actual:
            if actual["response_time"] <= predicted["response_time"] * 1.1:  # Within 10% tolerance
                success_criteria += 1
            total_criteria += 1
        
        if "quality_score" in predicted and "quality_score" in actual:
            if actual["quality_score"] >= predicted["quality_score"] * 0.9:  # Within 10% tolerance
                success_criteria += 1
            total_criteria += 1
        
        return (success_criteria / total_criteria) >= 0.5 if total_criteria > 0 else False
    
    # Public API Methods
    def get_optimization_status(self) -> Dict[str, Any]:
        """Returns current optimization status and statistics"""
        
        return {
            "current_strategy": self.current_optimization_strategy.value,
            "optimization_weights": self.optimization_weights.copy(),
            "statistics": self.optimization_statistics.copy(),
            "performance_history_size": len(self.performance_history),
            "last_optimization": self.last_optimization_time.isoformat(),
            "predictor_confidence": {
                predictor_name: self._calculate_prediction_confidence({})
                for predictor_name in self.performance_predictors.keys()
            },
            "resource_utilization": {
                model.value: self.model_selector.current_loads.get(model, 0)
                for model in AIModelType
            }
        }
    
    def get_performance_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Returns comprehensive performance analytics"""
        return self.analyze_optimization_effectiveness(days)


# =============================================================================
# EXPORTS
# =============================================================================


    # ✅ MISSING HELPER METHODS - ADDED
    def _calculate_occasion_complexity(self, occasion) -> float:
        """Calculate complexity score for occasion"""
        if not occasion:
            return 0.3
        
        complexity_map = {
            "birthday": 0.3,
            "christmas": 0.8,
            "wedding": 0.9,
            "anniversary": 0.6,
            "graduation": 0.7,
            "valentine": 0.7,
            "mother_day": 0.5,
            "father_day": 0.5
        }
        return complexity_map.get(str(occasion).lower(), 0.5)
    
    def _calculate_relationship_intimacy(self, relationship) -> float:
        """Calculate intimacy score for relationship"""
        if not relationship:
            return 0.5
        
        intimacy_map = {
            "partner": 0.9,
            "spouse": 1.0,
            "family": 0.8,
            "close_friend": 0.7,
            "friend": 0.5,
            "colleague": 0.3,
            "acquaintance": 0.2,
            "boss": 0.2,
            "employee": 0.3
        }
        return intimacy_map.get(str(relationship).lower(), 0.5)
    
    def _calculate_time_pressure(self, occasion_date) -> float:
        """Calculate time pressure based on occasion date"""
        if not occasion_date:
            return 0.0
        
        try:
            from datetime import datetime, date
            
            if isinstance(occasion_date, str):
                occasion_date = datetime.strptime(occasion_date, '%Y-%m-%d').date()
            elif isinstance(occasion_date, datetime):
                occasion_date = occasion_date.date()
            
            days_until = (occasion_date - date.today()).days
            
            if days_until <= 0:
                return 1.0  # Past due!
            elif days_until <= 1:
                return 0.9
            elif days_until <= 3:
                return 0.8
            elif days_until <= 7:
                return 0.6
            elif days_until <= 14:
                return 0.4
            elif days_until <= 30:
                return 0.2
            else:
                return 0.1
        except Exception as e:
            return 0.0

    def _calculate_personality_complexity(self, personality_data) -> float:
        """
        ✅ FEHLENDE METHODE: Berechnet Personality-Komplexität basierend auf Big Five Scores
        
        Args:
            personality_data: Dict mit personality scores
            
        Returns:
            float: Complexity score zwischen 0.0 und 1.0
        """
        if not personality_data:
            return 0.3  # Default moderate complexity
        
        try:
            # Extract Big Five scores
            big_five_scores = personality_data.get('big_five_scores', {})
            
            if not big_five_scores:
                return 0.3  # Default if no Big Five data
            
            # Get individual scores (default to neutral 3.5)
            scores = [
                big_five_scores.get('openness', 3.5),
                big_five_scores.get('conscientiousness', 3.5),
                big_five_scores.get('extraversion', 3.5),
                big_five_scores.get('agreeableness', 3.5),
                big_five_scores.get('neuroticism', 3.5)
            ]
            
            # Calculate complexity factors
            
            # 1. Variance from neutral (3.5) - higher variance = more complex
            variance_factor = sum(abs(score - 3.5) for score in scores) / len(scores) / 1.5  # Max 1.0
            
            # 2. Extreme scores - very high or very low scores increase complexity
            extreme_factor = sum(1 for score in scores if score <= 1.5 or score >= 4.5) / len(scores)
            
            # 3. Neuroticism adds complexity (emotional instability)
            neuroticism_factor = max(0, (big_five_scores.get('neuroticism', 3.5) - 3.0) / 2.0)
            
            # 4. High openness adds complexity (more nuanced preferences)
            openness_factor = max(0, (big_five_scores.get('openness', 3.5) - 3.0) / 2.0)
            
            # Combine factors (weighted)
            total_complexity = (
                variance_factor * 0.4 +      # Variance is most important
                extreme_factor * 0.3 +       # Extreme scores add complexity
                neuroticism_factor * 0.2 +   # Emotional complexity
                openness_factor * 0.1        # Preference complexity
            )
            
            # Normalize to 0.0-1.0 range
            return min(1.0, max(0.0, total_complexity))
            
        except Exception as e:
            logging.warning(f"Personality complexity calculation failed: {e}")
            return 0.5  # Safe fallback

__all__ = [
    # Enums
    'OptimizationObjective',
    'OptimizationHorizon',
    'PerformanceDimension',
    
    # Data Classes
    'OptimizationTarget',
    'OptimizationResult',
    'ResourceUtilizationSnapshot',
    
    # Main Class
    'OptimizationEngine'
]