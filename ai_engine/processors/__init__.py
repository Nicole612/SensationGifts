"""
AI Engine Processors Package
============================

Intelligent processing layer for SensationGifts AI recommendations.
Provides comprehensive prompt building, response parsing, model selection, and optimization.

This package provides:
- Dynamic prompt generation and optimization
- Structured AI response parsing and validation
- Intelligent AI model selection and orchestration
- Real-time performance optimization and learning

Usage Examples:
    # Complete AI processing pipeline
    from ai_engine.processors import AIProcessingPipeline
    
    # Individual components
    from ai_engine.processors import PromptBuilder, ResponseParser, ModelSelector, OptimizationEngine
    
    # High-level orchestration
    from ai_engine.processors import IntelligentProcessor
"""

# =============================================================================
# CORE PROCESSING COMPONENTS
# =============================================================================

from .prompt_builder import (
    # Strategy Types
    PromptBuildingStrategy,
    
    # Main Class
    DynamicPromptBuilder,
)

from .response_parser import (
    # Strategy and Result Types
    ParsingStrategy,
    OutputFormat,
    ValidationSeverity,
    ParsedResponse,
    ValidationResult,
    
    # Parser Classes
    ResponseParser,
)

from .model_selector import (
    # Selection Strategy Types
    SelectionStrategy,
    ModelHealthStatus,
    RequestComplexity,
    
    # Data Classes
    ModelPerformanceSnapshot,
    
    # Main Class
    ModelSelector,
)

from .optimization_engine import (
    # Optimization Types
    OptimizationObjective,
    PerformanceDimension,
    
    # Data Classes
    OptimizationTarget,
    OptimizationResult,
    ResourceUtilizationSnapshot,
    
    # Main Class
    OptimizationEngine,
)


# =============================================================================
# HIGH-LEVEL PROCESSING ORCHESTRATION
# =============================================================================

class IntelligentProcessor:
    """
    High-level orchestrator for AI processing pipeline
    
    Combines all processors for complete request handling:
    - Prompt optimization and building
    - Model selection and configuration
    - Response parsing and validation
    - Performance optimization and learning
    """
    
    def __init__(self):
        # 🚀 ADVANCED TECHNIQUES INTEGRATION
        try:
            from ai_engine.prompts.advanced_techniques import AdvancedTechniqueOrchestrator
            self.advanced_orchestrator = AdvancedTechniqueOrchestrator()
            print("✅ Advanced Technique Orchestrator integrated successfully")
        except Exception as e:
            print(f"❌ Advanced Techniques integration failed: {e}")
            self.advanced_orchestrator = None
        
        # Initialize core components
        self.prompt_builder = DynamicPromptBuilder()
        self.response_parser = ResponseParser()
        self.model_selector = ModelSelector()
        self.optimization_engine = OptimizationEngine(
            self.prompt_builder, self.response_parser, self.model_selector
        )
        
        # Processing statistics
        self.processing_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_processing_time": 0.0,
            "optimization_improvements": 0.0
        }
    
    async def process_gift_recommendation_request(
        self,
        request,
        optimization_preference=OptimizationObjective.BALANCED_ROI,
        context=None
    ):
        """
        Complete processing pipeline for gift recommendation requests
        
        Args:
            request: GiftRecommendationRequest
            optimization_preference: OptimizationObjective
            context: Optional context information
            
        Returns:
            Complete processing result with optimized configuration and response
        """
        
        from datetime import datetime
        processing_start = datetime.now()
        
        try:
            # 1. OPTIMIZATION PLANNING
            pipeline_config = await self.optimization_engine.optimize_request_pipeline(
                request, context, optimization_preference
            )
            
            # 2. PROMPT BUILDING
            prompt_result = self.prompt_builder.build_optimal_prompt(
                request=request,
                target_model=pipeline_config["model_configuration"]["selected_model"],
                optimization_goal=pipeline_config["prompt_configuration"]["optimization_goal"],
                complexity_level=pipeline_config["prompt_configuration"]["complexity_level"]
            )
            
            # 3. MODEL SELECTION CONFIRMATION
            model_selection = self.model_selector.select_optimal_model(
                request=request,
                optimization_goal=pipeline_config["prompt_configuration"]["optimization_goal"],
                strategy=pipeline_config["model_configuration"]["selection_strategy"],
                context=context
            )
            
            # 4. PROCESSING RESULT ASSEMBLY
            processing_result = {
                "pipeline_configuration": pipeline_config,
                "prompt_result": prompt_result,
                "model_selection": model_selection,
                "processing_metadata": {
                    "processing_time_ms": int((datetime.now() - processing_start).total_seconds() * 1000),
                    "optimization_applied": True,
                    "components_used": ["prompt_builder", "model_selector", "optimization_engine"],
                    "processing_timestamp": datetime.now().isoformat()
                },
                "execution_plan": {
                    "final_prompt": prompt_result["final_prompt"],
                    "target_model": model_selection["selected_model"],
                    "fallback_models": model_selection["fallback_chain"],
                    "expected_performance": pipeline_config["optimization_metadata"]["predicted_metrics"]
                }
            }
            
            # 5. UPDATE STATISTICS
            self._update_processing_statistics(processing_result, True)
            
            return processing_result
            
        except Exception as e:
            # Fallback processing
            fallback_result = await self._create_fallback_processing_result(request, str(e))
            self._update_processing_statistics(fallback_result, False)
            return fallback_result
    
    def parse_ai_response(
        self,
        raw_response: str,
        source_model,
        expected_schema=None,
        parsing_strategy=ParsingStrategy.HYBRID_PARSING
    ):
        """
        Parse and validate AI response using intelligent parsing
        
        Args:
            raw_response: Raw AI response string
            source_model: AI model that generated the response
            expected_schema: Expected Pydantic schema
            parsing_strategy: Parsing strategy to use
            
        Returns:
            ParsedResponse with structured data and validation
        """
        
        return self.response_parser.parse_gift_recommendation_response(
            raw_response=raw_response,
            source_model=source_model,
            expected_schema=expected_schema or self._infer_schema_from_context(),
            parsing_strategy=parsing_strategy
        )
    
    def record_actual_performance(self, processing_result, actual_response, actual_metrics):
        """
        Record actual performance for optimization learning
        
        Args:
            processing_result: Result from process_gift_recommendation_request
            actual_response: Actual AI response
            actual_metrics: Actual performance metrics
        """
        
        # Update optimization engine with actual performance
        self.optimization_engine.record_actual_performance(
            pipeline_config=processing_result["pipeline_configuration"],
            actual_response=actual_response,
            actual_metrics=actual_metrics
        )
        
        # Update model selector with performance data
        self.model_selector.record_model_performance(actual_metrics)
        
        # ✅ FIX: Update processing statistics
        # Check if this is a successful request based on actual_metrics
        request_successful = actual_metrics.request_successful if hasattr(actual_metrics, 'request_successful') else not actual_metrics.had_errors
        
        # Update stats
        self.processing_stats["total_requests"] += 1
        if request_successful:
            self.processing_stats["successful_requests"] += 1
        
        # Update average processing time if available
        if hasattr(actual_metrics, 'response_time_ms') and actual_metrics.response_time_ms:
            processing_time = actual_metrics.response_time_ms
            total_requests = self.processing_stats["total_requests"]
            current_avg = self.processing_stats["average_processing_time"]
            self.processing_stats["average_processing_time"] = \
                ((current_avg * (total_requests - 1)) + processing_time) / total_requests
    
    def get_processing_analytics(self):
        """Get comprehensive processing analytics"""
        
        return {
            "processing_statistics": self.processing_stats.copy(),
            "optimization_status": self.optimization_engine.get_optimization_status(),
            "model_health": self.model_selector.get_model_health_status(),
            "parsing_statistics": self.response_parser.get_parsing_statistics(),
            "prompt_building_statistics": getattr(self.prompt_builder, 'build_statistics', {}),
            "overall_system_health": self._calculate_system_health()
        }
    
    # Helper methods
    def _update_processing_statistics(self, result, success):
        """Update processing statistics"""
        self.processing_stats["total_requests"] += 1
        if success:
            self.processing_stats["successful_requests"] += 1
        
        # Update average processing time
        if "processing_metadata" in result:
            processing_time = result["processing_metadata"].get("processing_time_ms", 0)
            total_requests = self.processing_stats["total_requests"]
            current_avg = self.processing_stats["average_processing_time"]
            self.processing_stats["average_processing_time"] = \
                ((current_avg * (total_requests - 1)) + processing_time) / total_requests
    
    async def _create_fallback_processing_result(self, request, error):
        """Create fallback result when processing fails"""
        from ai_engine.schemas import AIModelType
        
        return {
            "pipeline_configuration": {
                "fallback_mode": True,
                "error": error
            },
            "prompt_result": {
                "final_prompt": f"Simple gift recommendation for {request.personality_data.get('person_name', 'Person')}",
                "metadata": {"fallback": True}
            },
            "model_selection": {
                "selected_model": AIModelType.GROQ_MIXTRAL,
                "fallback_reason": "processing_error"
            },
            "processing_metadata": {
                "processing_error": True,
                "error_message": error,
                "fallback_used": True
            },
            "execution_plan": {
                "final_prompt": f"Recommend gifts for {request.personality_data.get('person_name', 'Person')}",
                "target_model": AIModelType.GROQ_MIXTRAL,
                "fallback_models": [AIModelType.OPENAI_GPT4]
            }
        }
    
    def _infer_schema_from_context(self):
        """Infer expected schema from context"""
        from ai_engine.schemas import GiftRecommendationResponse
        return GiftRecommendationResponse
    
    def _calculate_system_health(self):
        """Calculate overall system health score"""
        success_rate = (self.processing_stats["successful_requests"] / 
                       max(1, self.processing_stats["total_requests"]))
        
        # Get component health scores
        optimization_health = 1.0  # Would calculate from optimization_engine
        model_health = 0.9  # Would calculate from model_selector
        parsing_health = 0.95  # Would calculate from response_parser
        
        overall_health = (success_rate + optimization_health + model_health + parsing_health) / 4
        
        return {
            "overall_score": overall_health,
            "success_rate": success_rate,
            "component_health": {
                "optimization": optimization_health,
                "model_selection": model_health,
                "response_parsing": parsing_health
            },
            "status": "healthy" if overall_health > 0.8 else "degraded" if overall_health > 0.6 else "unhealthy"
        }


class AIProcessingPipeline:
    """
    Simplified interface for complete AI processing pipeline
    
    Provides easy-to-use methods for common processing tasks
    """
    
    def __init__(self):
        self.processor = IntelligentProcessor()
    
    async def process_request(self, request, optimization="balanced"):
        """Process gift recommendation request with automatic optimization"""
        
        optimization_map = {
            "speed": OptimizationObjective.PERFORMANCE_MAXIMIZATION,
            "quality": OptimizationObjective.PERFORMANCE_MAXIMIZATION, 
            "cost": OptimizationObjective.COST_EFFICIENCY,
            "balanced": OptimizationObjective.BALANCED_ROI,
            "satisfaction": OptimizationObjective.USER_SATISFACTION
        }
        
        optimization_preference = optimization_map.get(optimization, OptimizationObjective.BALANCED_ROI)
        
        return await self.processor.process_gift_recommendation_request(
            request, optimization_preference
        )
    
    def parse_response(self, raw_response, source_model):
        """Parse AI response with automatic format detection"""
        return self.processor.parse_ai_response(raw_response, source_model)
    
    def get_analytics(self):
        """Get processing analytics"""
        return self.processor.get_processing_analytics()


# =============================================================================
# CONVENIENCE FACTORIES AND UTILITIES
# =============================================================================

class ProcessorFactory:
    """Factory for creating and configuring processors"""
    
    @staticmethod
    def create_production_processor():
        """Create processor optimized for production use"""
        return IntelligentProcessor()
    
    @staticmethod
    def create_development_processor():
        """Create processor optimized for development/testing"""
        processor = IntelligentProcessor()
        # Configure for more verbose logging, lower thresholds, etc.
        return processor
    
    @staticmethod
    def create_cost_optimized_processor():
        """Create processor optimized for cost efficiency"""
        processor = IntelligentProcessor()
        processor.optimization_engine.optimization_weights["cost"] = 0.6
        processor.optimization_engine.optimization_weights["quality"] = 0.2
        processor.optimization_engine.optimization_weights["speed"] = 0.2
        return processor
    
    @staticmethod
    def create_quality_optimized_processor():
        """Create processor optimized for highest quality"""
        processor = IntelligentProcessor()
        processor.optimization_engine.optimization_weights["quality"] = 0.6
        processor.optimization_engine.optimization_weights["cost"] = 0.2
        processor.optimization_engine.optimization_weights["speed"] = 0.2
        return processor


class ProcessorUtils:
    """Utility functions for processor operations"""
    
    @staticmethod
    def validate_request(request):
        """Validate request before processing"""
        # Implementation would validate request structure
        return True
    
    @staticmethod
    def estimate_processing_cost(request, target_model):
        """Estimate processing cost for request"""
        # Implementation would estimate costs
        return 0.03
    
    @staticmethod
    def compare_processing_strategies(request, strategies):
        """Compare different processing strategies for request"""
        # Implementation would run comparisons
        return {}


# =============================================================================
# VERSION & METADATA
# =============================================================================

__version__ = "2.1.0"
__author__ = "SensationGifts AI Team"
__description__ = "Intelligent AI processing pipeline with optimization and learning"

# Processing Capabilities
PROCESSING_FEATURES = [
    "dynamic_prompt_building",
    "intelligent_model_selection", 
    "structured_response_parsing",
    "real_time_optimization",
    "adaptive_learning",
    "cost_performance_analysis",
    "multi_model_orchestration"
]

# Supported Optimization Strategies
OPTIMIZATION_STRATEGIES = [
    "cost_efficiency",
    "performance_maximization",
    "balanced_roi",
    "user_satisfaction",
    "resource_utilization"
]

# =============================================================================
# PUBLIC API: Clean interface for other modules
# =============================================================================

__all__ = [
    # === CORE COMPONENTS ===
    "PromptBuilder",
    "ResponseParser", 
    "ModelSelector",
    "OptimizationEngine",
    
    # === HIGH-LEVEL ORCHESTRATION ===
    "IntelligentProcessor",
    "AIProcessingPipeline",
    
    # === STRATEGY ENUMS ===
    "PromptBuildingStrategy",
    "ParsingStrategy",
    "SelectionStrategy", 
    "OptimizationObjective",
    
    # === RESULT CLASSES ===
    "ParsedResponse",
    "ValidationResult",
    "ModelPerformanceSnapshot",
    "OptimizationResult",
    
    # === UTILITIES ===
    "ProcessorFactory",
    "ProcessorUtils",
    
    # === STATUS ENUMS ===
    "ModelHealthStatus",
    "ValidationSeverity",
    "OutputFormat",
    "RequestComplexity",
]


# =============================================================================
# DEVELOPMENT UTILITIES
# =============================================================================

def create_sample_processing_workflow():
    """Create sample workflow for testing and documentation"""
    
    workflow_steps = {
        "step_1": "Initialize IntelligentProcessor",
        "step_2": "Call process_gift_recommendation_request with request",
        "step_3": "Execute returned pipeline with AI model",
        "step_4": "Parse AI response with parse_ai_response",
        "step_5": "Record actual performance with record_actual_performance",
        "step_6": "Monitor analytics with get_processing_analytics"
    }
    
    return workflow_steps


def validate_processor_setup():
    """Validate that all processor components are properly configured"""
    
    try:
        # Test processor creation
        processor = IntelligentProcessor()
        
        # Test component availability
        assert processor.prompt_builder is not None
        assert processor.response_parser is not None
        assert processor.model_selector is not None
        assert processor.optimization_engine is not None
        
        return {
            "status": "success",
            "message": "All processor components are properly configured",
            "components": ["PromptBuilder", "ResponseParser", "ModelSelector", "OptimizationEngine"],
            "orchestration": ["IntelligentProcessor", "AIProcessingPipeline"],
            "capabilities": PROCESSING_FEATURES
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Processor setup validation failed: {str(e)}"
        }


def get_processor_capabilities():
    """Get comprehensive overview of processor capabilities"""
    
    return {
        "version": __version__,
        "processing_features": PROCESSING_FEATURES,
        "optimization_strategies": OPTIMIZATION_STRATEGIES,
        "supported_models": ["OpenAI GPT-4", "Groq Mixtral", "Anthropic Claude", "Google Gemini"],
        "parsing_formats": ["JSON", "Structured Text", "Mixed Format", "Unstructured Text"],
        "prompt_techniques": ["Few-Shot", "Chain-of-Thought", "Dynamic Generation", "Advanced Techniques"],
        "optimization_dimensions": ["Cost", "Quality", "Speed", "User Satisfaction", "Resource Efficiency"],
        "learning_capabilities": ["Real-time Adaptation", "Performance Prediction", "Weight Optimization"],
        "integration_interfaces": ["High-level Pipeline", "Component-level Access", "Factory Creation"]
    }


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class ProcessorMetrics:
    """Global metrics for processor performance monitoring"""
    
    def __init__(self):
        self.global_stats = {
            "total_processors_created": 0,
            "total_requests_processed": 0,
            "average_processing_time": 0.0,
            "optimization_success_rate": 0.0,
            "cost_savings_achieved": 0.0
        }
    
    def track_processor_creation(self):
        """Track processor creation"""
        self.global_stats["total_processors_created"] += 1
    
    def track_request_processing(self, processing_time_ms, success, cost_savings=0.0):
        """Track request processing"""
        self.global_stats["total_requests_processed"] += 1
        
        # Update average processing time
        total_requests = self.global_stats["total_requests_processed"]
        current_avg = self.global_stats["average_processing_time"]
        self.global_stats["average_processing_time"] = \
            ((current_avg * (total_requests - 1)) + processing_time_ms) / total_requests
        
        # Update optimization success rate
        if success:
            current_success_rate = self.global_stats["optimization_success_rate"]
            self.global_stats["optimization_success_rate"] = \
                ((current_success_rate * (total_requests - 1)) + 1.0) / total_requests
        
        # Update cost savings
        self.global_stats["cost_savings_achieved"] += cost_savings
    
    def get_global_metrics(self):
        """Get global processor metrics"""
        return self.global_stats.copy()


# Global metrics instance
_processor_metrics = ProcessorMetrics()

def get_global_processor_metrics():
    """Access global processor metrics"""
    return _processor_metrics.get_global_metrics()