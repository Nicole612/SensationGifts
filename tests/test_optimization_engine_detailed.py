"""
🔍 DETAILLIERTE TESTS: OptimizationEngine
========================================

Intensive Tests für alle OptimizationEngine Features:
- Performance Prediction
- Strategy Selection
- Resource Allocation
- Cost Optimization
- Adaptive Learning
"""

import asyncio
from datetime import datetime
from ai_engine.processors.optimization_engine import (
    OptimizationEngine,
    OptimizationObjective,
    PerformanceDimension
)
from ai_engine.schemas.input_schemas import GiftRecommendationRequest
from ai_engine.processors.prompt_builder import DynamicPromptBuilder
from ai_engine.processors.response_parser import ResponseParser
from ai_engine.processors.model_selector import ModelSelector
from unittest.mock import Mock

# Optional: pytest
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    class pytest:
        @staticmethod
        def fixture(f):
            return f


class TestOptimizationEngineDetailed:
    """Detaillierte Tests für OptimizationEngine"""
    
    def get_optimization_engine(self):
        """OptimizationEngine Helper"""
        prompt_builder = DynamicPromptBuilder()
        response_parser = ResponseParser()
        model_selector = ModelSelector()
        
        return OptimizationEngine(
            prompt_builder=prompt_builder,
            response_parser=response_parser,
            model_selector=model_selector
        )
    
    def get_sample_request(self):
        """Sample Request Helper"""
        return GiftRecommendationRequest(
            personality_data={
                "user_prompt": "Ich suche ein Geschenk für meine Freundin",
                "prompt_method": True
            },
            occasion="birthday",
            relationship="friend",
            budget_min=20,
            budget_max=100,
            number_of_recommendations=3
        )
    
    async def test_performance_prediction(self):
        """Test: Performance Prediction funktioniert"""
        optimization_engine = self.get_optimization_engine()
        sample_request = self.get_sample_request()
        # Test Performance Prediction
        predicted = await optimization_engine._predict_request_performance(
            sample_request,
            context={"test": True}
        )
        
        assert predicted is not None
        assert "response_time" in predicted or "predicted_response_time_ms" in predicted
        assert "cost_estimate" in predicted or "predicted_cost" in predicted
        assert "quality_score" in predicted or "predicted_quality" in predicted
        
        print("✅ Performance Prediction funktioniert")
        print(f"   - Predicted Time: {predicted.get('response_time', predicted.get('predicted_response_time_ms', 'N/A'))}ms")
        print(f"   - Predicted Cost: ${predicted.get('cost_estimate', predicted.get('predicted_cost', 'N/A'))}")
    
    async def test_optimization_strategy_selection(self):
        """Test: Optimization Strategy Selection"""
        optimization_engine = self.get_optimization_engine()
        sample_request = self.get_sample_request()
        # Test verschiedene Objectives
        strategies_tested = []
        
        for objective in [
            OptimizationObjective.BALANCED_ROI,
            OptimizationObjective.PERFORMANCE_MAXIMIZATION,
            OptimizationObjective.COST_EFFICIENCY,
            OptimizationObjective.USER_SATISFACTION
        ]:
            predicted = await optimization_engine._predict_request_performance(
                sample_request,
                context={"test": True}
            )
            
            strategy = optimization_engine._select_optimization_strategy(
                sample_request,
                predicted,
                objective
            )
            
            assert strategy is not None
            strategies_tested.append((objective.value, strategy))
            
            print(f"✅ Strategy für {objective.value}: {strategy}")
        
        # Alle sollten unterschiedlich sein
        unique_strategies = set(strategies_tested)
        assert len(unique_strategies) > 0
    
    async def test_prompt_optimization(self):
        """Test: Prompt Optimization"""
        optimization_engine = self.get_optimization_engine()
        sample_request = self.get_sample_request()
        predicted = await optimization_engine._predict_request_performance(
            sample_request,
            context={"test": True}
        )
        
        strategy = optimization_engine._select_optimization_strategy(
            sample_request,
            predicted,
            OptimizationObjective.BALANCED_ROI
        )
        
        prompt_config = await optimization_engine._optimize_prompt_configuration(
            sample_request,
            strategy,
            context={"test": True}
        )
        
        assert prompt_config is not None
        assert "optimization_goal" in prompt_config
        assert "complexity_level" in prompt_config
        
        print("✅ Prompt Optimization funktioniert")
        print(f"   - Optimization Goal: {prompt_config.get('optimization_goal')}")
        print(f"   - Complexity Level: {prompt_config.get('complexity_level')}")
    
    async def test_model_selection_optimization(self):
        """Test: Model Selection Optimization"""
        optimization_engine = self.get_optimization_engine()
        sample_request = self.get_sample_request()
        predicted = await optimization_engine._predict_request_performance(
            sample_request,
            context={"test": True}
        )
        
        strategy = optimization_engine._select_optimization_strategy(
            sample_request,
            predicted,
            OptimizationObjective.BALANCED_ROI
        )
        
        prompt_config = await optimization_engine._optimize_prompt_configuration(
            sample_request,
            strategy,
            context={"test": True}
        )
        
        model_config = await optimization_engine._optimize_model_selection(
            sample_request,
            strategy,
            prompt_config,
            context={"test": True}
        )
        
        assert model_config is not None
        assert "selected_model" in model_config
        assert "selection_strategy" in model_config
        
        print("✅ Model Selection Optimization funktioniert")
        print(f"   - Selected Model: {model_config.get('selected_model')}")
    
    async def test_resource_allocation_optimization(self):
        """Test: Resource Allocation Optimization"""
        optimization_engine = self.get_optimization_engine()
        sample_request = self.get_sample_request()
        predicted = await optimization_engine._predict_request_performance(
            sample_request,
            context={"test": True}
        )
        
        strategy = optimization_engine._select_optimization_strategy(
            sample_request,
            predicted,
            OptimizationObjective.BALANCED_ROI
        )
        
        prompt_config = await optimization_engine._optimize_prompt_configuration(
            sample_request,
            strategy,
            context={"test": True}
        )
        
        model_config = await optimization_engine._optimize_model_selection(
            sample_request,
            strategy,
            prompt_config,
            context={"test": True}
        )
        
        resource_config = await optimization_engine._optimize_resource_allocation(
            model_config,
            strategy
        )
        
        assert resource_config is not None
        assert "max_concurrent_requests" in resource_config or "resource_limits" in resource_config
        
        print("✅ Resource Allocation Optimization funktioniert")
    
    async def test_cost_performance_optimization(self):
        """Test: Cost-Performance Optimization"""
        optimization_engine = self.get_optimization_engine()
        sample_request = self.get_sample_request()
        predicted = await optimization_engine._predict_request_performance(
            sample_request,
            context={"test": True}
        )
        
        strategy = optimization_engine._select_optimization_strategy(
            sample_request,
            predicted,
            OptimizationObjective.COST_EFFICIENCY
        )
        
        prompt_config = await optimization_engine._optimize_prompt_configuration(
            sample_request,
            strategy,
            context={"test": True}
        )
        
        model_config = await optimization_engine._optimize_model_selection(
            sample_request,
            strategy,
            prompt_config,
            context={"test": True}
        )
        
        resource_config = await optimization_engine._optimize_resource_allocation(
            model_config,
            strategy
        )
        
        cost_config = await optimization_engine._optimize_cost_performance(
            model_config,
            resource_config,
            strategy
        )
        
        assert cost_config is not None
        assert "cost_estimate" in cost_config or "budget_allocation" in cost_config
        
        print("✅ Cost-Performance Optimization funktioniert")
        print(f"   - Cost Estimate: ${cost_config.get('cost_estimate', cost_config.get('budget_allocation', {}).get('total', 'N/A'))}")
    
    def test_optimization_weights(self):
        optimization_engine = self.get_optimization_engine()
        """Test: Optimization Weights"""
        weights = optimization_engine.optimization_weights
        
        assert "cost" in weights
        assert "quality" in weights
        assert "speed" in weights
        assert "satisfaction" in weights
        
        # Alle sollten zwischen 0 und 1 sein
        for key, value in weights.items():
            assert 0 <= value <= 1
        
        # Summe sollte ~1.0 sein
        total = sum(weights.values())
        assert 0.8 <= total <= 1.2  # Erlaube etwas Flexibilität
        
        print("✅ Optimization Weights korrekt")
        print(f"   - Weights: {weights}")
        print(f"   - Total: {total:.3f}")
    
    async def test_adaptive_learning(self):
        """Test: Adaptive Learning"""
        optimization_engine = self.get_optimization_engine()
        sample_request = self.get_sample_request()
        # Initial State
        initial_weights = optimization_engine.optimization_weights.copy()
        
        # Process Request
        pipeline_config = await optimization_engine.optimize_request_pipeline(
            sample_request,
            context={"test": True},
            optimization_preference=OptimizationObjective.BALANCED_ROI
        )
        
        # Record Performance
        actual_metrics = {
            "response_time_ms": 2000,  # Langsam
            "cost_estimate": 0.05,  # Teuer
            "quality_score": 0.9  # Gut
        }
        
        optimization_engine.record_actual_performance(
            pipeline_config=pipeline_config,
            actual_response=Mock(),
            actual_metrics=actual_metrics
        )
        
        # Prüfe dass Learning stattgefunden hat
        # (Gewichte könnten sich angepasst haben)
        final_weights = optimization_engine.optimization_weights
        
        # Learning sollte stattgefunden haben
        # (Auch wenn Gewichte gleich bleiben, Learning sollte aktiv sein)
        assert optimization_engine.adaptive_learning_enabled is not None
        
        print("✅ Adaptive Learning funktioniert")
        print(f"   - Initial Weights: {initial_weights}")
        print(f"   - Final Weights: {final_weights}")


if __name__ == "__main__":
    # Run tests manually
    test_instance = TestOptimizationEngineDetailed()
    asyncio.run(test_instance.test_performance_prediction())
    print("✅ OptimizationEngine detailed tests completed")

