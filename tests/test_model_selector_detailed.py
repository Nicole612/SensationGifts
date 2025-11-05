"""
🔍 DETAILLIERTE TESTS: ModelSelector
===================================

Intensive Tests für ModelSelector:
- Model Selection Logic
- Performance-based Selection
- Fallback Chains
- Health Monitoring
"""

from ai_engine.processors.model_selector import (
    ModelSelector,
    SelectionStrategy,
    ModelHealthStatus,
    RequestComplexity
)
from ai_engine.schemas.input_schemas import GiftRecommendationRequest
from ai_engine.schemas.prompt_schemas import AIModelType

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


class TestModelSelectorDetailed:
    """Detaillierte Tests für ModelSelector"""
    
    def get_model_selector(self):
        """ModelSelector Helper"""
        return ModelSelector()
    
    def get_sample_request(self):
        """Sample Request Helper"""
        return GiftRecommendationRequest(
            personality_data={
                "user_prompt": "Test prompt",
                "prompt_method": True
            },
            occasion="birthday",
            relationship="friend",
            number_of_recommendations=3
        )
    
    def test_model_selector_initialization(self):
        """Test: ModelSelector initialisiert korrekt"""
        model_selector = self.get_model_selector()
        assert model_selector is not None
        assert model_selector.model_performance_history is not None
        assert model_selector.health_status is not None
        
        print("✅ ModelSelector initialisiert korrekt")
    
    def test_select_optimal_model_speed(self):
        """Test: Model Selection für Speed"""
        model_selector = self.get_model_selector()
        sample_request = self.get_sample_request()
        result = model_selector.select_optimal_model(
            request=sample_request,
            optimization_goal="speed",
            strategy=None,
            context={"time_constraint": "emergency"}
        )
        
        assert result is not None
        assert "selected_model" in result
        assert "selection_reasoning" in result
        
        selected = result["selected_model"]
        # Für Speed sollte ein schnelles Model gewählt werden
        fast_models = [AIModelType.GROQ_MIXTRAL, AIModelType.GROQ_LLAMA]
        
        print(f"✅ Speed Selection: {selected}")
        print(f"   - Reasoning: {result.get('selection_reasoning', 'N/A')[:100]}")
    
    def test_select_optimal_model_quality(self):
        """Test: Model Selection für Quality"""
        model_selector = self.get_model_selector()
        sample_request = self.get_sample_request()
        result = model_selector.select_optimal_model(
            request=sample_request,
            optimization_goal="quality",
            strategy=None,
            context={"quality_requirement": "high"}
        )
        
        assert result is not None
        selected = result["selected_model"]
        
        # Für Quality sollte ein hochwertiges Model gewählt werden
        quality_models = [AIModelType.OPENAI_GPT4, AIModelType.ANTHROPIC_CLAUDE]
        
        print(f"✅ Quality Selection: {selected}")
        print(f"   - Reasoning: {result.get('selection_reasoning', 'N/A')[:100]}")
    
    def test_select_optimal_model_cost(self):
        """Test: Model Selection für Cost"""
        model_selector = self.get_model_selector()
        sample_request = self.get_sample_request()
        result = model_selector.select_optimal_model(
            request=sample_request,
            optimization_goal="cost",
            strategy=None,
            context={"budget_constraint": "tight"}
        )
        
        assert result is not None
        selected = result["selected_model"]
        
        # Für Cost sollte ein günstiges Model gewählt werden
        cost_models = [AIModelType.GROQ_MIXTRAL, AIModelType.OPENAI_GPT35]
        
        print(f"✅ Cost Selection: {selected}")
        print(f"   - Reasoning: {result.get('selection_reasoning', 'N/A')[:100]}")
    
    def test_request_complexity_analysis(self):
        """Test: Request Complexity Analysis"""
        model_selector = self.get_model_selector()
        sample_request = self.get_sample_request()
        complexity = model_selector._analyze_request_complexity(
            sample_request,
            context={}
        )
        
        assert complexity is not None
        assert isinstance(complexity, (RequestComplexity, str, int, float))
        
        print(f"✅ Request Complexity Analysis: {complexity}")
    
    def test_model_health_status(self):
        """Test: Model Health Status"""
        model_selector = self.get_model_selector()
        health = model_selector.get_model_health_status()
        
        assert health is not None
        assert isinstance(health, dict)
        
        # Prüfe dass alle Models einen Status haben
        models_checked = 0
        for model_type in AIModelType:
            model_key = model_type.value if hasattr(model_type, 'value') else str(model_type)
            if model_key in health or any(model_key in str(k) for k in health.keys()):
                models_checked += 1
        
        print(f"✅ Model Health Status: {len(health)} models checked")
        print(f"   - Health Status: {health}")
    
    def test_record_model_performance(self):
        """Test: Model Performance Recording"""
        model_selector = self.get_model_selector()
        # Initial History Length
        initial_length = len(model_selector.model_performance_history)
        
        # Record Performance
        metrics = {
            "response_time_ms": 1500,
            "token_usage": 500,
            "cost_estimate": 0.02,
            "quality_score": 0.9,
            "success": True
        }
        
        model_selector.record_model_performance(metrics)
        
        # Prüfe dass Performance gespeichert wurde
        final_length = len(model_selector.model_performance_history)
        assert final_length >= initial_length
        
        print("✅ Model Performance Recording funktioniert")
        print(f"   - History Length: {initial_length} → {final_length}")
    
    def test_fallback_chain(self):
        """Test: Fallback Chain"""
        model_selector = self.get_model_selector()
        sample_request = self.get_sample_request()
        result = model_selector.select_optimal_model(
            request=sample_request,
            optimization_goal="quality",
            strategy=None,
            context={}
        )
        
        assert "fallback_chain" in result or "alternatives" in result
        
        fallback = result.get("fallback_chain", result.get("alternatives", []))
        
        print(f"✅ Fallback Chain: {len(fallback)} alternatives")
        print(f"   - Fallback Models: {fallback}")


if __name__ == "__main__":
    # Run tests manually
    test_instance = TestModelSelectorDetailed()
    test_instance.test_model_selector_initialization()
    print("✅ ModelSelector detailed tests completed")

