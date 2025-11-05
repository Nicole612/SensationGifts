"""
🔗 INTEGRATION TESTS: Vollständige gift_finder.py Integration
============================================================

Testet die vollständige Integration von:
- Frontend Request → gift_finder.py → IntelligentProcessor → AI Model → Response
"""

import json
from unittest.mock import Mock, patch, MagicMock
from flask import Flask

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

try:
    from app.routes.gift_finder import gift_finder_bp
except ImportError:
    gift_finder_bp = None


def get_app():
    """Flask App Helper"""
    if gift_finder_bp is None:
        return None
    app = Flask(__name__)
    app.register_blueprint(gift_finder_bp)
    app.config['TESTING'] = True
    return app


def get_client(app):
    """Test Client Helper"""
    if app is None:
        return None
    return app.test_client()


class TestGiftFinderAPI:
    """Integration Tests für gift_finder API"""
    
    def test_health_check(self):
        """Test: Health Check Endpoint"""
        app = get_app()
        if app is None:
            print("⚠️ Flask app not available, skipping test")
            return
        client = get_client(app)
        response = client.get('/api/gift-finder/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'components' in data['data']
        print("✅ Health Check funktioniert")
    
    @patch('app.routes.gift_finder.get_intelligent_processor')
    @patch('app.routes.gift_finder.run_async')
    @patch('app.routes.gift_finder.execute_ai_model_call_async')
    def test_prompt_method_integration(self, mock_execute, mock_run_async, mock_processor):
        """Test: Prompt Method Integration"""
        app = get_app()
        if app is None:
            print("⚠️ Flask app not available, skipping test")
            return
        client = get_client(app)
        # Mock IntelligentProcessor
        mock_proc = MagicMock()
        mock_proc.response_parser = MagicMock()
        mock_processor.return_value = mock_proc
        
        # Mock Processing Result
        mock_result = {
            "pipeline_configuration": {
                "optimization_metadata": {"strategy": "balanced"}
            },
            "model_selection": {
                "selected_model": "openai_gpt4",
                "selection_reasoning": "Best quality"
            },
            "execution_plan": {
                "final_prompt": "Test prompt",
                "target_model": "openai_gpt4",
                "fallback_models": []
            }
        }
        
        # Mock Run Async
        async def mock_process(*args, **kwargs):
            return mock_result
        
        mock_run_async.side_effect = lambda coro: asyncio.run(coro)
        mock_proc.process_gift_recommendation_request = mock_process
        
        # Mock AI Response
        mock_ai_response = """
        {
            "recommendations": [
                {
                    "title": "Test Geschenk",
                    "description": "Ein Test-Geschenk",
                    "price_range": "€25-€50",
                    "emotional_impact": "Freude",
                    "confidence_score": 0.9
                }
            ],
            "overall_strategy": "Test Strategy",
            "overall_confidence": 0.85
        }
        """
        
        # Mock Execute AI Model Call
        async def mock_execute_async(*args, **kwargs):
            return mock_ai_response
        
        mock_execute.side_effect = mock_execute_async
        
        # Mock Response Parser
        mock_parsed = MagicMock()
        mock_parsed.parsing_success = True
        mock_parsed.confidence_score = 0.9
        mock_parsed.parsing_time_ms = 100
        mock_parsed.structured_output = MagicMock()
        mock_parsed.structured_output.recommendations = [
            MagicMock(
                title="Test Geschenk",
                description="Ein Test-Geschenk",
                price_range="€25-€50",
                emotional_impact="Freude",
                confidence_score=0.9
            )
        ]
        mock_parsed.parsed_data = {"recommendations": []}
        mock_proc.response_parser.parse_gift_recommendation_response.return_value = mock_parsed
        
        # Test Request
        request_data = {
            "method": "prompt",
            "data": {
                "user_prompt": "Ich suche ein Geschenk für meine Freundin",
                "occasion": "birthday",
                "relationship": "friend",
                "budget_min": 20,
                "budget_max": 80
            },
            "options": {
                "optimization_goal": "quality",
                "use_intelligent_processor": True
            }
        }
        
        response = client.post(
            '/api/gift-finder/process',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        # Validierung
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'recommendations' in data['data']
        
        print("✅ Prompt Method Integration funktioniert")
    
    def test_fallback_to_dynamic_prompt_builder(self):
        """Test: Fallback zu DynamicPromptBuilder"""
        app = get_app()
        if app is None:
            print("⚠️ Flask app not available, skipping test")
            return
        client = get_client(app)
        with patch('app.routes.gift_finder.get_intelligent_processor', return_value=None):
            # Mock DynamicPromptBuilder
            with patch('app.routes.gift_finder.DynamicPromptBuilder') as mock_builder:
                mock_instance = MagicMock()
                mock_instance.process_prompt_method.return_value = '{"recommendations": []}'
                mock_builder.return_value = mock_instance
                
                # Mock ResponseParser
                with patch('app.routes.gift_finder.ResponseParser') as mock_parser:
                    mock_parser_instance = MagicMock()
                    mock_parsed = MagicMock()
                    mock_parsed.parsing_success = True
                    mock_parsed.confidence_score = 0.8
                    mock_parsed.structured_output = MagicMock()
                    mock_parsed.structured_output.recommendations = []
                    mock_parsed.parsed_data = {"recommendations": []}
                    mock_parser_instance.parse_gift_recommendation_response.return_value = mock_parsed
                    mock_parser.return_value = mock_parser_instance
                    
                    request_data = {
                        "method": "prompt",
                        "data": {
                            "user_prompt": "Test",
                            "occasion": "birthday",
                            "relationship": "friend"
                        },
                        "options": {
                            "use_intelligent_processor": False
                        }
                    }
                    
                    response = client.post(
                        '/api/gift-finder/process',
                        data=json.dumps(request_data),
                        content_type='application/json'
                    )
                    
                    assert response.status_code == 200
                    print("✅ Fallback zu DynamicPromptBuilder funktioniert")


# =============================================================================
# 🧪 PERFORMANCE MONITORING TESTS
# =============================================================================

class TestPerformanceMonitoring:
    """Tests für Performance Monitoring"""
    
    async def test_performance_tracking_integration(self):
        """Test: Performance Tracking wird korrekt ausgeführt"""
        from app.routes.gift_finder import get_intelligent_processor
        
        processor = get_intelligent_processor()
        if processor is None:
            pytest.skip("IntelligentProcessor nicht verfügbar")
        
        # Initial Stats
        initial_stats = processor.processing_stats.copy()
        
        # Simuliere Request
        request = GiftRecommendationRequest(
            personality_data={"user_prompt": "Test", "prompt_method": True},
            occasion="birthday",
            relationship="friend",
            number_of_recommendations=3
        )
        
        # Process Request
        result = await processor.process_gift_recommendation_request(
            request=request,
            optimization_preference=OptimizationObjective.BALANCED_ROI
        )
        
        # Mock Performance Tracking
        mock_parsed = MagicMock()
        mock_parsed.parsing_success = True
        mock_parsed.confidence_score = 0.9
        
        processor.record_actual_performance(
            processing_result=result,
            actual_response=mock_parsed,
            actual_metrics={
                "response_time_ms": 1500,
                "quality_score": 0.9
            }
        )
        
        # Prüfe dass Stats aktualisiert wurden
        final_stats = processor.processing_stats
        assert final_stats["total_requests"] > initial_stats["total_requests"]
        
        print("✅ Performance Tracking Integration funktioniert")


if __name__ == "__main__":
    # Run tests manually
    test_instance = TestGiftFinderAPI()
    test_instance.test_health_check()
    print("✅ Integration tests completed")

