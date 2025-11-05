#!/usr/bin/env python3
"""
SIMPLE AI Test - Demo-Ready Version

ZWECK: Minimaler Test der AI-Engine
ZEIT: ~2 Minuten
ZIEL: Sicherstellen dass Basic AI-Funktionalität läuft
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test 1: Basic Imports"""
    print("🔍 Testing Basic Imports...")
    
    try:
        # Test core imports
        from ai_engine.models.base_client import AIModelType, GiftRecommendationSchema
        print("  ✅ Core types imported")
        
        # Test OpenAI client
        from ai_engine.models.openai_client import OpenAIClient
        print("  ✅ OpenAI client imported")
        
        # Test Groq client  
        from ai_engine.models.groq_client import GroqClient
        print("  ✅ Groq client imported")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_client_creation():
    """Test 2: Client Creation"""
    print("\n🛠️  Testing Client Creation...")
    
    try:
        from ai_engine.models.openai_client import OpenAIClient
        from ai_engine.models.groq_client import GroqClient
        from ai_engine.models.base_client import AIModelType
        
        # Test with dummy API keys (just creation, not actual calls)
        openai_client = OpenAIClient("dummy-key", AIModelType.OPENAI_GPT4)
        print("  ✅ OpenAI client created")
        
        groq_client = GroqClient("dummy-key", AIModelType.GROQ_MIXTRAL)  
        print("  ✅ Groq client created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Client creation failed: {e}")
        return False

def test_settings_import():
    """Test 3: Settings Import"""
    print("\n⚙️  Testing Settings...")
    
    try:
        from config.settings import get_settings
        settings = get_settings()
        print("  ✅ Settings imported")
        
        # Check API keys
        api_keys = {
            "OpenAI": bool(settings.openai_api_key),
            "Groq": bool(settings.groq_api_key),
            "Gemini": bool(settings.gemini_api_key),
            "Anthropic": bool(settings.anthropic_api_key)
        }
        
        configured_count = sum(api_keys.values())
        print(f"  📊 API keys configured: {configured_count}/4")
        
        for provider, configured in api_keys.items():
            status = "✅" if configured else "⚠️ "
            print(f"    {status} {provider}: {'Configured' if configured else 'Not configured'}")
        
        return configured_count > 0
        
    except Exception as e:
        print(f"  ❌ Settings test failed: {e}")
        return False

def test_actual_api_call():
    """Test 4: Real API Call (if API key available)"""
    print("\n🚀 Testing Real API Call...")
    
    try:
        from config.settings import get_settings
        from ai_engine.models.openai_client import OpenAIClient
        from ai_engine.models.groq_client import GroqClient
        from ai_engine.models.base_client import AIModelType
        
        settings = get_settings()
        
        # Try OpenAI first
        if settings.openai_api_key:
            print("  🧠 Testing OpenAI API call...")
            try:
                client = OpenAIClient(settings.openai_api_key, AIModelType.OPENAI_GPT4)
                start_time = time.time()
                response = client.generate_text("Say 'Hello World!'", max_tokens=10)
                response_time = time.time() - start_time
                
                if response.success:
                    print(f"    ✅ OpenAI SUCCESS ({response_time:.2f}s, ${response.cost:.4f})")
                    print(f"    📝 Response: {response.content[:50]}...")
                    return True
                else:
                    print(f"    ❌ OpenAI FAILED: {response.error}")
                    
            except Exception as e:
                print(f"    ❌ OpenAI Exception: {e}")
        
        # Try Groq if OpenAI failed
        if settings.groq_api_key:
            print("  ⚡ Testing Groq API call...")
            try:
                client = GroqClient(settings.groq_api_key, AIModelType.GROQ_MIXTRAL)
                start_time = time.time()
                response = client.generate_text("Say 'Hello World!'", max_tokens=10)
                response_time = time.time() - start_time
                
                if response.success:
                    print(f"    ✅ Groq SUCCESS ({response_time:.2f}s, ${response.cost:.4f})")
                    print(f"    📝 Response: {response.content[:50]}...")
                    return True
                else:
                    print(f"    ❌ Groq FAILED: {response.error}")
                    
            except Exception as e:
                print(f"    ❌ Groq Exception: {e}")
        
        print("  ⚠️  No working API calls - check API keys")
        return False
        
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        return False

def test_gift_recommendation():
    """Test 5: Gift Recommendation"""
    print("\n🎁 Testing Gift Recommendation...")
    
    try:
        from config.settings import get_settings
        from ai_engine.models.openai_client import OpenAIClient
        from ai_engine.models.groq_client import GroqClient
        from ai_engine.models.base_client import AIModelType
        
        settings = get_settings()
        
        # Demo personality profile
        demo_profile = {
            "personality_scores": {"openness": 0.8, "extraversion": 0.6},
            "hobbies": ["photography", "travel"], 
            "emotional_triggers": ["adventure", "creativity"]
        }
        
        # Try gift recommendation
        client = None
        if settings.openai_api_key:
            client = OpenAIClient(settings.openai_api_key, AIModelType.OPENAI_GPT4)
            method_name = "recommend_gift_with_reasoning"
        elif settings.groq_api_key:
            client = GroqClient(settings.groq_api_key, AIModelType.GROQ_MIXTRAL)
            method_name = "fast_gift_recommendation"
        
        if client:
            print(f"  🎯 Generating gift recommendation...")
            try:
                if hasattr(client, method_name):
                    method = getattr(client, method_name)
                    recommendation = method(
                        personality_profile=demo_profile,
                        occasion="birthday",
                        budget_range="50-150", 
                        relationship="friend"
                    )
                    
                    print(f"    ✅ GIFT RECOMMENDATION SUCCESS!")
                    print(f"    🎁 Gift: {recommendation.gift_name}")
                    print(f"    📊 Match Score: {recommendation.match_score:.2f}")
                    print(f"    🎯 Confidence: {recommendation.confidence:.2f}")
                    return True
                else:
                    print(f"    ⚠️  Method {method_name} not available, trying basic recommend_gift...")
                    recommendation = client.recommend_gift(
                        personality_profile=demo_profile,
                        occasion="birthday",
                        budget_range="50-150",
                        relationship="friend"
                    )
                    print(f"    ✅ BASIC GIFT RECOMMENDATION SUCCESS!")
                    print(f"    🎁 Gift: {recommendation.gift_name}")
                    return True
                    
            except Exception as e:
                print(f"    ❌ Gift recommendation failed: {e}")
                return False
        else:
            print("  ⚠️  No API clients available for gift recommendation")
            return False
            
    except Exception as e:
        print(f"  ❌ Gift recommendation test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("🎯 SensationGifts - Simple AI Engine Test")
    print("=" * 50)
    print("🎬 Demo-Ready Check...")
    print()
    
    test_results = []
    
    # Run all tests
    test_results.append(("Imports", test_basic_imports()))
    test_results.append(("Client Creation", test_client_creation()))
    test_results.append(("Settings", test_settings_import()))
    test_results.append(("API Call", test_actual_api_call()))
    test_results.append(("Gift Recommendation", test_gift_recommendation()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEMO READINESS SUMMARY")
    print("=" * 50)
    
    passed_tests = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed_tests += 1
    
    print(f"\nScore: {passed_tests}/{len(test_results)} tests passed")
    
    # Demo readiness verdict
    if passed_tests >= 3:  # At least imports, creation, and settings
        print("\n🎉 DEMO READY! 🎉")
        print("✅ Your AI-Engine is working!")
        
        if passed_tests >= 4:
            print("🚀 API calls working - EXCELLENT!")
        if passed_tests == 5:
            print("🏆 Full gift recommendations working - PERFECT!")
        
        print("\n📋 Demo Talking Points:")
        print("  • Multi-AI-Provider Architecture")
        print("  • Clean Code Structure") 
        print("  • Production-Ready Error Handling")
        if passed_tests >= 4:
            print("  • Live AI API Integration")
        if passed_tests == 5:
            print("  • Real Gift Recommendations")
        
        return True
    else:
        print("\n🚨 NOT DEMO READY")
        print("❌ Fix issues before")
        
        if passed_tests < 2:
            print("💡 Major import/structure issues - check file paths")
        elif passed_tests < 3:
            print("💡 Configuration issues - check .env file")
        else:
            print("💡 API issues - check API keys")
            
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)