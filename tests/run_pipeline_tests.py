#!/usr/bin/env python3
"""
🚀 VOLLSTÄNDIGE PIPELINE TEST RUNNER
====================================

Führt alle Tests für die AI-Processing-Pipeline aus.
"""

import sys
import os
import asyncio
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Tests
from tests.test_complete_pipeline import (
    TestIntelligentProcessor,
    TestOptimizationEngine,
    TestModelSelector,
    TestCompletePipeline,
    TestAsyncClients,
    TestGiftFinderIntegration,
    TestErrorHandling,
    TestPerformanceBenchmarks
)

# Import Components
from ai_engine.processors import IntelligentProcessor
from ai_engine.processors.optimization_engine import OptimizationObjective
from ai_engine.schemas.input_schemas import GiftRecommendationRequest


class TestRunner:
    """Test Runner für alle Pipeline Tests"""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
    
    def run_test_class(self, test_class, class_name):
        """Führt alle Tests einer Klasse aus"""
        print(f"\n{'='*80}")
        print(f"🧪 Testing: {class_name}")
        print(f"{'='*80}")
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_') and callable(getattr(test_instance, method))]
        
        passed = 0
        failed = 0
        errors = []
        
        for method_name in test_methods:
            try:
                print(f"\n  📝 {method_name}...", end=" ")
                method = getattr(test_instance, method_name)
                
                # Check if async
                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method())
                else:
                    method()
                
                print("✅ PASSED")
                passed += 1
                
            except AssertionError as e:
                print(f"❌ FAILED (Assertion)")
                print(f"     Error: {str(e)[:200]}")
                failed += 1
                errors.append(f"{method_name}: {str(e)}")
                
            except Exception as e:
                print(f"❌ ERROR")
                print(f"     Error: {str(e)[:200]}")
                failed += 1
                errors.append(f"{method_name}: {str(e)}")
        
        self.results[class_name] = {
            "passed": passed,
            "failed": failed,
            "errors": errors
        }
        
        print(f"\n  📊 Results: ✅ {passed} passed, ❌ {failed} failed")
        
        return passed, failed
    
    def run_all_tests(self):
        """Führt alle Tests aus"""
        self.start_time = time.time()
        
        print("=" * 80)
        print("🚀 VOLLSTÄNDIGE AI-PROCESSING-PIPELINE TESTS")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test Suites
        test_classes = [
            (TestIntelligentProcessor, "IntelligentProcessor"),
            (TestOptimizationEngine, "OptimizationEngine"),
            (TestModelSelector, "ModelSelector"),
            (TestCompletePipeline, "Complete Pipeline"),
            (TestAsyncClients, "Async Clients"),
            (TestGiftFinderIntegration, "Gift Finder Integration"),
            (TestErrorHandling, "Error Handling"),
            (TestPerformanceBenchmarks, "Performance Benchmarks")
        ]
        
        total_passed = 0
        total_failed = 0
        
        for test_class, class_name in test_classes:
            try:
                passed, failed = self.run_test_class(test_class, class_name)
                total_passed += passed
                total_failed += failed
            except Exception as e:
                print(f"❌ Error running {class_name}: {e}")
                total_failed += 1
        
        # Summary
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        for class_name, result in self.results.items():
            status = "✅" if result["failed"] == 0 else "⚠️"
            print(f"{status} {class_name:40} ✅ {result['passed']:3} ❌ {result['failed']:3}")
            if result["errors"]:
                for error in result["errors"][:3]:  # Show first 3 errors
                    print(f"      ⚠️  {error[:70]}")
        
        print("-" * 80)
        print(f"{'TOTAL':42} ✅ {total_passed:3} ❌ {total_failed:3}")
        print(f"Time Elapsed: {elapsed_time:.2f}s")
        print("=" * 80)
        
        # Success Rate
        total_tests = total_passed + total_failed
        if total_tests > 0:
            success_rate = (total_passed / total_tests) * 100
            print(f"\nSuccess Rate: {success_rate:.1f}%")
            
            if success_rate >= 80:
                print("🎉 EXCELLENT: Most tests passed!")
            elif success_rate >= 60:
                print("✅ GOOD: Most tests passed, some issues to fix")
            else:
                print("⚠️  NEEDS ATTENTION: Many tests failed")
        
        return total_failed == 0


def main():
    """Main Entry Point"""
    runner = TestRunner()
    success = runner.run_all_tests()
    
    # Exit Code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

