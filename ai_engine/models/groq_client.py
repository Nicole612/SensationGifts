"""
Groq Client - Ultra-Fast AI für Real-Time Gift Recommendations

SPEED FOCUS:
- 10x faster than OpenAI for many tasks
- Perfect für real-time user interactions
- Ideal für quick gift suggestions
- Optimized für speed over creativity

MODELS:
- Mixtral-8x7B: Best balance speed/quality
- Llama-3: Ultra-fast for simple tasks

USE CASES:
- Real-time gift suggestions while typing
- Quick personality insights
- Fast alternative generations
- Live chat recommendations
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    from groq import Groq
except ImportError:
    raise ImportError("Groq library not installed. Run: pip install groq")

from .base_client import (
    BaseAIClient, AIRequest, AIResponse, AIModelType,
    ModelCapability, ResponseFormat, GiftRecommendationSchema
)


class GroqClient(BaseAIClient):
    """
    Groq Ultra-Fast AI Client für Real-Time Gift Recommendations
    
    Optimized for:
    - Maximum speed (sub-second responses)
    - Real-time user interactions  
    - Quick gift suggestions
    - Live personality insights
    - Instant alternative generations
    """
    
    def __init__(self, 
                 api_key: str,
                 model_type: AIModelType = AIModelType.GROQ_MIXTRAL,
                 rate_limit_per_minute: int = 300):  # Groq has higher limits
        
        # Validate model type
        if model_type not in [AIModelType.GROQ_MIXTRAL, AIModelType.GROQ_LLAMA]:
            raise ValueError(f"Invalid model type for Groq: {model_type}")
        
        super().__init__(api_key, model_type, rate_limit_per_minute)
        
        # Initialize Groq client
        self.client = Groq(api_key=api_key)
        
        # Model configuration
        self.model_name = self._get_model_name()
        self.max_tokens_limit = self._get_max_tokens()
        
    def _get_model_name(self) -> str:
        """Maps our model type to Groq model name"""
        mapping = {
            AIModelType.GROQ_MIXTRAL: "llama3-70b-8192",
            AIModelType.GROQ_LLAMA: "llama3-70b-8192"
        }
        return mapping[self.model_type]
    
    def _get_max_tokens(self) -> int:
        """Get max tokens for this model"""
        limits = {
            AIModelType.GROQ_MIXTRAL: 8192,
            AIModelType.GROQ_LLAMA: 8192
        }
        return limits[self.model_type]
    
    def _get_model_capabilities(self) -> List[ModelCapability]:
        """Define what this Groq model can do"""
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.FAST_RESPONSE,
            ModelCapability.COST_EFFICIENT
        ]
        
        # Mixtral is better for JSON, Llama is faster for text
        if self.model_type == AIModelType.GROQ_MIXTRAL:
            capabilities.append(ModelCapability.JSON_OUTPUT)
        
        return capabilities
    
    def _get_model_pricing(self) -> Dict[str, float]:
        """Groq pricing (very cost-efficient!)"""
        pricing = {
            AIModelType.GROQ_MIXTRAL: {
                "input_per_token": 0.00000027,   # $0.27 per 1M tokens
                "output_per_token": 0.00000027   # Same for output
            },
            AIModelType.GROQ_LLAMA: {
                "input_per_token": 0.00000059,   # $0.59 per 1M tokens
                "output_per_token": 0.00000079   # $0.79 per 1M tokens
            }
        }
        return pricing[self.model_type]
    
    def _make_api_call(self, request: AIRequest) -> AIResponse:
        """
        Makes ultra-fast Groq API call
        """
        try:
            # Build messages
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            # Groq doesn't have native JSON mode, so we handle it via prompting
            if request.response_format == ResponseFormat.JSON:
                if "json" not in request.prompt.lower():
                    messages[-1]["content"] += "\n\nIMPORTANT: Respond with valid JSON only, no other text."
            
            # Build API parameters (optimized for speed)
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": min(request.max_tokens or 1000, self.max_tokens_limit),
                "stream": False  # No streaming for simplicity
            }
            
            # Make API call (should be very fast!)
            start_time = time.time()
            response = self.client.chat.completions.create(**api_params)
            response_time = time.time() - start_time
            
            # Extract response data
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Calculate cost (very cheap!)
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (
                input_tokens * self.pricing["input_per_token"] +
                output_tokens * self.pricing["output_per_token"]
            )
            
            # Parse JSON if requested
            parsed_json = None
            if request.response_format == ResponseFormat.JSON:
                try:
                    # Groq sometimes adds extra text, try to extract JSON
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_content = content[json_start:json_end]
                        parsed_json = json.loads(json_content)
                    else:
                        parsed_json = json.loads(content)
                except json.JSONDecodeError as e:
                    return AIResponse(
                        content=content,
                        model_type=self.model_type,
                        tokens_used=tokens_used,
                        cost=cost,
                        response_time=response_time,
                        timestamp=datetime.now(),
                        success=False,
                        error=f"Invalid JSON response: {e}"
                    )
            
            return AIResponse(
                content=content,
                model_type=self.model_type,
                tokens_used=tokens_used,
                cost=cost,
                response_time=response_time,
                timestamp=datetime.now(),
                success=True,
                parsed_json=parsed_json
            )
            
        except Exception as e:
            return AIResponse(
                content="",
                model_type=self.model_type,
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                timestamp=datetime.now(),
                success=False,
                error=f"Groq API Error: {e}"
            )
    
    # === SPEED-OPTIMIZED METHODS ===
    
    def quick_gift_suggestion(self,
                            personality_summary: str,
                            occasion: str,
                            budget: str,
                            max_suggestions: int = 3) -> List[str]:
        """
        Ultra-fast gift suggestions (< 1 second)
        
        Args:
            personality_summary: Brief personality description
            occasion: The occasion
            budget: Budget range as string
            max_suggestions: Number of suggestions
            
        Returns:
            List of quick gift ideas
        """
        prompt = f"""
        Quick gift ideas for:
        Person: {personality_summary}
        Occasion: {occasion}
        Budget: €{budget}
        
        Give {max_suggestions} specific, creative gift ideas.
        Keep each suggestion under 10 words.
        Focus on practical gifts they'd actually love.
        
        Format as JSON: {{"suggestions": ["idea1", "idea2", "idea3"]}}
        """
        
        response = self.generate_json(
            prompt=prompt,
            json_schema={"type": "object", "properties": {"suggestions": {"type": "array"}}},
            system_prompt="You are a fast gift consultant. Quick, practical suggestions only.",
            temperature=0.7
        )
        
        if response.success and response.parsed_json:
            return response.parsed_json.get("suggestions", [])
        return []
    
    def instant_personality_insights(self, personality_data: Dict) -> Dict[str, Any]:
        """
        Lightning-fast personality analysis for gifts
        """
        prompt = f"""
        Personality: {json.dumps(personality_data)}
        
        Quick insights for gift recommendations:
        1. What makes them happy? (2-3 words)
        2. Gift style? (practical/creative/experiential)
        3. Avoid what? (1-2 things)
        4. Personalization level? (low/medium/high)
        
        JSON format: {{"happy_triggers": [], "gift_style": "", "avoid": [], "personalization": ""}}
        """
        
        response = self.generate_json(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="Expert psychologist. Quick, actionable insights only.",
            temperature=0.4
        )
        
        return response.parsed_json if response.success else {}
    
    def real_time_alternatives(self, 
                             base_gift: str,
                             personality_hint: str,
                             price_range: str) -> List[str]:
        """
        Real-time alternative suggestions while user types
        """
        prompt = f"""
        Base gift: "{base_gift}"
        Person: {personality_hint}
        Budget: €{price_range}
        
        5 quick alternatives that are:
        - Similar emotional appeal
        - Different product category
        - Same price range
        - More personalized
        
        JSON: {{"alternatives": ["alt1", "alt2", "alt3", "alt4", "alt5"]}}
        """
        
        response = self.generate_json(
            prompt=prompt,
            json_schema={"type": "object", "properties": {"alternatives": {"type": "array"}}},
            temperature=0.8,  # More creative for alternatives
            system_prompt="Creative gift consultant. Think outside the box but stay practical."
        )
        
        if response.success and response.parsed_json:
            return response.parsed_json.get("alternatives", [])
        return []
    
    def validate_gift_idea(self,
                         gift_idea: str,
                         personality_profile: Dict,
                         occasion: str) -> Dict[str, Any]:
        """
        Quick validation if a gift idea is good
        """
        prompt = f"""
        Gift idea: "{gift_idea}"
        Person: {personality_profile.get('summary', 'Unknown')}
        Occasion: {occasion}
        
        Quick validation:
        - Match score 1-10?
        - Why good/bad? (1 sentence)
        - Better alternative? (if score < 7)
        
        JSON: {{"score": 8, "reason": "explanation", "alternative": "better idea or null"}}
        """
        
        response = self.generate_json(
            prompt=prompt,
            json_schema={"type": "object"},
            temperature=0.3,  # More consistent for validation
            system_prompt="Gift validation expert. Quick, honest assessments."
        )
        
        return response.parsed_json if response.success else {"score": 5, "reason": "Unknown", "alternative": None}
    
    # === OPTIMIZED GIFT RECOMMENDATION ===
    
    def fast_gift_recommendation(self,
                               personality_profile: Dict,
                               occasion: str,
                               budget_range: str,
                               relationship: str) -> GiftRecommendationSchema:
        """
        Speed-optimized version of gift recommendation
        Less detailed but much faster than OpenAI
        """
        prompt = f"""
        FAST GIFT RECOMMENDATION:
        
        Person: {personality_profile.get('summary', 'Creative, friendly person')}
        Traits: {personality_profile.get('personality_scores', {})}
        Hobbies: {personality_profile.get('hobbies', [])}
        Occasion: {occasion}
        Budget: €{budget_range}
        Relationship: {relationship}
        
        Quick recommendation with:
        - Specific gift name
        - 1-sentence reasoning
        - Match score 0-1
        - Emotional appeal
        - 2-3 personalization ideas
        - Price range
        - 2-3 alternatives
        - Confidence 0-1
        
        Respond in this JSON structure exactly:
        {{
            "gift_name": "specific gift name",
            "reasoning": "one sentence why it fits",
            "match_score": 0.8,
            "emotional_appeal": "what emotion it triggers",
            "personalization_ideas": ["idea1", "idea2"],
            "price_range": "budget category",
            "alternative_gifts": ["alt1", "alt2"],
            "confidence": 0.9
        }}
        """
        
        response = self.generate_json(
            prompt=prompt,
            json_schema=GiftRecommendationSchema.model_json_schema(),
            system_prompt="Fast gift expert. Quick but thoughtful recommendations.",
            temperature=0.6
        )
        
        if not response.success:
            # Fallback recommendation
            return GiftRecommendationSchema(
                gift_name="Personalized Photo Album",
                reasoning="Safe, thoughtful choice that works for most relationships",
                match_score=0.7,
                emotional_appeal="nostalgia",
                personalization_ideas=["Add favorite memories", "Custom cover design"],
                price_range=budget_range,
                alternative_gifts=["Custom mug", "Gift card"],
                confidence=0.6
            )
        
        try:
            return GiftRecommendationSchema.model_validate(response.parsed_json)
        except Exception as e:
            # Fallback on validation error
            return GiftRecommendationSchema(
                gift_name="Gift Card",
                reasoning="Flexible option when recommendation failed",
                match_score=0.5,
                emotional_appeal="choice",
                personalization_ideas=["Custom amount", "Favorite store"],
                price_range=budget_range,
                alternative_gifts=["Cash", "Experience voucher"],
                confidence=0.3
            )
    
    # === UTILITY METHODS ===
    
    def test_speed(self, num_requests: int = 5) -> Dict[str, float]:
        """
        Test Groq speed with multiple requests
        """
        times = []
        
        for i in range(num_requests):
            start = time.time()
            response = self.generate_text(
                prompt=f"Give me a creative gift idea #{i+1}",
                max_tokens=50
            )
            end = time.time()
            
            if response.success:
                times.append(end - start)
        
        if times:
            return {
                "avg_response_time": sum(times) / len(times),
                "min_response_time": min(times),
                "max_response_time": max(times),
                "successful_requests": len(times),
                "total_requests": num_requests
            }
        return {"error": "No successful requests"}
    
    def is_faster_than(self, other_client: BaseAIClient, test_prompt: str) -> bool:
        """
        Compare speed with another AI client
        """
        # Test Groq
        start = time.time()
        groq_response = self.generate_text(test_prompt, max_tokens=100)
        groq_time = time.time() - start
        
        # Test other client
        start = time.time()
        other_response = other_client.generate_text(test_prompt, max_tokens=100)
        other_time = time.time() - start
        
        return (groq_response.success and 
                other_response.success and 
                groq_time < other_time)
    
    def get_speed_metrics(self) -> Dict[str, Any]:
        """
        Get speed-focused metrics
        """
        base_metrics = self.get_metrics()
        return {
            "avg_response_time": base_metrics.avg_response_time,
            "total_requests": base_metrics.total_requests,
            "requests_per_minute": base_metrics.total_requests / max(1, 
                (datetime.now() - (base_metrics.last_request_time or datetime.now())).total_seconds() / 60),
            "avg_cost_per_request": base_metrics.cost_per_request,
            "model_type": self.model_type.value,
            "is_fast_model": True
        }


# === FACTORY FUNCTIONS ===

def create_groq_client(api_key: str, 
                      model_type: AIModelType = AIModelType.GROQ_MIXTRAL) -> GroqClient:
    """
    Factory function to create Groq client
    
    Args:
        api_key: Groq API key
        model_type: Which Groq model to use (Mixtral or Llama)
        
    Returns:
        Configured GroqClient instance
    """
    return GroqClient(api_key=api_key, model_type=model_type)


def create_speed_optimized_client(api_key: str) -> GroqClient:
    """
    Creates Groq client optimized for maximum speed
    """
    client = create_groq_client(api_key, AIModelType.GROQ_LLAMA)  # Faster model
    client.rate_limit_per_minute = 500  # Higher rate limit
    return client


# === TESTING UTILITIES ===

def test_groq_integration(api_key: str) -> Dict[str, Any]:
    """
    Comprehensive test of Groq integration focusing on speed
    """
    results = {
        "connection": False,
        "speed_test": False,
        "json_generation": False,
        "gift_recommendation": False,
        "avg_response_time": None,
        "errors": []
    }
    
    try:
        client = create_groq_client(api_key, AIModelType.GROQ_MIXTRAL)
        
        # Test 1: Connection & Speed
        start = time.time()
        text_response = client.generate_text("Hello Groq!", max_tokens=20)
        response_time = time.time() - start
        
        results["connection"] = text_response.success
        results["avg_response_time"] = response_time
        results["speed_test"] = response_time < 2.0  # Should be very fast
        
        if results["connection"]:
            # Test 2: JSON Generation
            json_response = client.generate_json(
                prompt="Give me 2 gift ideas in JSON format",
                json_schema={"type": "object", "properties": {"gifts": {"type": "array"}}}
            )
            results["json_generation"] = json_response.success
            
            # Test 3: Fast Gift Recommendation
            if results["json_generation"]:
                test_profile = {
                    "summary": "creative person who loves photography",
                    "personality_scores": {"openness": 0.8},
                    "hobbies": ["photography"]
                }
                
                try:
                    start = time.time()
                    recommendation = client.fast_gift_recommendation(
                        personality_profile=test_profile,
                        occasion="birthday",
                        budget_range="50-100",
                        relationship="friend"
                    )
                    recommendation_time = time.time() - start
                    
                    results["gift_recommendation"] = True
                    results["recommendation_time"] = recommendation_time
                    results["sample_recommendation"] = recommendation.model_dump()
                except Exception as e:
                    results["errors"].append(f"Gift recommendation failed: {e}")
        
    except Exception as e:
        results["errors"].append(f"Groq client creation failed: {e}")
    
    return results


def speed_comparison_test(openai_client, groq_client, test_prompts: List[str]) -> Dict[str, Any]:
    """
    Compare speed between OpenAI and Groq
    """
    results = {
        "openai_times": [],
        "groq_times": [],
        "groq_faster_count": 0,
        "avg_speedup": 0.0
    }
    
    for prompt in test_prompts:
        # Test OpenAI
        start = time.time()
        openai_response = openai_client.generate_text(prompt, max_tokens=100)
        openai_time = time.time() - start
        
        # Test Groq
        start = time.time()
        groq_response = groq_client.generate_text(prompt, max_tokens=100)
        groq_time = time.time() - start
        
        if openai_response.success and groq_response.success:
            results["openai_times"].append(openai_time)
            results["groq_times"].append(groq_time)
            
            if groq_time < openai_time:
                results["groq_faster_count"] += 1
    
    if results["openai_times"] and results["groq_times"]:
        avg_openai = sum(results["openai_times"]) / len(results["openai_times"])
        avg_groq = sum(results["groq_times"]) / len(results["groq_times"])
        results["avg_speedup"] = avg_openai / avg_groq if avg_groq > 0 else 0
    
    return results