"""
Async OpenAI Client - High-Performance Version des OpenAI Clients

HAUPTUNTERSCHIEDE zu openai_client.py:
- Erbt von AsyncBaseAIClient (nicht BaseAIClient)
- Alle Methoden sind async (mit async def)
- Kann mehrere OpenAI-Calls gleichzeitig verarbeiten
- Nutzt async OpenAI library
- Unterstützt Batch Processing

PERFORMANCE:
- 10x schneller für mehrere Empfehlungen
- Ideal für Batch-Gift-Recommendations
- Perfekt für Real-Time User Interfaces

VERWENDUNG:
async with AsyncOpenAIClient(api_key) as client:
    gift = await client.recommend_gift_async(profile, occasion, budget, relationship)
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# OpenAI Async Support
try:
    from openai import AsyncOpenAI  # Async Version der OpenAI Library
except ImportError:
    raise ImportError("Async OpenAI library not available. Update: pip install openai>=1.0.0")

# Import von der async_base_client.py
from .async_base_client import AsyncBaseAIClient, AIRequest, AIResponse
from .base_client import (
    AIModelType, ModelCapability, ResponseFormat, 
    GiftRecommendationSchema, PersonalityInsightSchema
)


class AsyncOpenAIClient(AsyncBaseAIClient):
    """
    Async OpenAI Client für High-Performance Gift Recommendations
    
    FEATURES:
    - Async GPT-4 & GPT-3.5 Support
    - Batch Processing für mehrere Empfehlungen
    - JSON Mode für structured outputs
    - Parallele API-Calls
    - Intelligente Error Handling
    
    PERFORMANCE BOOST:
    - 1 Empfehlung: ~2 Sekunden (gleich wie sync)
    - 5 Empfehlungen: ~2 Sekunden (statt 10 Sekunden sync!)
    - 10 Empfehlungen: ~3 Sekunden (statt 20 Sekunden sync!)
    """
    
    def __init__(self, 
                 api_key: str,
                 model_type: AIModelType = AIModelType.OPENAI_GPT4,
                 rate_limit_per_minute: int = 100,
                 max_concurrent_requests: int = 10):
        
        # Validiere model type (nur OpenAI Models)
        if model_type not in [AIModelType.OPENAI_GPT4, AIModelType.OPENAI_GPT35]:
            raise ValueError(f"Invalid model type for AsyncOpenAI: {model_type}")
        
        # Initialize parent (AsyncBaseAIClient)
        super().__init__(api_key, model_type, rate_limit_per_minute, max_concurrent_requests)
        
        # Async OpenAI Client
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Model configuration (gleich wie in dem sync Client)
        self.model_name = self._get_model_name()
        self.supports_json_mode = model_type == AIModelType.OPENAI_GPT4
        
        print(f"🚀 AsyncOpenAIClient initialized: {self.model_name}")
    
    def _get_model_name(self) -> str:
        """Maps model type to OpenAI model name (gleich wie sync)"""
        mapping = {
            AIModelType.OPENAI_GPT4: "gpt-4o-mini",
            AIModelType.OPENAI_GPT35: "gpt-4o-mini"
        }
        return mapping[self.model_type]
    
    def _get_model_capabilities(self) -> List[ModelCapability]:
        """Define capabilities (gleich wie sync)"""
        base_capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.HIGH_QUALITY,
            ModelCapability.FUNCTION_CALLING
        ]
        
        if self.model_type == AIModelType.OPENAI_GPT4:
            base_capabilities.extend([
                ModelCapability.JSON_OUTPUT,
                ModelCapability.VISION
            ])
        
        return base_capabilities
    
    def _get_model_pricing(self) -> Dict[str, float]:
        """OpenAI pricing (gleich wie sync)"""
        pricing = {
            AIModelType.OPENAI_GPT4: {
                "input_per_token": 0.00003,   # $0.03 per 1K tokens
                "output_per_token": 0.00006   # $0.06 per 1K tokens
            },
            AIModelType.OPENAI_GPT35: {
                "input_per_token": 0.000001,  # $0.001 per 1K tokens  
                "output_per_token": 0.000002  # $0.002 per 1K tokens
            }
        }
        return pricing[self.model_type]
    
    # === CORE ASYNC METHOD (das Herzstück!) ===
    
    async def _make_async_api_call(self, request: AIRequest) -> AIResponse:
        """
        Async OpenAI API Call - das ist der Kern!
        
        UNTERSCHIED zu sync Version:
        - Verwendet AsyncOpenAI statt OpenAI
        - await statt blocking call
        - Kann parallel zu anderen Calls laufen
        """
        
        try:
            # Build messages (gleich wie sync)
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            # API parameters (gleich wie sync)
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens
            }
            
            # JSON mode für structured outputs
            if (request.response_format == ResponseFormat.JSON and 
                self.supports_json_mode):
                api_params["response_format"] = {"type": "json_object"}
                
                # Ensure prompt asks for JSON
                if "json" not in request.prompt.lower():
                    messages[-1]["content"] += "\n\nRespond in valid JSON format."
            
            # 🔥 DER WICHTIGSTE UNTERSCHIED: await statt blocking!
            print(f"📡 Making async OpenAI API call...")
            start_time = time.time()
            
            # ASYNC CALL - läuft parallel zu anderen Calls!
            response = await self.client.chat.completions.create(**api_params)
            
            response_time = time.time() - start_time
            print(f"✅ Async OpenAI response received in {response_time:.2f}s")
            
            # Extract response data (gleich wie sync)
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Calculate cost (gleich wie sync)
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (
                input_tokens * self.pricing["input_per_token"] +
                output_tokens * self.pricing["output_per_token"]
            )
            
            # Parse JSON if requested (gleich wie sync)
            parsed_json = None
            if request.response_format == ResponseFormat.JSON:
                try:
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
            
            # Success response
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
            print(f"❌ Async OpenAI API error: {e}")
            return AIResponse(
                content="",
                model_type=self.model_type,
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                timestamp=datetime.now(),
                success=False,
                error=f"Async OpenAI API Error: {e}"
            )
    
    # === ASYNC GIFT RECOMMENDATION METHODS ===
    
    async def recommend_gift_with_reasoning_async(self,
                                                 personality_profile: Dict,
                                                 occasion: str,
                                                 budget_range: str,
                                                 relationship: str,
                                                 existing_gifts: Optional[List[str]] = None) -> GiftRecommendationSchema:
        """
        Async version of the recommend_gift_with_reasoning method
        
        VERWENDUNG:
        async with AsyncOpenAIClient(api_key) as client:
            gift = await client.recommend_gift_with_reasoning_async(
                personality_profile={"openness": 0.8, "hobbies": ["photography"]},
                occasion="birthday",
                budget_range="€50-€150",
                relationship="friend"
            )
        """
        
        print(f"🎁 Generating async gift recommendation for {occasion}...")
        
        # Build advanced prompt (gleich wie sync, aber mit mehr Context)
        prompt = self._build_advanced_gift_prompt(
            personality_profile, occasion, budget_range, relationship, existing_gifts
        )
        
        # Use specialized system prompt
        system_prompt = self._get_advanced_gift_expert_prompt()
        
        # Get JSON schema
        schema = GiftRecommendationSchema.model_json_schema()
        
        # 🔥 ASYNC CALL - macht Request ohne zu blockieren!
        response = await self.generate_json_async(
            prompt=prompt,
            json_schema=schema,
            system_prompt=system_prompt,
            temperature=0.6
        )
        
        if not response.success:
            raise Exception(f"Async OpenAI gift recommendation failed: {response.error}")
        
        # Validate and return
        try:
            recommendation = GiftRecommendationSchema.model_validate(response.parsed_json)
            print(f"✅ Async gift recommendation: {recommendation.gift_name}")
            return recommendation
            
        except Exception as e:
            raise Exception(f"Invalid async OpenAI response format: {e}")
    
    async def analyze_personality_for_gifts_async(self, personality_profile: Dict) -> Dict[str, Any]:
        """
        Async personality analysis for gift recommendations
        
        VERWENDUNG:
        insights = await client.analyze_personality_for_gifts_async(profile)
        """
        
        print("🧠 Analyzing personality for gift insights...")
        
        prompt = f"""
        Analyze this personality profile for gift recommendations:
        
        {json.dumps(personality_profile, indent=2)}
        
        Provide insights into:
        1. What types of gifts would resonate emotionally
        2. Gift categories to avoid
        3. Personalization opportunities
        4. Budget sensitivity
        5. Occasion preferences
        
        Respond in JSON format with actionable insights.
        """
        
        response = await self.generate_json_async(
            prompt=prompt,
            json_schema={
                "type": "object",
                "properties": {
                    "emotional_resonance": {"type": "array", "items": {"type": "string"}},
                    "avoid_categories": {"type": "array", "items": {"type": "string"}},
                    "personalization_opportunities": {"type": "array", "items": {"type": "string"}},
                    "budget_sensitivity": {"type": "string"},
                    "occasion_preferences": {"type": "object"}
                }
            },
            system_prompt="You are a psychology expert specializing in gift psychology.",
            temperature=0.4
        )
        
        return response.parsed_json if response.success else {}
    
    async def generate_creative_alternatives_async(self, 
                                                  base_gift: str,
                                                  personality_profile: Dict,
                                                  num_alternatives: int = 5) -> List[str]:
        """
        Async creative gift alternatives
        
        VERWENDUNG:
        alternatives = await client.generate_creative_alternatives_async(
            "Camera", profile, 5
        )
        """
        
        print(f"💡 Generating {num_alternatives} creative alternatives to '{base_gift}'...")
        
        prompt = f"""
        Given the base gift idea "{base_gift}" and this personality profile:
        {json.dumps(personality_profile, indent=2)}
        
        Generate {num_alternatives} creative alternative gift ideas that:
        1. Target the same emotional needs
        2. Match the personality better
        3. Are unique and thoughtful
        4. Range from practical to surprising
        
        Return as JSON array of gift ideas.
        """
        
        response = await self.generate_json_async(
            prompt=prompt,
            json_schema={
                "type": "object",
                "properties": {
                    "alternatives": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": num_alternatives,
                        "maxItems": num_alternatives
                    }
                }
            },
            system_prompt="You are a creative gift consultant known for unique, thoughtful ideas.",
            temperature=0.8  # Higher creativity
        )
        
        if response.success and response.parsed_json:
            return response.parsed_json.get("alternatives", [])
        return []
    
    # === BATCH PROCESSING SUPERPOWERS ===
    
    async def batch_gift_recommendations_for_family_async(self,
                                                         family_profiles: List[Dict]) -> List[GiftRecommendationSchema]:
        """
        Erstellt Geschenkempfehlungen für eine ganze Familie gleichzeitig!
        
        BEISPIEL:
        family = [
            {"name": "Mama", "profile": profile1, "occasion": "birthday", "budget": "€50-€150", "relationship": "mother"},
            {"name": "Papa", "profile": profile2, "occasion": "birthday", "budget": "€50-€150", "relationship": "father"},
            {"name": "Schwester", "profile": profile3, "occasion": "birthday", "budget": "€10-€50", "relationship": "sister"}
        ]
        
        gifts = await client.batch_gift_recommendations_for_family_async(family)
        
        # Statt 3 x 2 Sekunden = 6 Sekunden
        # Nur 2 Sekunden für alle 3 Empfehlungen!
        """
        
        print(f"👨‍👩‍👧‍👦 Generating batch gift recommendations for {len(family_profiles)} family members...")
        
        # Prepare batch data
        batch_data = []
        for member in family_profiles:
            batch_data.append({
                "personality_profile": member["profile"],
                "occasion": member["occasion"],
                "budget_range": member["budget"],
                "relationship": member["relationship"]
            })
        
        # 🔥 BATCH PROCESSING - alle gleichzeitig!
        recommendations = await self.batch_gift_recommendations_async(batch_data)
        
        # Add names for easier identification
        for i, recommendation in enumerate(recommendations):
            if i < len(family_profiles):
                recommendation.metadata = {"family_member": family_profiles[i]["name"]}
        
        print(f"✅ Generated {len(recommendations)} family gift recommendations")
        return recommendations
    
    async def quick_gift_brainstorm_async(self, 
                                         topic: str,
                                         num_ideas: int = 10) -> List[str]:
        """
        Schneller Geschenk-Brainstorm mit async Speed
        
        VERWENDUNG:
        ideas = await client.quick_gift_brainstorm_async("Tech-Lover", 10)
        """
        
        print(f"🧠 Quick brainstorm: {num_ideas} gift ideas for '{topic}'...")
        
        prompt = f"""
        Brainstorm {num_ideas} creative gift ideas for: {topic}
        
        Requirements:
        - Diverse price ranges
        - Mix of practical and fun
        - Some unique/unexpected options
        - Brief but specific descriptions
        
        Return as simple JSON array of gift ideas.
        """
        
        response = await self.generate_json_async(
            prompt=prompt,
            json_schema={
                "type": "object",
                "properties": {
                    "ideas": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            temperature=0.8
        )
        
        if response.success and response.parsed_json:
            return response.parsed_json.get("ideas", [])
        return []
    
    # === ADVANCED PROMPT BUILDING (von sync übernommen, aber optimiert) ===
    
    def _build_advanced_gift_prompt(self,
                               personality_profile: Dict,
                               occasion: str,
                               budget_range: str,
                               relationship: str,
                               existing_gifts: Optional[List[str]] = None) -> str:
        """Optimized prompt for GPT-4o-mini JSON output"""
        
        # Extract key personality insights
        traits = personality_profile.get('personality_scores', {})
        preferences = personality_profile.get('preferences', {})
        hobbies = personality_profile.get('hobbies', [])
        emotional_triggers = personality_profile.get('emotional_triggers', [])
        
        prompt = f"""Create a gift recommendation for this person:

    PERSON DETAILS:
    - Relationship: {relationship}
    - Occasion: {occasion}
    - Budget: €{budget_range}
    - Personality Traits: {traits}
    - Preferences: {preferences}
    - Hobbies: {', '.join(hobbies) if hobbies else 'None specified'}
    - Emotional Triggers: {', '.join(emotional_triggers) if emotional_triggers else 'None specified'}
    {f"- Avoid these gifts: {existing_gifts}" if existing_gifts else ""}

    ANALYSIS STEPS:
    1. Analyze their personality and identify what drives them emotionally
    2. Consider the occasion and relationship context  
    3. Think of gifts that would surprise and delight them
    4. Ensure it fits the budget and can be personalized

    RESPONSE FORMAT:
    You must respond with ONLY a JSON object in this EXACT format. Do not add any wrapper objects or additional text:

    {{
    "gift_name": "Specific gift name",
    "reasoning": "Detailed explanation why this gift fits their personality and the analysis steps above",
    "match_score": 0.85,
    "emotional_appeal": "What emotion this triggers",
    "personalization_ideas": ["idea 1", "idea 2", "idea 3"],
    "price_range": "Budget category",
    "alternative_gifts": ["alternative 1", "alternative 2"],
    "confidence": 0.9
    }}

    CRITICAL RULES:
    - Respond with ONLY the JSON object above
    - Do NOT wrap it in "giftRecommendation" or any other object
    - Do NOT add any text before or after the JSON
    - All string values must be in quotes
    - Numbers (match_score, confidence) should be between 0 and 1
    - Include detailed reasoning that shows your analysis steps
    """
        return prompt
    
    def _get_advanced_gift_expert_prompt(self) -> str:
            """Optimized system prompt for JSON compliance"""
            return """You are Dr. Elena Hartmann, a renowned gift psychology expert with 25 years of experience.

        Your expertise includes:
        - Understanding what makes gifts emotionally meaningful
        - Matching gifts to personality types using psychological insights
        - Creating personalized experiences, not just products
        - Balancing practicality with emotional impact

        CRITICAL JSON INSTRUCTIONS:
        - You MUST respond with valid JSON only
        - Do NOT use markdown code blocks (no ```)
        - Do NOT wrap the JSON in any parent object like "giftRecommendation"
        - Do NOT add explanatory text before or after the JSON
        - Use the exact field names specified in the user prompt
        - Ensure all JSON syntax is perfect (quotes, commas, brackets)

        Your recommendations should be thoughtful, personalized, and based on psychological insights.
        Respond with ONLY the requested JSON structure."""
    
    # === UTILITY METHODS ===
    
    async def test_async_speed(self, num_concurrent_requests: int = 5) -> Dict[str, Any]:
        """
        Test async performance with multiple concurrent requests
        
        VERWENDUNG:
        results = await client.test_async_speed(5)
        """
        
        print(f"🚀 Testing async speed with {num_concurrent_requests} concurrent requests...")
        
        test_requests = [
            AIRequest(
                prompt=f"Give me a creative gift idea #{i+1} for a creative person",
                max_tokens=100,
                temperature=0.7
            )
            for i in range(num_concurrent_requests)
        ]
        
        start_time = time.time()
        responses = await self.generate_batch_async(test_requests)
        total_time = time.time() - start_time
        
        successful_responses = [r for r in responses if isinstance(r, AIResponse) and r.success]
        
        results = {
            "total_requests": num_concurrent_requests,
            "successful_responses": len(successful_responses),
            "total_time": total_time,
            "avg_time_per_request": total_time / num_concurrent_requests,
            "requests_per_second": num_concurrent_requests / total_time,
            "speedup_vs_sequential": num_concurrent_requests / (total_time / 2.0)  # Assuming 2s per request
        }
        
        print(f"📊 Async Speed Test Results:")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Requests/Second: {results['requests_per_second']:.2f}")
        print(f"   Speedup vs Sequential: {results['speedup_vs_sequential']:.2f}x")
        
        return results
    
    async def close_async(self):
        """
        Properly close async resources
        """
        if hasattr(self.client, 'close'):
            await self.client.close()
        print("🔒 AsyncOpenAI client closed")


# === FACTORY FUNCTION ===

def create_async_openai_client(api_key: str, 
                              model_type: AIModelType = AIModelType.OPENAI_GPT4) -> AsyncOpenAIClient:
    """
    Factory function for AsyncOpenAIClient
    
    VERWENDUNG:
    client = create_async_openai_client(api_key, AIModelType.OPENAI_GPT4)
    """
    return AsyncOpenAIClient(api_key=api_key, model_type=model_type)


# === TESTING UTILITIES ===

async def test_async_openai_integration(api_key: str) -> Dict[str, Any]:
    """
    Comprehensive test of async OpenAI integration
    
    VERWENDUNG:
    results = await test_async_openai_integration(api_key)
    """
    
    results = {
        "connection": False,
        "text_generation": False,
        "json_generation": False,
        "gift_recommendation": False,
        "batch_processing": False,
        "performance_test": None,
        "errors": []
    }
    
    try:
        async with create_async_openai_client(api_key, AIModelType.OPENAI_GPT4) as client:
            
            # Test 1: Connection
            results["connection"] = await client.test_async_connection()
            
            if results["connection"]:
                # Test 2: Text Generation
                text_response = await client.generate_text_async(
                    "What makes a great gift?", max_tokens=50
                )
                results["text_generation"] = text_response.success
                
                # Test 3: JSON Generation
                if results["text_generation"]:
                    json_response = await client.generate_json_async(
                        prompt="List 3 gift categories in JSON format",
                        json_schema={"type": "object", "properties": {"categories": {"type": "array"}}}
                    )
                    results["json_generation"] = json_response.success
                    
                    # Test 4: Gift Recommendation
                    if results["json_generation"]:
                        test_profile = {
                            "personality_scores": {"openness": 0.8, "extraversion": 0.6},
                            "hobbies": ["photography", "travel"],
                            "emotional_triggers": ["adventure", "creativity"]
                        }
                        
                        try:
                            recommendation = await client.recommend_gift_with_reasoning_async(
                                personality_profile=test_profile,
                                occasion="birthday",
                                budget_range="€50-€150",
                                relationship="friend"
                            )
                            results["gift_recommendation"] = True
                            results["sample_recommendation"] = recommendation.model_dump()
                        except Exception as e:
                            results["errors"].append(f"Gift recommendation failed: {e}")
                        
                        # Test 5: Batch Processing
                        if results["gift_recommendation"]:
                            try:
                                batch_data = [
                                    {"personality_profile": test_profile, "occasion": "birthday", "budget_range": "€10-€50", "relationship": "friend"},
                                    {"personality_profile": test_profile, "occasion": "christmas", "budget_range": "€50-€150", "relationship": "family"}
                                ]
                                
                                batch_start = time.time()
                                batch_recommendations = await client.batch_gift_recommendations_async(batch_data)
                                batch_time = time.time() - batch_start
                                
                                results["batch_processing"] = len(batch_recommendations) == 2
                                results["batch_time"] = batch_time
                                
                                # Test 6: Performance Test
                                if results["batch_processing"]:
                                    perf_results = await client.test_async_speed(3)
                                    results["performance_test"] = perf_results
                                    
                            except Exception as e:
                                results["errors"].append(f"Batch processing failed: {e}")
        
    except Exception as e:
        results["errors"].append(f"AsyncOpenAI client creation failed: {e}")
    
    return results


# === USAGE EXAMPLES ===

async def example_usage():
    """
    Beispiele für die Verwendung des AsyncOpenAIClient
    """
    
    api_key = "your-openai-api-key"
    
    # Beispiel 1: Einzelne Empfehlung
    async with create_async_openai_client(api_key) as client:
        
        profile = {
            "personality_scores": {"openness": 0.8, "conscientiousness": 0.6},
            "hobbies": ["photography", "hiking"],
            "emotional_triggers": ["adventure", "creativity"]
        }
        
        # Async Gift Recommendation
        gift = await client.recommend_gift_with_reasoning_async(
            personality_profile=profile,
            occasion="birthday",
            budget_range="€50-€150",
            relationship="close_friend"
        )
        
        print(f"🎁 Recommended gift: {gift.gift_name}")
        print(f"💭 Reasoning: {gift.reasoning}")
    
    # Beispiel 2: Batch Processing für Familie
    async with create_async_openai_client(api_key) as client:
        
        family = [
            {"name": "Mama", "profile": profile, "occasion": "birthday", "budget": "€50-€150", "relationship": "mother"},
            {"name": "Papa", "profile": profile, "occasion": "birthday", "budget": "€50-€150", "relationship": "father"},
            {"name": "Schwester", "profile": profile, "occasion": "birthday", "budget": "€10-€50", "relationship": "sister"}
        ]
        
        # Alle Empfehlungen gleichzeitig generieren
        family_gifts = await client.batch_gift_recommendations_for_family_async(family)
        
        for gift in family_gifts:
            member_name = gift.metadata.get("family_member", "Unknown")
            print(f"👨‍👩‍👧‍👦 {member_name}: {gift.gift_name}")


if __name__ == "__main__":
    # Beispiel ausführen
    asyncio.run(example_usage())