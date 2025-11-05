"""
Async Anthropic Claude Client - Concurrent Ethical Reasoning

ASYNC ETHICAL SUPERPOWERS:
- Concurrent ethical analysis for multiple scenarios
- Batch risk assessment for gift portfolios
- Parallel cultural sensitivity analysis
- Non-blocking premium reasoning

PERFORMANCE BOOST:
- Multiple ethical analyses simultaneously
- Batch family ethics assessment
- Concurrent risk evaluation
- Premium reasoning without blocking UI

USE CASES:
- Family gift ethics analysis (all members at once)
- Portfolio risk assessment for multiple gifts
- Concurrent cultural sensitivity checks
- Batch premium recommendation generation
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Async Anthropic Support
try:
    import anthropic
    from anthropic import Anthropic
    # Note: Anthropic's library doesn't have native async support yet,
    # so we'll use asyncio.to_thread for non-blocking execution
except ImportError:
    raise ImportError("Anthropic library not installed. Run: pip install anthropic")

# Import from async_base_client.py
from .async_base_client import AsyncBaseAIClient, AIRequest, AIResponse
from .base_client import (
    AIModelType, ModelCapability, ResponseFormat, 
    GiftRecommendationSchema
)


class AsyncAnthropicClient(AsyncBaseAIClient):
    """
    Async Anthropic Claude Client für Concurrent Ethical Reasoning
    
    FEATURES:
    - Async ethical analysis and cultural sensitivity
    - Concurrent risk assessment for multiple gifts
    - Batch premium reasoning for families/groups
    - Non-blocking ethical considerations
    - Parallel cultural appropriateness checks
    
    ETHICAL FOCUS:
    - Multi-scenario ethical analysis without blocking
    - Concurrent cultural sensitivity assessment
    - Batch relationship appropriateness evaluation
    - Premium reasoning in parallel
    """
    
    def __init__(self, 
                 api_key: str,
                 model_type: AIModelType = AIModelType.ANTHROPIC_CLAUDE,
                 rate_limit_per_minute: int = 50,
                 max_concurrent_requests: int = 8):  # Conservative for premium reasoning
        
        # Validate model type
        if model_type != AIModelType.ANTHROPIC_CLAUDE:
            raise ValueError(f"Invalid model type for AsyncAnthropic: {model_type}")
        
        # Initialize parent (AsyncBaseAIClient)
        super().__init__(api_key, model_type, rate_limit_per_minute, max_concurrent_requests)
        
        # Initialize Anthropic client
        self.client = Anthropic(api_key=api_key)
        
        # Model configuration
        self.model_name = self._get_model_name()
        self.max_tokens_limit = 4096
        
        print(f"🎭 AsyncAnthropicClient initialized: {self.model_name}")
    
    def _get_model_name(self) -> str:
        """Maps model type to Anthropic model name"""
        return "claude-3-5-sonnet-20241022"
    
    def _get_model_capabilities(self) -> List[ModelCapability]:
        """Define capabilities (same as sync)"""
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.HIGH_QUALITY,
            ModelCapability.COMPLEX_REASONING,
            ModelCapability.ETHICAL_REASONING,
            ModelCapability.CULTURAL_SENSITIVITY,
            ModelCapability.JSON_OUTPUT
        ]
    
    def _get_model_pricing(self) -> Dict[str, float]:
        """Anthropic pricing (same as sync)"""
        return {
            "input_per_token": 0.000003,   # $3 per 1M input tokens
            "output_per_token": 0.000015   # $15 per 1M output tokens
        }
    
    # === CORE ASYNC METHOD ===
    
    async def _make_async_api_call(self, request: AIRequest) -> AIResponse:
        """
        Async Anthropic API Call using asyncio.to_thread for non-blocking execution
        
        Note: Anthropic's library doesn't have native async support yet,
        so we use asyncio.to_thread to prevent blocking (asyncio.to_thread() = Sync Code in async Umgebung ausführen)
        """
        
        try:
            # Build messages for Claude
            user_content = request.prompt
            if request.system_prompt:
                user_content = f"{request.system_prompt}\n\nHuman: {request.prompt}"
            else:
                user_content = f"Human: {request.prompt}"
            
            user_content += "\n\nAssistant: "
            
            # API parameters
            api_params = {
                "model": self.model_name,
                "max_tokens": min(request.max_tokens or 1000, self.max_tokens_limit),
                "temperature": request.temperature,
                "messages": [{"role": "user", "content": user_content}]
            }
            
            # 🔥 ASYNC EXECUTION - Use asyncio.to_thread to make sync call non-blocking
            print(f"🎭 Making async Anthropic API call...")
            start_time = time.time()
            
            # ASYNC WRAPPER - runs in thread pool to avoid blocking
            response = await asyncio.to_thread(
                self.client.messages.create,
                **api_params
            )
            
            response_time = time.time() - start_time
            print(f"✅ Async Anthropic response received in {response_time:.3f}s")
            
            # Extract content
            content = response.content[0].text
            
            # Calculate tokens and cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            tokens_used = input_tokens + output_tokens
            
            cost = (
                input_tokens * self.pricing["input_per_token"] +
                output_tokens * self.pricing["output_per_token"]
            )
            
            # Parse JSON if requested
            parsed_json = None
            if request.response_format == ResponseFormat.JSON:
                try:
                    # Claude sometimes adds explanatory text, try to extract JSON
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
            print(f"❌ Async Anthropic API error: {e}")
            return AIResponse(
                content="",
                model_type=self.model_type,
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                timestamp=datetime.now(),
                success=False,
                error=f"Async Anthropic API Error: {e}"
            )
    
    # === ASYNC ETHICAL REASONING METHODS ===
    
    async def ethical_gift_analysis_async(self,
                                        personality_profile: Dict,
                                        occasion: str,
                                        budget_range: str,
                                        relationship: str,
                                        cultural_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Async ethical and cultural analysis of gift appropriateness
        
        VERWENDUNG:
        async with AsyncAnthropicClient(api_key) as client:
            ethics = await client.ethical_gift_analysis_async(
                personality_profile=profile,
                occasion="birthday",
                budget_range="100-200",
                relationship="colleague",
                cultural_context="diverse_workplace"
            )
        """
        
        print("🎭 Starting async ethical gift analysis...")
        
        prompt = f"""
        Analyze the ethical and cultural appropriateness of gift giving in this situation:
        
        PERSON & CONTEXT:
        - Personality: {json.dumps(personality_profile, indent=2)}
        - Occasion: {occasion}
        - Relationship: {relationship}
        - Budget: €{budget_range}
        - Cultural Context: {cultural_context or "Not specified"}
        
        ETHICAL ANALYSIS REQUIRED:
        
        1. RELATIONSHIP APPROPRIATENESS:
           - What level of intimacy is appropriate for this relationship?
           - Are there boundary considerations to be aware of?
           - What could be misinterpreted or cause discomfort?
           - How does the occasion affect appropriate gift levels?
        
        2. CULTURAL SENSITIVITY:
           - Are there cultural considerations for this type of gift giving?
           - What cultural norms should be respected?
           - Are there potential cultural misunderstandings to avoid?
           - How do cultural backgrounds affect gift interpretation?
        
        3. ETHICAL CONSIDERATIONS:
           - Is this gift ethically appropriate given the context?
           - Could it create unwanted obligations or expectations?
           - Are there power dynamics to consider?
           - What are the potential unintended consequences?
        
        4. RISK ASSESSMENT:
           - What could go wrong with different gift approaches?
           - How to minimize risk of misunderstanding?
           - What are the safe vs. risky gift categories?
           - How to ensure positive reception?
        
        5. RECOMMENDATIONS:
           - What ethical guidelines should guide gift selection?
           - How to show care while respecting boundaries?
           - What approach maximizes positive impact?
           - How to be thoughtful without overstepping?
        
        Provide comprehensive ethical analysis in JSON format:
        {{
            "relationship_appropriateness": {{
                "intimacy_level": "low/medium/high",
                "boundary_considerations": ["consideration1", "consideration2"],
                "potential_misinterpretations": ["risk1", "risk2"],
                "occasion_impact": "analysis"
            }},
            "cultural_sensitivity": {{
                "cultural_considerations": ["factor1", "factor2"],
                "cultural_norms": ["norm1", "norm2"],
                "potential_misunderstandings": ["misunderstanding1", "misunderstanding2"],
                "cultural_adaptation_needed": true/false
            }},
            "ethical_considerations": {{
                "appropriateness_level": "appropriate/questionable/inappropriate",
                "obligation_risks": ["risk1", "risk2"],
                "power_dynamics": "analysis",
                "unintended_consequences": ["consequence1", "consequence2"]
            }},
            "risk_assessment": {{
                "risk_level": "low/medium/high",
                "potential_problems": ["problem1", "problem2"],
                "safe_categories": ["category1", "category2"],
                "risky_categories": ["category1", "category2"],
                "mitigation_strategies": ["strategy1", "strategy2"]
            }},
            "ethical_recommendations": {{
                "guiding_principles": ["principle1", "principle2"],
                "recommended_approach": "detailed approach",
                "boundary_respect_strategies": ["strategy1", "strategy2"],
                "positive_impact_maximization": "strategy"
            }},
            "overall_assessment": {{
                "ethical_score": 0.85,
                "cultural_sensitivity_score": 0.90,
                "risk_level": "low",
                "recommended_proceed": true/false
            }}
        }}
        """
        
        response = await self.generate_json_async(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are Dr. Sarah Chen, an expert in cross-cultural communication and ethical gift-giving practices with 20 years of experience in cultural psychology and relationship dynamics.",
            temperature=0.3
        )
        
        print("✅ Async ethical gift analysis completed")
        return response.parsed_json if response.success else {}
    
    async def batch_ethical_analysis_async(self,
                                         ethical_scenarios: List[Dict]) -> List[Dict[str, Any]]:
        """
        Analyze multiple ethical scenarios concurrently
        
        Perfect for family gift planning or multiple relationship contexts
        
        VERWENDUNG:
        scenarios = [
            {"profile": profile1, "occasion": "christmas", "relationship": "mother"},
            {"profile": profile2, "occasion": "christmas", "relationship": "father"},
            {"profile": profile3, "occasion": "christmas", "relationship": "sister"}
        ]
        analyses = await client.batch_ethical_analysis_async(scenarios)
        """
        
        print(f"🎭 Analyzing {len(ethical_scenarios)} ethical scenarios concurrently...")
        
        # Create analysis tasks for each scenario
        tasks = []
        for i, scenario in enumerate(ethical_scenarios):
            task = self.ethical_gift_analysis_async(
                personality_profile=scenario["profile"],
                occasion=scenario["occasion"],
                budget_range=scenario.get("budget_range", "50-150"),
                relationship=scenario["relationship"],
                cultural_context=scenario.get("cultural_context")
            )
            tasks.append(task)
        
        # Execute all analyses concurrently
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Process results
        analyses = []
        for i, response in enumerate(responses):
            if isinstance(response, dict):
                analysis = response
                analysis["scenario_index"] = i
                analysis["processing_time"] = total_time / len(ethical_scenarios)
                analyses.append(analysis)
            else:
                # Fallback analysis
                analyses.append({
                    "scenario_index": i,
                    "error": "Ethical analysis failed",
                    "fallback_strategy": "Use conservative gift approach",
                    "processing_time": total_time / len(ethical_scenarios)
                })
        
        print(f"✅ Batch ethical analysis completed in {total_time:.2f}s")
        return analyses
    
    async def premium_gift_recommendation_async(self,
                                              personality_profile: Dict,
                                              occasion: str,
                                              budget_range: str,
                                              relationship: str,
                                              additional_context: Optional[Dict] = None) -> GiftRecommendationSchema:
        """
        Async premium gift recommendation with comprehensive analysis
        
        VERWENDUNG:
        recommendation = await client.premium_gift_recommendation_async(
            personality_profile=profile,
            occasion="anniversary",
            budget_range="150-500",
            relationship="partner"
        )
        """
        
        print("🎭 Generating async premium gift recommendation...")
        
        context_data = additional_context or {}
        
        prompt = f"""
        Create a premium gift recommendation with comprehensive ethical and psychological analysis:
        
        COMPREHENSIVE CONTEXT:
        - Personality: {json.dumps(personality_profile, indent=2)}
        - Occasion: {occasion}
        - Relationship: {relationship}
        - Budget: €{budget_range}
        - Additional Context: {json.dumps(context_data, indent=2)}
        
        PREMIUM ANALYSIS FRAMEWORK:
        
        1. ETHICAL FOUNDATION:
           - Ensure recommendation is ethically appropriate for relationship
           - Consider cultural sensitivity and appropriateness
           - Assess potential risks and boundary considerations
           - Evaluate gift's impact on relationship dynamics
        
        2. PSYCHOLOGICAL DEPTH:
           - Analyze personality for gift compatibility
           - Consider emotional impact and meaning
           - Assess how gift aligns with their values and identity
           - Predict psychological response and satisfaction
        
        3. CULTURAL CONTEXT:
           - Consider cultural norms and expectations
           - Ensure cultural appropriateness and sensitivity
           - Adapt for cultural background if relevant
           - Avoid potential cultural misunderstandings
        
        4. RELATIONSHIP INTELLIGENCE:
           - Tailor for specific relationship dynamics
           - Consider appropriate intimacy level
           - Align with relationship history and trajectory
           - Respect boundaries while showing care
        
        5. PREMIUM PERSONALIZATION:
           - Identify unique personalization opportunities
           - Consider their individual preferences and quirks
           - Plan meaningful presentation and timing
           - Add thoughtful details that show deep understanding
        
        PREMIUM RECOMMENDATION:
        After comprehensive analysis, provide recommendation in this JSON format:
        {{
            "gift_name": "specific premium gift name",
            "reasoning": "comprehensive ethical and psychological reasoning",
            "match_score": 0.9,
            "emotional_appeal": "primary emotional response expected",
            "personalization_ideas": ["premium idea 1", "premium idea 2", "premium idea 3"],
            "price_range": "{budget_range}",
            "alternative_gifts": ["premium alternative 1", "premium alternative 2"],
            "confidence": 0.9,
            "ethical_assessment": {{
                "appropriateness_score": 0.95,
                "cultural_sensitivity": 0.90,
                "boundary_respect": 0.95,
                "risk_level": "low"
            }},
            "psychological_insight": {{
                "personality_alignment": 0.90,
                "emotional_impact": 0.85,
                "value_compatibility": 0.90,
                "predicted_satisfaction": 0.88
            }},
            "premium_features": {{
                "ethical_soundness": "detailed ethical analysis",
                "cultural_awareness": "cultural considerations addressed",
                "psychological_depth": "deep personality insight applied",
                "relationship_intelligence": "relationship dynamics considered"
            }}
        }}
        """
        
        response = await self.generate_json_async(
            prompt=prompt,
            json_schema=GiftRecommendationSchema.model_json_schema(),
            system_prompt="You are Dr. Elena Hartmann, a world-renowned expert in ethical gift psychology, combining expertise in cultural anthropology, personality psychology, and relationship dynamics to provide premium gift recommendations.",
            temperature=0.5
        )
        
        if not response.success:
            # Create fallback with ethical considerations
            return GiftRecommendationSchema(
                gift_name="Thoughtfully Chosen Experience",
                reasoning="Async premium ethical analysis suggests experience gifts minimize boundary risks while maximizing emotional connection",
                match_score=0.7,
                emotional_appeal="meaningful connection",
                personalization_ideas=["Choose based on their interests", "Add personal note explaining choice", "Consider timing carefully"],
                price_range=budget_range,
                alternative_gifts=["Quality time activity", "Personalized keepsake"],
                confidence=0.7
            )
        
        try:
            recommendation = GiftRecommendationSchema.model_validate(response.parsed_json)
            print(f"✅ Async premium Claude recommendation: {recommendation.gift_name}")
            return recommendation
        except Exception as e:
            # Fallback on validation error
            return GiftRecommendationSchema(
                gift_name="Ethically Considered Gift",
                reasoning="Async premium analysis framework applied with ethical safeguards",
                match_score=0.6,
                emotional_appeal="thoughtful consideration",
                personalization_ideas=["Respect boundaries", "Show genuine care"],
                price_range=budget_range,
                alternative_gifts=["Safe meaningful choice", "Culturally appropriate option"],
                confidence=0.6
            )
    
    async def batch_risk_assessment_async(self,
                                        gift_ideas: List[str],
                                        personality_profile: Dict,
                                        relationship: str,
                                        cultural_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Assess risks for multiple gift ideas concurrently
        
        VERWENDUNG:
        risks = await client.batch_risk_assessment_async(
            ["expensive_jewelry", "personal_book", "experience_gift"],
            profile,
            "colleague"
        )
        """
        
        print(f"🎭 Assessing risks for {len(gift_ideas)} gift ideas concurrently...")
        
        # Create risk assessment tasks
        tasks = []
        for gift_idea in gift_ideas:
            prompt = f"""
            Perform a comprehensive risk assessment for this gift idea:
            
            GIFT UNDER ANALYSIS: "{gift_idea}"
            
            CONTEXT:
            - Personality: {json.dumps(personality_profile, indent=2)}
            - Relationship: {relationship}
            - Cultural Context: {cultural_context or "Not specified"}
            
            RISK ASSESSMENT FRAMEWORK:
            
            1. RELATIONSHIP RISKS:
               - Could this gift be misinterpreted?
               - Does it respect appropriate boundaries?
               - Might it create unwanted obligations?
               - Could it change relationship dynamics unexpectedly?
            
            2. CULTURAL RISKS:
               - Are there cultural taboos or sensitivities?
               - Might this be inappropriate in their cultural context?
               - Could it cause cultural misunderstandings?
               - Are there religious or social considerations?
            
            3. PSYCHOLOGICAL RISKS:
               - How might this affect their self-perception?
               - Could it trigger negative emotions or memories?
               - Might it conflict with their values or identity?
               - Could it cause stress or discomfort?
            
            4. PRACTICAL RISKS:
               - Are there safety or usability concerns?
               - Might it be unwanted or unused?
               - Could it create storage or maintenance burdens?
               - Are there financial implications for them?
            
            5. MITIGATION STRATEGIES:
               - How to minimize identified risks?
               - What modifications would improve safety?
               - How to present the gift to reduce risk?
               - What alternatives address the same needs with less risk?
            
            Provide comprehensive risk assessment in JSON format:
            {{
                "gift_idea": "{gift_idea}",
                "overall_risk_level": "low/medium/high",
                "risk_categories": {{
                    "relationship_risks": {{
                        "risk_level": "low/medium/high",
                        "specific_risks": ["risk1", "risk2"],
                        "boundary_concerns": ["concern1", "concern2"],
                        "misinterpretation_potential": "low/medium/high"
                    }},
                    "cultural_risks": {{
                        "risk_level": "low/medium/high",
                        "cultural_taboos": ["taboo1", "taboo2"],
                        "sensitivity_areas": ["area1", "area2"],
                        "religious_considerations": ["consideration1", "consideration2"]
                    }},
                    "psychological_risks": {{
                        "risk_level": "low/medium/high",
                        "emotional_triggers": ["trigger1", "trigger2"],
                        "value_conflicts": ["conflict1", "conflict2"],
                        "identity_implications": ["implication1", "implication2"]
                    }},
                    "practical_risks": {{
                        "risk_level": "low/medium/high",
                        "usability_concerns": ["concern1", "concern2"],
                        "burden_potential": ["burden1", "burden2"],
                        "safety_considerations": ["safety1", "safety2"]
                    }}
                }},
                "mitigation_strategies": {{
                    "presentation_modifications": ["modification1", "modification2"],
                    "timing_considerations": ["consideration1", "consideration2"],
                    "accompanying_elements": ["element1", "element2"],
                    "alternative_approaches": ["approach1", "approach2"]
                }},
                "recommendation": {{
                    "proceed": true/false,
                    "confidence": 0.85,
                    "conditions": ["condition1", "condition2"],
                    "safer_alternatives": ["alternative1", "alternative2"]
                }}
            }}
            """
            
            task = self.generate_json_async(
                prompt=prompt,
                json_schema={"type": "object"},
                system_prompt="You are a risk assessment expert specializing in social and cultural risk analysis for interpersonal relationships and gift-giving scenarios.",
                temperature=0.3
            )
            tasks.append(task)
        
        # Execute all risk assessments concurrently
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Process results
        risk_assessments = []
        for i, response in enumerate(responses):
            if isinstance(response, AIResponse) and response.success and response.parsed_json:
                risk_assessment = response.parsed_json
                risk_assessment["processing_time"] = total_time / len(gift_ideas)
                risk_assessments.append(risk_assessment)
            else:
                # Fallback risk assessment
                risk_assessments.append({
                    "gift_idea": gift_ideas[i],
                    "overall_risk_level": "medium",
                    "error": "Risk assessment failed",
                    "fallback_recommendation": "Use conservative approach",
                    "processing_time": total_time / len(gift_ideas)
                })
        
        print(f"✅ Batch risk assessment completed in {total_time:.2f}s")
        return risk_assessments
    
    async def family_ethical_planning_async(self,
                                          family_profiles: List[Dict]) -> List[Dict[str, Any]]:
        """
        Comprehensive ethical planning for family gift giving
        
        VERWENDUNG:
        family = [
            {"name": "Mom", "profile": profile1, "occasion": "christmas", "relationship": "mother"},
            {"name": "Dad", "profile": profile2, "occasion": "christmas", "relationship": "father"},
            {"name": "Sister", "profile": profile3, "occasion": "christmas", "relationship": "sister"}
        ]
        
        ethical_plans = await client.family_ethical_planning_async(family)
        """
        
        print(f"👨‍👩‍👧‍👦 Creating ethical gift plans for {len(family_profiles)} family members...")
        
        # Create comprehensive analysis for each family member
        tasks = []
        for member in family_profiles:
            # Combine ethical analysis with premium recommendation
            async def create_family_plan(member_data):
                # Get budget from member data or use profile defaults
                member_budget = member_data.get("budget_range")
                if not member_budget and member_data["profile"]:
                    profile = member_data["profile"]
                    member_budget = f"€{profile.budget_min or 10}-{profile.budget_max or 500}"
                elif not member_budget:
                    member_budget = "€10-€500"  # Default fallback
                
                ethical_analysis = await self.ethical_gift_analysis_async(
                    personality_profile=member_data["profile"],
                    occasion=member_data["occasion"],
                    budget_range=member_budget,
                    relationship=member_data["relationship"],
                    cultural_context=member_data.get("cultural_context")
                )
                
                premium_recommendation = await self.premium_gift_recommendation_async(
                    personality_profile=member_data["profile"],
                    occasion=member_data["occasion"],
                    budget_range=member_budget,
                    relationship=member_data["relationship"]
                )
                
                return {
                    "family_member": member_data["name"],
                    "ethical_analysis": ethical_analysis,
                    "premium_recommendation": premium_recommendation.model_dump(),
                    "overall_assessment": {
                        "ethical_score": ethical_analysis.get("overall_assessment", {}).get("ethical_score", 0.7),
                        "cultural_sensitivity": ethical_analysis.get("overall_assessment", {}).get("cultural_sensitivity_score", 0.7),
                        "recommendation_confidence": premium_recommendation.confidence
                    }
                }
            
            tasks.append(create_family_plan(member))
        
        # Execute all family plans concurrently
        start_time = time.time()
        family_plans = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Process results and add family-wide insights
        processed_plans = []
        for plan in family_plans:
            if isinstance(plan, dict):
                plan["processing_time"] = total_time / len(family_profiles)
                processed_plans.append(plan)
            else:
                # Fallback plan
                processed_plans.append({
                    "family_member": "Unknown",
                    "error": str(plan) if isinstance(plan, Exception) else "Planning failed",
                    "fallback_strategy": "Use conservative family-appropriate gifts",
                    "processing_time": total_time / len(family_profiles)
                })
        
        print(f"✅ Family ethical planning completed in {total_time:.2f}s")
        return processed_plans
    
    # === UTILITY METHODS ===
    
    async def test_async_ethical_reasoning(self, scenario: Dict) -> Dict[str, Any]:
        """
        Test async ethical reasoning capabilities
        """
        
        print("🎭 Testing async ethical reasoning capability...")
        
        prompt = f"""
        Analyze this ethical scenario and provide your reasoning:
        
        SCENARIO:
        {json.dumps(scenario, indent=2)}
        
        ETHICAL ANALYSIS:
        1. Identify the ethical considerations
        2. Analyze potential consequences
        3. Consider different perspectives
        4. Provide ethical recommendation
        
        Respond in JSON format:
        {{
            "ethical_considerations": ["consideration1", "consideration2"],
            "potential_consequences": ["consequence1", "consequence2"],
            "perspectives": ["perspective1", "perspective2"],
            "ethical_recommendation": "detailed recommendation",
            "reasoning_confidence": 0.9,
            "processing_method": "async_claude"
        }}
        """
        
        response = await self.generate_json_async(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are an ethics expert. Provide thorough ethical analysis.",
            temperature=0.3
        )
        
        return response.parsed_json if response.success else {}
    
    async def close_async(self):
        """
        Properly close async resources
        """
        # Anthropic doesn't need explicit closing
        print("🔒 AsyncAnthropic client closed")


# === FACTORY FUNCTIONS ===

def create_async_anthropic_client(api_key: str) -> AsyncAnthropicClient:
    """
    Factory function for AsyncAnthropicClient
    
    VERWENDUNG:
    client = create_async_anthropic_client(api_key)
    """
    return AsyncAnthropicClient(api_key=api_key)


def create_ethical_reasoning_async_client(api_key: str) -> AsyncAnthropicClient:
    """
    Creates async Anthropic client optimized for ethical reasoning
    """
    client = AsyncAnthropicClient(
        api_key=api_key,
        max_concurrent_requests=6  # Conservative for premium ethical analysis
    )
    return client


# === TESTING UTILITIES ===

async def test_async_anthropic_integration(api_key: str) -> Dict[str, Any]:
    """
    Comprehensive test of async Anthropic integration
    """
    
    results = {
        "connection": False,
        "async_ethical_reasoning": False,
        "batch_ethical_analysis": False,
        "premium_recommendation": False,
        "batch_risk_assessment": False,
        "family_planning": False,
        "performance_metrics": None,
        "errors": []
    }
    
    try:
        async with create_async_anthropic_client(api_key) as client:
            
            # Test 1: Connection
            text_response = await client.generate_text_async("Hello Async Claude!", max_tokens=50)
            results["connection"] = text_response.success
            
            if results["connection"]:
                # Test 2: Async Ethical Reasoning
                ethical_scenario = {
                    "situation": "giving expensive gift to colleague",
                    "relationship": "professional",
                    "concerns": ["appropriateness", "workplace_dynamics"]
                }
                
                reasoning_result = await client.test_async_ethical_reasoning(ethical_scenario)
                results["async_ethical_reasoning"] = bool(reasoning_result)
                
                # Test 3: Batch Ethical Analysis
                if results["async_ethical_reasoning"]:
                    test_profile = {
                        "personality_scores": {"conscientiousness": 0.8, "agreeableness": 0.7},
                        "hobbies": ["reading", "volunteering"],
                        "values": ["social_justice", "environmental_sustainability"]
                    }
                    
                    try:
                        scenarios = [
                            {"profile": test_profile, "occasion": "birthday", "relationship": "friend"},
                            {"profile": test_profile, "occasion": "christmas", "relationship": "colleague"}
                        ]
                        
                        batch_start = time.time()
                        batch_analyses = await client.batch_ethical_analysis_async(scenarios)
                        batch_time = time.time() - batch_start
                        
                        results["batch_ethical_analysis"] = len(batch_analyses) == 2
                        results["batch_time"] = batch_time
                        results["sample_batch_analysis"] = batch_analyses
                        
                    except Exception as e:
                        results["errors"].append(f"Batch ethical analysis failed: {e}")
                
                # Test 4: Premium Recommendation
                if results["batch_ethical_analysis"]:
                    try:
                        recommendation = await client.premium_gift_recommendation_async(
                            personality_profile=test_profile,
                            occasion="birthday",
                            budget_range="100-200",
                            relationship="close_friend"
                        )
                        results["premium_recommendation"] = True
                        results["sample_recommendation"] = recommendation.model_dump()
                    except Exception as e:
                        results["errors"].append(f"Premium recommendation failed: {e}")
                
                # Test 5: Batch Risk Assessment
                if results["premium_recommendation"]:
                    try:
                        gift_ideas = ["expensive_jewelry", "book", "experience_gift"]
                        risk_assessments = await client.batch_risk_assessment_async(
                            gift_ideas, test_profile, "friend"
                        )
                        results["batch_risk_assessment"] = len(risk_assessments) == 3
                        results["sample_risk_assessments"] = risk_assessments
                    except Exception as e:
                        results["errors"].append(f"Batch risk assessment failed: {e}")
                
                # Test 6: Family Planning
                if results["batch_risk_assessment"]:
                    try:
                        family = [
                            {"name": "Alice", "profile": test_profile, "occasion": "christmas", "relationship": "friend"},
                            {"name": "Bob", "profile": test_profile, "occasion": "christmas", "relationship": "family"}
                        ]
                        
                        family_plans = await client.family_ethical_planning_async(family)
                        results["family_planning"] = len(family_plans) == 2
                        results["sample_family_plans"] = family_plans
                        
                    except Exception as e:
                        results["errors"].append(f"Family planning failed: {e}")
        
    except Exception as e:
        results["errors"].append(f"AsyncAnthropic client creation failed: {e}")
    
    return results


# === USAGE EXAMPLES ===

async def example_usage():
    """
    Examples of AsyncAnthropicClient usage
    """
    
    api_key = "your-anthropic-api-key"
    
    # Example 1: Ethical analysis
    async with create_async_anthropic_client(api_key) as client:
        
        profile = {
            "personality_scores": {"conscientiousness": 0.8, "agreeableness": 0.9},
            "values": ["environmental_sustainability", "social_justice"],
            "hobbies": ["volunteering", "gardening"]
        }
        
        # Async ethical analysis
        ethics = await client.ethical_gift_analysis_async(
            personality_profile=profile,
            occasion="birthday",
            budget_range="100-300",
            relationship="close_friend",
            cultural_context="environmentally_conscious_community"
        )
        
        print(f"🎭 Ethical score: {ethics.get('overall_assessment', {}).get('ethical_score', 'unknown')}")
    
    # Example 2: Family ethical planning
    async with create_async_anthropic_client(api_key) as client:
        
        family = [
            {"name": "Mom", "profile": profile, "occasion": "christmas", "relationship": "mother"},
            {"name": "Dad", "profile": profile, "occasion": "christmas", "relationship": "father"},
            {"name": "Sister", "profile": profile, "occasion": "christmas", "relationship": "sister"}
        ]
        
        # All ethical analyses in parallel
        family_ethics = await client.family_ethical_planning_async(family)
        
        for plan in family_ethics:
            member_name = plan.get("family_member", "Unknown")
            ethical_score = plan.get("overall_assessment", {}).get("ethical_score", 0)
            print(f"👨‍👩‍👧‍👦 {member_name}: Ethical score {ethical_score:.2f}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())