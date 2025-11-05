"""
Gemini Client - Google AI für Complex Reasoning & Gift Recommendations

GEMINI STRENGTHS:
- Excellent logical reasoning and analysis
- Strong context understanding
- Multimodal capabilities (text + images)
- Cost-efficient compared to OpenAI GPT-4
- Good balance of speed and quality

MODELS:
- Gemini Pro: Best balance speed/quality/cost
- Gemini Flash: Ultra-fast for simple tasks

USE CASES:
- Complex personality analysis
- Multi-constraint gift recommendations
- Logical reasoning about gift appropriateness
- Context-aware personalization
- Budget optimization with multiple factors
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    raise ImportError("Google GenerativeAI library not installed. Run: pip install google-generativeai")

from .base_client import (
    BaseAIClient, AIRequest, AIResponse, AIModelType,
    ModelCapability, ResponseFormat, GiftRecommendationSchema
)


class GeminiClient(BaseAIClient):
    """
    Google Gemini Client für Complex Reasoning in Gift Recommendations
    
    Spezialisiert auf:
    - Multi-factor analysis (personality + context + constraints)
    - Logical reasoning about gift appropriateness
    - Complex constraint satisfaction
    - Context-aware personalization
    - Budget optimization
    """
    
    def __init__(self, 
                 api_key: str,
                 model_type: AIModelType = AIModelType.GEMINI_PRO,
                 rate_limit_per_minute: int = 60):
        
        # Validate model type
        if model_type not in [AIModelType.GEMINI_PRO, AIModelType.GEMINI_FLASH]:
            raise ValueError(f"Invalid model type for Gemini: {model_type}")
        
        super().__init__(api_key, model_type, rate_limit_per_minute)
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Model configuration
        self.model_name = self._get_model_name()
        self.model = genai.GenerativeModel(self.model_name)
        
        # Safety settings (less restrictive for gift recommendations)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Generation config optimized for gift recommendations
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        print(f"🧠 GeminiClient initialized: {self.model_name}")
    
    def _get_model_name(self) -> str:
        """Maps our model type to Gemini model name"""
        mapping = {
            AIModelType.GEMINI_PRO: "gemini-1.5-flash-latest",
            AIModelType.GEMINI_FLASH: "gemini-1.5-flash-latest"
        }
        return mapping[self.model_type]
    
    def _get_model_capabilities(self) -> List[ModelCapability]:
        """Define what this Gemini model can do"""
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.COMPLEX_REASONING,
            ModelCapability.JSON_OUTPUT,
            ModelCapability.MULTIMODAL  # Can handle images
        ]
        
        if self.model_type == AIModelType.GEMINI_PRO:
            capabilities.append(ModelCapability.HIGH_QUALITY)
        elif self.model_type == AIModelType.GEMINI_FLASH:
            capabilities.append(ModelCapability.FAST_RESPONSE)
        
        return capabilities
    
    def _get_model_pricing(self) -> Dict[str, float]:
        """Gemini pricing (cost-efficient)"""
        pricing = {
            AIModelType.GEMINI_PRO: {
                "input_per_token": 0.00000125,   # $1.25 per 1M tokens
                "output_per_token": 0.00000375   # $3.75 per 1M tokens
            },
            AIModelType.GEMINI_FLASH: {
                "input_per_token": 0.000000075,  # $0.075 per 1M tokens
                "output_per_token": 0.0000003    # $0.3 per 1M tokens
            }
        }
        return pricing[self.model_type]
    
    def _make_api_call(self, request: AIRequest) -> AIResponse:
        """
        Makes Gemini API call with complex reasoning optimization
        """
        try:
            # Build prompt with system context
            full_prompt = self._build_full_prompt(request)
            
            # Update generation config with request parameters
            config = self.generation_config.copy()
            config["temperature"] = request.temperature
            config["top_p"] = request.top_p
            config["max_output_tokens"] = request.max_tokens or 2048
            
            # Make API call
            start_time = time.time()
            response = self.model.generate_content(
                full_prompt,
                generation_config=config,
                safety_settings=self.safety_settings
            )
            response_time = time.time() - start_time
            
            # Extract content
            content = response.text
            
            # Gemini doesn't provide token usage, so we estimate
            tokens_used = self._estimate_tokens(full_prompt + content)
            
            # Calculate cost
            input_tokens = self._estimate_tokens(full_prompt)
            output_tokens = self._estimate_tokens(content)
            cost = (
                input_tokens * self.pricing["input_per_token"] +
                output_tokens * self.pricing["output_per_token"]
            )
            
            # Parse JSON if requested
            parsed_json = None
            if request.response_format == ResponseFormat.JSON:
                try:
                    # Gemini sometimes adds explanatory text, try to extract JSON
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
                error=f"Gemini API Error: {e}"
            )
    
    def _build_full_prompt(self, request: AIRequest) -> str:
        """Build complete prompt with system context for Gemini"""
        prompt_parts = []
        
        if request.system_prompt:
            prompt_parts.append(f"SYSTEM CONTEXT:\n{request.system_prompt}\n")
        
        prompt_parts.append(f"USER REQUEST:\n{request.prompt}")
        
        if request.response_format == ResponseFormat.JSON:
            prompt_parts.append("\nIMPORTANT: Respond with valid JSON only, no additional text or markdown.")
        
        return "\n".join(prompt_parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation for Gemini"""
        # Gemini uses similar tokenization to GPT, roughly 4 chars per token
        return len(text) // 4
    
    # === SPECIALIZED METHODS FOR COMPLEX REASONING ===
    
    def analyze_gift_context_deeply(self,
                                   personality_profile: Dict,
                                   occasion: str,
                                   budget_range: str,
                                   relationship: str,
                                   additional_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Deep contextual analysis for gift recommendations
        
        Gemini's strength: Complex multi-factor analysis
        """
        
        context_data = additional_context or {}
        
        prompt = f"""
        Perform a deep contextual analysis for gift recommendation:
        
        PERSONALITY PROFILE:
        {json.dumps(personality_profile, indent=2)}
        
        GIFT CONTEXT:
        - Occasion: {occasion}
        - Relationship: {relationship}
        - Budget: €{budget_range}
        
        ADDITIONAL CONTEXT:
        {json.dumps(context_data, indent=2)}
        
        DEEP ANALYSIS REQUIRED:
        
        1. PERSONALITY INSIGHTS:
           - What are the core psychological drivers?
           - What emotional needs are most important?
           - How do their traits interact with each other?
           - What personality conflicts might affect gift reception?
        
        2. CONTEXTUAL FACTORS:
           - How does the occasion influence their expectations?
           - What does this relationship mean to them psychologically?
           - Are there cultural or social factors to consider?
           - How does the budget constraint affect the emotional impact?
        
        3. GIFT STRATEGY:
           - What approach would maximize emotional resonance?
           - What categories should be prioritized/avoided?
           - How should the gift be personalized?
           - What presentation style would be most effective?
        
        4. POTENTIAL CHALLENGES:
           - What could go wrong with certain gift types?
           - How to avoid common gifting mistakes for this person?
           - What backup strategies should be considered?
        
        5. SUCCESS FACTORS:
           - What would make this gift truly memorable?
           - How to exceed their expectations within budget?
           - What personalization would be most meaningful?
        
        Provide comprehensive analysis in JSON format:
        {{
            "personality_insights": {{
                "core_drivers": ["driver1", "driver2"],
                "emotional_needs": ["need1", "need2"],
                "trait_interactions": "analysis",
                "potential_conflicts": ["conflict1", "conflict2"]
            }},
            "contextual_factors": {{
                "occasion_influence": "analysis",
                "relationship_meaning": "analysis",
                "cultural_considerations": ["factor1", "factor2"],
                "budget_impact": "analysis"
            }},
            "gift_strategy": {{
                "optimal_approach": "strategy",
                "priority_categories": ["category1", "category2"],
                "avoid_categories": ["category1", "category2"],
                "personalization_approach": "strategy",
                "presentation_style": "recommendation"
            }},
            "potential_challenges": {{
                "risk_factors": ["risk1", "risk2"],
                "common_mistakes": ["mistake1", "mistake2"],
                "backup_strategies": ["strategy1", "strategy2"]
            }},
            "success_factors": {{
                "memorable_elements": ["element1", "element2"],
                "expectation_exceeding": "strategy",
                "meaningful_personalization": ["idea1", "idea2"]
            }},
            "confidence_assessment": {{
                "overall_confidence": 0.85,
                "analysis_completeness": 0.90,
                "recommendation_strength": 0.80
            }}
        }}
        """
        
        response = self.generate_json(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are a psychological gift analysis expert with deep understanding of human behavior, relationships, and cultural contexts. Provide thorough, nuanced analysis.",
            temperature=0.4  # Lower for more analytical consistency
        )
        
        return response.parsed_json if response.success else {}
    
    def multi_constraint_optimization(self,
                                    personality_profile: Dict,
                                    constraints: Dict[str, Any],
                                    preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize gift recommendation across multiple constraints
        
        Gemini's strength: Complex constraint satisfaction
        """
        
        prompt = f"""
        Solve this multi-constraint gift optimization problem:
        
        PERSONALITY PROFILE:
        {json.dumps(personality_profile, indent=2)}
        
        CONSTRAINTS:
        {json.dumps(constraints, indent=2)}
        
        PREFERENCES:
        {json.dumps(preferences, indent=2)}
        
        OPTIMIZATION TASK:
        Find the optimal gift recommendation that maximizes satisfaction across all constraints while respecting all hard limits.
        
        ANALYSIS FRAMEWORK:
        
        1. CONSTRAINT ANALYSIS:
           - Identify hard constraints (must be satisfied)
           - Identify soft constraints (preferences to optimize)
           - Analyze constraint interactions and conflicts
           - Determine constraint priorities
        
        2. SOLUTION SPACE:
           - Map viable gift options within constraints
           - Identify constraint trade-offs
           - Evaluate feasibility of different approaches
           - Consider creative solutions that satisfy multiple constraints
        
        3. OPTIMIZATION SCORING:
           - Weight different factors based on importance
           - Score potential gifts across all dimensions
           - Consider both objective and subjective factors
           - Account for risk and uncertainty
        
        4. RECOMMENDATION SYNTHESIS:
           - Select optimal primary recommendation
           - Provide alternative options with different trade-offs
           - Explain reasoning and constraint satisfaction
           - Suggest improvement opportunities
        
        Provide optimization results in JSON format:
        {{
            "constraint_analysis": {{
                "hard_constraints": ["constraint1", "constraint2"],
                "soft_constraints": ["preference1", "preference2"],
                "constraint_conflicts": ["conflict1", "conflict2"],
                "priority_ranking": ["priority1", "priority2"]
            }},
            "solution_space": {{
                "viable_options": ["option1", "option2"],
                "trade_offs": {{
                    "option1": {{"pros": ["pro1"], "cons": ["con1"]}},
                    "option2": {{"pros": ["pro1"], "cons": ["con1"]}}
                }},
                "creative_solutions": ["solution1", "solution2"]
            }},
            "optimization_scores": {{
                "option1": {{
                    "total_score": 0.85,
                    "constraint_satisfaction": 0.90,
                    "preference_match": 0.80,
                    "risk_factor": 0.15
                }},
                "option2": {{
                    "total_score": 0.82,
                    "constraint_satisfaction": 0.85,
                    "preference_match": 0.85,
                    "risk_factor": 0.20
                }}
            }},
            "final_recommendation": {{
                "primary_option": "option1",
                "reasoning": "detailed explanation",
                "alternative_options": ["option2", "option3"],
                "improvement_suggestions": ["suggestion1", "suggestion2"]
            }}
        }}
        """
        
        response = self.generate_json(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are an optimization expert specializing in multi-constraint problem solving. Use logical reasoning to find optimal solutions.",
            temperature=0.3  # Lower for more logical consistency
        )
        
        return response.parsed_json if response.success else {}
    
    def reasoning_based_recommendation(self,
                                     personality_profile: Dict,
                                     occasion: str,
                                     budget_range: str,
                                     relationship: str,
                                     reasoning_depth: str = "deep") -> GiftRecommendationSchema:
        """
        Generate gift recommendation using step-by-step logical reasoning
        
        Gemini's strength: Explicit reasoning chains
        """
        
        depth_instructions = {
            "simple": "Use 3-step reasoning process",
            "moderate": "Use 5-step reasoning process with examples",
            "deep": "Use comprehensive 7-step reasoning with detailed analysis"
        }
        
        prompt = f"""
        Generate a gift recommendation using systematic logical reasoning:
        
        PERSON ANALYSIS:
        {json.dumps(personality_profile, indent=2)}
        
        GIFT PARAMETERS:
        - Occasion: {occasion}
        - Relationship: {relationship}
        - Budget: €{budget_range}
        
        REASONING INSTRUCTIONS:
        {depth_instructions.get(reasoning_depth, depth_instructions["moderate"])}
        
        STEP-BY-STEP REASONING PROCESS:
        
        STEP 1: PERSONALITY ANALYSIS
        - Identify the 3 most important personality traits
        - Determine primary emotional drivers
        - Assess what brings them joy and satisfaction
        
        STEP 2: OCCASION CONTEXT
        - Analyze what this occasion means to them personally
        - Consider cultural and social expectations
        - Evaluate appropriate intimacy level for gift
        
        STEP 3: RELATIONSHIP DYNAMICS
        - Assess the nature of your relationship
        - Determine appropriate gift boundaries
        - Consider shared experiences and memories
        
        STEP 4: EMOTIONAL RESONANCE
        - Identify what emotions the gift should evoke
        - Consider their current life circumstances
        - Match gift to their emotional needs
        
        STEP 5: PRACTICAL CONSTRAINTS
        - Evaluate budget limitations and opportunities
        - Consider timing and availability
        - Assess personalization possibilities
        
        STEP 6: GIFT SYNTHESIS
        - Combine all factors into coherent recommendation
        - Ensure gift aligns with all analyzed factors
        - Maximize impact within constraints
        
        STEP 7: VALIDATION & ALTERNATIVES
        - Verify recommendation against all criteria
        - Identify potential issues or improvements
        - Generate suitable alternatives
        
        FINAL RECOMMENDATION:
        After completing all reasoning steps, provide your recommendation in this JSON format:
        {{
            "gift_name": "specific gift name",
            "reasoning": "detailed step-by-step reasoning showing your analysis process",
            "match_score": 0.85,
            "emotional_appeal": "primary emotional response expected",
            "personalization_ideas": ["idea1", "idea2", "idea3"],
            "price_range": "{budget_range}",
            "alternative_gifts": ["alternative1", "alternative2"],
            "confidence": 0.9,
            "reasoning_steps": {{
                "personality_analysis": "key insights",
                "occasion_context": "analysis",
                "relationship_dynamics": "insights",
                "emotional_resonance": "expected emotions",
                "practical_constraints": "considerations",
                "gift_synthesis": "combination logic",
                "validation": "final verification"
            }}
        }}
        """
        
        response = self.generate_json(
            prompt=prompt,
            json_schema=GiftRecommendationSchema.model_json_schema(),
            system_prompt="You are Dr. Elena Hartmann, a logical reasoning expert who uses systematic analysis to make perfect gift recommendations. Think step by step and show your reasoning.",
            temperature=0.5  # Balanced for reasoning
        )
        
        if not response.success:
            # Create fallback recommendation with basic reasoning
            return GiftRecommendationSchema(
                gift_name="Thoughtful Experience Gift",
                reasoning="Systematic analysis suggests experiential gifts create lasting memories and emotional connections",
                match_score=0.7,
                emotional_appeal="meaningful connection",
                personalization_ideas=["Customize to their interests", "Add personal note", "Choose timing carefully"],
                price_range=budget_range,
                alternative_gifts=["Personalized item", "Quality time activity"],
                confidence=0.6
            )
        
        try:
            return GiftRecommendationSchema.model_validate(response.parsed_json)
        except Exception as e:
            # Fallback on validation error
            return GiftRecommendationSchema(
                gift_name="Personalized Memory Gift",
                reasoning="Reasoning-based analysis failed, providing safe personal choice",
                match_score=0.6,
                emotional_appeal="personal connection",
                personalization_ideas=["Add personal touch", "Include shared memory"],
                price_range=budget_range,
                alternative_gifts=["Experience gift", "Quality item"],
                confidence=0.5
            )
    
    def compare_gift_options(self,
                           personality_profile: Dict,
                           gift_options: List[Dict],
                           evaluation_criteria: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare multiple gift options using complex reasoning
        
        Gemini's strength: Multi-factor comparison and analysis
        """
        
        prompt = f"""
        Compare and evaluate these gift options systematically:
        
        PERSONALITY PROFILE:
        {json.dumps(personality_profile, indent=2)}
        
        GIFT OPTIONS TO COMPARE:
        {json.dumps(gift_options, indent=2)}
        
        EVALUATION CRITERIA (with weights):
        {json.dumps(evaluation_criteria, indent=2)}
        
        COMPARISON FRAMEWORK:
        
        1. INDIVIDUAL ANALYSIS:
           For each gift option, analyze:
           - Personality fit (how well it matches their traits)
           - Emotional impact (what emotions it will evoke)
           - Practical suitability (budget, timing, appropriateness)
           - Personalization potential (how it can be customized)
           - Risk factors (what could go wrong)
        
        2. COMPARATIVE ANALYSIS:
           - Direct comparison across all criteria
           - Identify strengths and weaknesses of each option
           - Consider synergies and trade-offs
           - Evaluate differentiation factors
        
        3. WEIGHTED SCORING:
           - Score each option on each criterion (0-1)
           - Apply weights to calculate overall scores
           - Consider both quantitative and qualitative factors
           - Account for uncertainty and risk
        
        4. RECOMMENDATION RANKING:
           - Rank options from best to worst
           - Explain reasoning for rankings
           - Identify best option for different scenarios
           - Suggest improvements for lower-ranked options
        
        Provide comprehensive comparison in JSON format:
        {{
            "individual_analysis": {{
                "option1": {{
                    "personality_fit": 0.8,
                    "emotional_impact": 0.9,
                    "practical_suitability": 0.7,
                    "personalization_potential": 0.8,
                    "risk_factors": ["risk1", "risk2"],
                    "strengths": ["strength1", "strength2"],
                    "weaknesses": ["weakness1", "weakness2"]
                }},
                "option2": {{
                    "personality_fit": 0.7,
                    "emotional_impact": 0.8,
                    "practical_suitability": 0.9,
                    "personalization_potential": 0.6,
                    "risk_factors": ["risk1"],
                    "strengths": ["strength1", "strength2"],
                    "weaknesses": ["weakness1"]
                }}
            }},
            "weighted_scores": {{
                "option1": {{
                    "total_score": 0.82,
                    "criterion_scores": {{"personality_fit": 0.8, "emotional_impact": 0.9}},
                    "confidence": 0.85
                }},
                "option2": {{
                    "total_score": 0.78,
                    "criterion_scores": {{"personality_fit": 0.7, "emotional_impact": 0.8}},
                    "confidence": 0.90
                }}
            }},
            "final_ranking": [
                {{
                    "rank": 1,
                    "option": "option1",
                    "score": 0.82,
                    "reasoning": "detailed explanation why this is best"
                }},
                {{
                    "rank": 2,
                    "option": "option2",
                    "score": 0.78,
                    "reasoning": "detailed explanation of position"
                }}
            ],
            "recommendations": {{
                "best_overall": "option1",
                "best_for_safety": "option2",
                "best_for_impact": "option1",
                "improvement_suggestions": {{
                    "option1": ["suggestion1", "suggestion2"],
                    "option2": ["suggestion1", "suggestion2"]
                }}
            }}
        }}
        """
        
        response = self.generate_json(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are a systematic comparison expert who evaluates options using logical frameworks and clear reasoning.",
            temperature=0.4  # Lower for more consistent comparisons
        )
        
        return response.parsed_json if response.success else {}
    
    # === UTILITY METHODS ===
    
    def test_reasoning_capability(self, test_scenario: Dict) -> Dict[str, Any]:
        """
        Test Gemini's reasoning capabilities with a sample scenario
        """
        
        prompt = f"""
        Test scenario for reasoning capability:
        
        SCENARIO:
        {json.dumps(test_scenario, indent=2)}
        
        REASONING CHALLENGE:
        Analyze this scenario and provide your reasoning process step by step.
        
        Show your thinking by:
        1. Identifying key factors
        2. Analyzing relationships between factors
        3. Drawing logical conclusions
        4. Evaluating confidence in your reasoning
        
        Respond in JSON format:
        {{
            "key_factors": ["factor1", "factor2"],
            "factor_relationships": "analysis",
            "logical_conclusions": ["conclusion1", "conclusion2"],
            "reasoning_confidence": 0.8,
            "reasoning_quality": "assessment of own reasoning"
        }}
        """
        
        response = self.generate_json(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are a reasoning expert. Show your systematic thinking process.",
            temperature=0.3
        )
        
        return response.parsed_json if response.success else {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns Gemini model information
        """
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "primary_strength": "complex reasoning",
            "use_cases": ["deep_analysis", "constraint_optimization", "logical_reasoning", "multi_factor_comparison"],
            "max_tokens": 2048,
            "capabilities": [cap.value for cap in self.capabilities],
            "pricing": self.pricing,
            "multimodal": True,
            "expected_response_time": "2-4 seconds"
        }


# === FACTORY FUNCTIONS ===

def create_gemini_client(api_key: str, 
                        model_type: AIModelType = AIModelType.GEMINI_PRO) -> GeminiClient:
    """
    Factory function to create Gemini client
    
    Args:
        api_key: Google API key
        model_type: Which Gemini model to use
        
    Returns:
        Configured GeminiClient instance
    """
    return GeminiClient(api_key=api_key, model_type=model_type)


def create_reasoning_optimized_client(api_key: str) -> GeminiClient:
    """
    Creates Gemini client optimized for complex reasoning
    """
    client = create_gemini_client(api_key, AIModelType.GEMINI_PRO)
    
    # Optimize for reasoning
    client.generation_config.update({
        "temperature": 0.4,  # Lower for more logical consistency
        "top_p": 0.7,        # More focused responses
        "top_k": 30          # Reduce randomness
    })
    
    return client


# === TESTING UTILITIES ===

def test_gemini_integration(api_key: str) -> Dict[str, Any]:
    """
    Comprehensive test of Gemini integration focusing on reasoning
    """
    results = {
        "connection": False,
        "reasoning_test": False,
        "json_generation": False,
        "gift_recommendation": False,
        "deep_analysis": False,
        "errors": []
    }
    
    try:
        client = create_gemini_client(api_key, AIModelType.GEMINI_PRO)
        
        # Test 1: Connection
        text_response = client.generate_text("Hello Gemini!", max_tokens=50)
        results["connection"] = text_response.success
        
        if results["connection"]:
            # Test 2: Reasoning Test
            test_scenario = {
                "person": "creative, introverted, loves books",
                "occasion": "birthday",
                "budget": "€50-€100",
                "challenge": "find perfect gift"
            }
            
            reasoning_result = client.test_reasoning_capability(test_scenario)
            results["reasoning_test"] = bool(reasoning_result)
            
            # Test 3: JSON Generation
            json_response = client.generate_json(
                prompt="Analyze personality traits for gift giving",
                json_schema={"type": "object", "properties": {"traits": {"type": "array"}}}
            )
            results["json_generation"] = json_response.success
            
            # Test 4: Gift Recommendation
            if results["json_generation"]:
                test_profile = {
                    "personality_scores": {"openness": 0.8, "conscientiousness": 0.6},
                    "hobbies": ["reading", "writing"],
                    "emotional_triggers": ["creativity", "nostalgia"]
                }
                
                try:
                    recommendation = client.reasoning_based_recommendation(
                        personality_profile=test_profile,
                        occasion="birthday",
                        budget_range="€50-€100",
                        relationship="friend",
                        reasoning_depth="moderate"
                    )
                    results["gift_recommendation"] = True
                    results["sample_recommendation"] = recommendation.model_dump()
                except Exception as e:
                    results["errors"].append(f"Gift recommendation failed: {e}")
            
            # Test 5: Deep Analysis
            if results["gift_recommendation"]:
                try:
                    analysis = client.analyze_gift_context_deeply(
                        personality_profile=test_profile,
                        occasion="birthday",
                        budget_range="€50-€100",
                        relationship="friend"
                    )
                    results["deep_analysis"] = bool(analysis)
                    results["sample_analysis"] = analysis
                except Exception as e:
                    results["errors"].append(f"Deep analysis failed: {e}")
        
    except Exception as e:
        results["errors"].append(f"Gemini client creation failed: {e}")
    
    return results


def reasoning_quality_test(gemini_client: GeminiClient, 
                         test_cases: List[Dict]) -> Dict[str, Any]:
    """
    Test reasoning quality across multiple scenarios
    """
    results = {
        "total_tests": len(test_cases),
        "successful_tests": 0,
        "average_confidence": 0.0,
        "reasoning_quality": [],
        "failed_tests": []
    }
    
    confidences = []
    
    for i, test_case in enumerate(test_cases):
        try:
            reasoning_result = gemini_client.test_reasoning_capability(test_case)
            if reasoning_result:
                results["successful_tests"] += 1
                confidence = reasoning_result.get("reasoning_confidence", 0.5)
                confidences.append(confidence)
                results["reasoning_quality"].append({
                    "test_case": i,
                    "confidence": confidence,
                    "quality": reasoning_result.get("reasoning_quality", "unknown")
                })
        except Exception as e:
            results["failed_tests"].append({"test_case": i, "error": str(e)})
    
    if confidences:
        results["average_confidence"] = sum(confidences) / len(confidences)
    
    return results