"""
Async Gemini Client - Concurrent Complex Reasoning für Gift Recommendations

ASYNC REASONING SUPERPOWERS:
- Concurrent complex analysis
- Multi-constraint optimization in parallel
- Batch personality analysis
- Async context understanding

PERFORMANCE BOOST:
- Multiple deep analyses simultaneously
- Complex reasoning without blocking
- Perfect for comprehensive gift strategies
- Batch family personality analysis

USE CASES:
- Deep personality analysis for multiple people
- Complex constraint optimization
- Multi-factor gift comparisons
- Comprehensive gift strategy development
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Async Google AI Support
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    # Note: Google's library doesn't have native async support yet,
    # so we'll use asyncio.to_thread for non-blocking execution
except ImportError:
    raise ImportError("Google GenerativeAI library not installed. Run: pip install google-generativeai")

# Import from async_base_client.py
from .async_base_client import AsyncBaseAIClient, AIRequest, AIResponse
from .base_client import (
    AIModelType, ModelCapability, ResponseFormat, 
    GiftRecommendationSchema
)


class AsyncGeminiClient(AsyncBaseAIClient):
    """
    Async Gemini Client für Concurrent Complex Reasoning
    
    FEATURES:
    - Async complex reasoning and analysis
    - Concurrent multi-constraint optimization
    - Batch personality insights
    - Parallel gift option comparisons
    - Non-blocking context analysis
    
    REASONING FOCUS:
    - Multi-factor analysis without blocking
    - Complex constraint satisfaction in parallel
    - Deep personality understanding concurrently
    - Systematic gift comparisons
    """
    
    def __init__(self, 
                 api_key: str,
                 model_type: AIModelType = AIModelType.GEMINI_PRO,
                 rate_limit_per_minute: int = 60,
                 max_concurrent_requests: int = 10):
        
        # Validate model type
        if model_type not in [AIModelType.GEMINI_PRO, AIModelType.GEMINI_FLASH]:
            raise ValueError(f"Invalid model type for AsyncGemini: {model_type}")
        
        # Initialize parent (AsyncBaseAIClient)
        super().__init__(api_key, model_type, rate_limit_per_minute, max_concurrent_requests)
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Model configuration
        self.model_name = self._get_model_name()
        self.model = genai.GenerativeModel(self.model_name)
        
        # Safety settings (same as sync)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Generation config optimized for reasoning
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        print(f"🧠 AsyncGeminiClient initialized: {self.model_name}")
    
    def _get_model_name(self) -> str:
        """Maps model type to Gemini model name"""
        mapping = {
            AIModelType.GEMINI_PRO: "gemini-1.5-pro-latest",
            AIModelType.GEMINI_FLASH: "gemini-1.5-flash-latest"
        }
        return mapping[self.model_type]
    
    def _get_model_capabilities(self) -> List[ModelCapability]:
        """Define capabilities (same as sync)"""
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.COMPLEX_REASONING,
            ModelCapability.JSON_OUTPUT,
            ModelCapability.MULTIMODAL
        ]
        
        if self.model_type == AIModelType.GEMINI_PRO:
            capabilities.append(ModelCapability.HIGH_QUALITY)
        elif self.model_type == AIModelType.GEMINI_FLASH:
            capabilities.append(ModelCapability.FAST_RESPONSE)
        
        return capabilities
    
    def _get_model_pricing(self) -> Dict[str, float]:
        """Gemini pricing (same as sync)"""
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
    
    # === CORE ASYNC METHOD ===
    
    async def _make_async_api_call(self, request: AIRequest) -> AIResponse:
        """
        Async Gemini API Call using asyncio.to_thread for non-blocking execution
        
        Note: Google's library doesn't have native async support yet,
        so we use asyncio.to_thread to prevent blocking
        """
        
        try:
            # Build full prompt
            full_prompt = self._build_full_prompt(request)
            
            # Update generation config
            config = self.generation_config.copy()
            config["temperature"] = request.temperature
            config["top_p"] = request.top_p
            config["max_output_tokens"] = request.max_tokens or 2048
            
            # 🔥 ASYNC EXECUTION - Use asyncio.to_thread to make sync call non-blocking
            print(f"🧠 Making async Gemini API call...")
            start_time = time.time()
            
            # ASYNC WRAPPER - runs in thread pool to avoid blocking
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=config,
                safety_settings=self.safety_settings
            )
            
            response_time = time.time() - start_time
            print(f"✅ Async Gemini response received in {response_time:.3f}s")
            
            # Extract content
            content = response.text
            
            # Estimate tokens and cost
            tokens_used = self._estimate_tokens(full_prompt + content)
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
            print(f"❌ Async Gemini API error: {e}")
            return AIResponse(
                content="",
                model_type=self.model_type,
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                timestamp=datetime.now(),
                success=False,
                error=f"Async Gemini API Error: {e}"
            )
    
    def _build_full_prompt(self, request: AIRequest) -> str:
        """Build complete prompt (same as sync)"""
        prompt_parts = []
        
        if request.system_prompt:
            prompt_parts.append(f"SYSTEM CONTEXT:\n{request.system_prompt}\n")
        
        prompt_parts.append(f"USER REQUEST:\n{request.prompt}")
        
        if request.response_format == ResponseFormat.JSON:
            prompt_parts.append("\nIMPORTANT: Respond with valid JSON only, no additional text or markdown.")
        
        return "\n".join(prompt_parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """Token estimation (same as sync)"""
        return len(text) // 4
    
    # === ASYNC COMPLEX REASONING METHODS ===
    
    async def analyze_gift_context_deeply_async(self,
                                              personality_profile: Dict,
                                              occasion: str,
                                              budget_range: str,
                                              relationship: str,
                                              additional_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Async deep contextual analysis for gift recommendations
        
        VERWENDUNG:
        async with AsyncGeminiClient(api_key) as client:
            analysis = await client.analyze_gift_context_deeply_async(
                personality_profile=profile,
                occasion="birthday",
                budget_range="100-200",
                relationship="partner"
            )
        """
        
        print("🧠 Starting async deep context analysis...")
        
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
        
        response = await self.generate_json_async(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are a psychological gift analysis expert with deep understanding of human behavior, relationships, and cultural contexts. Provide thorough, nuanced analysis.",
            temperature=0.4
        )
        
        print("✅ Async deep context analysis completed")
        return response.parsed_json if response.success else {}
    
    async def batch_personality_analysis_async(self,
                                             personality_profiles: List[Dict],
                                             analysis_focus: str = "gift_recommendations") -> List[Dict[str, Any]]:
        """
        Analyze multiple personality profiles concurrently
        
        Perfect for family gift planning or group recommendations
        
        VERWENDUNG:
        family_profiles = [profile1, profile2, profile3]
        analyses = await client.batch_personality_analysis_async(family_profiles)
        """
        
        print(f"🧠 Analyzing {len(personality_profiles)} personality profiles concurrently...")
        
        # Create analysis tasks for each profile
        tasks = []
        for i, profile in enumerate(personality_profiles):
            prompt = f"""
            Analyze this personality profile for {analysis_focus}:
            
            PERSONALITY PROFILE #{i+1}:
            {json.dumps(profile, indent=2)}
            
            ANALYSIS FOCUS: {analysis_focus}
            
            Provide insights into:
            1. Core personality drivers
            2. Emotional triggers and needs
            3. Gift preference patterns
            4. Potential gifting challenges
            5. Recommended personalization approach
            
            JSON format:
            {{
                "profile_id": {i+1},
                "core_drivers": ["driver1", "driver2"],
                "emotional_triggers": ["trigger1", "trigger2"],
                "gift_preferences": {{
                    "preferred_categories": ["category1", "category2"],
                    "avoid_categories": ["category1", "category2"],
                    "personalization_level": "high/medium/low",
                    "experience_vs_material": "experiences/materials/both"
                }},
                "potential_challenges": ["challenge1", "challenge2"],
                "personalization_approach": "strategy description",
                "gift_strategy_summary": "brief strategy overview"
            }}
            """
            
            task = self.generate_json_async(
                prompt=prompt,
                json_schema={"type": "object"},
                system_prompt="You are a personality analysis expert specializing in gift psychology.",
                temperature=0.4
            )
            tasks.append(task)
        
        # Execute all analyses concurrently
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Process results
        analyses = []
        for i, response in enumerate(responses):
            if isinstance(response, AIResponse) and response.success and response.parsed_json:
                analysis = response.parsed_json
                analysis["processing_time"] = total_time / len(personality_profiles)
                analyses.append(analysis)
            else:
                # Fallback analysis
                analyses.append({
                    "profile_id": i+1,
                    "error": "Analysis failed",
                    "fallback_strategy": "Use general gift recommendations"
                })
        
        print(f"✅ Batch personality analysis completed in {total_time:.2f}s")
        return analyses
    
    async def multi_constraint_optimization_async(self,
                                                personality_profile: Dict,
                                                constraints: Dict[str, Any],
                                                preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async multi-constraint optimization for gift recommendations
        
        VERWENDUNG:
        optimization = await client.multi_constraint_optimization_async(
            personality_profile=profile,
            constraints={"budget": "€50-€100", "delivery_time": "1_week", "occasion": "birthday"},
            preferences={"personalization": True, "eco_friendly": True}
        )
        """
        
        print("🧠 Starting async multi-constraint optimization...")
        
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
        
        response = await self.generate_json_async(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are an optimization expert specializing in multi-constraint problem solving. Use logical reasoning to find optimal solutions.",
            temperature=0.3
        )
        
        print("✅ Async multi-constraint optimization completed")
        return response.parsed_json if response.success else {}
    
    async def compare_gift_options_async(self,
                                       personality_profile: Dict,
                                       gift_options: List[Dict],
                                       evaluation_criteria: Dict[str, float]) -> Dict[str, Any]:
        """
        Async comparison of multiple gift options
        
        VERWENDUNG:
        comparison = await client.compare_gift_options_async(
            personality_profile=profile,
            gift_options=[
                {"name": "Camera", "price": 200, "category": "tech"},
                {"name": "Art Kit", "price": 80, "category": "creative"}
            ],
            evaluation_criteria={"personality_fit": 0.4, "value": 0.3, "uniqueness": 0.3}
        )
        """
        
        print(f"🧠 Comparing {len(gift_options)} gift options async...")
        
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
                }}
            }},
            "weighted_scores": {{
                "option1": {{
                    "total_score": 0.82,
                    "criterion_scores": {{"personality_fit": 0.8, "emotional_impact": 0.9}},
                    "confidence": 0.85
                }}
            }},
            "final_ranking": [
                {{
                    "rank": 1,
                    "option": "option1",
                    "score": 0.82,
                    "reasoning": "detailed explanation why this is best"
                }}
            ],
            "recommendations": {{
                "best_overall": "option1",
                "best_for_safety": "option2",
                "best_for_impact": "option1",
                "improvement_suggestions": {{
                    "option1": ["suggestion1", "suggestion2"]
                }}
            }}
        }}
        """
        
        response = await self.generate_json_async(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are a systematic comparison expert who evaluates options using logical frameworks and clear reasoning.",
            temperature=0.4
        )
        
        print("✅ Async gift options comparison completed")
        return response.parsed_json if response.success else {}
    
    async def reasoning_based_recommendation_async(self,
                                                 personality_profile: Dict,
                                                 occasion: str,
                                                 budget_range: str,
                                                 relationship: str,
                                                 reasoning_depth: str = "deep") -> GiftRecommendationSchema:
        """
        Async reasoning-based gift recommendation
        
        VERWENDUNG:
        recommendation = await client.reasoning_based_recommendation_async(
            personality_profile=profile,
            occasion="birthday",
            budget_range="50-150",
            relationship="friend",
            reasoning_depth="deep"
        )
        """
        
        print(f"🧠 Generating async reasoning-based recommendation ({reasoning_depth})...")
        
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
        
        response = await self.generate_json_async(
            prompt=prompt,
            json_schema=GiftRecommendationSchema.model_json_schema(),
            system_prompt="You are Dr. Elena Hartmann, a logical reasoning expert who uses systematic analysis to make perfect gift recommendations. Think step by step and show your reasoning.",
            temperature=0.5
        )
        
        if not response.success:
            # Create fallback recommendation
            return GiftRecommendationSchema(
                gift_name="Thoughtful Experience Gift",
                reasoning="Async systematic analysis suggests experiential gifts create lasting memories",
                match_score=0.7,
                emotional_appeal="meaningful connection",
                personalization_ideas=["Customize to interests", "Add personal note"],
                price_range=budget_range,
                alternative_gifts=["Personalized item", "Quality time activity"],
                confidence=0.6
            )
        
        try:
            recommendation = GiftRecommendationSchema.model_validate(response.parsed_json)
            print(f"✅ Async reasoning-based recommendation: {recommendation.gift_name}")
            return recommendation
        except Exception as e:
            # Fallback on validation error
            return GiftRecommendationSchema(
                gift_name="Personalized Memory Gift",
                reasoning="Async reasoning analysis failed, providing safe personal choice",
                match_score=0.6,
                emotional_appeal="personal connection",
                personalization_ideas=["Add personal touch"],
                price_range=budget_range,
                alternative_gifts=["Experience gift", "Quality item"],
                confidence=0.5
            )
    
    # === BATCH REASONING OPERATIONS ===
    
    async def batch_reasoning_analysis_async(self,
                                           analysis_requests: List[Dict]) -> List[Dict[str, Any]]:
        """
        Process multiple reasoning analyses concurrently
        
        VERWENDUNG:
        requests = [
            {"type": "personality_analysis", "data": profile1},
            {"type": "gift_comparison", "data": {"profile": profile2, "options": [...]}}
        ]
        results = await client.batch_reasoning_analysis_async(requests)
        """
        
        print(f"🧠 Processing {len(analysis_requests)} reasoning analyses concurrently...")
        
        # Create tasks based on analysis type
        tasks = []
        for request in analysis_requests:
            if request["type"] == "personality_analysis":
                task = self.analyze_gift_context_deeply_async(
                    personality_profile=request["data"],
                    occasion=request.get("occasion", "general"),
                    budget_range=request.get("budget_range", "flexible"),
                    relationship=request.get("relationship", "friend")
                )
            elif request["type"] == "gift_comparison":
                task = self.compare_gift_options_async(
                    personality_profile=request["data"]["profile"],
                    gift_options=request["data"]["options"],
                    evaluation_criteria=request["data"].get("criteria", {"fit": 1.0})
                )
            elif request["type"] == "constraint_optimization":
                task = self.multi_constraint_optimization_async(
                    personality_profile=request["data"]["profile"],
                    constraints=request["data"]["constraints"],
                    preferences=request["data"]["preferences"]
                )
            else:
                # Default to simple analysis
                task = self.analyze_gift_context_deeply_async(
                    personality_profile=request["data"],
                    occasion="general",
                    budget_range="flexible",
                    relationship="friend"
                )
            
            tasks.append(task)
        
        # Execute all analyses concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, dict):
                result["analysis_time"] = total_time / len(analysis_requests)
                result["request_index"] = i
                processed_results.append(result)
            else:
                processed_results.append({
                    "request_index": i,
                    "error": str(result) if isinstance(result, Exception) else "Analysis failed",
                    "analysis_time": total_time / len(analysis_requests)
                })
        
        print(f"✅ Batch reasoning analysis completed in {total_time:.2f}s")
        return processed_results
    
    # === UTILITY METHODS ===
    
    async def test_async_reasoning_capability(self, test_scenario: Dict) -> Dict[str, Any]:
        """
        Test async reasoning capabilities
        """
        
        print("🧠 Testing async reasoning capability...")
        
        prompt = f"""
        Test scenario for async reasoning capability:
        
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
            "reasoning_quality": "assessment of own reasoning",
            "processing_method": "async_gemini"
        }}
        """
        
        response = await self.generate_json_async(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are a reasoning expert. Show your systematic thinking process.",
            temperature=0.3
        )
        
        return response.parsed_json if response.success else {}
    
    async def close_async(self):
        """
        Properly close async resources
        """
        # Gemini doesn't need explicit closing
        print("🔒 AsyncGemini client closed")


# === FACTORY FUNCTIONS ===

def create_async_gemini_client(api_key: str, 
                             model_type: AIModelType = AIModelType.GEMINI_PRO) -> AsyncGeminiClient:
    """
    Factory function for AsyncGeminiClient
    
    VERWENDUNG:
    client = create_async_gemini_client(api_key, AIModelType.GEMINI_PRO)
    """
    return AsyncGeminiClient(api_key=api_key, model_type=model_type)


def create_reasoning_optimized_async_client(api_key: str) -> AsyncGeminiClient:
    """
    Creates async Gemini client optimized for complex reasoning
    """
    client = AsyncGeminiClient(
        api_key=api_key, 
        model_type=AIModelType.GEMINI_PRO,
        max_concurrent_requests=8  # Moderate concurrency for complex reasoning
    )
    
    # Optimize for reasoning
    client.generation_config.update({
        "temperature": 0.4,  # Lower for more logical consistency
        "top_p": 0.7,        # More focused responses
        "top_k": 30          # Reduce randomness
    })
    
    return client


# === TESTING UTILITIES ===

async def test_async_gemini_integration(api_key: str) -> Dict[str, Any]:
    """
    Comprehensive test of async Gemini integration
    """
    
    results = {
        "connection": False,
        "async_reasoning": False,
        "batch_analysis": False,
        "deep_context_analysis": False,
        "multi_constraint_optimization": False,
        "performance_metrics": None,
        "errors": []
    }
    
    try:
        async with create_async_gemini_client(api_key, AIModelType.GEMINI_PRO) as client:
            
            # Test 1: Connection
            text_response = await client.generate_text_async("Hello Async Gemini!", max_tokens=50)
            results["connection"] = text_response.success
            
            if results["connection"]:
                # Test 2: Async Reasoning
                test_scenario = {
                    "person": "analytical, introverted, loves puzzles",
                    "occasion": "birthday",
                    "budget": "75-125",
                    "challenge": "find intellectually stimulating gift"
                }
                
                reasoning_result = await client.test_async_reasoning_capability(test_scenario)
                results["async_reasoning"] = bool(reasoning_result)
                
                # Test 3: Deep Context Analysis
                if results["async_reasoning"]:
                    test_profile = {
                        "personality_scores": {"openness": 0.8, "conscientiousness": 0.7},
                        "hobbies": ["reading", "puzzles", "chess"],
                        "emotional_triggers": ["intellectual_challenge", "mastery"]
                    }
                    
                    try:
                        analysis = await client.analyze_gift_context_deeply_async(
                            personality_profile=test_profile,
                            occasion="birthday",
                            budget_range="75-125",
                            relationship="friend"
                        )
                        results["deep_context_analysis"] = bool(analysis)
                        results["sample_analysis"] = analysis
                    except Exception as e:
                        results["errors"].append(f"Deep context analysis failed: {e}")
                
                # Test 4: Multi-constraint Optimization
                if results["deep_context_analysis"]:
                    try:
                        optimization = await client.multi_constraint_optimization_async(
                            personality_profile=test_profile,
                            constraints={
                                "budget": "75-125",
                                "delivery_time": "1_week",
                                "complexity": "high"
                            },
                            preferences={
                                "intellectual_challenge": True,
                                "personalization": True
                            }
                        )
                        results["multi_constraint_optimization"] = bool(optimization)
                        results["sample_optimization"] = optimization
                    except Exception as e:
                        results["errors"].append(f"Multi-constraint optimization failed: {e}")
                
                # Test 5: Batch Analysis
                if results["multi_constraint_optimization"]:
                    try:
                        batch_requests = [
                            {"type": "personality_analysis", "data": test_profile, "occasion": "birthday"},
                            {"type": "personality_analysis", "data": test_profile, "occasion": "christmas"}
                        ]
                        
                        batch_start = time.time()
                        batch_results = await client.batch_reasoning_analysis_async(batch_requests)
                        batch_time = time.time() - batch_start
                        
                        results["batch_analysis"] = len(batch_results) == 2
                        results["batch_time"] = batch_time
                        
                    except Exception as e:
                        results["errors"].append(f"Batch analysis failed: {e}")
        
    except Exception as e:
        results["errors"].append(f"AsyncGemini client creation failed: {e}")
    
    return results


# === USAGE EXAMPLES ===

async def example_usage():
    """
    Examples of AsyncGeminiClient usage
    """
    
    api_key = "your-google-api-key"
    
    # Example 1: Deep reasoning analysis
    async with create_async_gemini_client(api_key) as client:
        
        profile = {
            "personality_scores": {"openness": 0.9, "conscientiousness": 0.8},
            "hobbies": ["philosophy", "chess", "classical_music"],
            "emotional_triggers": ["intellectual_stimulation", "mastery"]
        }
        
        # Deep async analysis
        analysis = await client.analyze_gift_context_deeply_async(
            personality_profile=profile,
            occasion="birthday",
            budget_range="100-300",
            relationship="close_friend"
        )
        
        print(f"🧠 Deep analysis insights: {analysis.get('personality_insights', {})}")
    
    # Example 2: Batch family analysis
    async with create_async_gemini_client(api_key) as client:
        
        family_profiles = [profile, profile, profile]  # 3 family members
        
        # All analyses in parallel
        family_analyses = await client.batch_personality_analysis_async(family_profiles)
        
        for i, analysis in enumerate(family_analyses):
            print(f"👨‍👩‍👧‍👦 Family member {i+1}: {analysis.get('gift_strategy_summary', 'Unknown')}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())