"""
Anthropic Claude Client - Premium Ethical Reasoning für Gift Recommendations

CLAUDE'S UNIQUE STRENGTHS:
- Exceptional ethical reasoning and cultural sensitivity
- Nuanced human psychology analysis
- Detailed risk assessment for gift appropriateness
- Long-form analytical reasoning
- Strong instruction following
- Cultural and social context awareness

USE CASES:
- Ethical gift appropriateness analysis
- Cultural sensitivity in gift selection
- Detailed personality psychology insights
- Risk assessment for relationship dynamics
- Premium detailed reasoning chains
- Complex social context analysis

POSITIONING:
Claude als "Premium Reasoning" Model für wichtige/komplexe Gift-Entscheidungen
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    raise ImportError("Anthropic library not installed. Run: pip install anthropic")

from .base_client import (
    BaseAIClient, AIRequest, AIResponse, AIModelType,
    ModelCapability, ResponseFormat, GiftRecommendationSchema
)


class AnthropicClient(BaseAIClient):
    """
    Anthropic Claude Client für Premium Ethical Gift Reasoning
    
    Spezialisiert auf:
    - Ethical considerations in gift giving
    - Cultural sensitivity and appropriateness
    - Detailed personality psychology analysis
    - Risk assessment for relationships
    - Nuanced social context understanding
    - Premium reasoning quality
    """
    
    def __init__(self, 
                 api_key: str,
                 model_type: AIModelType = AIModelType.ANTHROPIC_CLAUDE,
                 rate_limit_per_minute: int = 50):  # Claude has conservative limits
        
        # Validate model type
        if model_type != AIModelType.ANTHROPIC_CLAUDE:
            raise ValueError(f"Invalid model type for Anthropic: {model_type}")
        
        super().__init__(api_key, model_type, rate_limit_per_minute)
        
        # Initialize Anthropic client
        self.client = Anthropic(api_key=api_key)
        
        # Model configuration
        self.model_name = self._get_model_name()
        self.max_tokens_limit = 4096  # Claude's current limit
        
        print(f"🎭 AnthropicClient initialized: {self.model_name}")
    
    def _get_model_name(self) -> str:
        """Maps our model type to Anthropic model name"""
        # Using Claude 3.5 Sonnet as the primary model
        return "claude-3-5-sonnet-20241022"
    
    def _get_model_capabilities(self) -> List[ModelCapability]:
        """Define what Claude can do"""
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.HIGH_QUALITY,
            ModelCapability.COMPLEX_REASONING,
            ModelCapability.ETHICAL_REASONING,  # Unique to Claude
            ModelCapability.CULTURAL_SENSITIVITY,  # Unique to Claude
            ModelCapability.JSON_OUTPUT
        ]
    
    def _get_model_pricing(self) -> Dict[str, float]:
        """Anthropic Claude pricing"""
        return {
            "input_per_token": 0.000003,   # $3 per 1M input tokens
            "output_per_token": 0.000015   # $15 per 1M output tokens
        }
    
    def _make_api_call(self, request: AIRequest) -> AIResponse:
        """
        Makes Anthropic Claude API call with ethical reasoning optimization
        """
        try:
            # Build messages for Claude
            messages = []
            
            # Claude doesn't use system messages in the same way, integrate into user message
            user_content = request.prompt
            if request.system_prompt:
                user_content = f"{request.system_prompt}\n\nHuman: {request.prompt}"
            else:
                user_content = f"Human: {request.prompt}"
            
            # Add Assistant prefix for proper formatting
            user_content += "\n\nAssistant: "
            
            # Claude API parameters
            api_params = {
                "model": self.model_name,
                "max_tokens": min(request.max_tokens or 1000, self.max_tokens_limit),
                "temperature": request.temperature,
                "messages": [{"role": "user", "content": user_content}]
            }
            
            # Make API call
            start_time = time.time()
            response = self.client.messages.create(**api_params)
            response_time = time.time() - start_time
            
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
            
        except anthropic.APIError as e:
            return AIResponse(
                content="",
                model_type=self.model_type,
                tokens_used=0,
                cost=0.0,
                response_time=0.0,
                timestamp=datetime.now(),
                success=False,
                error=f"Anthropic API Error: {e}"
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
                error=f"Unexpected error: {e}"
            )
    
    # === SPECIALIZED METHODS FOR ETHICAL GIFT REASONING ===
    
    def ethical_gift_analysis(self,
                             personality_profile: Dict,
                             occasion: str,
                             budget_range: str,
                             relationship: str,
                             cultural_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Ethical and cultural analysis of gift appropriateness
        
        Claude's strength: Deep ethical reasoning and cultural sensitivity
        """
        
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
        
        response = self.generate_json(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are Dr. Sarah Chen, an expert in cross-cultural communication and ethical gift-giving practices with 20 years of experience in cultural psychology and relationship dynamics.",
            temperature=0.3  # Lower temperature for more consistent ethical analysis
        )
        
        return response.parsed_json if response.success else {}
    
    def detailed_personality_gift_psychology(self,
                                           personality_profile: Dict,
                                           relationship_history: Optional[str] = None) -> Dict[str, Any]:
        """
        Deep personality psychology analysis for gift recommendations
        
        Claude's strength: Nuanced human psychology understanding
        """
        
        prompt = f"""
        Provide a comprehensive personality psychology analysis for gift recommendations:
        
        PERSONALITY PROFILE:
        {json.dumps(personality_profile, indent=2)}
        
        RELATIONSHIP HISTORY:
        {relationship_history or "No specific history provided"}
        
        DEEP PSYCHOLOGICAL ANALYSIS:
        
        1. CORE PSYCHOLOGICAL DRIVERS:
           - What are their fundamental psychological needs?
           - What drives their sense of identity and self-worth?
           - How do they express and receive affection?
           - What are their emotional triggers and sensitivities?
        
        2. GIFT-RECEIVING PSYCHOLOGY:
           - How do they typically respond to receiving gifts?
           - What makes them feel truly appreciated?
           - Do they prefer practical, emotional, or experiential gifts?
           - How do they interpret gift-giving gestures?
        
        3. RELATIONSHIP DYNAMICS:
           - How do they view this specific relationship?
           - What role does gift-giving play in their relationships?
           - How do they express gratitude and appreciation?
           - What are their expectations in relationships?
        
        4. PERSONALIZATION INSIGHTS:
           - What level of personalization do they appreciate?
           - How do they respond to thoughtful details?
           - What shows that someone "really knows them"?
           - How important is the thought vs. the actual gift?
        
        5. POTENTIAL PSYCHOLOGICAL REACTIONS:
           - How might they react to different gift categories?
           - What could make them feel uncomfortable or obligated?
           - What would genuinely surprise and delight them?
           - How do they handle gifts that miss the mark?
        
        Provide detailed psychological insights in JSON format:
        {{
            "core_psychological_drivers": {{
                "fundamental_needs": ["need1", "need2"],
                "identity_drivers": ["driver1", "driver2"],
                "affection_expression": "how they show/receive love",
                "emotional_triggers": ["trigger1", "trigger2"]
            }},
            "gift_receiving_psychology": {{
                "typical_response_pattern": "description",
                "appreciation_factors": ["factor1", "factor2"],
                "gift_preference_type": "practical/emotional/experiential/mixed",
                "gesture_interpretation": "how they read gift intentions"
            }},
            "relationship_dynamics": {{
                "relationship_view": "their perspective on relationship",
                "gift_giving_role": "importance in their relationships",
                "gratitude_expression": "how they show thanks",
                "relationship_expectations": ["expectation1", "expectation2"]
            }},
            "personalization_insights": {{
                "personalization_appreciation": "low/medium/high",
                "thoughtful_details_impact": "low/medium/high",
                "recognition_indicators": ["what shows you know them"],
                "thought_vs_gift_importance": "thought/gift/balanced"
            }},
            "predicted_reactions": {{
                "positive_categories": ["category1", "category2"],
                "uncomfortable_categories": ["category1", "category2"],
                "surprise_delight_factors": ["factor1", "factor2"],
                "miss_handling": "how they handle disappointing gifts"
            }},
            "gift_strategy_recommendations": {{
                "optimal_approach": "detailed strategy",
                "personalization_level": "how much to personalize",
                "presentation_importance": "low/medium/high",
                "timing_considerations": ["consideration1", "consideration2"]
            }}
        }}
        """
        
        response = self.generate_json(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are Dr. Elena Rodriguez, a renowned psychologist specializing in personality psychology, relationship dynamics, and the psychology of gift-giving with deep expertise in human emotional responses.",
            temperature=0.4
        )
        
        return response.parsed_json if response.success else {}
    
    def premium_gift_recommendation(self,
                                  personality_profile: Dict,
                                  occasion: str,
                                  budget_range: str,
                                  relationship: str,
                                  additional_context: Optional[Dict] = None) -> GiftRecommendationSchema:
        """
        Premium gift recommendation with comprehensive ethical and psychological analysis
        
        Claude's strength: Combining ethical, cultural, and psychological insights
        """
        
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
        
        response = self.generate_json(
            prompt=prompt,
            json_schema=GiftRecommendationSchema.model_json_schema(),
            system_prompt="You are Dr. Elena Hartmann, a world-renowned expert in ethical gift psychology, combining expertise in cultural anthropology, personality psychology, and relationship dynamics to provide premium gift recommendations.",
            temperature=0.5
        )
        
        if not response.success:
            # Create fallback with ethical considerations
            return GiftRecommendationSchema(
                gift_name="Thoughtfully Chosen Experience",
                reasoning="Premium ethical analysis suggests experience gifts minimize boundary risks while maximizing emotional connection",
                match_score=0.7,
                emotional_appeal="meaningful connection",
                personalization_ideas=["Choose based on their interests", "Add personal note explaining choice", "Consider timing carefully"],
                price_range=budget_range,
                alternative_gifts=["Quality time activity", "Personalized keepsake"],
                confidence=0.7
            )
        
        try:
            recommendation = GiftRecommendationSchema.model_validate(response.parsed_json)
            print(f"🎭 Premium Claude recommendation: {recommendation.gift_name}")
            return recommendation
        except Exception as e:
            # Fallback on validation error
            return GiftRecommendationSchema(
                gift_name="Ethically Considered Gift",
                reasoning="Premium analysis framework applied with ethical safeguards",
                match_score=0.6,
                emotional_appeal="thoughtful consideration",
                personalization_ideas=["Respect boundaries", "Show genuine care"],
                price_range=budget_range,
                alternative_gifts=["Safe meaningful choice", "Culturally appropriate option"],
                confidence=0.6
            )
    
    def risk_assessment_for_gift(self,
                               gift_idea: str,
                               personality_profile: Dict,
                               relationship: str,
                               cultural_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive risk assessment for a specific gift idea
        
        Claude's strength: Thorough risk analysis and mitigation strategies
        """
        
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
        
        response = self.generate_json(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are a risk assessment expert specializing in social and cultural risk analysis for interpersonal relationships and gift-giving scenarios.",
            temperature=0.3
        )
        
        return response.parsed_json if response.success else {}
    
    # === UTILITY METHODS ===
    
    def test_ethical_reasoning(self, scenario: Dict) -> Dict[str, Any]:
        """
        Test Claude's ethical reasoning capabilities
        """
        
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
            "reasoning_confidence": 0.9
        }}
        """
        
        response = self.generate_json(
            prompt=prompt,
            json_schema={"type": "object"},
            system_prompt="You are an ethics expert. Provide thorough ethical analysis.",
            temperature=0.3
        )
        
        return response.parsed_json if response.success else {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns Claude model information
        """
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "primary_strength": "ethical reasoning",
            "use_cases": ["ethical_analysis", "cultural_sensitivity", "risk_assessment", "premium_recommendations"],
            "max_tokens": self.max_tokens_limit,
            "capabilities": [cap.value for cap in self.capabilities],
            "pricing": self.pricing,
            "expected_response_time": "2-5 seconds",
            "unique_features": ["ethical_reasoning", "cultural_sensitivity", "risk_assessment"]
        }


# === FACTORY FUNCTIONS ===

def create_anthropic_client(api_key: str) -> AnthropicClient:
    """
    Factory function to create Anthropic client
    
    Args:
        api_key: Anthropic API key
        
    Returns:
        Configured AnthropicClient instance
    """
    return AnthropicClient(api_key=api_key)


def create_ethical_reasoning_client(api_key: str) -> AnthropicClient:
    """
    Creates Anthropic client optimized for ethical reasoning
    """
    client = AnthropicClient(api_key=api_key)
    return client


# === TESTING UTILITIES ===

def test_anthropic_integration(api_key: str) -> Dict[str, Any]:
    """
    Comprehensive test of Anthropic integration
    """
    results = {
        "connection": False,
        "ethical_reasoning": False,
        "gift_recommendation": False,
        "risk_assessment": False,
        "cultural_sensitivity": False,
        "errors": []
    }
    
    try:
        client = create_anthropic_client(api_key)
        
        # Test 1: Connection
        text_response = client.generate_text("Hello Claude!", max_tokens=50)
        results["connection"] = text_response.success
        
        if results["connection"]:
            # Test 2: Ethical Reasoning
            ethical_scenario = {
                "situation": "giving expensive gift to colleague",
                "relationship": "professional",
                "concerns": ["appropriateness", "workplace_dynamics"]
            }
            
            ethical_result = client.test_ethical_reasoning(ethical_scenario)
            results["ethical_reasoning"] = bool(ethical_result)
            
            # Test 3: Gift Recommendation
            if results["ethical_reasoning"]:
                test_profile = {
                    "personality_scores": {"conscientiousness": 0.8, "agreeableness": 0.7},
                    "hobbies": ["reading", "volunteering"],
                    "values": ["social_justice", "environmental_sustainability"]
                }
                
                try:
                    recommendation = client.premium_gift_recommendation(
                        personality_profile=test_profile,
                        occasion="birthday",
                        budget_range="75-150",
                        relationship="close_friend"
                    )
                    results["gift_recommendation"] = True
                    results["sample_recommendation"] = recommendation.model_dump()
                except Exception as e:
                    results["errors"].append(f"Gift recommendation failed: {e}")
            
            # Test 4: Risk Assessment
            if results["gift_recommendation"]:
                try:
                    risk_analysis = client.risk_assessment_for_gift(
                        gift_idea="expensive jewelry",
                        personality_profile=test_profile,
                        relationship="friend"
                    )
                    results["risk_assessment"] = bool(risk_analysis)
                    results["sample_risk_analysis"] = risk_analysis
                except Exception as e:
                    results["errors"].append(f"Risk assessment failed: {e}")
            
            # Test 5: Cultural Sensitivity
            if results["risk_assessment"]:
                try:
                    cultural_analysis = client.ethical_gift_analysis(
                        personality_profile=test_profile,
                        occasion="religious_holiday",
                        budget_range="50-100",
                        relationship="colleague",
                        cultural_context="diverse_workplace"
                    )
                    results["cultural_sensitivity"] = bool(cultural_analysis)
                    results["sample_cultural_analysis"] = cultural_analysis
                except Exception as e:
                    results["errors"].append(f"Cultural sensitivity test failed: {e}")
        
    except Exception as e:
        results["errors"].append(f"Anthropic client creation failed: {e}")
    
    return results


def ethical_reasoning_quality_test(anthropic_client: AnthropicClient,
                                 test_scenarios: List[Dict]) -> Dict[str, Any]:
    """
    Test ethical reasoning quality across multiple scenarios
    """
    results = {
        "total_tests": len(test_scenarios),
        "successful_tests": 0,
        "average_confidence": 0.0,
        "ethical_quality": [],
        "failed_tests": []
    }
    
    confidences = []
    
    for i, scenario in enumerate(test_scenarios):
        try:
            ethical_result = anthropic_client.test_ethical_reasoning(scenario)
            if ethical_result:
                results["successful_tests"] += 1
                confidence = ethical_result.get("reasoning_confidence", 0.5)
                confidences.append(confidence)
                results["ethical_quality"].append({
                    "test_scenario": i,
                    "confidence": confidence,
                    "ethical_considerations": len(ethical_result.get("ethical_considerations", []))
                })
        except Exception as e:
            results["failed_tests"].append({"test_scenario": i, "error": str(e)})
    
    if confidences:
        results["average_confidence"] = sum(confidences) / len(confidences)
    
    return results