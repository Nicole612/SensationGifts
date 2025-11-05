"""
Advanced Prompt Engineering Techniques - State-of-the-Art AI Optimization
========================================================================

Cutting-edge prompt strategies für maximale AI-Performance in SensationGifts.
Implementiert modernste Techniken aus der AI-Forschung.

Advanced Features:
- Meta-Prompting (AI generates own prompts)
- Self-Correction and Validation
- Ensemble Prompting (multiple strategies)
- Adaptive Learning from feedback
- Retrieval-Augmented Prompting
- Constitutional AI principles
"""

from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from decimal import Decimal
from enum import Enum
import json

from ai_engine.schemas import (
    BasePromptTemplate,
    PromptTechnique,
    PromptComplexity,
    AIModelType,
    PromptOptimizationGoal,
    ContextInjection,
    PromptExample,
    PromptPerformanceMetrics
)


# =============================================================================
# ADVANCED TECHNIQUE TYPES
# =============================================================================

class AdvancedTechnique(str, Enum):
    """State-of-the-Art Prompt Engineering Techniken"""
    META_PROMPTING = "meta_prompting"              # AI generates own prompts
    SELF_CORRECTION = "self_correction"            # AI validates and improves
    ENSEMBLE_PROMPTING = "ensemble_prompting"      # Multiple approaches combined
    CONSTITUTIONAL_AI = "constitutional_ai"        # Principle-based reasoning
    RETRIEVAL_AUGMENTED = "retrieval_augmented"    # External knowledge integration
    ADAPTIVE_LEARNING = "adaptive_learning"        # Learning from feedback
    MULTI_STEP_REASONING = "multi_step_reasoning"  # Complex reasoning chains
    PROMPT_CHAINING = "prompt_chaining"            # Sequential prompt execution


class ValidationCriteria(str, Enum):
    """Kriterien für AI Self-Validation"""
    PERSONALITY_ACCURACY = "personality_accuracy"  # Passt zu Persönlichkeit?
    BUDGET_COMPLIANCE = "budget_compliance"        # Im Budget?
    RELATIONSHIP_APPROPRIATE = "relationship_appropriate" # Beziehungsangemessen?
    OCCASION_SUITABLE = "occasion_suitable"        # Passend zum Anlass?
    CULTURAL_SENSITIVE = "cultural_sensitive"      # Kulturell angemessen?
    CREATIVITY_LEVEL = "creativity_level"          # Kreativ genug?
    PRACTICALITY = "practicality"                  # Praktisch umsetzbar?


class EnsembleStrategy(str, Enum):
    """Strategien für Ensemble Prompting"""
    MAJORITY_VOTING = "majority_voting"            # Mehrheitsentscheidung
    WEIGHTED_CONSENSUS = "weighted_consensus"      # Gewichteter Konsens
    BEST_OF_N = "best_of_n"                       # Bestes von N Ergebnissen
    HYBRID_COMBINATION = "hybrid_combination"      # Kombiniere beste Teile
    CONFIDENCE_WEIGHTED = "confidence_weighted"    # Nach Konfidenz gewichtet


# =============================================================================
# META-PROMPTING: AI GENERATES OWN PROMPTS
# =============================================================================

class MetaPromptingEngine:
    """
    Meta-Prompting: AI analysiert Situation und generiert optimale Prompts
    
    Revolutionär: Statt vordefinierte Prompts → AI erstellt situative Prompts
    """
    
    @staticmethod
    def create_meta_prompt_template() -> BasePromptTemplate:
        """
        Template das AI beibringt, eigene Prompts zu erstellen
        """
        return BasePromptTemplate(
            template_name="meta_prompt_generator",
            template_version="3.0",
            description="AI generates optimal prompts for gift recommendation scenarios",
            
            technique=PromptTechnique.DYNAMIC_GENERATION,
            complexity=PromptComplexity.EXPERT,
            target_model=AIModelType.OPENAI_GPT4,  # Needs high reasoning capability
            optimization_goal=PromptOptimizationGoal.QUALITY,
            
            system_prompt="""
            You are a meta-prompt engineering expert. Your job is to analyze gift recommendation scenarios 
            and generate the most effective prompt for that specific situation.
            
            Your expertise includes:
            - Understanding what makes prompts effective for different AI models
            - Adapting prompt style to situation complexity and constraints  
            - Incorporating advanced prompting techniques (CoT, Few-Shot, etc.)
            - Optimizing for specific goals (speed vs quality vs creativity)
            
            You generate prompts that will produce the best possible gift recommendations.
            """,
            
            instruction_prompt="""
            SCENARIO ANALYSIS:
            Given this gift recommendation scenario, generate the optimal prompt:
            
            Situation: {scenario_description}
            Target AI Model: {target_model}
            Optimization Goal: {optimization_goal}
            Complexity Required: {complexity_level}
            Available Context: {available_context}
            
            META-PROMPT GENERATION PROCESS:
            
            1. ANALYZE SCENARIO REQUIREMENTS:
               - What type of reasoning is needed?
               - What information is most critical?
               - What are the main challenges/constraints?
               - What prompting technique would work best?
            
            2. DESIGN OPTIMAL PROMPT STRUCTURE:
               - Choose appropriate prompting technique
               - Determine necessary context integration
               - Plan reasoning flow and steps
               - Consider model-specific optimizations
            
            3. GENERATE CUSTOMIZED PROMPT:
               - Create system prompt that establishes expertise
               - Design instruction that guides reasoning process
               - Include relevant examples if using few-shot
               - Specify desired output format
               - Add validation criteria
            
            4. OPTIMIZE FOR TARGET MODEL:
               - Adjust complexity for model capabilities
               - Optimize token usage for speed/cost goals
               - Include model-specific best practices
            
            Generate the complete, ready-to-use prompt for this scenario.
            """,
            
            output_format_instructions="""
            Return the generated prompt in this structure:
            {
              "scenario_analysis": {
                "key_challenges": ["challenge1", "challenge2"],
                "reasoning_type_needed": "type of reasoning required",
                "optimal_technique": "best prompting technique for this scenario",
                "model_considerations": "model-specific factors"
              },
              "generated_prompt": {
                "system_prompt": "system prompt for AI role/expertise",
                "instruction_prompt": "main instruction with reasoning guidance", 
                "output_format": "specified output format",
                "validation_criteria": ["criteria1", "criteria2"]
              },
              "expected_performance": {
                "quality_prediction": "0.0-1.0 expected quality score",
                "speed_estimate": "estimated response time category", 
                "optimization_rationale": "why this prompt should work well"
              }
            }
            """,
            
            max_tokens=3500,
            temperature=0.7  # Creative prompt generation
        )
    
    def apply_self_correction(self, prompt: str, input_data: Dict[str, Any]) -> str:
        """
        Wendet Self-Correction auf einen Prompt an
        """
        try:
            # Erstelle Validierungs-Prompt
            validation_template = self.create_validation_prompt()
            
            # Erstelle Validierungs-Kontext
            validation_context = {
                "original_recommendation": prompt,
                "person_context": input_data.get("personality_data", "N/A"),
                "occasion_context": input_data.get("occasion", "N/A"),
                "relationship_context": input_data.get("relationship", "N/A"),
                "budget_context": input_data.get("budget_range", "N/A"),
                "cultural_context": input_data.get("cultural_context", "N/A")
            }
            
            # Validiere und verbessere
            corrected_prompt = f"""
{validation_template.system_prompt}

VALIDATION CONTEXT:
{validation_context}

ORIGINAL PROMPT:
{prompt}

VALIDATION INSTRUCTIONS:
{validation_template.instruction_prompt}

Please validate and improve the above prompt for better gift recommendations.
"""
            
            return corrected_prompt
            
        except Exception as e:
            # Fallback: Original prompt zurückgeben
            return prompt
    
    @staticmethod
    def generate_situational_prompt(
        scenario_description: str,
        target_model: AIModelType,
        optimization_goal: PromptOptimizationGoal,
        available_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generiert optimalen Prompt für spezifische Situation
        
        Uses meta-prompting to create custom prompts on-the-fly
        """
        
        # This would call the AI with the meta-prompt template
        # For now, return example structure
        return {
            "scenario_analysis": {
                "key_challenges": ["time_constraint", "budget_limitation"],
                "reasoning_type_needed": "practical_optimization",
                "optimal_technique": "chain_of_thought_with_constraints",
                "model_considerations": f"optimized_for_{target_model.value}"
            },
            "generated_prompt": {
                "system_prompt": "You are an expert gift consultant specializing in time-sensitive, budget-conscious recommendations...",
                "instruction_prompt": "Given the constraints, systematically analyze...",
                "output_format": "JSON with prioritized recommendations",
                "validation_criteria": ["budget_compliance", "time_feasibility"]
            },
            "expected_performance": {
                "quality_prediction": 0.87,
                "speed_estimate": "fast",
                "optimization_rationale": "Focused on practical constraints while maintaining quality"
            }
        }


# =============================================================================
# SELF-CORRECTION: AI VALIDATES AND IMPROVES ITS ANSWERS
# =============================================================================

class SelfCorrectionEngine:
    """
    Self-Correction: AI überprüft und verbessert ihre eigenen Empfehlungen
    
    Zwei-Phasen Prozess:
    1. Erste Empfehlung generieren
    2. Kritisch bewerten und verbessern
    """
    
    @staticmethod
    def create_validation_prompt() -> BasePromptTemplate:
        """
        Template für AI Self-Validation ihrer Geschenkempfehlungen
        """
        return BasePromptTemplate(
            template_name="self_validation_critic",
            template_version="2.4",
            description="AI validates and improves its own gift recommendations",
            
            technique=PromptTechnique.TEMPLATE_BASED,
            complexity=PromptComplexity.COMPLEX,
            target_model=AIModelType.ANTHROPIC_CLAUDE,  # Good at critical analysis
            optimization_goal=PromptOptimizationGoal.QUALITY,
            
            system_prompt="""
            You are a critical gift recommendation reviewer. Your job is to analyze and improve gift recommendations.
            
            Your expertise:
            - Spotting mismatches between personality and gift suggestions
            - Identifying budget, relationship, or occasion inappropriateness
            - Recognizing cultural insensitivity or practical issues
            - Suggesting specific improvements for better recommendations
            
            You are thorough but constructive in your criticism.
            """,
            
            instruction_prompt="""
            CRITICAL REVIEW PROCESS:
            
            Original Recommendation to Review:
            {original_recommendation}
            
            Context for Evaluation:
            Person: {person_context}
            Occasion: {occasion_context} 
            Relationship: {relationship_context}
            Budget: {budget_context}
            Cultural: {cultural_context}
            
            VALIDATION CRITERIA:
            
            1. PERSONALITY ACCURACY (Weight: 30%):
               - Do recommendations align with Big Five + Limbic personality profile?
               - Are trait-based preferences properly considered?
               - Is the personality reasoning sound?
               
            2. RELATIONSHIP APPROPRIATENESS (Weight: 25%):
               - Are gifts appropriate for this relationship type?
               - Is the intimacy level suitable?
               - Does it respect relationship boundaries?
               
            3. OCCASION SUITABILITY (Weight: 20%):
               - Do gifts match the occasion's tone and purpose?
               - Is timing and seasonal context considered?
               - Are traditions and expectations respected?
               
            4. BUDGET COMPLIANCE (Weight: 15%):
               - Are all recommendations within budget range?
               - Is value-for-money reasonable?
               - Are there good options across price points?
               
            5. PRACTICALITY & FEASIBILITY (Weight: 10%):
               - Can these gifts actually be obtained/delivered?
               - Are they practical for the recipient's lifestyle?
               - Is availability and delivery realistic?
            
            CRITICAL ANALYSIS:
            For each recommendation, identify:
            - Strengths that should be preserved
            - Weaknesses that need improvement  
            - Missing elements that should be added
            - Better alternatives that would work better
            
            IMPROVEMENT SUGGESTIONS:
            Provide specific, actionable improvements for each weak area.
            """,
            
            output_format_instructions="""
            Provide critical analysis in this format:
            {
              "overall_assessment": {
                "total_score": "0.0-1.0 overall quality score",
                "key_strengths": ["strength1", "strength2"],
                "major_weaknesses": ["weakness1", "weakness2"],
                "improvement_potential": "how much better this could be"
              },
              "detailed_validation": {
                "personality_accuracy": {
                  "score": "0.0-1.0",
                  "issues": ["issue1", "issue2"],
                  "improvements": ["improvement1", "improvement2"]
                },
                "relationship_appropriateness": {
                  "score": "0.0-1.0", 
                  "issues": ["issue1", "issue2"],
                  "improvements": ["improvement1", "improvement2"]
                },
                "occasion_suitability": {
                  "score": "0.0-1.0",
                  "issues": ["issue1", "issue2"], 
                  "improvements": ["improvement1", "improvement2"]
                },
                "budget_compliance": {
                  "score": "0.0-1.0",
                  "issues": ["issue1", "issue2"],
                  "improvements": ["improvement1", "improvement2"]
                },
                "practicality": {
                  "score": "0.0-1.0",
                  "issues": ["issue1", "issue2"],
                  "improvements": ["improvement1", "improvement2"]
                }
              },
              "improved_recommendations": [
                {
                  "original": "original recommendation",
                  "improved": "improved version",
                  "changes_made": "what was changed and why",
                  "expected_improvement": "how this is better"
                }
              ],
              "additional_suggestions": [
                "new recommendation 1 with reasoning",
                "new recommendation 2 with reasoning"
              ]
            }
            """,
            
            max_tokens=4000,
            temperature=0.4  # Analytical and critical
        )
    
    @staticmethod
    def create_improvement_prompt() -> BasePromptTemplate:
        """
        Template für AI um basierend auf Kritik zu verbessern
        """
        return BasePromptTemplate(
            template_name="self_improvement_generator",
            template_version="2.1",
            description="AI generates improved recommendations based on validation feedback",
            
            technique=PromptTechnique.TEMPLATE_BASED,
            complexity=PromptComplexity.COMPLEX,
            target_model=AIModelType.OPENAI_GPT4,
            optimization_goal=PromptOptimizationGoal.QUALITY,
            
            system_prompt="""
            You are a gift recommendation improvement specialist. You take validation feedback 
            and generate significantly better recommendations.
            
            Your expertise:
            - Incorporating critical feedback into better suggestions
            - Addressing specific weaknesses while preserving strengths
            - Creative problem-solving for challenging constraints
            - Generating alternatives that solve identified problems
            """,
            
            instruction_prompt="""
            IMPROVEMENT GENERATION:
            
            Validation Feedback:
            {validation_results}
            
            Original Context:
            {original_context}
            
            IMPROVEMENT STRATEGY:
            
            1. ADDRESS CRITICAL WEAKNESSES:
               - Fix personality mismatches identified
               - Correct relationship inappropriateness
               - Solve budget or practical issues
               - Improve occasion alignment
            
            2. PRESERVE IDENTIFIED STRENGTHS:
               - Keep elements that scored well
               - Maintain good reasoning approaches
               - Preserve creative or thoughtful elements
            
            3. ENHANCE OVERALL QUALITY:
               - Add missing elements identified in feedback
               - Increase personalization and thoughtfulness
               - Improve explanation and reasoning quality
               - Consider additional factors not originally addressed
            
            4. GENERATE CREATIVE ALTERNATIVES:
               - Think beyond original approach if needed
               - Consider unconventional but appropriate solutions
               - Provide backup options for different scenarios
            
            Focus on creating recommendations that would score 0.9+ on all validation criteria.
            """,
            
            output_format_instructions="""
            Provide improved recommendations:
            {
              "improvement_summary": {
                "changes_made": "overview of major improvements",
                "problems_solved": ["problem1", "problem2"],
                "enhanced_elements": ["enhancement1", "enhancement2"]
              },
              "improved_recommendations": [
                {
                  "title": "improved gift name",
                  "price": "€XX",
                  "category": "category",
                  "personality_match": "how it perfectly matches personality",
                  "relationship_appropriateness": "why it's perfect for this relationship",
                  "occasion_alignment": "how it fits the occasion perfectly",
                  "reasoning": "comprehensive explanation of why this is an excellent choice",
                  "confidence": "0.9+ expected confidence score",
                  "improvements_made": "specific improvements from original"
                }
              ],
              "alternative_approaches": [
                "creative alternative 1 with reasoning",
                "creative alternative 2 with reasoning"
              ],
              "quality_prediction": "expected validation score for improved recommendations"
            }
            """,
            
            max_tokens=3500,
            temperature=0.8  # Creative improvement
        )
    
    def create_ensemble_prompt(self, base_prompt: str, techniques: List[AdvancedTechnique]) -> str:
        """
        Erstellt Ensemble-Prompt mit mehreren Strategien
        """
        try:
            strategies = self.get_ensemble_strategies()
            
            # Wähle relevante Strategien basierend auf Techniken
            selected_strategies = []
            for technique in techniques:
                if technique == AdvancedTechnique.ENSEMBLE_PROMPTING:
                    selected_strategies.extend(strategies[:3])  # Top 3 Strategien
                elif technique == AdvancedTechnique.MULTI_STEP_REASONING:
                    selected_strategies.append(strategies[0])  # Personality-focused
            
            # Erstelle Ensemble-Prompt
            ensemble_prompt = f"""
{base_prompt}

ENSEMBLE STRATEGIES TO APPLY:
"""
            
            for strategy in selected_strategies:
                ensemble_prompt += f"""
- {strategy['name'].upper()}: {strategy['description']}
  Technique: {strategy['technique'].value}
  Weight: {strategy['weight']}
  Speciality: {strategy['speciality']}
"""
            
            ensemble_prompt += """
COMBINE ALL STRATEGIES:
Integrate insights from all strategies to create the most comprehensive and accurate gift recommendations.
"""
            
            return ensemble_prompt
            
        except Exception as e:
            # Fallback: Original prompt zurückgeben
            return base_prompt


# =============================================================================
# ENSEMBLE PROMPTING: MULTIPLE AI STRATEGIES COMBINED
# =============================================================================

class EnsemblePromptingEngine:
    """
    Ensemble Prompting: Kombiniert multiple AI-Strategien für beste Ergebnisse
    
    Verwendet verschiedene Prompting-Ansätze und kombiniert die besten Elemente
    """
    
    @staticmethod
    def get_ensemble_strategies() -> List[Dict[str, Any]]:
        """
        Definiert verschiedene Prompting-Strategien für Ensemble
        """
        return [
            {
                "name": "personality_focused",
                "description": "Deep Big Five + Limbic analysis with personality-first approach",
                "technique": PromptTechnique.CHAIN_OF_THOUGHT,
                "weight": 0.3,
                "speciality": "personality_accuracy"
            },
            {
                "name": "relationship_focused", 
                "description": "Relationship dynamics and appropriateness first",
                "technique": PromptTechnique.ROLE_PLAYING,
                "weight": 0.25,
                "speciality": "relationship_appropriateness"
            },
            {
                "name": "practical_focused",
                "description": "Budget, timing, and practical constraints first",
                "technique": PromptTechnique.TEMPLATE_BASED,
                "weight": 0.2,
                "speciality": "practicality"
            },
            {
                "name": "creative_focused",
                "description": "Creative and unique gift suggestions",
                "technique": PromptTechnique.FEW_SHOT,
                "weight": 0.15,
                "speciality": "creativity"
            },
            {
                "name": "cultural_focused",
                "description": "Cultural sensitivity and appropriateness",
                "technique": PromptTechnique.CONTEXT_INJECTION,
                "weight": 0.1,
                "speciality": "cultural_sensitivity"
            }
        ]
    
    @staticmethod
    def create_ensemble_coordinator_prompt() -> BasePromptTemplate:
        """
        Coordinator-Prompt der Ensemble-Ergebnisse kombiniert
        """
        return BasePromptTemplate(
            template_name="ensemble_coordinator",
            template_version="1.9",
            description="Coordinates and combines multiple AI recommendation strategies",
            
            technique=PromptTechnique.DYNAMIC_GENERATION,
            complexity=PromptComplexity.EXPERT,
            target_model=AIModelType.OPENAI_GPT4,
            optimization_goal=PromptOptimizationGoal.QUALITY,
            
            system_prompt="""
            You are an ensemble coordination expert who combines multiple AI recommendation approaches 
            to create optimal gift suggestions.
            
            Your expertise:
            - Identifying the best elements from different approaches
            - Resolving conflicts between different recommendation strategies
            - Weighing different factors appropriately for the situation
            - Creating coherent final recommendations from diverse inputs
            """,
            
            instruction_prompt="""
            ENSEMBLE COORDINATION TASK:
            
            Multiple AI strategies have generated gift recommendations:
            
            Strategy Results:
            {ensemble_results}
            
            Coordination Process:
            
            1. ANALYZE STRATEGY OUTPUTS:
               - What are the unique strengths of each approach?
               - Where do strategies agree vs. disagree?
               - Which strategies performed best for this scenario?
               - What insights can be combined?
            
            2. IDENTIFY BEST ELEMENTS:
               - Best personality insights
               - Most appropriate relationship considerations
               - Most practical solutions
               - Most creative suggestions
               - Best cultural considerations
            
            3. RESOLVE CONFLICTS:
               - When strategies disagree, determine best approach
               - Balance different priorities appropriately
               - Ensure final recommendations are coherent
               - Maintain consistency across suggestions
            
            4. SYNTHESIZE OPTIMAL RECOMMENDATIONS:
               - Combine best elements from each strategy
               - Ensure recommendations complement each other
               - Provide clear reasoning for choices made
               - Include confidence levels based on ensemble agreement
            
            Create final recommendations that leverage the collective intelligence of all strategies.
            """,
            
            output_format_instructions="""
            Provide ensemble-coordinated recommendations:
            {
              "ensemble_analysis": {
                "strategy_performance": {
                  "personality_focused": "performance assessment",
                  "relationship_focused": "performance assessment", 
                  "practical_focused": "performance assessment",
                  "creative_focused": "performance assessment",
                  "cultural_focused": "performance assessment"
                },
                "agreement_areas": ["where strategies agreed"],
                "conflict_resolutions": ["how conflicts were resolved"],
                "synthesis_rationale": "approach used to combine strategies"
              },
              "coordinated_recommendations": [
                {
                  "title": "synthesized gift recommendation",
                  "price": "€XX",
                  "category": "category",
                  "ensemble_confidence": "0.0-1.0 based on strategy agreement",
                  "contributing_strategies": ["which strategies contributed to this"],
                  "personality_reasoning": "personality insights used",
                  "relationship_reasoning": "relationship considerations",
                  "practical_reasoning": "practical factors",
                  "creative_elements": "creative aspects included",
                  "cultural_considerations": "cultural factors addressed",
                  "overall_reasoning": "comprehensive explanation"
                }
              ],
              "ensemble_insights": [
                "key insight 1 from combining approaches",
                "key insight 2 from ensemble analysis"
              ]
            }
            """,
            
            max_tokens=4500,
            temperature=0.6  # Balanced synthesis
        )
    
    def apply_constitutional_ai(self, prompt: str, input_data: Dict[str, Any]) -> str:
        """
        Wendet Constitutional AI Prinzipien auf einen Prompt an
        """
        try:
            principles = self.get_gift_recommendation_principles()
            
            # Erstelle Constitutional AI Prompt
            constitutional_prompt = f"""
{prompt}

CONSTITUTIONAL AI PRINCIPLES TO APPLY:
"""
            
            for principle in principles[:5]:  # Top 5 Prinzipien
                constitutional_prompt += f"""
- {principle['principle']}: {principle['description']}
  Application: {principle['application']}
"""
            
            constitutional_prompt += """
ETHICAL VALIDATION:
Ensure all recommendations follow these principles and are ethically sound, respectful, and responsible.
"""
            
            return constitutional_prompt
            
        except Exception as e:
            # Fallback: Original prompt zurückgeben
            return prompt


# =============================================================================
# CONSTITUTIONAL AI: PRINCIPLE-BASED REASONING
# =============================================================================

class ConstitutionalAIEngine:
    """
    Constitutional AI: Principle-basierte Empfehlungen mit ethischen Guidelines
    
    Stellt sicher dass Empfehlungen ethisch, respektvoll und verantwortlich sind
    """
    
    @staticmethod
    def get_gift_recommendation_principles() -> List[Dict[str, str]]:
        """
        Ethische Prinzipien für Geschenkempfehlungen
        """
        return [
            {
                "principle": "Respect Individual Autonomy",
                "description": "Gifts should respect the recipient's choices, values, and independence",
                "application": "Don't suggest gifts that impose lifestyle changes or values"
            },
            {
                "principle": "Cultural Sensitivity",
                "description": "Honor cultural backgrounds and avoid cultural appropriation",
                "application": "Research cultural significance and appropriateness of suggestions"
            },
            {
                "principle": "Relationship Boundaries",
                "description": "Respect appropriate intimacy levels for each relationship type",
                "application": "Ensure gifts don't cross relationship boundaries or create discomfort"
            },
            {
                "principle": "Financial Responsibility",
                "description": "Don't encourage overspending or financial strain",
                "application": "Stay within stated budgets and suggest reasonable financial expectations"
            },
            {
                "principle": "Environmental Consideration",
                "description": "Consider environmental impact when possible",
                "application": "Suggest sustainable options and experiences over material waste"
            },
            {
                "principle": "Inclusivity and Accessibility",
                "description": "Consider diverse abilities and circumstances",
                "application": "Ensure gifts are accessible and don't exclude based on abilities"
            },
            {
                "principle": "Avoiding Harm",
                "description": "Never suggest gifts that could cause physical, emotional, or social harm",
                "application": "Consider safety, allergies, phobias, and psychological impact"
            },
            {
                "principle": "Authenticity Over Materialism",
                "description": "Emphasize meaningful connections over expensive items",
                "application": "Value thoughtfulness and personal meaning over price tags"
            }
        ]
    
    @staticmethod
    def create_constitutional_prompt() -> BasePromptTemplate:
        """
        Template für principle-basierte Geschenkempfehlungen
        """
        return BasePromptTemplate(
            template_name="constitutional_gift_recommendations",
            template_version="2.7",
            description="Principle-based gift recommendations following ethical guidelines",
            
            technique=PromptTechnique.TEMPLATE_BASED,
            complexity=PromptComplexity.COMPLEX,
            target_model=AIModelType.ANTHROPIC_CLAUDE,  # Best for ethical reasoning
            optimization_goal=PromptOptimizationGoal.BALANCE,
            
            system_prompt="""
            You are an ethically-minded gift consultant who follows constitutional principles 
            for responsible recommendation-making.
            
            Your core principles:
            - Respect individual autonomy and values
            - Honor cultural backgrounds and sensitivities  
            - Maintain appropriate relationship boundaries
            - Encourage financial responsibility
            - Consider environmental and social impact
            - Promote inclusivity and accessibility
            - Prioritize authenticity over materialism
            - Never suggest anything potentially harmful
            
            Every recommendation must align with these ethical guidelines.
            """,
            
            instruction_prompt="""
            CONSTITUTIONAL GIFT RECOMMENDATION PROCESS:
            
            Request Context:
            {request_context}
            
            PRINCIPLE-BASED ANALYSIS:
            
            1. AUTONOMY RESPECT CHECK:
               - Does this honor the recipient's known values and choices?
               - Does it support their autonomy rather than imposing change?
               - Would they feel respected by these suggestions?
            
            2. CULTURAL SENSITIVITY REVIEW:
               - Are suggestions culturally appropriate and respectful?
               - Do they avoid appropriation while honoring heritage?
               - Are cultural factors properly considered?
            
            3. RELATIONSHIP BOUNDARY ASSESSMENT:
               - Are intimacy levels appropriate for this relationship?
               - Do suggestions respect professional/personal boundaries?
               - Would this feel comfortable for both parties?
            
            4. FINANCIAL RESPONSIBILITY EVALUATION:
               - Are suggestions within reasonable financial bounds?
               - Do they encourage responsible spending?
               - Are there good options across price ranges?
            
            5. ENVIRONMENTAL CONSIDERATION:
               - Can we suggest experiences over material items?
               - Are there sustainable or eco-friendly alternatives?
               - Does this promote mindful consumption?
            
            6. INCLUSIVITY CHECK:
               - Are suggestions accessible to people with different abilities?
               - Do they work for diverse lifestyles and circumstances?
               - Are we avoiding assumptions or exclusions?
            
            7. HARM PREVENTION:
               - Could any suggestion cause physical, emotional, or social harm?
               - Are safety considerations addressed?
               - Are potential negative consequences considered?
            
            8. AUTHENTICITY EMPHASIS:
               - Do suggestions prioritize meaning over cost?
               - Do they strengthen relationships rather than impress?
               - Is the focus on genuine thoughtfulness?
            
            Generate recommendations that exemplify these principles.
            """,
            
            output_format_instructions="""
            Provide constitutional recommendations:
            {
              "principle_compliance": {
                "autonomy_respect": "how recommendations respect individual autonomy",
                "cultural_sensitivity": "cultural considerations addressed", 
                "relationship_boundaries": "appropriate boundary maintenance",
                "financial_responsibility": "responsible financial approach",
                "environmental_consideration": "environmental factors considered",
                "inclusivity": "inclusivity and accessibility addressed",
                "harm_prevention": "potential harms avoided",
                "authenticity_focus": "emphasis on meaning over materialism"
              },
              "ethical_recommendations": [
                {
                  "title": "principled gift suggestion",
                  "price": "€XX",
                  "category": "category",
                  "constitutional_alignment": "which principles this exemplifies",
                  "ethical_reasoning": "why this is ethically sound",
                  "potential_concerns": "any ethical considerations to be aware of",
                  "alternative_if_concerns": "alternative if recipient has different values",
                  "relationship_impact": "how this strengthens the relationship ethically",
                  "broader_impact": "consideration of wider social/environmental impact"
                }
              ],
              "ethical_guidance": [
                "general guidance for ethical gift-giving in this context",
                "how to approach the gift-giving conversation respectfully"
              ]
            }
            """,
            
            max_tokens=4000,
            temperature=0.5  # Thoughtful and principled
        )


# =============================================================================
# ADAPTIVE LEARNING: PROMPTS IMPROVE BASED ON FEEDBACK
# =============================================================================

class AdaptiveLearningEngine:
    """
    Adaptive Learning: Prompts verbessern sich basierend auf User-Feedback
    
    Sammelt Performance-Daten und passt Prompts automatisch an
    """
    
    def __init__(self):
        self.performance_history: List[PromptPerformanceMetrics] = []
        self.adaptation_rules: Dict[str, Callable] = {}
        self.learning_rate: float = 0.1
    
    def record_performance(self, metrics: PromptPerformanceMetrics):
        """Zeichnet Performance-Metriken auf für Learning"""
        self.performance_history.append(metrics)
    
    def analyze_performance_patterns(self) -> Dict[str, Any]:
        """
        Analysiert Performance-Patterns für Verbesserungen
        """
        if not self.performance_history:
            return {"status": "insufficient_data"}
        
        # Analyze patterns in the data
        recent_metrics = self.performance_history[-20:]  # Last 20 entries
        
        avg_quality = sum(m.output_quality_score for m in recent_metrics if m.output_quality_score) / len(recent_metrics)
        avg_speed = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_user_satisfaction = sum(m.user_satisfaction for m in recent_metrics if m.user_satisfaction) / len([m for m in recent_metrics if m.user_satisfaction])
        
        # Identify improvement opportunities
        low_quality_scenarios = [m for m in recent_metrics if m.output_quality_score and m.output_quality_score < 0.7]
        slow_responses = [m for m in recent_metrics if m.response_time_ms > 5000]
        
        return {
            "performance_summary": {
                "avg_quality": avg_quality,
                "avg_speed_ms": avg_speed,
                "avg_user_satisfaction": avg_user_satisfaction,
                "total_samples": len(recent_metrics)
            },
            "improvement_opportunities": {
                "low_quality_count": len(low_quality_scenarios),
                "slow_response_count": len(slow_responses),
                "common_failure_patterns": self._identify_failure_patterns(low_quality_scenarios)
            },
            "adaptation_suggestions": self._generate_adaptation_suggestions(recent_metrics)
        }
    
    def _identify_failure_patterns(self, low_quality_scenarios: List[PromptPerformanceMetrics]) -> List[str]:
        """Identifiziert häufige Muster bei schlechter Performance"""
        patterns = []
        
        # Analyze test scenarios for patterns
        scenario_types = [m.test_scenario for m in low_quality_scenarios]
        
        # Simple pattern detection (could be more sophisticated)
        if scenario_types.count("emergency_gift") > len(scenario_types) * 0.3:
            patterns.append("struggles_with_time_constraints")
        
        if scenario_types.count("cultural_context") > len(scenario_types) * 0.3:
            patterns.append("needs_better_cultural_awareness")
        
        if scenario_types.count("budget_minimal") > len(scenario_types) * 0.3:
            patterns.append("difficulty_with_budget_constraints")
        
        return patterns
    
    def _generate_adaptation_suggestions(self, metrics: List[PromptPerformanceMetrics]) -> List[str]:
        """Generiert Vorschläge für Prompt-Verbesserungen"""
        suggestions = []
        
        # Analyze what works well vs poorly
        high_performers = [m for m in metrics if m.output_quality_score and m.output_quality_score > 0.85]
        low_performers = [m for m in metrics if m.output_quality_score and m.output_quality_score < 0.65]
        
        if len(high_performers) > len(low_performers):
            suggestions.append("identify_and_replicate_high_performance_patterns")
        
        if len(low_performers) > len(metrics) * 0.3:
            suggestions.append("review_and_improve_prompt_templates")
        
        return suggestions
    
    def apply_adaptive_learning(self, prompt: str, request_type: str) -> str:
        """
        Wendet Adaptive Learning auf einen Prompt an
        """
        try:
            # Erstelle Adaptive Learning Prompt
            adaptive_prompt = f"""
{prompt}

ADAPTIVE LEARNING OPTIMIZATION:
Based on previous performance data for {request_type} requests, optimize this prompt for:
- Better accuracy in gift recommendations
- Improved personalization scores
- Enhanced user satisfaction
- Faster response times
- Higher confidence scores

Apply learned best practices and avoid previously identified issues.
"""
            
            return adaptive_prompt
            
        except Exception as e:
            # Fallback: Original prompt zurückgeben
            return prompt
            suggestions.append("investigate_and_fix_common_failure_modes")
        
        # Speed vs Quality trade-offs
        avg_speed = sum(m.response_time_ms for m in metrics) / len(metrics)
        if avg_speed > 3000:
            suggestions.append("optimize_for_speed_without_sacrificing_quality")
        
        return suggestions
    
    @staticmethod
    def create_adaptive_prompt_template() -> BasePromptTemplate:
        """
        Template das sich basierend auf Feedback anpasst
        """
        return BasePromptTemplate(
            template_name="adaptive_learning_gift_recommendations",
            template_version="3.2",
            description="Self-improving prompt template that adapts based on performance feedback",
            
            technique=PromptTechnique.DYNAMIC_GENERATION,
            complexity=PromptComplexity.EXPERT,
            target_model=AIModelType.AUTO_SELECT,
            optimization_goal=PromptOptimizationGoal.BALANCE,
            
            system_prompt="""
            You are an adaptive gift recommendation AI that learns from experience.
            
            Your capabilities:
            - Learning from previous recommendation feedback
            - Adapting approach based on what works best
            - Improving reasoning based on success patterns
            - Self-correcting based on performance metrics
            
            Recent Performance Insights:
            {performance_insights}
            
            Adaptation Focus Areas:
            {adaptation_focus}
            """,
            
            instruction_prompt="""
            ADAPTIVE RECOMMENDATION PROCESS:
            
            Request: {gift_request}
            
            LEARNING-INFORMED APPROACH:
            
            1. APPLY LEARNED PATTERNS:
               - Use successful strategies from similar past scenarios
               - Avoid approaches that have performed poorly
               - Emphasize elements that users have rated highly
            
            2. ADAPT TO PERFORMANCE FEEDBACK:
               {adaptation_rules}
            
            3. CONTINUOUS IMPROVEMENT:
               - Be more specific in areas where past recommendations were vague
               - Improve reasoning in areas where logic was weak
               - Enhance personalization where generic responses failed
            
            4. SELF-MONITORING:
               - Evaluate your own reasoning as you go
               - Check against known successful patterns
               - Adjust approach if not meeting quality standards
            
            Generate recommendations using your learned experience for optimal results.
            """,
            
            output_format_instructions="""
            Provide adaptive recommendations with learning indicators:
            {
              "learning_application": {
                "successful_patterns_used": ["pattern1", "pattern2"],
                "avoided_failure_modes": ["failure1", "failure2"],
                "adaptations_made": ["adaptation1", "adaptation2"]
              },
              "recommendations": [
                {
                  "title": "learned and adapted gift recommendation",
                  "reasoning": "reasoning that incorporates learned patterns",
                  "confidence": "confidence based on similarity to successful past cases",
                  "learning_basis": "which past experiences inform this recommendation"
                }
              ],
              "self_assessment": {
                "expected_performance": "predicted quality score based on learning",
                "uncertainty_areas": "areas where more learning is needed",
                "improvement_indicators": "what would indicate this is working well"
              }
            }
            """,
            
            max_tokens=3500,
            temperature=0.7  # Balanced learning
        )


# =============================================================================
# ADVANCED TECHNIQUE ORCHESTRATOR
# =============================================================================

class AdvancedTechniqueOrchestrator:
    """
    Orchestrator der alle Advanced Techniques koordiniert
    
    Entscheidet welche Techniken für welche Situation optimal sind
    """
    
    def __init__(self):
        self.meta_engine = MetaPromptingEngine()
        self.self_correction = SelfCorrectionEngine()
        self.ensemble = EnsemblePromptingEngine()
        self.constitutional = ConstitutionalAIEngine()
        self.adaptive = AdaptiveLearningEngine()
    
    def select_optimal_techniques(
        self,
        scenario_complexity: PromptComplexity,
        optimization_goal: PromptOptimizationGoal,
        available_resources: Dict[str, Any]
    ) -> List[AdvancedTechnique]:
        """
        Wählt optimale Advanced Techniques für Szenario
        """
        techniques = []
        
        # High complexity always benefits from ensemble
        if scenario_complexity == PromptComplexity.EXPERT:
            techniques.append(AdvancedTechnique.ENSEMBLE_PROMPTING)
        
        # Quality focus benefits from self-correction
        if optimization_goal == PromptOptimizationGoal.QUALITY:
            techniques.append(AdvancedTechnique.SELF_CORRECTION)
        
        # Cultural or ethical contexts need constitutional AI
        if available_resources.get("cultural_context") or available_resources.get("ethical_concerns"):
            techniques.append(AdvancedTechnique.CONSTITUTIONAL_AI)
        
        # Sufficient historical data enables adaptive learning
        if available_resources.get("performance_history") and len(available_resources["performance_history"]) > 10:
            techniques.append(AdvancedTechnique.ADAPTIVE_LEARNING)
        
        # Complex novel scenarios benefit from meta-prompting
        if scenario_complexity == PromptComplexity.EXPERT and not available_resources.get("similar_scenarios"):
            techniques.append(AdvancedTechnique.META_PROMPTING)
        
        return techniques
    
    def orchestrate_advanced_recommendation(
        self,
        request_context: Dict[str, Any],
        selected_techniques: List[AdvancedTechnique],
        target_model: AIModelType
    ) -> Dict[str, Any]:
        """
        Orchestriert Advanced Techniques für optimale Empfehlung
        """
        
        results = {
            "techniques_used": selected_techniques,
            "orchestration_strategy": "sequential_with_integration",
            "final_recommendations": [],
            "technique_contributions": {}
        }
        
        # Execute selected techniques in optimal order
        intermediate_results = {}
        
        # 1. Meta-prompting first if selected (generates custom prompts)
        if AdvancedTechnique.META_PROMPTING in selected_techniques:
            intermediate_results["meta_prompting"] = self._execute_meta_prompting(request_context, target_model)
        
        # 2. Constitutional AI for ethical foundation
        if AdvancedTechnique.CONSTITUTIONAL_AI in selected_techniques:
            intermediate_results["constitutional"] = self._execute_constitutional_ai(request_context)
        
        # 3. Ensemble prompting for multiple perspectives
        if AdvancedTechnique.ENSEMBLE_PROMPTING in selected_techniques:
            intermediate_results["ensemble"] = self._execute_ensemble_prompting(request_context)
        
        # 4. Self-correction for quality improvement
        if AdvancedTechnique.SELF_CORRECTION in selected_techniques:
            intermediate_results["self_correction"] = self._execute_self_correction(intermediate_results)
        
        # 5. Adaptive learning for continuous improvement
        if AdvancedTechnique.ADAPTIVE_LEARNING in selected_techniques:
            intermediate_results["adaptive"] = self._execute_adaptive_learning(request_context, intermediate_results)
        
        # Integrate all results
        results["final_recommendations"] = self._integrate_technique_results(intermediate_results)
        results["technique_contributions"] = intermediate_results
        
        return results
    
    def _execute_meta_prompting(self, context: Dict[str, Any], target_model: AIModelType) -> Dict[str, Any]:
        """Executes meta-prompting technique"""
        return self.meta_engine.generate_situational_prompt(
            scenario_description=context.get("scenario", ""),
            target_model=target_model,
            optimization_goal=PromptOptimizationGoal.QUALITY,
            available_context=context
        )
    
    def _execute_constitutional_ai(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes constitutional AI technique"""
        # Would execute constitutional prompt with context
        return {"constitutional_recommendations": [], "ethical_guidelines_applied": []}
    
    def _execute_ensemble_prompting(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes ensemble prompting technique"""
        # Would execute multiple strategies and coordinate results
        return {"ensemble_results": [], "strategy_performance": {}}
    
    def _execute_self_correction(self, intermediate_results: Dict[str, Any]) -> Dict[str, Any]:
        """Executes self-correction technique"""
        # Would validate and improve previous results
        return {"validated_recommendations": [], "improvements_made": []}
    
    def _execute_adaptive_learning(self, context: Dict[str, Any], intermediate_results: Dict[str, Any]) -> Dict[str, Any]:
        """Executes adaptive learning technique"""
        # Would apply learned patterns and adaptations
        return {"adapted_recommendations": [], "learning_applied": []}
    
    def _integrate_technique_results(self, intermediate_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Integrates results from all techniques into final recommendations"""
        # Would intelligently combine all technique outputs
        return [{"integrated_recommendation": "placeholder"}]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'AdvancedTechnique',
    'ValidationCriteria',
    'EnsembleStrategy',
    
    # Engines
    'MetaPromptingEngine',
    'SelfCorrectionEngine', 
    'EnsemblePromptingEngine',
    'ConstitutionalAIEngine',
    'AdaptiveLearningEngine',
    
    # Orchestrator
    'AdvancedTechniqueOrchestrator'
]