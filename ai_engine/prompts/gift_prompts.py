"""
Gift Recommendation Prompts - Advanced Few-Shot Learning
=======================================================

Intelligente Prompt-Templates für perfekte Geschenkempfehlungen.
Verwendet Few-Shot Learning um AI aus Beispielen lernen zu lassen.

Features:
- Model-spezifische Optimierungen (OpenAI, Groq, Claude)
- Big Five + Limbic-Persönlichkeit → Geschenk Mapping
- Dynamische Kontext-Integration
- Performance-optimierte Prompts
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from decimal import Decimal

from ai_engine.schemas import (
    FewShotPromptTemplate, 
    PromptExample,
    ContextInjection,
    PromptTechnique,
    PromptComplexity,
    AIModelType,
    PromptOptimizationGoal,
    GiftOccasion,
    AgeGroup
)

from ..schemas.relationship_types import RelationshipType, RelationshipAnalyzer


# =============================================================================
# FEW-SHOT EXAMPLES: High-Quality Training Data für AI
# =============================================================================

class GiftExamples:
    """
    Kuratierte Few-Shot Beispiele für verschiedene Szenarien
    
    Jedes Beispiel zeigt der AI: "So sollst du denken und antworten"
    """
    
    @staticmethod
    def get_personality_based_examples() -> List[PromptExample]:
        """
        Beispiele die zeigen wie Persönlichkeit → Geschenk funktioniert
        
        Basiert auf Big Five + Limbic-Dimensionen
        """
        return [
            PromptExample(
                input_example="""
                Person: Lisa, 28, Close Friend
                Occasion: Birthday
                Personality: High Openness (4.5/5), High Extraversion (4.2/5), Medium Conscientiousness (3.1/5)
                Budget: €30-60
                Preferences: Art, travel, spontaneous activities
                """,
                expected_output="""
                {
                  "recommendations": [
                    {
                      "title": "Watercolor Art Workshop Experience",
                      "price": "€45",
                      "category": "experiences",
                      "personality_match": "Perfect for high openness - creative exploration",
                      "confidence": 0.92,
                      "reasoning": "High openness loves new artistic experiences, extraversion enjoys social workshops"
                    },
                    {
                      "title": "Spontaneous City Adventure Voucher",
                      "price": "€35", 
                      "category": "experiences",
                      "personality_match": "Medium conscientiousness + high extraversion = loves unplanned social fun",
                      "confidence": 0.87,
                      "reasoning": "Spontaneous nature + social energy = perfect adventure gift"
                    },
                    {
                      "title": "Travel-Inspired Art Supplies Kit",
                      "price": "€52",
                      "category": "art_crafts",
                      "personality_match": "Combines travel passion with creative openness",
                      "confidence": 0.84,
                      "reasoning": "Bridges her love for travel and art, portable for her spontaneous lifestyle"
                    }
                  ]
                }
                """,
                explanation="Zeigt AI wie Persönlichkeits-Dimensionen zu spezifischen Geschenken führen",
                difficulty_level=PromptComplexity.MODERATE,
                tags=["personality_mapping", "openness", "extraversion", "experiences"]
            ),
            
            PromptExample(
                input_example="""
                Person: Marcus, 35, Romantic Partner  
                Occasion: Anniversary
                Personality: High Conscientiousness (4.3/5), Low Extraversion (2.1/5), High Honesty-Humility (4.1/5)
                Budget: €80-150
                Preferences: Reading, quality time, minimalist lifestyle
                """,
                expected_output="""
                {
                  "recommendations": [
                    {
                      "title": "Premium Leather-Bound Journal Set",
                      "price": "€95",
                      "category": "books", 
                      "personality_match": "High conscientiousness appreciates quality, organization tools",
                      "confidence": 0.89,
                      "reasoning": "Organized personality + reading love = perfect journaling system"
                    },
                    {
                      "title": "Private In-Home Couples Cooking Class",
                      "price": "€120",
                      "category": "experiences",
                      "personality_match": "Low extraversion prefers intimate settings, conscientiousness likes learning",
                      "confidence": 0.91,
                      "reasoning": "Intimate setting respects introversion, structured learning appeals to conscientiousness"
                    },
                    {
                      "title": "Minimalist Reading Nook Upgrade Kit",
                      "price": "€85",
                      "category": "home_decor",
                      "personality_match": "Honest-humble + minimalist = appreciates simple, functional beauty",
                      "confidence": 0.86,
                      "reasoning": "Enhances his reading passion while respecting minimalist values"
                    }
                  ]
                }
                """,
                explanation="Zeigt wie Introversion + Conscientiousness zu intimen, strukturierten Geschenken führt",
                difficulty_level=PromptComplexity.MODERATE,
                tags=["conscientiousness", "introversion", "honesty_humility", "romantic"]
            ),
            
            PromptExample(
                input_example="""
                Person: Emma, 22, Family Sister
                Occasion: Graduation
                Personality: High Emotionality (4.4/5), High Agreeableness (4.0/5), Medium Openness (3.2/5)
                Budget: €40-80
                Preferences: Sentimental items, photos, memories with family
                """,
                expected_output="""
                {
                  "recommendations": [
                    {
                      "title": "Custom Family Memory Photo Book",
                      "price": "€55",
                      "category": "personalized",
                      "personality_match": "High emotionality treasures sentimental, memory-based gifts",
                      "confidence": 0.94,
                      "reasoning": "Emotional personality + family bond + graduation milestone = perfect memory gift"
                    },
                    {
                      "title": "Personalized Graduation Charm Bracelet",
                      "price": "€68",
                      "category": "jewelry",
                      "personality_match": "Agreeable personality appreciates thoughtful, personal touches",
                      "confidence": 0.88,
                      "reasoning": "Sentimental jewelry marks achievement while honoring family connection"
                    },
                    {
                      "title": "Sister Bond Memory Jar Kit",
                      "price": "€42",
                      "category": "personalized",
                      "personality_match": "Emotionality + agreeableness = loves relationship-focused, heartfelt gifts",
                      "confidence": 0.85,
                      "reasoning": "Creates ongoing emotional connection through shared memories"
                    }
                  ]
                }
                """,
                explanation="Zeigt wie Emotionality + Agreeableness zu sentimentalen, beziehungsorientierten Geschenken führt",
                difficulty_level=PromptComplexity.MODERATE,
                tags=["emotionality", "agreeableness", "sentimental", "family"]
            )
        ]
    
    @staticmethod 
    def get_quick_recommendation_examples() -> List[PromptExample]:
        """
        Beispiele für schnelle Empfehlungen (optimiert für Groq)
        
        Fokus auf Speed über Details
        """
        return [
            PromptExample(
                input_example="""
                Quick Request: Birthday, Close Friend, 25-30 age, €30-50 budget, likes fitness and tech
                """,
                expected_output="""
                1. Fitness Smartwatch Band (€35) - Tech + fitness combo
                2. Workout Playlist Speaker (€42) - Music + motivation  
                3. Fitness App Premium Subscription (€48) - Digital convenience
                """,
                explanation="Schnelle, präzise Empfehlungen ohne lange Erklärungen",
                difficulty_level=PromptComplexity.SIMPLE,
                tags=["quick", "groq_optimized", "fitness", "tech"]
            ),
            
            PromptExample(
                input_example="""
                Quick Request: Christmas, Colleague, 40+ age, under €25, professional setting
                """,
                expected_output="""
                1. Premium Coffee Subscription (€22) - Professional, universally appreciated
                2. Elegant Desk Plant Set (€18) - Office-appropriate, stress-reducing
                3. Quality Notebook & Pen Set (€24) - Practical, sophisticated
                """,
                explanation="Professionelle, sichere Geschenke für Arbeitskontext",
                difficulty_level=PromptComplexity.SIMPLE,
                tags=["quick", "professional", "christmas", "colleague"]
            )
        ]
    
    @staticmethod
    def get_complex_scenario_examples() -> List[PromptExample]:
        """
        Komplexe Szenarien für Chain-of-Thought Learning
        
        Zeigt mehrstufiges Reasoning
        """
        return [
            PromptExample(
                input_example="""
                Complex Scenario:
                Person: Dr. Sarah Chen, 42, Professional Colleague
                Occasion: Promotion Congratulations  
                Personality: Very High Conscientiousness (4.8/5), Medium Extraversion (3.0/5), High Openness (4.2/5)
                Context: She's a research scientist, recently promoted to department head
                Budget: €60-120
                Cultural: Asian heritage, values balance between tradition and innovation
                Constraints: Needs to be delivered within 3 days, appropriate for professional setting
                """,
                expected_output="""
                {
                  "reasoning_process": {
                    "step_1": "Analyze personality: Very high conscientiousness + promotion = appreciates excellence and achievement recognition",
                    "step_2": "Consider cultural context: Asian heritage values thoughtful, respectful gifts that honor success",
                    "step_3": "Professional setting: Must be appropriate for workplace, not too personal",
                    "step_4": "Time constraint: 3-day delivery limits options to available items",
                    "step_5": "Innovation + tradition balance: Something that honors both modern success and cultural values"
                  },
                  "recommendations": [
                    {
                      "title": "Executive Bamboo Desk Organizer with Personalized Nameplate",
                      "price": "€95",
                      "category": "professional",
                      "reasoning": "Bamboo respects Asian cultural appreciation for natural materials, organization appeals to conscientiousness, nameplate celebrates new leadership role",
                      "confidence": 0.91,
                      "delivery": "Next-day available"
                    },
                    {
                      "title": "Premium Tea Ceremony Set for Office",
                      "price": "€78",
                      "category": "wellness",
                      "reasoning": "Honors cultural heritage while providing stress relief for new leadership responsibilities",
                      "confidence": 0.87,
                      "delivery": "2-day shipping"
                    },
                    {
                      "title": "Innovation in Leadership Book Collection",
                      "price": "€65",
                      "category": "books",
                      "reasoning": "Appeals to openness for learning, supports her new leadership role, professionally appropriate",
                      "confidence": 0.83,
                      "delivery": "Same-day digital + 3-day physical"
                    }
                  ]
                }
                """,
                explanation="Zeigt komplexes multi-faktorielles Reasoning für schwierige Geschenk-Situationen",
                difficulty_level=PromptComplexity.EXPERT,
                tags=["complex_reasoning", "cultural_sensitivity", "professional", "time_constraints"]
            )
        ]


# =============================================================================
# MODEL-SPECIFIC PROMPT TEMPLATES
# =============================================================================

class OpenAIOptimizedTemplates:
    """
    Prompts optimiert für OpenAI GPT-4
    
    Fokus: Beste Qualität, kreatives Reasoning, detaillierte Analysen
    """
    
    @staticmethod
    def create_premium_gift_template() -> FewShotPromptTemplate:
        """
        Premium Template für OpenAI - Beste Qualität
        
        Verwendet alle verfügbaren Beispiele für maximale Lernwirkung
        """
        return FewShotPromptTemplate(
            template_name="openai_premium_gift_recommendations",
            template_version="2.1",
            description="Premium OpenAI template for highest quality gift recommendations",
            
            # Template Configuration
            technique=PromptTechnique.FEW_SHOT,
            complexity=PromptComplexity.COMPLEX,
            target_model=AIModelType.OPENAI_GPT4,
            optimization_goal=PromptOptimizationGoal.QUALITY,
            
            # System Prompt: Definiert AI-Rolle
            system_prompt="""
            You are an expert gift consultant with 15+ years of experience in personalized recommendations.
            You specialize in Big Five + Limbic personality analysis and cultural sensitivity.
            Your gift recommendations are known for their accuracy and thoughtfulness.

            Key Expertise:
            - Big Five + Limbic personality → gift preference mapping
            - Cultural and contextual awareness  
            - Budget optimization strategies
            - Relationship-appropriate suggestions
            - Timing and delivery considerations
            """,
            
            # Main Instruction
            instruction_prompt="""
            Analyze the provided personality profile and context to recommend perfect gifts.
            
            Follow this reasoning process:
            1. Map Big Five + Limbic dimensions to gift preferences
            2. Consider relationship dynamics and appropriateness
            3. Integrate occasion-specific requirements  
            4. Apply budget constraints intelligently
            5. Account for cultural and timing factors
            
            Provide 3-5 recommendations with detailed reasoning for each.
            Include confidence scores and alternative options.
            """,
            
            # Output Format
            output_format_instructions="""
            Return a valid JSON object with this structure:
            {
              "personality_analysis": "Brief summary of key traits affecting gift choice",
              "recommendations": [
                {
                  "title": "Specific gift name",
                  "price": "€XX",
                  "category": "gift_category", 
                  "personality_match": "How this matches their personality",
                  "confidence": 0.XX,
                  "reasoning": "Detailed explanation of why this is perfect",
                  "where_to_buy": ["shop1", "shop2"],
                  "delivery_time": "X days"
                }
              ],
              "alternatives": ["backup options if main recommendations unavailable"],
              "gift_wrapping_suggestion": "How to present this gift beautifully"
            }
            """,
            
            # Few-Shot Examples
            examples=GiftExamples.get_personality_based_examples() + GiftExamples.get_complex_scenario_examples(),
            
            # OpenAI-specific optimizations
            max_tokens=3000,
            temperature=0.7,  # Balance creativity with consistency
            
            # Performance tracking
            success_rate=0.94  # Based on user feedback
        )


class GroqSpeedTemplates:
    """
    Prompts optimiert für Groq Mixtral
    
    Fokus: Ultra-schnelle Antworten, effiziente Token-Nutzung
    """
    
    @staticmethod
    def create_quick_recommendation_template() -> FewShotPromptTemplate:
        """
        Speed-optimized Template für Groq
        
        Minimale Token, maximale Geschwindigkeit
        """
        return FewShotPromptTemplate(
            template_name="groq_speed_gift_recommendations",
            template_version="1.5",
            description="Ultra-fast gift recommendations optimized for Groq",
            
            # Speed Configuration
            technique=PromptTechnique.FEW_SHOT,
            complexity=PromptComplexity.SIMPLE,
            target_model=AIModelType.GROQ_MIXTRAL,
            optimization_goal=PromptOptimizationGoal.SPEED,
            
            # Minimal System Prompt
            system_prompt="You are a quick gift recommendation expert. Provide fast, accurate suggestions.",
            
            # Concise Instruction
            instruction_prompt="""
            Quick gift recommendations based on: person type, occasion, budget, preferences.
            Format: Simple list with price and brief reason.
            3 recommendations maximum.
            """,
            
            # Minimal Output Format
            output_format_instructions="""
            Format:
            1. Gift Name (€price) - Brief reason
            2. Gift Name (€price) - Brief reason  
            3. Gift Name (€price) - Brief reason
            """,
            
            # Speed-optimized Examples
            examples=GiftExamples.get_quick_recommendation_examples(),
            
            # Speed Settings
            max_tokens=500,  # Very limited for speed
            temperature=0.3,  # Low for consistency and speed
            
            # Performance
            success_rate=0.87  # Slightly lower quality but much faster
        )


class ClaudeReasoningTemplates:
    """
    Prompts optimiert für Anthropic Claude
    
    Fokus: Ethisches Reasoning, kulturelle Sensitivität, durchdachte Analysen
    """
    
    @staticmethod
    def create_thoughtful_recommendation_template() -> FewShotPromptTemplate:
        """
        Claude-optimized Template für durchdachte Empfehlungen
        
        Betont ethische Überlegungen und kulturelle Sensitivität
        """
        return FewShotPromptTemplate(
            template_name="claude_thoughtful_gift_recommendations",
            template_version="1.8",
            description="Thoughtful, ethically-aware gift recommendations optimized for Claude",
            
            # Claude Configuration
            technique=PromptTechnique.FEW_SHOT,
            complexity=PromptComplexity.MODERATE,
            target_model=AIModelType.ANTHROPIC_CLAUDE,
            optimization_goal=PromptOptimizationGoal.BALANCE,
            
            # Ethical System Prompt
            system_prompt="""
            You are a thoughtful gift consultant who prioritizes ethical considerations and cultural sensitivity.
            
            Core Principles:
            - Respect cultural differences and traditions
            - Consider environmental and ethical sourcing
            - Promote meaningful connections over materialism
            - Ensure recommendations are appropriate and respectful
            - Balance personal desires with relationship dynamics
            """,
            
            # Balanced Instruction
            instruction_prompt="""
            Provide thoughtful gift recommendations that consider:
            
            1. Personality fit and genuine appeal
            2. Cultural appropriateness and sensitivity  
            3. Relationship boundaries and expectations
            4. Ethical sourcing and sustainability when possible
            5. Long-term value and meaningfulness
            
            Explain your reasoning clearly and considerately.
            """,
            
            # Structured Output
            output_format_instructions="""
            Provide a JSON response with thoughtful analysis:
            {
              "cultural_considerations": "Any cultural factors to consider",
              "relationship_analysis": "How relationship affects gift appropriateness", 
              "recommendations": [
                {
                  "title": "Gift name",
                  "price": "€XX",
                  "category": "category",
                  "ethical_notes": "Sustainability/sourcing considerations",
                  "cultural_appropriateness": "Why this respects cultural context",
                  "relationship_fit": "Why this suits the relationship",
                  "confidence": 0.XX,
                  "reasoning": "Comprehensive explanation"
                }
              ],
              "general_advice": "Thoughtful suggestions for gift-giving approach"
            }
            """,
            
            # Thoughtful Examples
            examples=GiftExamples.get_personality_based_examples(),
            
            # Claude Settings
            max_tokens=2500,
            temperature=0.6,  # Balanced creativity
            
            # Performance
            success_rate=0.91  # High quality, ethical awareness
        )


# =============================================================================
# DYNAMIC CONTEXT INTEGRATION
# =============================================================================

class ContextualPromptBuilder:
    """
    Intelligente Kontext-Integration für personalisierte Prompts
    
    Passt Prompts dynamisch an Situation, Persönlichkeit und Präferenzen an
    """
    
    @staticmethod
    def build_personality_context(big_five_scores: Dict[str, float], limbic_scores: Dict[str, float] = None) -> ContextInjection:
      """
      Konvertiert Big Five + Limbic-Scores zu intelligentem Prompt-Kontext
      
      Args:
          big_five_scores: Dict mit Big Five-Dimensionen und Werten (0-1 scale)
          limbic_scores: Dict mit Limbic-Dimensionen und Werten (0-1 scale)
      """
      personality_insights = []
      
      # Analyse der Big Five-Dimensionen
      high_traits = [trait for trait, score in big_five_scores.items() if score >= 0.7]
      low_traits = [trait for trait, score in big_five_scores.items() if score <= 0.3]
      
      # Big Five Insights
      if big_five_scores.get('openness', 0.5) >= 0.7:
          personality_insights.append("High openness suggests preference for creative, novel, and experiential gifts")
      
      if big_five_scores.get('extraversion', 0.5) >= 0.7:
          personality_insights.append("High extraversion indicates enjoyment of social experiences and group activities")
      elif big_five_scores.get('extraversion', 0.5) <= 0.3:
          personality_insights.append("Lower extraversion suggests preference for intimate, private, or solo experiences")
      
      if big_five_scores.get('conscientiousness', 0.5) >= 0.7:
          personality_insights.append("High conscientiousness appreciates organized, planned, and high-quality items")
      
      if big_five_scores.get('agreeableness', 0.5) >= 0.7:
          personality_insights.append("High agreeableness enjoys gifts that facilitate connection and shared experiences")
      
      if big_five_scores.get('neuroticism', 0.5) >= 0.7:
          personality_insights.append("High neuroticism benefits from stress-reducing and comforting gifts")
      elif big_five_scores.get('neuroticism', 0.5) <= 0.3:
          personality_insights.append("Low neuroticism (high emotional stability) can handle challenging or intense experiences")
      
      # Limbic System Insights
      if limbic_scores:
          if limbic_scores.get('stimulanz', 0.5) >= 0.7:
              personality_insights.append("High stimulanz needs exciting, intense, and adrenaline-rich gifts")
          
          if limbic_scores.get('dominanz', 0.5) >= 0.7:
              personality_insights.append("High dominanz appreciates status, exclusivity, and premium gifts")
          
          if limbic_scores.get('balance', 0.5) >= 0.7:
              personality_insights.append("High balance seeks harmonious, mindful, and well-balanced gifts")
      
      return ContextInjection(
          personality_context="; ".join(personality_insights),
          emotional_context=f"Dominant traits: {', '.join(high_traits)}; Lower traits: {', '.join(low_traits)}"
      )
    
    @staticmethod
    def build_occasion_context(occasion: GiftOccasion, date: Optional[datetime] = None) -> ContextInjection:
        """
        Erstellt occasions-spezifischen Kontext
        """
        
        occasion_insights = {
            GiftOccasion.BIRTHDAY: "Personal celebration - focus on individual preferences and making them feel special",
            GiftOccasion.CHRISTMAS: "Traditional holiday - balance personal taste with seasonal appropriateness", 
            GiftOccasion.ANNIVERSARY: "Relationship milestone - emphasize shared memories and future together",
            GiftOccasion.GRADUATION: "Achievement recognition - celebrate accomplishment and future potential",
            GiftOccasion.WEDDING: "Life transition - practical items for new life phase or symbolic gifts",
            GiftOccasion.VALENTINES: "Romantic expression - intimate and personally meaningful gifts",
            GiftOccasion.JUST_BECAUSE: "Spontaneous gesture - fun, surprising, or thoughtful everyday items"
        }
        
        return ContextInjection(
            occasion_context=occasion_insights.get(occasion, "General gift-giving occasion")
        )
    
    @staticmethod
    def build_relationship_context(relationship: RelationshipType) -> ContextInjection:
        """
        Erstellt beziehungs-spezifischen Kontext
        Erstellt beziehungs-spezifischen Kontext mit der neuen RelationshipAnalyzer
        """
        
        relationship_guidelines = {
            RelationshipType.ROMANTIC_PARTNER: "Intimate relationship - personal, meaningful gifts appropriate",
            RelationshipType.SPOUSE: "Life partner - practical, shared experience, or deeply personal gifts",
            RelationshipType.FAMILY_PARENT: "Respect and gratitude - thoughtful, quality items showing appreciation",
            RelationshipType.FAMILY_SIBLING: "Familiar bond - can be playful, personal, or shared-memory based",
            RelationshipType.FRIEND_CLOSE: "Strong friendship - personal but not overly intimate gifts",
            RelationshipType.FRIEND_CASUAL: "Moderate friendship - thoughtful but not too personal",
            RelationshipType.COLLEAGUE: "Professional relationship - appropriate, respectful, universal appeal",
            RelationshipType.BOSS: "Professional hierarchy - respectful, high-quality, appropriate gifts"
        }
        
        analyzer = RelationshipAnalyzer(relationship_type=relationship)
    
        return ContextInjection(
            relationship_context=analyzer.to_ai_context()
        )
        


# =============================================================================
# TEMPLATE FACTORY: Intelligente Template-Auswahl
# =============================================================================

class GiftPromptFactory:
    """
    Factory für intelligente Prompt-Template Auswahl
    
    Wählt optimal Template basierend auf:
    - AI-Model Capabilities  
    - Optimization Goals
    - Complexity Requirements
    """
    
    @staticmethod
    def get_optimal_template(
        model_type: AIModelType,
        optimization_goal: PromptOptimizationGoal,
        complexity: PromptComplexity
    ) -> FewShotPromptTemplate:
        """
        Wählt optimales Template für gegebene Parameter
        """
        
        if model_type == AIModelType.OPENAI_GPT4:
            if optimization_goal == PromptOptimizationGoal.QUALITY:
                return OpenAIOptimizedTemplates.create_premium_gift_template()
        
        elif model_type == AIModelType.GROQ_MIXTRAL:
            if optimization_goal == PromptOptimizationGoal.SPEED:
                return GroqSpeedTemplates.create_quick_recommendation_template()
        
        elif model_type == AIModelType.ANTHROPIC_CLAUDE:
            return ClaudeReasoningTemplates.create_thoughtful_recommendation_template()
        
        # Fallback to OpenAI premium template
        return OpenAIOptimizedTemplates.create_premium_gift_template()
    
    @staticmethod
    def build_contextual_prompt(
        template: FewShotPromptTemplate,
        personality_context: ContextInjection,
        occasion_context: ContextInjection,
        relationship_context: ContextInjection
    ) -> str:
        """
        Kombiniert Template mit Kontext zu finalem Prompt
        
        Dies ist wo die Magie passiert - dynamische Prompt-Generierung!
        """
        
        # System Prompt
        final_prompt = f"SYSTEM: {template.system_prompt}\n\n"
        
        # Context Integration
        final_prompt += "CONTEXT:\n"
        if personality_context.personality_context:
            final_prompt += f"Personality: {personality_context.personality_context}\n"
        if occasion_context.occasion_context:
            final_prompt += f"Occasion: {occasion_context.occasion_context}\n"
        if relationship_context.relationship_context:
            final_prompt += f"Relationship: {relationship_context.relationship_context}\n"
        final_prompt += "\n"
        
        # Few-Shot Examples
        final_prompt += "EXAMPLES:\n"
        for i, example in enumerate(template.examples, 1):
            final_prompt += f"\nExample {i}:\n"
            final_prompt += f"Input: {example.input_example}\n"
            final_prompt += f"Output: {example.expected_output}\n"
            final_prompt += template.example_separator + "\n"
        
        # Main Instruction
        final_prompt += f"\nINSTRUCTION:\n{template.instruction_prompt}\n\n"
        
        # Output Format
        if template.output_format_instructions:
            final_prompt += f"OUTPUT FORMAT:\n{template.output_format_instructions}\n\n"
        
        final_prompt += "Now, analyze the following request and provide your recommendation:\n\n"
        
        return final_prompt


# =============================================================================
# EXPORTS: Clean Interface
# =============================================================================

__all__ = [
    'GiftExamples',
    'OpenAIOptimizedTemplates',
    'GroqSpeedTemplates', 
    'ClaudeReasoningTemplates',
    'ContextualPromptBuilder',
    'GiftPromptFactory'
]