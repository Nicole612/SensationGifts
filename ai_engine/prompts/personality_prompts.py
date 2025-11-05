"""
Personality Analysis Prompts - Chain-of-Thought Big Five + Limbic Reasoning
========================================================================

INNOVATION: Advanced Chain-of-Thought für Big Five + Limbic System
- Wissenschaftlich validierte Big Five Dimensionen
- Neuropsychologisches Limbic System für emotionale Kauftrigger  
- Cross-Dimensional Interaction Analysis
- Personality → Gift Category Mapping optimiert für E-Commerce

Features:
- Strukturiertes Schritt-für-Schritt Denken
- Big Five Einzelanalyse + Limbic Overlay
- Emotionale Trigger durch Limbic System
- Erweiterte Gift-Personalisierung
"""

from typing import Dict, List, Optional, Any
from decimal import Decimal

from ai_engine.schemas import (
    ChainOfThoughtTemplate,
    ChainOfThoughtStep, 
    ContextInjection,
    PromptTechnique,
    PromptComplexity,
    AIModelType,
    PromptOptimizationGoal
)


# =============================================================================
# CHAIN-OF-THOUGHT REASONING STEPS: Big Five + Limbic Analysis
# =============================================================================

class BigFiveLimbicAnalysisSteps:
    """
    Strukturierte Denkschritte für Big Five + Limbic Persönlichkeitsanalyse
    
    INNOVATION: Kombiniert wissenschaftliche Big Five mit emotionalen Limbic Triggern
    für optimierte E-Commerce Geschenkempfehlungen
    """
    
    @staticmethod
    def get_comprehensive_analysis_steps() -> List[ChainOfThoughtStep]:
        """
        Vollständige Big Five + Limbic Analyse mit Cross-Dimensional Reasoning
        
        5 Big Five Dimensionen + 3 Limbic Dimensionen + Interaktions-Analyse
        """
        return [
            ChainOfThoughtStep(
                step_number=1,
                step_description="Analyze Openness to Experience for innovation vs tradition preference",
                step_prompt="""
                First, let me analyze the Openness to Experience score of {openness}/5.
                
                Consider:
                - High Openness (4.0+): Loves novelty, creativity, new experiences, artistic pursuits
                - Medium Openness (2.5-3.9): Balanced between familiar and new experiences
                - Low Openness (below 2.5): Prefers traditional, familiar, proven choices
                
                Based on this score, should gifts be more innovative/creative or traditional/familiar?
                What types of gift categories align with this openness level?
                """,
                expected_reasoning="AI should map Openness to innovation tolerance and creative gift preferences",
                dependencies=[]
            ),
            
            ChainOfThoughtStep(
                step_number=2,
                step_description="Analyze Conscientiousness for structured vs spontaneous gift preferences",
                step_prompt="""
                Next, let me examine the Conscientiousness score of {conscientiousness}/5.
                
                Consider:
                - High Conscientiousness (4.0+): Values organization, planning, quality, structured experiences
                - Medium Conscientiousness (2.5-3.9): Flexible between planned and spontaneous activities
                - Low Conscientiousness (below 2.5): Prefers flexible, spontaneous, low-maintenance gifts
                
                Should gifts be more organized/planned or flexible/spontaneous?
                What level of complexity and structure is appropriate?
                """,
                expected_reasoning="AI should determine appropriate structure level and quality expectations",
                dependencies=[1]
            ),
            
            ChainOfThoughtStep(
                step_number=3,
                step_description="Analyze Extraversion for social vs private gift preferences", 
                step_prompt="""
                Now, let me consider the Extraversion score of {extraversion}/5.
                
                Consider:
                - High Extraversion (4.0+): Enjoys social experiences, group activities, public recognition
                - Medium Extraversion (2.5-3.9): Comfortable with both social and private experiences
                - Low Extraversion (below 2.5): Prefers intimate, private, solo experiences
                
                Should the recommended gifts be more social/public or private/intimate in nature?
                How does this affect the social context of gift experiences?
                """,
                expected_reasoning="AI should determine social context appropriateness for gifts",
                dependencies=[1, 2]
            ),
            
            ChainOfThoughtStep(
                step_number=4,
                step_description="Analyze Agreeableness for cooperation vs individual-focused gifts",
                step_prompt="""
                Let me evaluate the Agreeableness score of {agreeableness}/5.
                
                Consider:
                - High Agreeableness (4.0+): Values harmony, cooperation, shared experiences, helping others
                - Medium Agreeableness (2.5-3.9): Balanced between group and individual preferences  
                - Low Agreeableness (below 2.5): More independent, competitive, individual-focused
                
                Should gifts emphasize sharing/cooperation or individual achievement/enjoyment?
                How important is the social harmony aspect of the gift?
                """,
                expected_reasoning="AI should balance group-oriented vs individual-focused gift characteristics",
                dependencies=[1, 2, 3]
            ),
            
            ChainOfThoughtStep(
                step_number=5,
                step_description="Analyze Neuroticism for emotional stability and stress considerations",
                step_prompt="""
                Now examining the Neuroticism score of {neuroticism}/5.
                
                Consider:
                - High Neuroticism (4.0+): More emotionally sensitive, may appreciate comfort/stress-relief gifts
                - Medium Neuroticism (2.5-3.9): Balanced emotional responses
                - Low Neuroticism (below 2.5): Emotionally stable, can handle challenging/intense experiences
                
                Should gifts be more comforting/stress-relieving or can they be challenging/intense?
                What emotional support level should the gift provide?
                """,
                expected_reasoning="AI should assess emotional support needs and stress considerations",
                dependencies=[1, 2, 3, 4]
            ),
            
            ChainOfThoughtStep(
                step_number=6,
                step_description="Analyze Limbic Stimulanz for stimulation-seeking behavior",
                step_prompt="""
                Now I'll analyze the Limbic Stimulanz (stimulation-seeking) score of {stimulanz}/5.
                
                Consider:
                - High Stimulanz (4.0+): Craves excitement, novelty, intense experiences, adrenaline
                - Medium Stimulanz (2.5-3.9): Enjoys moderate stimulation with some calm periods
                - Low Stimulanz (below 2.5): Prefers calm, peaceful, low-intensity experiences
                
                What level of excitement and intensity should the gift provide?
                Should it be high-energy/thrilling or calm/peaceful?
                """,
                expected_reasoning="AI should determine appropriate stimulation level for gifts",
                dependencies=[1, 2, 3, 4, 5]
            ),
            
            ChainOfThoughtStep(
                step_number=7,
                step_description="Analyze Limbic Dominanz for control and leadership preferences",
                step_prompt="""
                Let me examine the Limbic Dominanz (dominance-seeking) score of {dominanz}/5.
                
                Consider:
                - High Dominanz (4.0+): Values control, leadership, status, premium/exclusive items
                - Medium Dominanz (2.5-3.9): Comfortable with both leading and following
                - Low Dominanz (below 2.5): Prefers supportive, humble, collaborative gifts
                
                Should gifts emphasize status/control or humility/collaboration?
                What level of exclusivity and prestige is appropriate?
                """,
                expected_reasoning="AI should assess status and control preferences for gift selection",
                dependencies=[1, 2, 3, 4, 5, 6]
            ),
            
            ChainOfThoughtStep(
                step_number=8,
                step_description="Analyze Limbic Balance for harmony and emotional equilibrium",
                step_prompt="""
                Finally, let me consider the Limbic Balance score of {balance}/5.
                
                Consider:
                - High Balance (4.0+): Values harmony, equilibrium, mindful choices, sustainable options
                - Medium Balance (2.5-3.9): Seeks some balance but can handle occasional extremes
                - Low Balance (below 2.5): Comfortable with extremes, intense emotions, dramatic experiences
                
                Should gifts promote balance/harmony or can they be more extreme/dramatic?
                How important is emotional equilibrium in the gift choice?
                """,
                expected_reasoning="AI should determine balance vs intensity preferences for gifts",
                dependencies=[1, 2, 3, 4, 5, 6, 7]
            ),
            
            ChainOfThoughtStep(
                step_number=9,
                step_description="Cross-dimensional Big Five + Limbic interaction analysis",
                step_prompt="""
                Now I need to analyze how Big Five and Limbic dimensions interact:
                
                Key Interaction Patterns:
                - High Openness + High Stimulanz = Creative thrill-seeking experiences
                - High Conscientiousness + High Balance = Quality mindful products
                - High Extraversion + High Dominanz = Social leadership experiences
                - Low Neuroticism + High Stimulanz = Extreme adventure activities
                - High Agreeableness + High Balance = Harmonious group experiences
                - High Openness + Low Balance = Edgy creative experiences
                
                What unique personality-emotion combination emerges from these scores?
                How do cognitive traits (Big Five) interact with emotional drives (Limbic)?
                """,
                expected_reasoning="AI should identify personality-emotion archetypes and unique combinations",
                dependencies=[1, 2, 3, 4, 5, 6, 7, 8]
            ),
            
            ChainOfThoughtStep(
                step_number=10,
                step_description="Determine Limbic Type and emotional purchase triggers",
                step_prompt="""
                Based on the Limbic analysis, I need to determine the Limbic Type:
                
                Limbic Types:
                - Disciplined: High Balance + Low Stimulanz (values quality, mindfulness)
                - Traditionalist: Low Stimulanz + Low Dominanz (prefers familiar, modest)
                - Performer: High Dominanz + High Stimulanz (wants status, excitement)
                - Adventurer: High Stimulanz + Medium Balance (seeks new experiences)
                - Harmonizer: High Balance + Low Dominanz (values peace, cooperation)
                - Hedonist: High Stimulanz + Low Balance (wants immediate pleasure)
                - Pioneer: High in all three (innovative leader seeking balanced excellence)
                
                What Limbic Type best describes this person and what are their core emotional triggers?
                """,
                expected_reasoning="AI should classify Limbic Type and identify emotional purchase motivations",
                dependencies=[6, 7, 8, 9]
            ),
            
            ChainOfThoughtStep(
                step_number=11,
                step_description="Synthesize into optimized gift category preferences",
                step_prompt="""
                Based on the complete Big Five + Limbic analysis, let me synthesize optimal gift categories:
                
                Primary Categories (best personality-emotion fit):
                Secondary Categories (good alternatives):  
                Avoid Categories (likely poor fit):
                
                Consider both cognitive preferences (Big Five) and emotional triggers (Limbic).
                Focus on categories that satisfy both personality traits and emotional drives.
                """,
                expected_reasoning="AI should create prioritized gift recommendations based on personality-emotion integration",
                dependencies=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            )
        ]
    
    @staticmethod
    def get_quick_analysis_steps() -> List[ChainOfThoughtStep]:
        """
        Schnelle Big Five + Limbic Analyse für Speed-optimierte Verarbeitung
        
        Fokus auf dominante Traits und praktische Limbic Triggers
        """
        return [
            ChainOfThoughtStep(
                step_number=1,
                step_description="Identify dominant Big Five traits",
                step_prompt="""
                Looking at the Big Five scores, let me identify the strongest traits (4.0+ scores):
                Openness: {openness}/5
                Conscientiousness: {conscientiousness}/5  
                Extraversion: {extraversion}/5
                Agreeableness: {agreeableness}/5
                Neuroticism: {neuroticism}/5
                
                Which 2-3 Big Five traits are most dominant and will drive gift preferences?
                """,
                expected_reasoning="AI should identify the Big Five personality drivers",
                dependencies=[]
            ),
            
            ChainOfThoughtStep(
                step_number=2,
                step_description="Determine Limbic emotional profile",
                step_prompt="""
                Now let me analyze the Limbic emotional drivers:
                Stimulanz (stimulation-seeking): {stimulanz}/5
                Dominanz (dominance-seeking): {dominanz}/5
                Balance (harmony-seeking): {balance}/5
                
                What's the primary Limbic Type and key emotional triggers?
                How do these emotional drives complement the Big Five traits?
                """,
                expected_reasoning="AI should identify emotional purchase motivations",
                dependencies=[1]
            ),
            
            ChainOfThoughtStep(
                step_number=3,
                step_description="Generate quick category recommendations",
                step_prompt="""
                Based on dominant Big Five traits + Limbic Type, recommend:
                1. Primary category (best personality-emotion fit)
                2. Secondary category (good alternative)
                3. Tertiary category (backup option)
                
                Brief reasoning connecting personality traits to emotional triggers.
                """,
                expected_reasoning="AI should provide quick, actionable recommendations integrating personality and emotion",
                dependencies=[1, 2]
            )
        ]


# =============================================================================
# SPECIALIZED TEMPLATES FOR BIG FIVE + LIMBIC
# =============================================================================

class ComprehensiveBigFiveLimbicTemplate:
    """
    Template für tiefgehende Big Five + Limbic Persönlichkeitsanalyse
    
    INNOVATION: Erste Integration von wissenschaftlichem Big Five mit 
    neuropsychologischem Limbic System für E-Commerce Optimization
    """
    
    @staticmethod
    def create_template() -> ChainOfThoughtTemplate:
        return ChainOfThoughtTemplate(
            template_name="comprehensive_bigfive_limbic_analysis",
            template_version="3.0",
            description="Deep Big Five + Limbic personality analysis with emotion-behavior mapping",
            
            # Configuration
            technique=PromptTechnique.CHAIN_OF_THOUGHT,
            complexity=PromptComplexity.EXPERT,
            target_model=AIModelType.OPENAI_GPT4,
            optimization_goal=PromptOptimizationGoal.QUALITY,
            
            # Enhanced System Prompt
            system_prompt="""
            You are an expert personality psychologist and consumer behavior analyst specializing in:
            - Big Five personality model (OCEAN) analysis and interpretation
            - Limbic System neuroscience for emotional decision-making
            - Personality-emotion integration for consumer preferences
            - E-commerce personalization through psychological profiling
            
            Your expertise includes:
            - Big Five dimensional analysis and behavioral predictions
            - Limbic System emotional triggers and purchase motivations  
            - Cross-dimensional personality-emotion interactions
            - Scientific personality → consumer behavior mapping
            - Cultural and contextual considerations in gift-giving
            """,
            
            # Instruction
            instruction_prompt="""
            Perform a comprehensive Big Five + Limbic System analysis to understand gift preferences.
            
            This analysis combines:
            1. Big Five personality dimensions (cognitive traits)
            2. Limbic System dimensions (emotional drives)
            3. Cross-dimensional interactions between cognition and emotion
            4. Practical implications for gift personalization
            
            Follow the structured reasoning steps provided.
            For each step, consider:
            1. What this dimension reveals about the person
            2. How it affects gift preferences and purchase behavior
            3. How it interacts with other dimensions
            4. What practical gift implications this has
            5. How emotional drives complement cognitive traits
            
            Be thorough, scientific, and actionable in your analysis.
            """,
            
            # Chain-of-Thought Steps
            reasoning_steps=BigFiveLimbicAnalysisSteps.get_comprehensive_analysis_steps(),
            
            # CoT Configuration
            step_connector="Let me think through this systematically, analyzing both personality and emotional dimensions:",
            conclusion_prompt="Based on this comprehensive Big Five + Limbic analysis, here are my insights:",
            encourage_reasoning=True,
            
            # Output Format
            output_format_instructions="""
            Provide your analysis in this JSON structure:
            {
              "personality_summary": "2-3 sentence overview integrating Big Five traits with Limbic drives",
              "big_five_analysis": {
                "dominant_traits": ["trait1", "trait2", "trait3"],
                "openness": "detailed analysis of creativity/novelty preferences",
                "conscientiousness": "analysis of structure/quality preferences",
                "extraversion": "analysis of social/energy preferences", 
                "agreeableness": "analysis of cooperation/harmony preferences",
                "neuroticism": "analysis of emotional stability/stress preferences"
              },
              "limbic_analysis": {
                "limbic_type": "primary Limbic Type (Disciplined/Performer/etc.)",
                "stimulanz": "analysis of stimulation-seeking behavior",
                "dominanz": "analysis of control/status preferences",
                "balance": "analysis of harmony/equilibrium needs",
                "emotional_triggers": ["trigger1", "trigger2", "trigger3"]
              },
              "personality_emotion_integration": {
                "interaction_effects": "How Big Five traits interact with Limbic drives",
                "purchase_motivations": "Primary emotional triggers for gift purchases",
                "decision_making_style": "How this person likely makes gift-related decisions"
              },
              "gift_optimization": {
                "primary_categories": ["category1", "category2"],
                "secondary_categories": ["category3", "category4"], 
                "avoid_categories": ["category5", "category6"],
                "personalization_strategies": ["strategy1", "strategy2"],
                "emotional_appeals": ["appeal1", "appeal2"]
              },
              "confidence_assessment": "Confidence in analysis (0.0-1.0) with reasoning"
            }
            """,
            
            # Performance Settings
            max_tokens=5000,
            temperature=0.65,
            success_rate=0.95
        )


class QuickBigFiveLimbicTemplate:
    """
    Template für schnelle Big Five + Limbic Analyse
    
    Speed-optimiert für Groq mit Fokus auf praktische Empfehlungen
    """
    
    @staticmethod  
    def create_template() -> ChainOfThoughtTemplate:
        return ChainOfThoughtTemplate(
            template_name="quick_bigfive_limbic_analysis",
            template_version="2.0",
            description="Fast Big Five + Limbic analysis with focused emotion-behavior insights",
            
            # Speed Configuration
            technique=PromptTechnique.CHAIN_OF_THOUGHT,
            complexity=PromptComplexity.MODERATE,
            target_model=AIModelType.GROQ_MIXTRAL,
            optimization_goal=PromptOptimizationGoal.SPEED,
            
            # Streamlined System Prompt
            system_prompt="You are a personality-emotion analyst focused on quick, accurate gift insights using Big Five + Limbic System.",
            
            # Focused Instruction
            instruction_prompt="""
            Quickly analyze Big Five personality + Limbic emotional drivers for gift recommendations.
            Focus on dominant traits and practical emotional triggers.
            """,
            
            # Shortened Steps
            reasoning_steps=BigFiveLimbicAnalysisSteps.get_quick_analysis_steps(),
            
            # Speed Settings
            step_connector="Quick analysis:",
            conclusion_prompt="Key insights:",
            encourage_reasoning=False,
            
            # Minimal Output
            output_format_instructions="""
            JSON format:
            {
              "dominant_big_five": ["trait1", "trait2"],
              "limbic_type": "Primary Limbic Type",
              "emotional_triggers": ["trigger1", "trigger2"],
              "gift_preferences": "Brief summary of preferences",
              "top_categories": ["cat1", "cat2", "cat3"], 
              "avoid": ["avoid1", "avoid2"]
            }
            """,
            
            # Speed Optimization
            max_tokens=1200,
            temperature=0.4,
            success_rate=0.88
        )


class EmotionalIntelligenceTemplate:
    """
    Template für emotionale Intelligenz und Limbic-fokussierte Analyse
    
    Optimiert für Anthropic Claude (Emotional Reasoning)
    Fokus auf emotionale Trigger und empathische Gift-Empfehlungen
    """
    
    @staticmethod
    def create_template() -> ChainOfThoughtTemplate:
        return ChainOfThoughtTemplate(
            template_name="emotional_intelligence_limbic_analysis",
            template_version="2.5", 
            description="Emotionally intelligent Big Five + Limbic analysis with empathetic gift insights",
            
            # Claude Configuration
            technique=PromptTechnique.CHAIN_OF_THOUGHT,
            complexity=PromptComplexity.COMPLEX,
            target_model=AIModelType.ANTHROPIC_CLAUDE,
            optimization_goal=PromptOptimizationGoal.BALANCE,
            
            # Empathetic System Prompt
            system_prompt="""
            You are an emotionally intelligent personality analyst who understands:
            - The deep emotional needs behind personality traits
            - How Limbic System drives influence meaningful connections
            - Empathetic gift-giving that truly resonates with recipients
            - Cultural sensitivity in emotional expression and gift preferences
            - The psychology of meaningful, lasting gift experiences
            """,
            
            # Emotionally Aware Instruction
            instruction_prompt="""
            Analyze this person's Big Five personality and Limbic emotional drivers with deep empathy and understanding.
            
            Focus on:
            - What this person truly needs emotionally
            - How their personality shapes their deepest preferences
            - What kind of gifts would make them feel truly seen and appreciated
            - How to honor both their cognitive traits and emotional drives
            - Cultural and individual sensitivity in gift recommendations
            
            Remember: Great gifts connect with both who someone is (personality) and what they feel (emotions).
            """,
            
            # Empathy-Enhanced Steps
            reasoning_steps=BigFiveLimbicAnalysisSteps.get_comprehensive_analysis_steps() + [
                ChainOfThoughtStep(
                    step_number=12,
                    step_description="Emotional empathy and deep needs analysis",
                    step_prompt="""
                    Finally, let me consider the deeper emotional needs:
                    - What does this personality-emotion combination tell me about their core needs?
                    - What kind of gift would make them feel truly understood and appreciated?
                    - How can gifts honor both their personality and emotional authenticity?
                    - What cultural or individual sensitivities should be considered?
                    """,
                    expected_reasoning="AI should identify deep emotional needs and empathetic gift approaches",
                    dependencies=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                )
            ],
            
            # Thoughtful Configuration
            step_connector="Let me thoughtfully consider both personality and emotions:",
            conclusion_prompt="Bringing together personality, emotions, and deep human understanding:",
            encourage_reasoning=True,
            
            # Empathetic Output
            output_format_instructions="""
            Provide thoughtful, empathetic analysis:
            {
              "personality_insights": "Deep understanding of this person's cognitive and emotional nature",
              "core_emotional_needs": "What this person most needs to feel understood and appreciated",
              "gift_philosophy": "The underlying approach that will resonate most deeply",
              "empathetic_recommendations": {
                "deeply_meaningful": ["option1", "option2"],
                "personality_aligned": ["option3", "option4"],
                "emotionally_resonant": ["option5", "option6"]
              },
              "cultural_considerations": "Important cultural or individual sensitivities",
              "gift_giving_approach": "How to present and contextualize gifts for maximum emotional impact"
            }
            """,
            
            # Balanced Settings
            max_tokens=4000,
            temperature=0.6,
            success_rate=0.92
        )


# =============================================================================
# BIG FIVE + LIMBIC MAPPING LOGIC
# =============================================================================

class BigFiveLimbicMapper:
    """
    Intelligentes Mapping von Big Five + Limbic Scores zu Geschenk-Kategorien
    
    INNOVATION: Erste wissenschaftliche Integration von Persönlichkeit + Emotion
    für E-Commerce Personalisierung
    """
    
    @staticmethod
    def map_bigfive_limbic_to_categories(
        big_five_scores: Dict[str, float], 
        limbic_scores: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """
        Konvertiert Big Five + Limbic Scores zu priorisierten Geschenk-Kategorien
        
        Args:
            big_five_scores: Dict with openness, conscientiousness, extraversion, agreeableness, neuroticism
            limbic_scores: Dict with stimulanz, dominanz, balance
            
        Returns:
            Dict mit 'primary', 'secondary', 'avoid' Kategorien
        """
        
        primary_categories = []
        secondary_categories = []
        avoid_categories = []
        
        # Big Five Mappings
        if big_five_scores.get('openness', 0) >= 0.6:
            primary_categories.extend(['art_crafts', 'experiences', 'books', 'innovative_tech'])
            if big_five_scores.get('openness', 0) >= 0.8:
                primary_categories.extend(['experimental_art', 'avant_garde', 'cutting_edge_tech'])
        elif big_five_scores.get('openness', 0) <= 0.4:
            primary_categories.extend(['traditional_gifts', 'classic_items', 'proven_products'])
            avoid_categories.extend(['experimental_items', 'avant_garde_art'])
        
        if big_five_scores.get('conscientiousness', 0) >= 0.6:
            primary_categories.extend(['quality_tools', 'organization', 'premium_products', 'planning_tools'])
            secondary_categories.extend(['professional_development', 'skill_building'])
        elif big_five_scores.get('conscientiousness', 0) <= 0.4:
            primary_categories.extend(['spontaneous_experiences', 'flexible_gifts'])
            avoid_categories.extend(['complex_organization_tools', 'rigid_planning_systems'])
        
        if big_five_scores.get('extraversion', 0) >= 0.6:
            primary_categories.extend(['social_experiences', 'group_activities', 'party_supplies', 'public_events'])
            secondary_categories.extend(['networking_tools', 'presentation_equipment'])
        elif big_five_scores.get('extraversion', 0) <= 0.4:
            primary_categories.extend(['solo_hobbies', 'private_experiences', 'home_comfort'])
            avoid_categories.extend(['group_games', 'public_performances'])
        
        if big_five_scores.get('agreeableness', 0) >= 0.6:
            primary_categories.extend(['collaborative_gifts', 'charity_donations', 'family_oriented'])
            secondary_categories.extend(['team_building', 'community_projects'])
        elif big_five_scores.get('agreeableness', 0) <= 0.4:
            primary_categories.extend(['individual_achievement', 'competitive_items'])
            avoid_categories.extend(['overly_collaborative_gifts'])
        
        if big_five_scores.get('neuroticism', 0) >= 0.6:
            primary_categories.extend(['stress_relief', 'comfort_items', 'wellness', 'stability_focused'])
            secondary_categories.extend(['meditation_tools', 'calming_experiences'])
        elif big_five_scores.get('neuroticism', 0) <= 0.4:
            primary_categories.extend(['challenging_experiences', 'risk_activities', 'intense_sports'])
            secondary_categories.extend(['thrill_seeking', 'adventure_gear'])
        
        # Limbic System Mappings (INNOVATION!)
        if limbic_scores.get('stimulanz', 0) >= 0.6:
            primary_categories.extend(['high_energy_activities', 'extreme_sports', 'exciting_experiences'])
            if limbic_scores.get('stimulanz', 0) >= 0.8:
                primary_categories.extend(['adrenaline_rushes', 'intense_stimulation'])
        elif limbic_scores.get('stimulanz', 0) <= 0.4:
            primary_categories.extend(['calm_activities', 'peaceful_experiences', 'low_stimulation'])
            avoid_categories.extend(['high_intensity_activities', 'extreme_sports'])
        
        if limbic_scores.get('dominanz', 0) >= 0.6:
            primary_categories.extend(['luxury_items', 'status_symbols', 'leadership_tools', 'premium_brands'])
            secondary_categories.extend(['exclusive_experiences', 'VIP_access'])
        elif limbic_scores.get('dominanz', 0) <= 0.4:
            primary_categories.extend(['humble_gifts', 'collaborative_tools', 'modest_options'])
            avoid_categories.extend(['status_symbols', 'luxury_displays'])
        
        if limbic_scores.get('balance', 0) >= 0.6:
            primary_categories.extend(['harmonious_items', 'balanced_lifestyle', 'mindful_products', 'sustainable_gifts'])
            secondary_categories.extend(['zen_items', 'equilibrium_focused'])
        elif limbic_scores.get('balance', 0) <= 0.4:
            primary_categories.extend(['intense_experiences', 'dramatic_items', 'extreme_emotions'])
            avoid_categories.extend(['overly_balanced_items', 'too_zen'])
        
        # Cross-Dimensional Combinations (ADVANCED INNOVATION!)
        
        # Creative Stimulation Seeker
        if (big_five_scores.get('openness', 0) >= 0.6 and 
            limbic_scores.get('stimulanz', 0) >= 0.6):
            primary_categories.extend(['creative_adventures', 'artistic_challenges', 'innovative_art_tech'])
        
        # Organized Leader  
        if (big_five_scores.get('conscientiousness', 0) >= 0.6 and 
            limbic_scores.get('dominanz', 0) >= 0.6):
            primary_categories.extend(['premium_organization_tools', 'executive_accessories', 'leadership_development'])
        
        # Social Harmonizer
        if (big_five_scores.get('extraversion', 0) >= 0.6 and 
            big_five_scores.get('agreeableness', 0) >= 0.6 and
            limbic_scores.get('balance', 0) >= 0.6):
            primary_categories.extend(['group_harmony_activities', 'community_building', 'social_wellness'])
        
        # Stable Adventurer
        if (big_five_scores.get('neuroticism', 0) <= 0.4 and 
            limbic_scores.get('stimulanz', 0) >= 0.6):
            primary_categories.extend(['controlled_adventures', 'safe_thrills', 'structured_excitement'])
        
        # Remove duplicates and prioritize
        primary_categories = list(set(primary_categories))
        secondary_categories = list(set(secondary_categories) - set(primary_categories))
        avoid_categories = list(set(avoid_categories))
        
        return {
            'primary': primary_categories[:6],  # Top 6
            'secondary': secondary_categories[:4],  # Top 4 alternatives
            'avoid': avoid_categories[:3]  # Top 3 to avoid
        }
    
    @staticmethod
    def determine_limbic_type(limbic_scores: Dict[str, float]) -> str:
        """
        Bestimmt Limbic Type basierend auf Stimulanz, Dominanz, Balance Scores
        """
        stimulanz = limbic_scores.get('stimulanz', 0.5)
        dominanz = limbic_scores.get('dominanz', 0.5)
        balance = limbic_scores.get('balance', 0.5)
        
        # Schwellenwerte
        threshold = 0.6
        
        high_stim = stimulanz >= threshold
        high_dom = dominanz >= threshold
        high_bal = balance >= threshold
        
        if high_bal and not high_stim:
            return "disciplined"
        elif not high_stim and not high_dom:
            return "traditionalist"
        elif high_dom and high_stim:
            if high_bal:
                return "pioneer"
            else:
                return "performer"
        elif high_stim:
            if high_bal:
                return "adventurer"
            else:
                return "hedonist"
        elif high_bal and not high_dom:
            return "harmonizer"
        else:
            return "balanced"  # Default for ambiguous cases

    @staticmethod
    def generate_personality_context(personality_scores) -> ContextInjection:
        """
        Generiert personality context aus Big Five + Limbic Scores
        """
        
        # Big Five Analysis
        openness = personality_scores.openness
        conscientiousness = personality_scores.conscientiousness
        extraversion = personality_scores.extraversion
        agreeableness = personality_scores.agreeableness
        neuroticism = personality_scores.neuroticism
        
        # Generate personality context text
        personality_traits = []
        
        if openness >= 4.0:
            personality_traits.append("highly creative and open to new experiences")
        elif openness <= 2.0:
            personality_traits.append("prefers traditional and familiar choices")
        
        if conscientiousness >= 4.0:
            personality_traits.append("values quality and organization")
        elif conscientiousness <= 2.0:
            personality_traits.append("prefers flexible and spontaneous options")
        
        if extraversion >= 4.0:
            personality_traits.append("enjoys social and energetic experiences")
        elif extraversion <= 2.0:
            personality_traits.append("prefers private and calm activities")
        
        if agreeableness >= 4.0:
            personality_traits.append("values harmony and cooperation")
        elif agreeableness <= 2.0:
            personality_traits.append("prefers individual achievements")
        
        if neuroticism >= 4.0:
            personality_traits.append("benefits from comfort and stress-relief")
        elif neuroticism <= 2.0:
            personality_traits.append("comfortable with challenging experiences")
        
        personality_context = f"This person is {', '.join(personality_traits)}."
        
        # Generate preference context
        preference_context = "Gift preferences align with personality traits and emotional needs."
        
        return ContextInjection(
            personality_context=personality_context,
            preference_context=preference_context,
            occasion_context="",
            budget_context="",
            relationship_context="",
            cultural_context=""
        )

# =============================================================================
# TEMPLATE FACTORY FOR BIG FIVE + LIMBIC
# =============================================================================

class BigFiveLimbicPromptFactory:
    """
    Factory für intelligente Big Five + Limbic Template Auswahl
    """
    
    @staticmethod
    def get_optimal_template(
        model_type: AIModelType,
        complexity_required: PromptComplexity,
        focus_emotional: bool = False
    ) -> ChainOfThoughtTemplate:
        """
        Wählt optimales Big Five + Limbic Template
        """
        
        if focus_emotional and model_type == AIModelType.ANTHROPIC_CLAUDE:
            return EmotionalIntelligenceTemplate.create_template()
        
        elif complexity_required == PromptComplexity.EXPERT and model_type == AIModelType.OPENAI_GPT4:
            return ComprehensiveBigFiveLimbicTemplate.create_template()
        
        elif model_type == AIModelType.GROQ_MIXTRAL:
            return QuickBigFiveLimbicTemplate.create_template()
        
        else:
            # Default to comprehensive analysis
            return ComprehensiveBigFiveLimbicTemplate.create_template()
    
    @staticmethod
    def build_personality_prompt(
        template: ChainOfThoughtTemplate,
        big_five_scores: Dict[str, float],
        limbic_scores: Dict[str, float],
        context: Optional[ContextInjection] = None
    ) -> str:
        """
        Baut finalen Big Five + Limbic Prompt aus Template + Scores
        """
        
        # System Prompt
        final_prompt = f"SYSTEM: {template.system_prompt}\n\n"
        
        # Context if provided
        if context:
            final_prompt += f"CONTEXT: {context.personality_context}\n\n"
        
        # Big Five Data
        final_prompt += "BIG FIVE PERSONALITY DATA:\n"
        final_prompt += f"Openness to Experience: {big_five_scores.get('openness', 'N/A')}/5\n"
        final_prompt += f"Conscientiousness: {big_five_scores.get('conscientiousness', 'N/A')}/5\n"
        final_prompt += f"Extraversion: {big_five_scores.get('extraversion', 'N/A')}/5\n"
        final_prompt += f"Agreeableness: {big_five_scores.get('agreeableness', 'N/A')}/5\n"
        final_prompt += f"Neuroticism: {big_five_scores.get('neuroticism', 'N/A')}/5\n\n"
        
        # Limbic Data
        final_prompt += "LIMBIC SYSTEM DATA:\n"
        final_prompt += f"Stimulanz (Stimulation-Seeking): {limbic_scores.get('stimulanz', 'N/A')}/5\n"
        final_prompt += f"Dominanz (Dominance-Seeking): {limbic_scores.get('dominanz', 'N/A')}/5\n"
        final_prompt += f"Balance (Harmony-Seeking): {limbic_scores.get('balance', 'N/A')}/5\n\n"
        
        # Limbic Type
        limbic_type = BigFiveLimbicMapper.determine_limbic_type(limbic_scores)
        final_prompt += f"Determined Limbic Type: {limbic_type}\n\n"
        
        # Chain-of-Thought Steps
        final_prompt += f"{template.step_connector}\n\n"
        
        for step in template.reasoning_steps:
            # Variable substitution in step prompts
            step_prompt = step.step_prompt.format(
                openness=big_five_scores.get('openness', 'N/A'),
                conscientiousness=big_five_scores.get('conscientiousness', 'N/A'),
                extraversion=big_five_scores.get('extraversion', 'N/A'),
                agreeableness=big_five_scores.get('agreeableness', 'N/A'),
                neuroticism=big_five_scores.get('neuroticism', 'N/A'),
                stimulanz=limbic_scores.get('stimulanz', 'N/A'),
                dominanz=limbic_scores.get('dominanz', 'N/A'),
                balance=limbic_scores.get('balance', 'N/A')
            )
            
            final_prompt += f"Step {step.step_number}: {step.step_description}\n"
            final_prompt += f"{step_prompt}\n\n"
        
        # Conclusion
        final_prompt += f"{template.conclusion_prompt}\n\n"
        
        # Output Format
        if template.output_format_instructions:
            final_prompt += f"OUTPUT FORMAT:\n{template.output_format_instructions}\n\n"
        
        return final_prompt


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BigFiveLimbicAnalysisSteps',
    'ComprehensiveBigFiveLimbicTemplate',
    'QuickBigFiveLimbicTemplate', 
    'EmotionalIntelligenceTemplate',
    'BigFiveLimbicMapper',
    'BigFiveLimbicPromptFactory'
]