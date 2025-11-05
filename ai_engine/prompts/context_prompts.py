"""
Context-Aware Prompts - Dynamic Situational Adaptation
=====================================================

Intelligente Prompt-Anpassung basierend auf Kontext, Situation und Constraints.
Kombiniert Persönlichkeit mit situativen Faktoren für optimale Empfehlungen.

Features:
- Occasion-specific prompt adaptation
- Budget-sensitive recommendations  
- Time-constraint optimization
- Cultural context integration
- Relationship-appropriate suggestions
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

from ai_engine.schemas import (
    DynamicPromptTemplate,
    ContextInjection,
    PromptTechnique,
    PromptComplexity,
    AIModelType,
    PromptOptimizationGoal,
    GiftOccasion,
    RelationshipType,
    AgeGroup,
    GenderIdentity
)


# =============================================================================
# CONTEXTUAL SITUATION TYPES
# =============================================================================

class TimeConstraint(str, Enum):
    """Zeitliche Beschränkungen für Geschenkauswahl"""
    MONTHS_AHEAD = "months_ahead"      # 2+ Monate Vorlauf
    WEEKS_AHEAD = "weeks_ahead"        # 2-8 Wochen Vorlauf
    LAST_MINUTE = "last_minute"        # 1-7 Tage
    EMERGENCY = "emergency"            # Heute oder morgen
    DIGITAL_INSTANT = "digital_instant" # Sofort verfügbar (digital)


class BudgetContext(str, Enum):
    """Budget-Situationen die Empfehlungen beeinflussen"""
    UNLIMITED = "unlimited"            # Geld spielt keine Rolle
    GENEROUS = "generous"              # €200+ Budget
    MODERATE = "moderate"              # €50-200 Budget  
    TIGHT = "tight"                    # €20-50 Budget
    MINIMAL = "minimal"                # Unter €20
    FREE_OR_HANDMADE = "free_or_handmade" # Kein Budget für Kauf


class SeasonalContext(str, Enum):
    """Jahreszeitliche Kontexte"""
    SPRING = "spring"                  # März-Mai
    SUMMER = "summer"                  # Juni-August
    AUTUMN = "autumn"                  # September-November
    WINTER = "winter"                  # Dezember-Februar
    HOLIDAY_SEASON = "holiday_season"  # Nov-Jan (Feiertage)


class LocationContext(str, Enum):
    """Wo findet die Geschenkübergabe statt?"""
    HOME_PRIVATE = "home_private"      # Zuhause, privat
    OFFICE_WORKPLACE = "office_workplace" # Arbeitsplatz
    RESTAURANT_PUBLIC = "restaurant_public" # Restaurant, öffentlich
    PARTY_EVENT = "party_event"        # Party oder Event
    VIRTUAL_ONLINE = "virtual_online"  # Online-Übergabe
    SURPRISE_DELIVERY = "surprise_delivery" # Überraschungslieferung


class EmotionalContext(str, Enum):
    """Emotionale Situation des Anlasses"""
    CELEBRATION = "celebration"        # Fröhliche Feier
    MILESTONE = "milestone"            # Wichtiger Meilenstein  
    COMFORT = "comfort"                # Trost spenden
    APOLOGY = "apology"                # Entschuldigung
    GRATITUDE = "gratitude"            # Dankbarkeit ausdrücken
    ENCOURAGEMENT = "encouragement"    # Ermutigung geben
    ROMANTIC = "romantic"              # Romantische Geste


# =============================================================================
# DYNAMIC CONTEXT ANALYSIS
# =============================================================================

class ContextualSituationAnalyzer:
    """
    Analysiert komplexe Situationen und erstellt optimierte Prompt-Kontexte
    
    Kombiniert multiple Kontext-Dimensionen für intelligente Empfehlungen
    """
    
    @staticmethod
    def analyze_time_sensitivity(
        occasion_date: Optional[date],
        current_date: Optional[date] = None
    ) -> TimeConstraint:
        """
        Bestimmt zeitliche Dringlichkeit basierend auf Datum
        """
        if not occasion_date:
            return TimeConstraint.WEEKS_AHEAD  # Default
        
        if not current_date:
            current_date = date.today()
        
        days_until = (occasion_date - current_date).days
        
        if days_until <= 1:
            return TimeConstraint.EMERGENCY
        elif days_until <= 7:
            return TimeConstraint.LAST_MINUTE
        elif days_until <= 14:
            return TimeConstraint.WEEKS_AHEAD
        else:
            return TimeConstraint.MONTHS_AHEAD
    
    @staticmethod
    def analyze_budget_pressure(
        budget_min: Optional[Decimal],
        budget_max: Optional[Decimal],
        relationship: RelationshipType
    ) -> BudgetContext:
        """
        Analysiert Budget-Kontext basierend auf Beziehung und verfügbaren Mitteln
        """
        if not budget_max or budget_max == 0:
            return BudgetContext.FREE_OR_HANDMADE
        
        # Relationship-adjusted expectations
        relationship_expectations = {
            RelationshipType.SPOUSE: 100,
            RelationshipType.ROMANTIC_PARTNER: 80,
            RelationshipType.FAMILY_PARENT: 60,
            RelationshipType.FRIEND_CLOSE: 40,
            RelationshipType.FAMILY_SIBLING: 35,
            RelationshipType.FRIEND_CASUAL: 25,
            RelationshipType.COLLEAGUE: 20,
            RelationshipType.BOSS: 30,
            RelationshipType.ACQUAINTANCE: 15
        }
        
        expected_budget = relationship_expectations.get(relationship, 30)
        
        if budget_max >= expected_budget * 3:
            return BudgetContext.UNLIMITED
        elif budget_max >= expected_budget * 1.5:
            return BudgetContext.GENEROUS
        elif budget_max >= expected_budget * 0.8:
            return BudgetContext.MODERATE
        elif budget_max >= expected_budget * 0.4:
            return BudgetContext.TIGHT
        else:
            return BudgetContext.MINIMAL
    
    @staticmethod
    def determine_seasonal_context(occasion_date: Optional[date] = None) -> SeasonalContext:
        """
        Bestimmt jahreszeitlichen Kontext
        """
        if not occasion_date:
            occasion_date = date.today()
        
        month = occasion_date.month
        
        if month in [12, 1, 2]:
            if month == 12 or (month == 1 and occasion_date.day <= 15):
                return SeasonalContext.HOLIDAY_SEASON
            return SeasonalContext.WINTER
        elif month in [3, 4, 5]:
            return SeasonalContext.SPRING
        elif month in [6, 7, 8]:
            return SeasonalContext.SUMMER
        else:  # 9, 10, 11
            return SeasonalContext.AUTUMN
    
    @staticmethod
    def infer_emotional_context(
        occasion: GiftOccasion,
        relationship: RelationshipType
    ) -> EmotionalContext:
        """
        Leitet emotionalen Kontext aus Anlass und Beziehung ab
        """
        occasion_emotions = {
            GiftOccasion.BIRTHDAY: EmotionalContext.CELEBRATION,
            GiftOccasion.CHRISTMAS: EmotionalContext.CELEBRATION,
            GiftOccasion.ANNIVERSARY: EmotionalContext.ROMANTIC if relationship in [RelationshipType.SPOUSE, RelationshipType.ROMANTIC_PARTNER] else EmotionalContext.MILESTONE,
            GiftOccasion.WEDDING: EmotionalContext.MILESTONE,
            GiftOccasion.GRADUATION: EmotionalContext.MILESTONE,
            GiftOccasion.VALENTINES: EmotionalContext.ROMANTIC,
            GiftOccasion.MOTHERS_DAY: EmotionalContext.GRATITUDE,
            GiftOccasion.FATHERS_DAY: EmotionalContext.GRATITUDE,
            GiftOccasion.JUST_BECAUSE: EmotionalContext.ENCOURAGEMENT,
            GiftOccasion.APOLOGY: EmotionalContext.APOLOGY,
            GiftOccasion.CONGRATULATIONS: EmotionalContext.CELEBRATION
        }
        
        return occasion_emotions.get(occasion, EmotionalContext.CELEBRATION)


# =============================================================================
# CONTEXT-SPECIFIC PROMPT TEMPLATES
# =============================================================================

class AgeGroupTemplate:
    """
    Template für altersgruppenspezifische Geschenkempfehlungen
    
    Fokus: Altersgerechte Interessen, Entwicklungsstufen und Sicherheit
    """
    
    @staticmethod
    def create_template() -> DynamicPromptTemplate:
        return DynamicPromptTemplate(
            template_name="age_group_specific_gift_recommendations",
            template_version="1.0",
            description="Age-appropriate gift recommendations considering developmental stages and interests",
            
            # Age Group Configuration
            technique=PromptTechnique.DYNAMIC_GENERATION,
            complexity=PromptComplexity.MODERATE,
            target_model=AIModelType.ANTHROPIC_CLAUDE,  # Good at understanding age-appropriate content
            optimization_goal=PromptOptimizationGoal.QUALITY,
            
            # System Prompt
            system_prompt="""
            You are an expert in age-appropriate gift selection who understands developmental stages, interests, and safety considerations.
            
            Your expertise includes:
            - Age-specific interests and hobbies
            - Developmental milestones and learning needs
            - Safety considerations for different age groups
            - Age-appropriate complexity and engagement levels
            - Cultural and generational preferences within age groups
            """,
            
            # Age Group Instruction
            instruction_prompt="""
            AGE GROUP CONTEXT: {age_group}
            Recipient Age: {recipient_age}
            Developmental Stage: {developmental_stage}
            
            Consider age-specific factors:
            
            CHILD (0-12 years):
            - Educational and developmental value
            - Safety and durability requirements
            - Age-appropriate complexity
            - Interactive and engaging elements
            - Parent approval considerations
            
            TEENAGER (13-17 years):
            - Identity formation and self-expression
            - Technology and social media integration
            - Independence and personal interests
            - Peer influence and trends
            - Educational and career exploration
            
            YOUNG_ADULT (18-25 years):
            - Independence and life transitions
            - Career and education focus
            - Social connections and experiences
            - Personal development and hobbies
            - Budget-conscious but quality-focused
            
            ADULT (26-45 years):
            - Career and family responsibilities
            - Quality and practicality preferences
            - Time-saving and convenience needs
            - Personal interests and hobbies
            - Investment in lasting value
            
            MIDDLE_AGED (46-65 years):
            - Life experience and wisdom appreciation
            - Health and wellness focus
            - Legacy and meaningful experiences
            - Comfort and quality preferences
            - Technology adaptation considerations
            
            SENIOR (65+ years):
            - Accessibility and ease of use
            - Health and wellness priorities
            - Legacy and family connections
            - Comfort and safety considerations
            - Technology that's easy to use
            
            Always consider safety, appropriateness, and developmental value for the specific age group.
            """,
            
            # Age Group Variables
            variable_placeholders={
                "age_group": "Specific age group (child, teenager, young_adult, adult, middle_aged, senior)",
                "recipient_age": "Exact age of recipient",
                "developmental_stage": "Key developmental characteristics for this age",
                "interests_typical": "Typical interests for this age group"
            },
            
            # Age-Specific Sections
            conditional_sections={
                "safety_considerations": "Include safety and appropriateness guidelines",
                "developmental_value": "Explain how gift supports development",
                "parent_guidance": "Include guidance for parents when relevant",
                "accessibility_features": "Consider accessibility for older recipients"
            },
            
            # Adaptation Settings
            personality_adaptation=True,   # Personality still important within age group
            relationship_adaptation=True,  # Relationships matter at all ages
            occasion_adaptation=True,      # Occasions have age-specific meanings
            
            # Quality Settings
            max_tokens=2500,
            temperature=0.6  # Balanced creativity and appropriateness
        )


class GenderIdentityTemplate:
    """
    Template für gender-inklusive Geschenkempfehlungen
    
    Fokus: Respektvolle, nicht-stereotypische Empfehlungen
    """
    
    @staticmethod
    def create_template() -> DynamicPromptTemplate:
        return DynamicPromptTemplate(
            template_name="gender_inclusive_gift_recommendations",
            template_version="1.0",
            description="Gender-inclusive gift recommendations that avoid stereotypes and respect identity",
            
            # Gender Identity Configuration
            technique=PromptTechnique.DYNAMIC_GENERATION,
            complexity=PromptComplexity.MODERATE,
            target_model=AIModelType.ANTHROPIC_CLAUDE,  # Best for inclusive and sensitive content
            optimization_goal=PromptOptimizationGoal.QUALITY,
            
            # System Prompt
            system_prompt="""
            You are a gender-inclusive gift consultant who respects all gender identities and avoids stereotypes.
            
            Your approach:
            - Focus on individual interests, not gender assumptions
            - Respect stated gender identity preferences
            - Avoid traditional gender stereotypes
            - Consider cultural and personal context
            - Provide inclusive and respectful recommendations
            """,
            
            # Gender Identity Instruction
            instruction_prompt="""
            GENDER IDENTITY CONTEXT: {gender_identity}
            Personal Preferences: {personal_preferences}
            Cultural Context: {cultural_context}
            
            Consider gender-inclusive factors:
            
            FEMALE:
            - Focus on individual interests, not assumptions
            - Consider career, hobbies, and personal style
            - Respect professional and personal achievements
            - Avoid stereotypical "girly" assumptions
            
            MALE:
            - Focus on individual interests, not assumptions
            - Consider career, hobbies, and personal style
            - Respect professional and personal achievements
            - Avoid stereotypical "masculine" assumptions
            
            NON_BINARY:
            - Focus entirely on individual interests and preferences
            - Avoid gender-specific assumptions
            - Consider unique personal style and expression
            - Respect chosen identity and preferences
            
            PREFER_NOT_TO_SAY:
            - Focus entirely on personality and interests
            - Avoid any gender-based assumptions
            - Use neutral and inclusive language
            - Respect privacy and comfort level
            
            GUIDELINES:
            - Always prioritize individual interests over gender
            - Use inclusive language in descriptions
            - Consider the person's stated preferences
            - Avoid traditional gender stereotypes
            - Focus on what makes the gift meaningful to this specific person
            
            Explain why each recommendation is suitable for this individual, not their gender.
            """,
            
            # Gender Identity Variables
            variable_placeholders={
                "gender_identity": "Stated gender identity (female, male, non_binary, prefer_not_to_say)",
                "personal_preferences": "Individual interests and preferences",
                "cultural_context": "Cultural considerations for gender expression",
                "comfort_level": "Comfort level with gender-specific items"
            },
            
            # Gender-Inclusive Sections
            conditional_sections={
                "inclusive_language": "Use inclusive and respectful language",
                "individual_focus": "Focus on individual interests and achievements",
                "stereotype_avoidance": "Explicitly avoid gender stereotypes",
                "personal_meaning": "Explain personal meaning over gender assumptions"
            },
            
            # Adaptation Settings
            personality_adaptation=True,   # Personality is more important than gender
            relationship_adaptation=True,  # Relationships matter regardless of gender
            occasion_adaptation=True,      # Occasions are gender-neutral
            
            # Quality Settings
            max_tokens=2500,
            temperature=0.5  # Balanced and respectful
        )


class EmergencyGiftTemplate:
    """
    Template für Notfall-Geschenkempfehlungen
    
    Fokus: Sofort verfügbar, trotzdem durchdacht
    """
    
    @staticmethod
    def create_template() -> DynamicPromptTemplate:
        return DynamicPromptTemplate(
            template_name="emergency_gift_recommendations",
            template_version="1.8",
            description="Last-minute gift solutions that still show thoughtfulness",
            
            # Emergency Configuration
            technique=PromptTechnique.DYNAMIC_GENERATION,
            complexity=PromptComplexity.MODERATE,
            target_model=AIModelType.GROQ_MIXTRAL,  # Speed required
            optimization_goal=PromptOptimizationGoal.SPEED,
            
            # System Prompt
            system_prompt="""
            You are an expert in last-minute gift solutions that maintain thoughtfulness.
            Your specialty: Finding meaningful gifts available immediately or within 24 hours.
            """,
            
            # Emergency Instruction
            instruction_prompt="""
            EMERGENCY GIFT SITUATION: {time_constraint}
            
            Focus on gifts that are:
            1. Available TODAY or with next-day delivery
            2. Still thoughtful and personalized
            3. Appropriate for the relationship and occasion
            4. Can be presented beautifully despite time constraints
            5. Age-appropriate and safe for the recipient
            6. Respectful of gender identity and personal preferences
            
            Prioritize:
            - Digital gifts (immediate delivery)
            - Local store pickup options
            - Experience vouchers (printable)
            - Subscription services
            - Same-day delivery options
            
            AGE CONSIDERATIONS:
            - Ensure age-appropriate content and safety
            - Consider developmental stage and interests
            - Respect age-specific preferences and needs
            
            GENDER INCLUSIVITY:
            - Focus on individual interests, not gender assumptions
            - Use inclusive language and respectful recommendations
            - Avoid stereotypes and respect stated preferences
            
            Provide immediate solutions with backup options.
            """,
            
            # Dynamic Variables
            variable_placeholders={
                "time_constraint": "How much time is available (emergency, last_minute, etc.)",
                "location_context": "Where gift will be given",
                "budget_pressure": "Budget constraints",
                "relationship_appropriateness": "What's appropriate for this relationship"
            },
            
            # Conditional Sections
            conditional_sections={
                "digital_focus": "If time_constraint == 'emergency', emphasize digital and instant options",
                "local_pickup": "If time_constraint == 'last_minute', include local store options",
                "presentation_tips": "Quick gift wrapping and presentation ideas for rushed situations"
            },
            
            # Emergency Adaptations
            personality_adaptation=False,  # Less personality focus due to time
            relationship_adaptation=True,  # Still important for appropriateness
            occasion_adaptation=True,      # Critical for emergency context
            
            # Speed Settings
            max_tokens=1500,
            temperature=0.4  # Focused and practical
        )


class BudgetConstrainedTemplate:
    """
    Template für budget-bewusste Geschenkempfehlungen
    
    Fokus: Maximaler Wert bei minimalem Budget
    """
    
    @staticmethod
    def create_template() -> DynamicPromptTemplate:
        return DynamicPromptTemplate(
            template_name="budget_conscious_gift_recommendations",
            template_version="2.1",
            description="Thoughtful gifts that maximize impact within budget constraints",
            
            # Budget Configuration
            technique=PromptTechnique.DYNAMIC_GENERATION,
            complexity=PromptComplexity.MODERATE,
            target_model=AIModelType.ANTHROPIC_CLAUDE,  # Thoughtful recommendations
            optimization_goal=PromptOptimizationGoal.BALANCE,
            
            # System Prompt
            system_prompt="""
            You are a budget-conscious gift expert who believes meaningful gifts don't require high prices.
            Your expertise: Finding creative, thoughtful solutions within tight budgets.
            
            Core principles:
            - Creativity over cost
            - Personal meaning over price tags
            - Handmade and DIY solutions
            - Experience gifts over material items
            - Thoughtful presentation regardless of budget
            """,
            
            # Budget-Focused Instruction
            instruction_prompt="""
            BUDGET-CONSCIOUS GIFT SITUATION:
            Budget Category: {budget_context}
            Maximum Budget: {budget_max}
            
            Focus on maximizing thoughtfulness and impact:
            
            For MINIMAL budgets (under €20):
            - Handmade options with personal touch
            - Digital creations (playlists, photo albums)
            - Small but meaningful items
            - Acts of service as gifts
            
            For TIGHT budgets (€20-50):
            - Creative combinations of small items
            - Experience gifts that don't cost much
            - Personalized items within budget
            - Group gifts for larger items
            
            For MODERATE budgets (€50-200):
            - Quality over quantity
            - Meaningful experiences
            - Investment in relationships
            - Items with lasting value
            
            AGE-APPROPRIATE BUDGET CONSIDERATIONS:
            - Children: Focus on educational value and safety
            - Teenagers: Consider technology and social interests
            - Adults: Balance quality with practical needs
            - Seniors: Prioritize accessibility and ease of use
            
            GENDER-INCLUSIVE BUDGET APPROACH:
            - Focus on individual interests regardless of budget
            - Avoid gender-stereotyped budget assumptions
            - Consider personal preferences over traditional categories
            
            Always suggest creative presentation and explain why each gift is special.
            """,
            
            # Budget Variables
            variable_placeholders={
                "budget_context": "Budget pressure level (minimal, tight, moderate)",
                "budget_max": "Maximum available budget",
                "value_priorities": "What kind of value to prioritize (emotional, practical, experiential)",
                "handmade_feasibility": "Whether handmade options are realistic"
            },
            
            # Budget-Specific Sections
            conditional_sections={
                "diy_options": "Include DIY and handmade alternatives",
                "group_gift_suggestion": "If budget is very tight, suggest group gifting",
                "free_additions": "Always include free ways to enhance any gift",
                "presentation_on_budget": "Beautiful presentation ideas that cost little"
            },
            
            # Adaptation Settings
            personality_adaptation=True,   # Important for meaningful gifts
            relationship_adaptation=True,  # Critical for appropriateness
            occasion_adaptation=True,      # Helps justify budget choices
            
            # Thoughtful Settings
            max_tokens=2500,
            temperature=0.7  # Creative solutions needed
        )


class CulturalContextTemplate:
    """
    Template für kulturell-sensitive Geschenkempfehlungen
    
    Fokus: Kulturelle Angemessenheit und Respekt
    """
    
    @staticmethod
    def create_template() -> DynamicPromptTemplate:
        return DynamicPromptTemplate(
            template_name="culturally_aware_gift_recommendations",
            template_version="1.6",
            description="Culturally sensitive gift recommendations that respect traditions and values",
            
            # Cultural Configuration
            technique=PromptTechnique.DYNAMIC_GENERATION,
            complexity=PromptComplexity.COMPLEX,
            target_model=AIModelType.ANTHROPIC_CLAUDE,  # Best for cultural sensitivity
            optimization_goal=PromptOptimizationGoal.QUALITY,
            
            # Cultural System Prompt
            system_prompt="""
            You are a culturally aware gift consultant with deep understanding of:
            - Gift-giving traditions across cultures
            - Religious considerations in gift selection
            - Cultural taboos and sensitivities
            - Respectful ways to honor cultural backgrounds
            - Universal gestures of thoughtfulness
            
            Your approach is always respectful, inclusive, and culturally informed.
            """,
            
            # Cultural Instruction
            instruction_prompt="""
            CULTURAL CONTEXT AWARENESS:
            Cultural Background: {cultural_context}
            Religious Considerations: {religious_context}
            Regional Customs: {regional_context}
            
            Consider these cultural factors:
            
            1. GIFT-GIVING TRADITIONS:
               - What are appropriate gifts in this culture?
               - Are there traditional gift-giving customs to honor?
               - What colors, numbers, or symbols have special meaning?
            
            2. CULTURAL SENSITIVITIES:
               - What should be avoided due to cultural or religious reasons?
               - Are there taboos around certain materials or items?
               - How does the cultural context affect timing and presentation?
            
            3. RESPECTFUL INTEGRATION:
               - How can we honor the cultural background while being personally meaningful?
               - What universal human values can we celebrate?
               - How do we show respect without appropriation?
            
            4. INCLUSIVE RECOMMENDATIONS:
               - Ensure recommendations work across cultural contexts
               - Explain cultural significance when relevant
               - Offer alternatives that respect different comfort levels
            
            Always explain cultural considerations and provide respectful alternatives.
            """,
            
            # Cultural Variables
            variable_placeholders={
                "cultural_context": "Cultural background and traditions",
                "religious_context": "Religious considerations",
                "regional_context": "Regional customs and practices",
                "acculturation_level": "How traditional vs. modern the approach should be"
            },
            
            # Cultural Sections
            conditional_sections={
                "traditional_options": "Include traditional gift ideas that honor heritage",
                "modern_fusion": "Blend traditional and contemporary approaches",
                "universal_appeal": "Focus on universally appropriate gifts",
                "cultural_education": "Explain cultural significance of recommendations"
            },
            
            # Cultural Adaptation
            personality_adaptation=True,   # Still important individual consideration
            relationship_adaptation=True,  # Relationships have cultural context
            occasion_adaptation=True,      # Occasions may have cultural significance
            
            # Quality Settings
            max_tokens=3500,
            temperature=0.5  # Balanced and respectful
        )


class SeasonalContextTemplate:
    """
    Template für jahreszeitlich angepasste Geschenkempfehlungen
    
    Fokus: Saisonale Angemessenheit und Verfügbarkeit
    """
    
    @staticmethod
    def create_template() -> DynamicPromptTemplate:
        return DynamicPromptTemplate(
            template_name="seasonal_gift_recommendations", 
            template_version="2.2",
            description="Season-appropriate gifts that align with weather, activities, and mood",
            
            # Seasonal Configuration
            technique=PromptTechnique.DYNAMIC_GENERATION,
            complexity=PromptComplexity.MODERATE,
            target_model=AIModelType.OPENAI_GPT4,  # Good at seasonal reasoning
            optimization_goal=PromptOptimizationGoal.BALANCE,
            
            # Seasonal System Prompt
            system_prompt="""
            You are a seasonal gift expert who understands how weather, activities, and seasonal moods affect gift preferences.
            
            Your expertise includes:
            - Seasonal activity patterns and preferences
            - Weather-appropriate gift selections
            - Holiday and seasonal traditions
            - Seasonal availability of products and experiences
            - Mood and energy changes throughout the year
            """,
            
            # Seasonal Instruction
            instruction_prompt="""
            SEASONAL CONTEXT: {seasonal_context}
            Current Season: {current_season}
            Holiday Context: {holiday_context}
            
            Consider seasonal factors:
            
            SPRING (March-May):
            - Renewal and fresh starts
            - Outdoor activities resuming
            - Lighter, brighter preferences
            - Spring cleaning and organization
            - Gardening and nature themes
            
            SUMMER (June-August):
            - Outdoor experiences and travel
            - Vacation and leisure focus
            - Cooling and comfort items
            - Social gatherings and parties
            - Active and adventure themes
            
            AUTUMN (September-November):
            - Back-to-school and new beginnings
            - Cozy and comfort preferences
            - Harvest and gratitude themes
            - Preparation for winter
            - Warm colors and textures
            
            WINTER (December-February):
            - Indoor activities and comfort
            - Holiday season (if applicable)
            - Warmth and coziness priorities
            - Reflection and planning themes
            - Light and brightness to combat darkness
            
            HOLIDAY SEASON (November-January):
            - Traditional holiday considerations
            - Gift-giving season dynamics
            - Family gathering themes
            - Year-end reflection and new year planning
            
            Align recommendations with seasonal energy and practical needs.
            """,
            
            # Seasonal Variables
            variable_placeholders={
                "seasonal_context": "Current season and its characteristics",
                "current_season": "Specific season (spring, summer, autumn, winter)",
                "holiday_context": "Relevant holidays and seasonal celebrations",
                "weather_considerations": "Weather-related factors affecting gift use"
            },
            
            # Seasonal Sections
            conditional_sections={
                "weather_appropriate": "Ensure gifts work with current weather",
                "seasonal_activities": "Align with popular seasonal activities",
                "holiday_integration": "Consider holiday themes if relevant",
                "seasonal_availability": "Focus on seasonally available items"
            },
            
            # Adaptation Settings
            personality_adaptation=True,   # Personality still important
            relationship_adaptation=True,  # Relationships unchanged by season
            occasion_adaptation=True,      # Occasions may have seasonal elements
            
            # Balanced Settings
            max_tokens=2500,
            temperature=0.6  # Creative seasonal thinking
        )


# =============================================================================
# CONTEXTUAL PROMPT FACTORY
# =============================================================================

class ContextualPromptFactory:
    """
    Intelligente Factory für kontextuelle Prompt-Auswahl
    
    Analysiert Situation und wählt optimales Template
    """
    
    @staticmethod
    def analyze_situation_complexity(
        time_constraint: TimeConstraint,
        budget_context: BudgetContext,
        cultural_complexity: bool = False,
        multiple_constraints: bool = False
    ) -> PromptComplexity:
        """
        Bestimmt erforderliche Prompt-Komplexität basierend auf Situation
        """
        complexity_factors = 0
        
        if time_constraint in [TimeConstraint.EMERGENCY, TimeConstraint.LAST_MINUTE]:
            complexity_factors += 1
        
        if budget_context in [BudgetContext.MINIMAL, BudgetContext.FREE_OR_HANDMADE]:
            complexity_factors += 1
        
        if cultural_complexity:
            complexity_factors += 2
        
        if multiple_constraints:
            complexity_factors += 1
        
        if complexity_factors >= 4:
            return PromptComplexity.EXPERT
        elif complexity_factors >= 2:
            return PromptComplexity.COMPLEX
        else:
            return PromptComplexity.MODERATE
    
    @staticmethod
    def select_optimal_template(
        time_constraint: TimeConstraint,
        budget_context: BudgetContext,
        seasonal_context: SeasonalContext,
        cultural_considerations: bool = False,
        age_group: Optional[AgeGroup] = None,
        gender_identity: Optional[GenderIdentity] = None,
        primary_optimization: PromptOptimizationGoal = PromptOptimizationGoal.BALANCE
    ) -> DynamicPromptTemplate:
        """
        Wählt optimales Template basierend auf situativen Faktoren
        
        🚀 ENHANCED: Age Group und Gender Identity Integration
        """
        
        # Emergency situations override other considerations
        if time_constraint in [TimeConstraint.EMERGENCY, TimeConstraint.LAST_MINUTE]:
            return EmergencyGiftTemplate.create_template()
        
        # Age Group considerations are high priority for children and seniors
        if age_group in [AgeGroup.CHILD, AgeGroup.SENIOR]:
            return AgeGroupTemplate.create_template()
        
        # Cultural considerations are high priority
        if cultural_considerations:
            return CulturalContextTemplate.create_template()
        
        # Gender identity considerations for non-binary and prefer_not_to_say
        if gender_identity in [GenderIdentity.NON_BINARY, GenderIdentity.PREFER_NOT_TO_SAY]:
            return GenderIdentityTemplate.create_template()
        
        # Budget constraints need special handling
        if budget_context in [BudgetContext.MINIMAL, BudgetContext.TIGHT]:
            return BudgetConstrainedTemplate.create_template()
        
        # Seasonal considerations for normal situations
        return SeasonalContextTemplate.create_template()
    
    @staticmethod
    def build_comprehensive_context(
        occasion: GiftOccasion,
        relationship: RelationshipType,
        occasion_date: Optional[date] = None,
        budget_min: Optional[Decimal] = None,
        budget_max: Optional[Decimal] = None,
        cultural_background: Optional[str] = None,
        location_context: Optional[LocationContext] = None,
        age_group: Optional[AgeGroup] = None,
        gender_identity: Optional[GenderIdentity] = None,
        recipient_age: Optional[int] = None
    ) -> ContextInjection:
        """
        Erstellt umfassenden Kontext aus allen verfügbaren Informationen
        
        🚀 ENHANCED: Age Group und Gender Identity Integration
        """
        
        # Analyze contextual factors
        time_constraint = ContextualSituationAnalyzer.analyze_time_sensitivity(occasion_date)
        budget_context = ContextualSituationAnalyzer.analyze_budget_pressure(budget_min, budget_max, relationship)
        seasonal_context = ContextualSituationAnalyzer.determine_seasonal_context(occasion_date)
        emotional_context = ContextualSituationAnalyzer.infer_emotional_context(occasion, relationship)
        
        # Build context strings
        occasion_context_parts = [
            f"Occasion: {occasion.value}",
            f"Emotional tone: {emotional_context.value}",
            f"Time constraint: {time_constraint.value}",
            f"Season: {seasonal_context.value}"
        ]
        
        budget_context_parts = [
            f"Budget context: {budget_context.value}",
        ]
        if budget_max:
            budget_context_parts.append(f"Maximum budget: €{budget_max}")
        
        relationship_context_parts = [
            f"Relationship: {relationship.value}",
        ]
        if location_context:
            relationship_context_parts.append(f"Gift location: {location_context.value}")
        
        cultural_context_parts = []
        if cultural_background:
            cultural_context_parts.append(f"Cultural background: {cultural_background}")
        
        # 🚀 NEW: Age Group Context
        age_context_parts = []
        if age_group:
            age_context_parts.append(f"Age group: {age_group.value}")
        if recipient_age:
            age_context_parts.append(f"Recipient age: {recipient_age}")
        
        # 🚀 NEW: Gender Identity Context
        gender_context_parts = []
        if gender_identity:
            gender_context_parts.append(f"Gender identity: {gender_identity.value}")
        
        return ContextInjection(
            occasion_context=" | ".join(occasion_context_parts),
            budget_context=" | ".join(budget_context_parts),
            relationship_context=" | ".join(relationship_context_parts),
            cultural_context=" | ".join(cultural_context_parts) if cultural_context_parts else None,
            age_context=" | ".join(age_context_parts) if age_context_parts else None,
            gender_context=" | ".join(gender_context_parts) if gender_context_parts else None
        )
    
    @staticmethod
    def generate_contextual_prompt(
        template: DynamicPromptTemplate,
        context: ContextInjection,
        variables: Dict[str, Any]
    ) -> str:
        """
        Generiert finalen kontextuellen Prompt
        
        Kombiniert Template + Kontext + Variablen zu intelligentem Prompt
        """
        
        # System Prompt
        final_prompt = f"SYSTEM: {template.system_prompt}\n\n"
        
        # Context Integration
        final_prompt += "SITUATIONAL CONTEXT:\n"
        if context.occasion_context:
            final_prompt += f"Occasion: {context.occasion_context}\n"
        if context.budget_context:
            final_prompt += f"Budget: {context.budget_context}\n"
        if context.relationship_context:
            final_prompt += f"Relationship: {context.relationship_context}\n"
        if context.cultural_context:
            final_prompt += f"Cultural: {context.cultural_context}\n"
        if context.age_context:
            final_prompt += f"Age: {context.age_context}\n"
        if context.gender_context:
            final_prompt += f"Gender: {context.gender_context}\n"
        final_prompt += "\n"
        
        # Dynamic Instruction with Variable Substitution
        instruction = template.instruction_prompt
        for placeholder, value in variables.items():
            instruction = instruction.replace(f"{{{placeholder}}}", str(value))
        
        final_prompt += f"INSTRUCTION:\n{instruction}\n\n"
        
        # Conditional Sections based on context
        for section_name, section_content in template.conditional_sections.items():
            # Simple condition evaluation (could be enhanced)
            if any(keyword in str(variables.values()) for keyword in ["emergency", "minimal", "cultural"]):
                final_prompt += f"SPECIAL CONSIDERATION - {section_name}:\n{section_content}\n\n"
        
        final_prompt += "Provide contextually appropriate recommendations that address all situational factors:\n\n"
        
        return final_prompt


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'TimeConstraint',
    'BudgetContext', 
    'SeasonalContext',
    'LocationContext',
    'EmotionalContext',
    
    # Analyzers
    'ContextualSituationAnalyzer',
    
    # Templates
    'AgeGroupTemplate',
    'GenderIdentityTemplate',
    'EmergencyGiftTemplate',
    'BudgetConstrainedTemplate',
    'CulturalContextTemplate', 
    'SeasonalContextTemplate',
    
    # Factory
    'ContextualPromptFactory'
]