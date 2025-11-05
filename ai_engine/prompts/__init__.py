"""
AI Engine Prompts Package
=========================

Advanced Prompt Engineering für SensationGifts AI-System.
Exportiert alle Prompt-Templates, Factories und Advanced Techniques.

Usage:
    # Einfache Imports
    from ai_engine.prompts import GiftPromptFactory, BigFiveLimbicPromptFactory
    from ai_engine.prompts import AdvancedTechniqueOrchestrator
    
    # Convenience Functions
    from ai_engine.prompts import create_optimal_prompt, build_contextual_prompt
"""

# =============================================================================
# GIFT PROMPTS: Few-Shot Learning Templates
# =============================================================================

from .gift_prompts import (
    # Example Collections
    GiftExamples,
    
    # Model-Specific Templates
    OpenAIOptimizedTemplates,
    GroqSpeedTemplates,
    ClaudeReasoningTemplates,
    
    # Context Builders
    ContextualPromptBuilder,
    
    # Main Factory
    GiftPromptFactory,
)

# =============================================================================
# PERSONALITY PROMPTS: Chain-of-Thought Big Five + Limbic Analysis
# =============================================================================

from .personality_prompts import (
    # Analysis Steps
    BigFiveLimbicAnalysisSteps,
    
    # Templates
    ComprehensiveBigFiveLimbicTemplate,
    QuickBigFiveLimbicTemplate, 
    EmotionalIntelligenceTemplate,
    
    # Mapping Logic
    BigFiveLimbicMapper,
    
    # Factory
    BigFiveLimbicPromptFactory
)

# =============================================================================
# CONTEXT PROMPTS: Dynamic Situational Adaptation
# =============================================================================

from .context_prompts import (
    # Enums
    TimeConstraint,
    BudgetContext,
    SeasonalContext,
    LocationContext,
    EmotionalContext,
    
    # Analyzer
    ContextualSituationAnalyzer,
    
    # Templates
    EmergencyGiftTemplate,
    BudgetConstrainedTemplate,
    CulturalContextTemplate,
    SeasonalContextTemplate,
    
    # Factory
    ContextualPromptFactory,
)

# =============================================================================
# ADVANCED TECHNIQUES: State-of-the-Art AI Engineering
# =============================================================================

from .advanced_techniques import (
    # Enums
    AdvancedTechnique,
    ValidationCriteria,
    EnsembleStrategy,
    
    # Engines
    MetaPromptingEngine,
    SelfCorrectionEngine,
    EnsemblePromptingEngine,
    ConstitutionalAIEngine,
    AdaptiveLearningEngine,
    
    # Orchestrator
    AdvancedTechniqueOrchestrator,
)


# =============================================================================
# CONVENIENCE FUNCTIONS: Häufig verwendete Kombinationen
# =============================================================================

def create_optimal_prompt(
    request_type: str,
    model_type: str,
    complexity: str = "moderate",
    optimization_goal: str = "balance"
):
    """
    One-liner für optimale Prompt-Erstellung
    
    Usage:
        prompt = create_optimal_prompt("gift_recommendation", "openai_gpt4", "complex", "quality")
    """
    from ai_engine.schemas import AIModelType, PromptComplexity, PromptOptimizationGoal
    
    # Convert strings to enums
    model_enum = AIModelType(model_type)
    complexity_enum = PromptComplexity(complexity)
    goal_enum = PromptOptimizationGoal(optimization_goal)
    
    if request_type == "gift_recommendation":
        return GiftPromptFactory.get_optimal_template(model_enum, goal_enum, complexity_enum)
    elif request_type == "personality_analysis":
        return BigFiveLimbicPromptFactory.get_optimal_template(model_enum, complexity_enum)
    else:
        raise ValueError(f"Unknown request_type: {request_type}")


def build_contextual_prompt(
    template,
    personality_data=None,
    occasion_context=None,
    relationship_context=None,
    budget_context=None,
    cultural_context=None
):
    """
    One-liner für kontextuelle Prompt-Erstellung
    
    Usage:
        final_prompt = build_contextual_prompt(
            template=my_template,
            personality_data=big_five_limbic_scores,
            occasion_context="birthday_celebration",
            relationship_context="parent_child",
            budget_context="premium",
            cultural_context="german"
        )
    """
    # Build comprehensive context from all parameters
    context = {}
    
    # Add personality context
    if personality_data:
        context['personality'] = ContextualPromptBuilder.build_personality_context(personality_data)
    
    # Add occasion context
    if occasion_context:
        context['occasion'] = occasion_context
    
    # Add relationship context
    if relationship_context:
        context['relationship'] = relationship_context
    
    # Add budget context
    if budget_context:
        context['budget'] = budget_context
    
    # Add cultural context
    if cultural_context:
        context['cultural'] = cultural_context
    
    # Use appropriate factory method with full context
    if hasattr(template, 'build_contextual_prompt'):
        return template.build_contextual_prompt(context)
    else:
        # Fallback to GiftPromptFactory with comprehensive context
        return GiftPromptFactory.build_contextual_prompt(
            template, 
            context.get('personality'), 
            context.get('occasion'),
            context.get('relationship'),
            context.get('budget'),
            context.get('cultural')
        )


def get_advanced_recommendation_system():
    """
    One-liner für vollständiges Advanced AI System
    
    Usage:
        ai_system = get_advanced_recommendation_system()
        result = ai_system.orchestrate_advanced_recommendation(context, techniques, model)
    """
    return AdvancedTechniqueOrchestrator()


# =============================================================================
# PROMPT TEMPLATE REGISTRY: Zentrale Template-Verwaltung
# =============================================================================

class PromptTemplateRegistry:
    """
    Zentrale Registry für alle verfügbaren Prompt-Templates
    
    Ermöglicht easy Zugriff und Verwaltung aller Templates
    """
    
    def __init__(self):
        self._templates = {}
        self._register_default_templates()
    
    def _register_default_templates(self):
        """Registriert alle Standard-Templates"""
        
        # Gift Recommendation Templates
        self._templates["openai_premium_gifts"] = OpenAIOptimizedTemplates.create_premium_gift_template()
        self._templates["groq_speed_gifts"] = GroqSpeedTemplates.create_quick_recommendation_template()
        self._templates["claude_thoughtful_gifts"] = ClaudeReasoningTemplates.create_thoughtful_recommendation_template()
        
        # Personality Analysis Templates
        self._templates["comprehensive_personality"] = ComprehensiveBigFiveLimbicTemplate.create_template()
        self._templates["quick_personality"] = QuickBigFiveLimbicTemplate.create_template()
        self._templates["cultural_personality"] = EmotionalIntelligenceTemplate.create_template()
        
        # Context-Specific Templates
        self._templates["emergency_gifts"] = EmergencyGiftTemplate.create_template()
        self._templates["budget_constrained"] = BudgetConstrainedTemplate.create_template()
        self._templates["cultural_context"] = CulturalContextTemplate.create_template()
        self._templates["seasonal_context"] = SeasonalContextTemplate.create_template()
        
        # Advanced Technique Templates
        self._templates["meta_prompting"] = MetaPromptingEngine.create_meta_prompt_template()
        self._templates["self_validation"] = SelfCorrectionEngine.create_validation_prompt()
        self._templates["self_improvement"] = SelfCorrectionEngine.create_improvement_prompt()
        self._templates["ensemble_coordinator"] = EnsemblePromptingEngine.create_ensemble_coordinator_prompt()
        self._templates["constitutional_ai"] = ConstitutionalAIEngine.create_constitutional_prompt()
        self._templates["adaptive_learning"] = AdaptiveLearningEngine.create_adaptive_prompt_template()
    
    def get_template(self, template_name: str):
        """Holt Template by Name"""
        return self._templates.get(template_name)
    
    def list_templates(self) -> list:
        """Listet alle verfügbaren Templates"""
        return list(self._templates.keys())
    
    def register_custom_template(self, name: str, template):
        """Registriert custom Template"""
        self._templates[name] = template
    
    def get_templates_by_category(self, category: str) -> dict:
        """Holt Templates nach Kategorie"""
        category_filters = {
            "gift_recommendation": ["openai_premium_gifts", "groq_speed_gifts", "claude_thoughtful_gifts"],
            "personality_analysis": ["comprehensive_personality", "quick_personality", "cultural_personality"],
            "contextual": ["emergency_gifts", "budget_constrained", "cultural_context", "seasonal_context"],
            "advanced": ["meta_prompting", "self_validation", "ensemble_coordinator", "constitutional_ai"]
        }
        
        template_names = category_filters.get(category, [])
        return {name: self._templates[name] for name in template_names if name in self._templates}


# Global Registry Instance
_template_registry = PromptTemplateRegistry()

def get_template_registry() -> PromptTemplateRegistry:
    """Zugriff auf globale Template-Registry"""
    return _template_registry


# =============================================================================
# SMART PROMPT SELECTION: Intelligente Template-Auswahl
# =============================================================================

def select_optimal_prompt_strategy(
    scenario_description: str,
    constraints: dict = None,
    optimization_priorities: list = None
):
    """
    Intelligente Auswahl der besten Prompt-Strategie für Szenario
    
    Args:
        scenario_description: Beschreibung der Situation
        constraints: Dict mit Constraints (budget, time, etc.)
        optimization_priorities: Liste von Prioritäten (speed, quality, etc.)
    
    Returns:
        Dict mit empfohlener Strategie und Templates
    """
    constraints = constraints or {}
    optimization_priorities = optimization_priorities or ["balance"]
    
    strategy = {
        "primary_template": None,
        "fallback_template": None,
        "advanced_techniques": [],
        "reasoning": ""
    }
    
    # Analyze scenario for optimal approach
    if "emergency" in scenario_description.lower() or constraints.get("time_constraint") == "emergency":
        strategy["primary_template"] = "groq_speed_gifts"
        strategy["fallback_template"] = "emergency_gifts"
        strategy["reasoning"] = "Emergency situation requires speed optimization"
    
    elif "cultural" in scenario_description.lower() or constraints.get("cultural_context"):
        strategy["primary_template"] = "claude_thoughtful_gifts"
        strategy["advanced_techniques"] = [AdvancedTechnique.CONSTITUTIONAL_AI]
        strategy["reasoning"] = "Cultural context requires thoughtful, sensitive approach"
    
    elif "quality" in optimization_priorities:
        strategy["primary_template"] = "openai_premium_gifts"
        strategy["advanced_techniques"] = [AdvancedTechnique.SELF_CORRECTION, AdvancedTechnique.ENSEMBLE_PROMPTING]
        strategy["reasoning"] = "Quality priority justifies advanced techniques"
    
    else:
        strategy["primary_template"] = "openai_premium_gifts"
        strategy["fallback_template"] = "groq_speed_gifts"
        strategy["reasoning"] = "Balanced approach for general use case"
    
    return strategy


# =============================================================================
# PACKAGE METADATA
# =============================================================================

__version__ = "2.1.0"
__author__ = "SensationGifts AI Team"
__description__ = "Advanced Prompt Engineering für AI-powered Gift Recommendations"

# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================

__all__ = [
    # === FACTORIES & MAIN CLASSES ===
    "GiftPromptFactory",
    "BigFiveLimbicPromptFactory", 
    "ContextualPromptFactory",
    "AdvancedTechniqueOrchestrator",
    
    # === TEMPLATE CLASSES ===
    "OpenAIOptimizedTemplates",
    "GroqSpeedTemplates",
    "ClaudeReasoningTemplates",
    "ComprehensiveBigFiveLimbicTemplate",
    "QuickBigFiveLimbicTemplate",
    "EmotionalIntelligenceTemplate",
    "EmergencyGiftTemplate",
    "BudgetConstrainedTemplate",
    "CulturalContextTemplate",
    "SeasonalContextTemplate",
    
    # === BUILDERS & ANALYZERS ===
    "ContextualPromptBuilder",
    "ContextualSituationAnalyzer",
    "BigFiveLimbicMapper",
    
    # === ADVANCED ENGINES ===
    "MetaPromptingEngine",
    "SelfCorrectionEngine",
    "EnsemblePromptingEngine", 
    "ConstitutionalAIEngine",
    "AdaptiveLearningEngine",
    
    # === EXAMPLE COLLECTIONS ===
    "GiftExamples",
    "BigFiveLimbicAnalysisSteps",
    
    # === ENUMS ===
    "TimeConstraint",
    "BudgetContext",
    "SeasonalContext",
    "LocationContext", 
    "EmotionalContext",
    "AdvancedTechnique",
    "ValidationCriteria",
    "EnsembleStrategy",
    
    # === CONVENIENCE FUNCTIONS ===
    "create_optimal_prompt",
    "build_contextual_prompt",
    "get_advanced_recommendation_system",
    "select_optimal_prompt_strategy",
    
    # === REGISTRY ===
    "PromptTemplateRegistry",
    "get_template_registry",
]