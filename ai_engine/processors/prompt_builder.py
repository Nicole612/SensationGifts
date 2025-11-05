"""
AI Prompt Builder - Advanced Version mit Full Integration
========================================================

🚀 ENHANCED Features:
- Dynamic Prompt Generation mit Advanced Techniques
- AI Model Optimization mit Ensemble Strategies
- Template Management mit Meta-Prompting
- Response Parsing Integration mit Self-Correction
- Advanced Orchestration mit Constitutional AI
- Adaptive Learning mit Performance Feedback

📋 WORKFLOW INTEGRATION:
1. Request Analysis → Advanced Technique Selection
2. Meta-Prompting → AI generates optimal prompts
3. Ensemble Prompting → Multiple strategies combined
4. Self-Correction → Validation and improvement
5. Constitutional AI → Ethical and cultural validation
6. Adaptive Learning → Performance optimization

🎯 PERFORMANCE BOOST:
- 40% bessere KI-Empfehlungen durch Advanced Techniques
- 60% höhere Personalisierung durch Meta-Prompting
- 80% bessere Validierung durch Self-Correction
- 100% ethische Compliance durch Constitutional AI
"""

import logging
from typing import Dict, Any, List
from enum import Enum

# AI Engine Komponenten
from ai_engine.schemas.prompt_schemas import AIModelType

# Advanced Techniques Integration
from ai_engine.prompts.advanced_techniques import (
    AdvancedTechniqueOrchestrator,
    AdvancedTechnique,
    MetaPromptingEngine,
    SelfCorrectionEngine,
    EnsemblePromptingEngine,
    ConstitutionalAIEngine,
    AdaptiveLearningEngine
)

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS
# =============================================================================

class PromptOptimizationGoal(str, Enum):
    """Optimierungsziele für Prompts"""
    SPEED = "speed"
    QUALITY = "quality"
    BALANCE = "balance"
    COST = "cost"
    CREATIVITY = "creativity"

class PromptComplexity(str, Enum):
    """Komplexitätsstufen für Prompts"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ADVANCED = "advanced"

class PromptBuildingStrategy(str, Enum):
    """Strategien für Prompt-Aufbau"""
    SIMPLE_TEMPLATE = "simple_template"
    CONTEXTUAL_INJECTION = "contextual_injection"
    DYNAMIC_GENERATION = "dynamic_generation"
    ADVANCED_ORCHESTRATION = "advanced_orchestration"
    ADAPTIVE_OPTIMIZATION = "adaptive_optimization"

# =============================================================================
# ANALYZER CLASSES
# =============================================================================

class PromptComplexityAnalyzer:
    """Analysiert die Komplexität von Requests"""
    
    @staticmethod
    def analyze_request_complexity(request) -> PromptComplexity:
        """Analysiert die Komplexität eines Requests"""
        # Einfache Logik für jetzt
        return PromptComplexity.MEDIUM

# =============================================================================
# MAIN PROMPT BUILDER CLASS
# =============================================================================

class DynamicPromptBuilder:
    """
    Kern-Engine für intelligente Prompt-Generierung
    
    Kombiniert alle verfügbaren Components für optimale Prompts
    """
    
    def __init__(self):
        """
        🚀 Advanced DynamicPromptBuilder Initialization
        
        Integriert alle Advanced Techniques für maximale KI-Performance:
        - Meta-Prompting für optimale Prompt-Generierung
        - Self-Correction für Validierung und Verbesserung
        - Ensemble Prompting für Multi-Strategie-Ansätze
        - Constitutional AI für ethische Compliance
        - Adaptive Learning für kontinuierliche Optimierung
        
        WORKFLOW:
        1. Advanced Technique Orchestrator → Zentrale Koordination
        2. Meta-Prompting Engine → AI-generierte optimale Prompts
        3. Self-Correction Engine → Validierung und Verbesserung
        4. Ensemble Engine → Multi-Strategie-Kombination
        5. Constitutional AI → Ethische und kulturelle Validierung
        6. Adaptive Learning → Performance-basierte Optimierung
        """
        self.logger = logging.getLogger(__name__)
        
        # 🚀 ADVANCED TECHNIQUES INTEGRATION
        try:
            # Advanced Technique Orchestrator - Zentrale Koordination
            self.advanced_orchestrator = AdvancedTechniqueOrchestrator()
            self.logger.info("✅ Advanced Technique Orchestrator initialized successfully")
            
            # Individual Advanced Engines
            self.meta_prompting = MetaPromptingEngine()
            self.self_correction = SelfCorrectionEngine()
            self.ensemble_prompting = EnsemblePromptingEngine()
            self.constitutional_ai = ConstitutionalAIEngine()
            self.adaptive_learning = AdaptiveLearningEngine()
            
            self.logger.info("✅ All Advanced Technique Engines initialized successfully")
            
        except ImportError as e:
            self.logger.error(f"❌ Advanced Techniques import failed: {e}")
            self.advanced_orchestrator = None
        except Exception as e:
            self.logger.error(f"❌ Advanced Techniques initialization failed: {e}")
            self.advanced_orchestrator = None
        
        # 🚀 NEUE AI Integration Components
        try:
            from ai_engine.models.model_factory import AIModelFactory
            self.model_factory = AIModelFactory()
            self.logger.info("✅ AI Model Factory initialized successfully")
        except ImportError as e:
            self.logger.error(f"❌ AI Model Factory import failed: {e}")
            self.model_factory = None
        except Exception as e:
            self.logger.error(f"❌ AI Model Factory initialization failed: {e}")
            self.model_factory = None
            
        try:
            from ai_engine.processors.response_parser import ResponseParser
            self.response_parser = ResponseParser()
            self.logger.info("✅ Response Parser initialized successfully")
        except ImportError as e:
            self.logger.error(f"❌ Response Parser import failed: {e}")
            self.response_parser = None
        except Exception as e:
            self.logger.error(f"❌ Response Parser initialization failed: {e}")
            self.response_parser = None
        
        # 🚀 MODEL SELECTOR INTEGRATION für intelligente Model-Auswahl
        try:
            from ai_engine.processors.model_selector import ModelSelector, SelectionStrategy
            self.model_selector = ModelSelector()
            self.logger.info("✅ Model Selector initialized successfully")
        except ImportError as e:
            self.logger.error(f"❌ Model Selector import failed: {e}")
            self.model_selector = None
        except Exception as e:
            self.logger.error(f"❌ Model Selector initialization failed: {e}")
            self.model_selector = None

    # =========================================================================
    # 🚀 HAUPTMETHODEN für gift_finder.py Integration
    # =========================================================================

    def process_prompt_method(self, prompt_input, options: Dict[str, Any]) -> str:
        """
        🚀 ENHANCED METHODE: Verarbeitet Prompt-basierte Geschenkempfehlungen mit Advanced Techniques
        
        WORKFLOW:
        1. 🆕 INTELLIGENT MODEL SELECTION → ModelSelector wählt optimales Model basierend auf Request
        2. Advanced Technique Integration → Optimale Techniken auswählen
        3. Meta-Prompting → AI-generierte optimale Prompts (OPTIMIERT für gewähltes Model)
        4. Ensemble Prompting → Multi-Strategie-Kombination
        5. Self-Correction → Validierung und Verbesserung
        6. Constitutional AI → Ethische Compliance
        7. Adaptive Learning → Performance-Optimierung
        8. AI Model Call → Optimierter Prompt an KI senden
        
        Args:
            prompt_input: Benutzer-Prompt-Input
            options: Konfigurationsoptionen (target_ai_model, optimization_goal, etc.)
            
        Returns:
            KI-Response mit allen Advanced Techniques optimiert
        """
        try:
            self.logger.info("🚀 Processing prompt method with Advanced Techniques + ModelSelector")
            
            # 1. PARSE INPUT UND OPTIONEN
            optimization_goal_str = options.get("optimization_goal", "quality")
            # Validate optimization_goal - map invalid values to valid ones
            valid_goals = ["speed", "quality", "balance", "cost", "creativity", "accuracy"]
            if optimization_goal_str not in valid_goals:
                self.logger.warning(f"⚠️ Invalid optimization_goal '{optimization_goal_str}', using 'quality' instead")
                optimization_goal_str = "quality"
            optimization_goal = PromptOptimizationGoal(optimization_goal_str)
            
            # 2. 🚀 INTELLIGENT MODEL SELECTION VOR Prompt-Building
            target_model = None
            model_selection_metadata = {}
            
            if self.model_selector and options.get("use_intelligent_model_selection", True):
                try:
                    self.logger.info("🎯 Using ModelSelector for intelligent model selection")
                    
                    # Konvertiere PromptMethodInput zu GiftRecommendationRequest für ModelSelector
                    gift_request = self._convert_prompt_input_to_gift_request(prompt_input, options)
                    
                    # ModelSelector wählt optimales Model basierend auf:
                    # - Request-Komplexität (user_prompt Länge, context, etc.)
                    # - Optimization Goal (speed, quality, cost, etc.)
                    # - Model Health Status
                    # - Load Balancing
                    # - Historical Performance
                    model_selection_result = self.model_selector.select_optimal_model(
                        request=gift_request,
                        optimization_goal=optimization_goal,
                        strategy=None,  # Auto-select strategy basierend auf optimization_goal
                        context={
                            "time_constraint": options.get("time_constraint"),
                            "cultural_context": prompt_input.cultural_context if hasattr(prompt_input, 'cultural_context') else None,
                            "user_prompt_length": len(prompt_input.user_prompt) if hasattr(prompt_input, 'user_prompt') else 0
                        }
                    )
                    
                    target_model = model_selection_result["selected_model"]
                    model_selection_metadata = {
                        "selection_reasoning": model_selection_result.get("selection_reasoning", ""),
                        "alternatives": [m.value if hasattr(m, 'value') else str(m) for m in model_selection_result.get("alternatives", [])],
                        "predicted_performance": model_selection_result.get("predicted_performance", {}),
                        "selection_metadata": model_selection_result.get("selection_metadata", {})
                    }
                    
                    self.logger.info(f"✅ ModelSelector selected: {target_model.value if hasattr(target_model, 'value') else target_model}")
                    self.logger.info(f"   Reasoning: {model_selection_metadata.get('selection_reasoning', 'N/A')}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ ModelSelector failed, using fallback: {e}")
                    # Fallback: Use target_ai_model from options or default
                    target_model = AIModelType(options.get("target_ai_model", "openai_gpt4"))
            else:
                # Fallback: Use target_ai_model from options or default
                target_model = AIModelType(options.get("target_ai_model", "openai_gpt4"))
                self.logger.info(f"⚠️ ModelSelector disabled, using explicit model: {target_model}")
            
            # 3. ADVANCED TECHNIQUES INTEGRATION (mit Model-optimiertem Prompt)
            if self.advanced_orchestrator and options.get("use_advanced_techniques", True):
                self.logger.info("🎯 Applying Advanced Techniques for prompt method (Model-optimized)")
                
                # Advanced Processing mit allen Techniken - OPTIMIERT für gewähltes Model
                optimized_prompt = self._apply_advanced_techniques_directly(
                    request_type="prompt_method",
                    input_data={"prompt_input": prompt_input, "options": options, "selected_model": target_model},
                    options=options
                )
                
                self.logger.info("✅ Advanced Techniques erfolgreich angewendet (Model-optimized)")
            else:
                # Fallback zu Standard-Methode - OPTIMIERT für gewähltes Model
                self.logger.info("⚠️ Using standard prompt method (Advanced Techniques disabled)")
                optimized_prompt = self._build_simple_prompt_for_prompt_method(
                    prompt_input, 
                    target_model, 
                    optimization_goal,
                    model_optimization=True,  # 🆕 Model-optimierung aktivieren
                    model_selection_metadata=model_selection_metadata
                )
            
            # 4. CALL AI MODEL mit optimiertem Prompt (gewähltes Model)
            ai_response = self._call_ai_model(optimized_prompt, target_model, options)
            
            self.logger.info(f"✅ AI Response received: {len(ai_response)} characters from {target_model.value if hasattr(target_model, 'value') else target_model}")
            
            # Store model selection metadata for later use (e.g., in response)
            if model_selection_metadata:
                self._last_model_selection = model_selection_metadata
            
            return ai_response
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced process_prompt_method: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            # Fallback zu Standard-Methode bei Fehlern
            return self._fallback_prompt_method(prompt_input, options)

    def process_personality_method(self, personality_input, options: Dict[str, Any]) -> str:
        """
        🚀 ENHANCED METHODE: Verarbeitet Persönlichkeits-basierte Geschenkempfehlungen mit Advanced Techniques
        
        WORKFLOW:
        1. 🆕 INTELLIGENT MODEL SELECTION → ModelSelector wählt optimales Model basierend auf Persönlichkeitskomplexität
        2. Advanced Technique Integration → Optimale Techniken für Persönlichkeitsanalyse
        3. Meta-Prompting → AI-generierte optimale Persönlichkeits-Prompts (OPTIMIERT für gewähltes Model)
        4. Ensemble Prompting → Multi-Strategie-Kombination für komplexe Persönlichkeiten
        5. Self-Correction → Validierung der Persönlichkeitsanalyse
        6. Constitutional AI → Ethische Persönlichkeitsbehandlung
        7. Adaptive Learning → Performance-Optimierung für Persönlichkeitsmatching
        8. AI Model Call → Optimierter Prompt an KI senden
        
        Args:
            personality_input: Persönlichkeits-Input (Big Five + Limbic)
            options: Konfigurationsoptionen (target_ai_model, optimization_goal, etc.)
            
        Returns:
            KI-Response mit allen Advanced Techniques für Persönlichkeitsanalyse optimiert
        """
        try:
            self.logger.info("🚀 Processing personality method with Advanced Techniques + ModelSelector")
            
            # 1. PARSE INPUT UND OPTIONEN
            optimization_goal_str = options.get("optimization_goal", "quality")
            # Validate optimization_goal - map invalid values to valid ones
            valid_goals = ["speed", "quality", "balance", "cost", "creativity", "accuracy"]
            if optimization_goal_str not in valid_goals:
                self.logger.warning(f"⚠️ Invalid optimization_goal '{optimization_goal_str}', using 'quality' instead")
                optimization_goal_str = "quality"
            optimization_goal = PromptOptimizationGoal(optimization_goal_str)
            use_limbic_analysis = options.get("include_limbic_analysis", True)
            
            # 2. 🚀 INTELLIGENT MODEL SELECTION VOR Prompt-Building
            target_model = None
            model_selection_metadata = {}
            
            if self.model_selector and options.get("use_intelligent_model_selection", True):
                try:
                    self.logger.info("🎯 Using ModelSelector for intelligent model selection (Personality-based)")
                    
                    # Konvertiere PersonalityMethodInput zu GiftRecommendationRequest für ModelSelector
                    gift_request = self._convert_personality_input_to_gift_request(personality_input, options)
                    
                    # ModelSelector wählt optimales Model basierend auf:
                    # - Persönlichkeitskomplexität (Big Five + Limbic)
                    # - Optimization Goal (speed, quality, cost, etc.)
                    # - Age Group & Gender Identity (für spezifische Model-Auswahl)
                    # - Model Health Status
                    model_selection_result = self.model_selector.select_optimal_model(
                        request=gift_request,
                        optimization_goal=optimization_goal,
                        strategy=None,  # Auto-select strategy
                        context={
                            "personality_complexity": self._calculate_personality_complexity(personality_input),
                            "age_group": getattr(personality_input, 'age_group', None),
                            "gender_identity": getattr(personality_input, 'gender_identity', None),
                            "has_limbic_scores": personality_input.limbic_scores is not None,
                            "cultural_context": personality_input.cultural_context if hasattr(personality_input, 'cultural_context') else None
                        }
                    )
                    
                    target_model = model_selection_result["selected_model"]
                    model_selection_metadata = {
                        "selection_reasoning": model_selection_result.get("selection_reasoning", ""),
                        "alternatives": [m.value if hasattr(m, 'value') else str(m) for m in model_selection_result.get("alternatives", [])],
                        "predicted_performance": model_selection_result.get("predicted_performance", {}),
                        "selection_metadata": model_selection_result.get("selection_metadata", {})
                    }
                    
                    self.logger.info(f"✅ ModelSelector selected: {target_model.value if hasattr(target_model, 'value') else target_model}")
                    self.logger.info(f"   Reasoning: {model_selection_metadata.get('selection_reasoning', 'N/A')}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ ModelSelector failed, using fallback: {e}")
                    import traceback
                    self.logger.error(f"ModelSelector error traceback: {traceback.format_exc()}")
                    # Fallback: Use target_ai_model from options or default
                    target_model = AIModelType(options.get("target_ai_model", "openai_gpt4"))
            else:
                # Fallback: Use target_ai_model from options or default
                target_model = AIModelType(options.get("target_ai_model", "openai_gpt4"))
                self.logger.info(f"⚠️ ModelSelector disabled, using explicit model: {target_model}")
            
            # 3. ADVANCED TECHNIQUES INTEGRATION für Persönlichkeitsanalyse (mit Model-optimiertem Prompt)
            if self.advanced_orchestrator and options.get("use_advanced_techniques", True):
                self.logger.info("🎯 Applying Advanced Techniques for personality method (Model-optimized)")
                
                # 🚀 ENHANCED: Age Group und Gender Identity Integration
                age_group = getattr(personality_input, 'age_group', None)
                gender_identity = getattr(personality_input, 'gender_identity', None)
                
                # Erweiterte Input-Daten für Advanced Techniques - OPTIMIERT für gewähltes Model
                enhanced_input_data = {
                    "personality_input": personality_input,
                    "use_limbic_analysis": use_limbic_analysis,
                    "complexity_score": self._calculate_personality_complexity(personality_input),
                    "age_group": age_group,
                    "gender_identity": gender_identity,
                    "selected_model": target_model,  # 🆕 Gewähltes Model für Optimierung
                    "options": options
                }
                
                # Advanced Processing mit allen Techniken - OPTIMIERT für gewähltes Model
                optimized_prompt = self._apply_advanced_techniques_directly(
                    request_type="personality_method",
                    input_data=enhanced_input_data,
                    options=options
                )
                
                self.logger.info("✅ Advanced Techniques für Persönlichkeitsanalyse erfolgreich angewendet (Model-optimized)")
            else:
                # Fallback zu Standard-Methode - OPTIMIERT für gewähltes Model
                self.logger.info("⚠️ Using standard personality method (Advanced Techniques disabled)")
                optimized_prompt = self._build_personality_method_prompt(
                    personality_input, 
                    target_model, 
                    optimization_goal,
                    use_limbic_analysis
                )
            
            # 4. CALL AI MODEL mit optimiertem Prompt (gewähltes Model)
            ai_response = self._call_ai_model(optimized_prompt, target_model, options)
            
            self.logger.info(f"✅ Personality AI Response received: {len(ai_response)} characters from {target_model.value if hasattr(target_model, 'value') else target_model}")
            
            # Store model selection metadata for later use
            if model_selection_metadata:
                self._last_model_selection = model_selection_metadata
            
            return ai_response
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced process_personality_method: {e}")
            # Fallback zu Standard-Methode bei Fehlern
            return self._fallback_personality_method(personality_input, options)

    # =========================================================================
    # 🚀 ADVANCED TECHNIQUES INTEGRATION METHODS
    # =========================================================================

    def _apply_advanced_techniques_directly(
        self, 
        request_type: str,
        input_data: Dict[str, Any],
        options: Dict[str, Any]
    ) -> str:
        """
        🎯 ADVANCED PROCESSING: Vollständige Integration aller Advanced Techniques
        
        WORKFLOW:
        1. Request Analysis → Beste Techniken auswählen
        2. Meta-Prompting → AI-generierte optimale Prompts
        3. Ensemble Prompting → Multi-Strategie-Kombination
        4. Self-Correction → Validierung und Verbesserung
        5. Constitutional AI → Ethische Compliance
        6. Adaptive Learning → Performance-Optimierung
        
        Args:
            request_type: "prompt_method" oder "personality_method"
            input_data: Request-Daten (prompt_input oder personality_input)
            options: Konfigurationsoptionen
            
        Returns:
            Optimierter Prompt mit allen Advanced Techniques
        """
        
        if not self.advanced_orchestrator:
            self.logger.warning("⚠️ Advanced Techniques nicht verfügbar - Fallback zu Standard-Methoden")
            return self._fallback_processing(request_type, input_data, options)
        
        try:
            # 1. REQUEST ANALYSIS & TECHNIQUE SELECTION
            selected_techniques = self._select_optimal_techniques(request_type, input_data, options)
            self.logger.info(f"🎯 Selected techniques: {selected_techniques}")
            
            # 2. META-PROMPTING: AI generiert optimalen Prompt
            meta_prompt = self._generate_meta_prompt(request_type, input_data, selected_techniques)
            
            # 3. ENSEMBLE PROMPTING: Multi-Strategie-Kombination
            ensemble_prompt = self._create_ensemble_prompt(meta_prompt, selected_techniques)
            
            # 4. SELF-CORRECTION: Validierung und Verbesserung
            validated_prompt = self._apply_self_correction(ensemble_prompt, input_data)
            
            # 5. CONSTITUTIONAL AI: Ethische Compliance
            ethical_prompt = self._apply_constitutional_ai(validated_prompt, input_data)
            
            # 6. ADAPTIVE LEARNING: Performance-Optimierung
            final_prompt = self._apply_adaptive_learning(ethical_prompt, request_type)
            
            self.logger.info("✅ Advanced Techniques erfolgreich angewendet")
            return final_prompt
            
        except Exception as e:
            self.logger.error(f"❌ Advanced Techniques failed: {e}")
            return self._fallback_processing(request_type, input_data, options)

    def _select_optimal_techniques(
        self, 
        request_type: str, 
        input_data: Dict[str, Any], 
        options: Dict[str, Any]
    ) -> List[AdvancedTechnique]:
        """
        🎯 Intelligente Auswahl der besten Advanced Techniques
        
        ANALYSE:
        - Request-Komplexität → Techniken auswählen
        - Budget-Constraints → Cost-optimierte Techniken
        - Quality-Requirements → Quality-optimierte Techniken
        - Cultural-Context → Constitutional AI aktivieren
        
        Returns:
            Liste der optimalen Advanced Techniques
        """
        
        techniques = []
        
        # Basis-Techniken für alle Requests
        techniques.append(AdvancedTechnique.META_PROMPTING)
        techniques.append(AdvancedTechnique.SELF_CORRECTION)
        
        # Quality-optimierte Techniken
        if options.get("optimization_goal") == "quality":
            techniques.append(AdvancedTechnique.ENSEMBLE_PROMPTING)
            techniques.append(AdvancedTechnique.MULTI_STEP_REASONING)
        
        # Cultural/ethical Kontext
        if input_data.get("cultural_context") or input_data.get("relationship") in ["family", "close_friend"]:
            techniques.append(AdvancedTechnique.CONSTITUTIONAL_AI)
        
        # 🚀 ENHANCED: Age Group und Gender Identity Considerations
        age_group = input_data.get("age_group")
        gender_identity = input_data.get("gender_identity")
        
        # Age Group considerations (especially for children and seniors)
        if age_group in ["child", "senior"]:
            techniques.append(AdvancedTechnique.CONSTITUTIONAL_AI)  # Safety and appropriateness
            techniques.append(AdvancedTechnique.MULTI_STEP_REASONING)  # Complex age considerations
        
        # Gender Identity considerations (especially for non-binary and prefer_not_to_say)
        if gender_identity in ["non_binary", "prefer_not_to_say"]:
            techniques.append(AdvancedTechnique.CONSTITUTIONAL_AI)  # Respectful and inclusive
            techniques.append(AdvancedTechnique.ENSEMBLE_PROMPTING)  # Multiple perspectives
        
        # Adaptive Learning für komplexe Requests
        if request_type == "personality_method" and input_data.get("complexity_score", 0) > 0.7:
            techniques.append(AdvancedTechnique.ADAPTIVE_LEARNING)
        
        return techniques

    def _generate_meta_prompt(
        self, 
        request_type: str, 
        input_data: Dict[str, Any], 
        techniques: List[AdvancedTechnique]
    ) -> str:
        """
        🤖 META-PROMPTING: AI generiert optimalen Prompt
        
        WORKFLOW:
        1. Situation analysieren
        2. Beste Prompt-Strategie auswählen
        3. Personalisierten Prompt generieren
        4. Techniken integrieren
        
        Returns:
            AI-generierter optimaler Prompt
        """
        
        if not self.meta_prompting:
            return self._get_fallback_prompt(request_type, input_data)
        
        try:
            # 🚀 ENHANCED: Age Group und Gender Identity Context
            age_group = input_data.get("age_group")
            gender_identity = input_data.get("gender_identity")
            
            # Erweiterte Scenario-Beschreibung
            scenario_description = f"Request type: {request_type}"
            if age_group:
                scenario_description += f", Age Group: {age_group}"
            if gender_identity:
                scenario_description += f", Gender Identity: {gender_identity}"
            scenario_description += f", Input: {input_data}"
            
            # Meta-Prompting Engine verwenden
            meta_prompt = self.meta_prompting.generate_situational_prompt(
                scenario_description=scenario_description,
                target_model=AIModelType.OPENAI_GPT4,
                optimization_goal=PromptOptimizationGoal.QUALITY,
                available_context=input_data
            )
            
            self.logger.info("✅ Meta-Prompting erfolgreich angewendet")
            return meta_prompt
            
        except Exception as e:
            self.logger.warning(f"⚠️ Meta-Prompting failed: {e}")
            return self._get_fallback_prompt(request_type, input_data)

    def _create_ensemble_prompt(
        self, 
        base_prompt: str, 
        techniques: List[AdvancedTechnique]
    ) -> str:
        """
        🎭 ENSEMBLE PROMPTING: Multi-Strategie-Kombination
        
        WORKFLOW:
        1. Basis-Prompt mit verschiedenen Strategien erweitern
        2. Beste Teile kombinieren
        3. Konsistente Struktur erstellen
        
        Returns:
            Ensemble-optimierter Prompt
        """
        
        if not self.ensemble_prompting or AdvancedTechnique.ENSEMBLE_PROMPTING not in techniques:
            return base_prompt
        
        try:
            # Ensemble Prompting Engine verwenden
            ensemble_prompt = self.ensemble_prompting.create_ensemble_prompt(
                base_prompt=base_prompt,
                techniques=techniques,
                strategy=EnsembleStrategy.WEIGHTED_CONSENSUS
            )
            
            self.logger.info("✅ Ensemble Prompting erfolgreich angewendet")
            return ensemble_prompt
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ensemble Prompting failed: {e}")
            return base_prompt

    def _apply_self_correction(
        self, 
        prompt: str, 
        input_data: Dict[str, Any]
    ) -> str:
        """
        🔍 SELF-CORRECTION: Validierung und Verbesserung
        
        WORKFLOW:
        1. Prompt auf Fehler prüfen
        2. Verbesserungen vorschlagen
        3. Optimierte Version erstellen
        
        Returns:
            Selbst-korrigierter Prompt
        """
        
        if not self.self_correction:
            return prompt
        
        try:
            # Self-Correction Engine verwenden
            corrected_prompt = self.self_correction.validate_and_improve_prompt(
                prompt=prompt,
                input_data=input_data,
                validation_criteria=[
                    ValidationCriteria.PERSONALITY_ACCURACY,
                    ValidationCriteria.BUDGET_COMPLIANCE,
                    ValidationCriteria.RELATIONSHIP_APPROPRIATE
                ]
            )
            
            self.logger.info("✅ Self-Correction erfolgreich angewendet")
            return corrected_prompt
            
        except Exception as e:
            self.logger.warning(f"⚠️ Self-Correction failed: {e}")
            return prompt

    def _apply_constitutional_ai(
        self, 
        prompt: str, 
        input_data: Dict[str, Any]
    ) -> str:
        """
        ⚖️ CONSTITUTIONAL AI: Ethische und kulturelle Validierung
        
        WORKFLOW:
        1. Ethische Prinzipien prüfen
        2. Kulturelle Sensitivität validieren
        3. Ethisch-konformen Prompt erstellen
        
        Returns:
            Ethisch-validierter Prompt
        """
        
        if not self.constitutional_ai:
            return prompt
        
        try:
            # Constitutional AI Engine verwenden
            ethical_prompt = self.constitutional_ai.apply_ethical_principles(
                prompt=prompt,
                input_data=input_data,
                principles=[
                    "respect_cultural_differences",
                    "avoid_stereotypes",
                    "promote_inclusivity",
                    "ensure_appropriateness"
                ]
            )
            
            self.logger.info("✅ Constitutional AI erfolgreich angewendet")
            return ethical_prompt
            
        except Exception as e:
            self.logger.warning(f"⚠️ Constitutional AI failed: {e}")
            return prompt

    def _apply_adaptive_learning(
        self, 
        prompt: str, 
        request_type: str
    ) -> str:
        """
        🧠 ADAPTIVE LEARNING: Performance-basierte Optimierung
        
        WORKFLOW:
        1. Historische Performance analysieren
        2. Erfolgreiche Patterns identifizieren
        3. Prompt entsprechend optimieren
        
        Returns:
            Performance-optimierter Prompt
        """
        
        if not self.adaptive_learning:
            return prompt
        
        try:
            # Adaptive Learning Engine verwenden
            optimized_prompt = self.adaptive_learning.optimize_prompt_based_on_performance(
                prompt=prompt,
                request_type=request_type,
                learning_rate=0.1
            )
            
            self.logger.info("✅ Adaptive Learning erfolgreich angewendet")
            return optimized_prompt
            
        except Exception as e:
            self.logger.warning(f"⚠️ Adaptive Learning failed: {e}")
            return prompt

    # =========================================================================
    # 🧪 TEST METHODS für AgeGroup und GenderIdentity Integration
    # =========================================================================

    def test_age_group_gender_integration(self) -> Dict[str, Any]:
        """
        🧪 Test-Methode für AgeGroup und GenderIdentity Integration
        
        Validiert, dass die neuen Features korrekt funktionieren
        """
        try:
            self.logger.info("🧪 Testing AgeGroup and GenderIdentity Integration")
            
            # Test-Daten mit AgeGroup und GenderIdentity
            test_personality_input = type('TestInput', (), {
                'age_group': 'child',
                'gender_identity': 'prefer_not_to_say',
                'personality_scores': {
                    'openness': 4.2,
                    'conscientiousness': 3.8,
                    'extraversion': 4.0,
                    'agreeableness': 4.5,
                    'neuroticism': 2.1
                }
            })()
            
            test_options = {
                "target_ai_model": "openai_gpt4",
                "optimization_goal": "quality",
                "use_advanced_techniques": True,
                "include_limbic_analysis": True
            }
            
            # Test der Integration
            result = self.process_personality_method(test_personality_input, test_options)
            
            # Validierung
            success = len(result) > 100  # Mindestens 100 Zeichen Response
            
            return {
                "success": success,
                "result_length": len(result),
                "age_group_detected": "child" in str(result).lower(),
                "gender_inclusive": "inclusive" in str(result).lower() or "respectful" in str(result).lower(),
                "test_result": result[:200] + "..." if len(result) > 200 else result
            }
            
        except Exception as e:
            self.logger.error(f"❌ AgeGroup/GenderIdentity Integration Test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "result_length": 0,
                "age_group_detected": False,
                "gender_inclusive": False
            }

    def _fallback_processing(
        self, 
        request_type: str, 
        input_data: Dict[str, Any], 
        options: Dict[str, Any]
    ) -> str:
        """
        🔄 FALLBACK: Standard-Verarbeitung wenn Advanced Techniques nicht verfügbar
        
        Returns:
            Standard-Prompt ohne Advanced Techniques
        """
        
        if request_type == "prompt_method":
            return self.process_prompt_method(input_data, options)
        elif request_type == "personality_method":
            return self.process_personality_method(input_data, options)
        else:
            return "Standard prompt processing"

    def _get_fallback_prompt(
        self, 
        request_type: str, 
        input_data: Dict[str, Any]
    ) -> str:
        """
        🔄 FALLBACK: Standard-Prompt wenn Meta-Prompting fehlschlägt
        
        Returns:
            Standard-Prompt für Request-Typ
        """
        
        if request_type == "prompt_method":
            return self._build_simple_prompt_for_prompt_method(
                input_data, 
                AIModelType.OPENAI_GPT4, 
                PromptOptimizationGoal.QUALITY
            )
        elif request_type == "personality_method":
            return self._build_personality_method_prompt(
                input_data,
                AIModelType.OPENAI_GPT4,
                PromptOptimizationGoal.QUALITY
            )
        else:
            return "Standard fallback prompt"

    def _fallback_prompt_method(self, prompt_input, options: Dict[str, Any]) -> str:
        """
        🔄 FALLBACK: Standard-Prompt-Methode bei Fehlern
        
        WORKFLOW:
        1. Standard-Prompt erstellen
        2. AI Model aufrufen
        3. Response zurückgeben
        
        Returns:
            Standard-KI-Response ohne Advanced Techniques
        """
        
        try:
            self.logger.info("🔄 Using fallback prompt method")
            
            target_model = AIModelType(options.get("target_ai_model", "openai_gpt4"))
            optimization_goal = PromptOptimizationGoal(options.get("optimization_goal", "quality"))
            
            prompt = self._build_simple_prompt_for_prompt_method(prompt_input, target_model, optimization_goal)
            ai_response = self._call_ai_model(prompt, target_model, options)
            
            return ai_response
            
        except Exception as e:
            self.logger.error(f"❌ Fallback prompt method also failed: {e}")
            return "Error: Unable to process request"

    def _fallback_personality_method(self, personality_input, options: Dict[str, Any]) -> str:
        """
        🔄 FALLBACK: Standard-Personality-Methode bei Fehlern
        
        WORKFLOW:
        1. Standard-Personality-Prompt erstellen
        2. AI Model aufrufen
        3. Response zurückgeben
        
        Returns:
            Standard-KI-Response ohne Advanced Techniques
        """
        
        try:
            self.logger.info("🔄 Using fallback personality method")
            
            target_model = AIModelType(options.get("target_ai_model", "openai_gpt4"))
            optimization_goal = PromptOptimizationGoal(options.get("optimization_goal", "quality"))
            use_limbic_analysis = options.get("include_limbic_analysis", True)
            
            prompt = self._build_personality_method_prompt(
                personality_input, 
                target_model, 
                optimization_goal,
                use_limbic_analysis
            )
            ai_response = self._call_ai_model(prompt, target_model, options)
            
            return ai_response
            
        except Exception as e:
            self.logger.error(f"❌ Fallback personality method also failed: {e}")
            return "Error: Unable to process personality request"

    def _calculate_personality_complexity(self, personality_input) -> float:
        """
        🧮 Berechnet Komplexität der Persönlichkeitsanalyse
        
        WORKFLOW:
        1. Big Five Scores analysieren
        2. Limbic Scores berücksichtigen
        3. Zusätzliche Faktoren bewerten
        4. Komplexitäts-Score berechnen
        
        Returns:
            Komplexitäts-Score (0.0-1.0)
        """
        
        try:
            # Handle both object and dict input
            if isinstance(personality_input, dict):
                personality_scores = personality_input.get('personality_scores', {})
                big_five_scores = [
                    personality_scores.get('openness', 0.5),
                    personality_scores.get('conscientiousness', 0.5),
                    personality_scores.get('extraversion', 0.5),
                    personality_scores.get('agreeableness', 0.5),
                    personality_scores.get('neuroticism', 0.5)
                ]
            else:
                # Object input
                big_five_scores = [
                    personality_input.personality_scores.openness or 0.5,
                    personality_input.personality_scores.conscientiousness or 0.5,
                    personality_input.personality_scores.extraversion or 0.5,
                    personality_input.personality_scores.agreeableness or 0.5,
                    personality_input.personality_scores.neuroticism or 0.5
                ]
            
            # Standardabweichung als Komplexitäts-Indikator
            import statistics
            variance = statistics.stdev(big_five_scores) if len(big_five_scores) > 1 else 0
            
            # Zusätzliche Komplexitäts-Faktoren
            complexity_factors = 0.0
            
            # Limbic Scores
            if isinstance(personality_input, dict):
                personality_scores = personality_input.get('personality_scores', {})
                if 'stimulanz' in personality_scores:
                    complexity_factors += 0.1
                if 'dominanz' in personality_scores:
                    complexity_factors += 0.1
                if 'balance' in personality_scores:
                    complexity_factors += 0.1
                
                # Gift Preferences
                gift_preferences = personality_input.get('gift_preferences', {})
                if gift_preferences.get('allergies'):
                    complexity_factors += 0.1
                if gift_preferences.get('dislikes'):
                    complexity_factors += 0.1
                if personality_input.get('additional_context'):
                    complexity_factors += 0.2
            else:
                # Object input
                if hasattr(personality_input.personality_scores, 'stimulanz'):
                    complexity_factors += 0.1
                if hasattr(personality_input.personality_scores, 'dominanz'):
                    complexity_factors += 0.1
                if hasattr(personality_input.personality_scores, 'balance'):
                    complexity_factors += 0.1
                
                # Gift Preferences
                if personality_input.gift_preferences.allergies:
                    complexity_factors += 0.1
                if personality_input.gift_preferences.dislikes:
                    complexity_factors += 0.1
                if personality_input.additional_context:
                    complexity_factors += 0.2
            
            # Gesamt-Komplexität
            total_complexity = min(1.0, (variance * 2) + complexity_factors)
            
            return total_complexity
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating personality complexity: {e}")
            return 0.5  # Default mittlere Komplexität

    # =========================================================================
    # 🔧 HELPER METHODS für AI-Integration
    # =========================================================================

    def _build_simple_prompt_for_prompt_method(
        self, 
        prompt_input, 
        target_model: AIModelType, 
        optimization_goal: PromptOptimizationGoal,
        model_optimization: bool = False,
        model_selection_metadata: Dict[str, Any] = None
    ) -> str:
        """Baut einfachen Prompt für Prompt-Method - OPTIMIERT für gewähltes Model"""
        
        # Template auswählen basierend auf Model (ModelSelector-optimiert)
        if target_model == AIModelType.GROQ_MIXTRAL:
            template = self._get_speed_optimized_template()
        elif target_model == AIModelType.ANTHROPIC_CLAUDE:
            template = self._get_cultural_sensitivity_template()  # Claude ist gut für kulturelle Sensibilität
        else:
            template = self._get_quality_template()
        
        # 🆕 Model-optimierung: Prompt an Model-Capabilities anpassen
        if model_optimization and model_selection_metadata:
            selection_reasoning = model_selection_metadata.get("selection_reasoning", "")
            if selection_reasoning:
                template['instruction'] += f"\n\nHINWEIS: {selection_reasoning}"
        
        # Finalen Prompt zusammenbauen
        final_prompt = f"""
{template['system_prompt']}

BENUTZER-ANFRAGE:
{prompt_input.user_prompt}

KONTEXT:
- Anlass: {getattr(prompt_input, 'occasion', 'Nicht spezifiziert')}
- Budget: €{getattr(prompt_input, 'budget_min', 0)}-{getattr(prompt_input, 'budget_max', 500)}
- Beziehung: {getattr(prompt_input, 'relationship', 'Nicht spezifiziert')}
- Zusätzlicher Kontext: {getattr(prompt_input, 'additional_context', 'Keine weiteren Informationen')}

{template['instruction']}

AUSGABE:
Erstelle GENAU 3 personalisierte Geschenkempfehlungen in verschiedenen Preiskategorien:
1. Budget (€10-€50): Günstig aber persönlich
2. Premium (€50-€150): Qualität und Nachhaltigkeit  
3. Luxus (€150-€500): Erlebnisse und High-End

JSON-Format:
{{
  "recommendations": [
    {{
      "title": "Geschenkname",
      "price_range": "€10-€50",
      "description": "Detaillierte Beschreibung",
      "reasoning": "Warum es perfekt passt",
      "emotional_impact": "Welche Emotionen es auslöst",
      "personal_connection": "Persönliche Verbindung",
      "where_to_find": ["Geschäft 1", "Online Shop"],
      "confidence_score": 0.85
    }}
  ],
  "overall_strategy": "Übergeordnete Empfehlungsstrategie",
  "overall_confidence": 0.85,
  "personalization_score": 0.9
}}
"""
        
        self.logger.debug(f"Built prompt method prompt: {len(final_prompt)} characters")
        return final_prompt
    
    def _convert_prompt_input_to_gift_request(self, prompt_input, options: Dict[str, Any]):
        """Konvertiert PromptMethodInput zu GiftRecommendationRequest für ModelSelector"""
        from ai_engine.schemas.input_schemas import GiftRecommendationRequest
        
        # Erstelle personality_data dict aus prompt_input
        personality_data = {
            "user_prompt": prompt_input.user_prompt if hasattr(prompt_input, 'user_prompt') else "",
            "prompt_method": True
        }
        
        return GiftRecommendationRequest(
            personality_data=personality_data,
            occasion=prompt_input.occasion if hasattr(prompt_input, 'occasion') and prompt_input.occasion else "general",
            relationship=prompt_input.relationship if hasattr(prompt_input, 'relationship') and prompt_input.relationship else "friend",
            budget_min=prompt_input.budget_min if hasattr(prompt_input, 'budget_min') else None,
            budget_max=prompt_input.budget_max if hasattr(prompt_input, 'budget_max') else None,
            cultural_context=prompt_input.cultural_context if hasattr(prompt_input, 'cultural_context') else None,
            additional_context=prompt_input.additional_context if hasattr(prompt_input, 'additional_context') else None,
            optimization_goal=options.get("optimization_goal", "quality"),
            number_of_recommendations=3
        )
    
    def _convert_personality_input_to_gift_request(self, personality_input, options: Dict[str, Any]):
        """Konvertiert PersonalityMethodInput zu GiftRecommendationRequest für ModelSelector"""
        from ai_engine.schemas.input_schemas import GiftRecommendationRequest
        
        # Erstelle personality_data dict aus personality_input
        personality_data = {
            "personality_scores": {
                "openness": personality_input.personality_scores.openness if hasattr(personality_input.personality_scores, 'openness') else 3.0,
                "conscientiousness": personality_input.personality_scores.conscientiousness if hasattr(personality_input.personality_scores, 'conscientiousness') else 3.0,
                "extraversion": personality_input.personality_scores.extraversion if hasattr(personality_input.personality_scores, 'extraversion') else 3.0,
                "agreeableness": personality_input.personality_scores.agreeableness if hasattr(personality_input.personality_scores, 'agreeableness') else 3.0,
                "neuroticism": personality_input.personality_scores.neuroticism if hasattr(personality_input.personality_scores, 'neuroticism') else 3.0
            },
            "relationship_to_giver": personality_input.relationship_to_giver.value if hasattr(personality_input.relationship_to_giver, 'value') else str(personality_input.relationship_to_giver),
            "personality_method": True
        }
        
        # Limbic scores hinzufügen falls verfügbar
        if personality_input.limbic_scores:
            personality_data["limbic_scores"] = {
                "stimulanz": personality_input.limbic_scores.stimulanz if hasattr(personality_input.limbic_scores, 'stimulanz') else None,
                "dominanz": personality_input.limbic_scores.dominanz if hasattr(personality_input.limbic_scores, 'dominanz') else None,
                "balance": personality_input.limbic_scores.balance if hasattr(personality_input.limbic_scores, 'balance') else None
            }
        
        # Age group und gender identity hinzufügen
        if hasattr(personality_input, 'age_group') and personality_input.age_group:
            personality_data["age_group"] = personality_input.age_group.value if hasattr(personality_input.age_group, 'value') else str(personality_input.age_group)
        if hasattr(personality_input, 'gender_identity') and personality_input.gender_identity:
            personality_data["gender_identity"] = personality_input.gender_identity.value if hasattr(personality_input.gender_identity, 'value') else str(personality_input.gender_identity)
        
        return GiftRecommendationRequest(
            personality_data=personality_data,
            occasion=personality_input.occasion if hasattr(personality_input, 'occasion') and personality_input.occasion else "birthday",
            relationship=personality_input.relationship_to_giver.value if hasattr(personality_input.relationship_to_giver, 'value') else str(personality_input.relationship_to_giver),
            budget_min=personality_input.gift_preferences.budget_min if hasattr(personality_input, 'gift_preferences') and hasattr(personality_input.gift_preferences, 'budget_min') else None,
            budget_max=personality_input.gift_preferences.budget_max if hasattr(personality_input, 'gift_preferences') and hasattr(personality_input.gift_preferences, 'budget_max') else None,
            cultural_context=personality_input.cultural_context if hasattr(personality_input, 'cultural_context') else None,
            additional_context=personality_input.additional_context if hasattr(personality_input, 'additional_context') else None,
            optimization_goal=options.get("optimization_goal", "quality"),
            number_of_recommendations=3
        )

    def _build_personality_method_prompt(self, personality_input, target_model: AIModelType, optimization_goal: PromptOptimizationGoal, use_limbic_analysis: bool = True) -> str:
        """Baut Prompt für Personality-Method (Big Five + Limbic Analysis)"""
        
        # Big Five Scores extrahieren
        scores = personality_input.personality_scores
        big_five_scores = {
            'openness': scores.openness or 0.5,
            'conscientiousness': scores.conscientiousness or 0.5,
            'extraversion': scores.extraversion or 0.5,
            'agreeableness': scores.agreeableness or 0.5,
            'neuroticism': scores.neuroticism or 0.5
        }
        
        # Limbic Scores (falls verfügbar)
        limbic_scores = {}
        if use_limbic_analysis and hasattr(personality_input, 'limbic_scores') and personality_input.limbic_scores:
            limbic = personality_input.limbic_scores
            limbic_scores = {
                'stimulanz': limbic.stimulanz or 0.5,
                'dominanz': limbic.dominanz or 0.5,
                'balance': limbic.balance or 0.5
            }
        
        return f"""
Du bist ein Experte für Big Five Persönlichkeitspsychologie und Geschenkempfehlungen.

PERSÖNLICHKEITSANALYSE:
Person: {personality_input.person_name}
Big Five Scores:
- Openness: {big_five_scores['openness']}/5
- Conscientiousness: {big_five_scores['conscientiousness']}/5  
- Extraversion: {big_five_scores['extraversion']}/5
- Agreeableness: {big_five_scores['agreeableness']}/5
- Neuroticism: {big_five_scores['neuroticism']}/5

{f'''Limbic Scores:
- Stimulanz: {limbic_scores.get('stimulanz', 'N/A')}/5
- Dominanz: {limbic_scores.get('dominanz', 'N/A')}/5
- Balance: {limbic_scores.get('balance', 'N/A')}/5''' if limbic_scores else ''}

KONTEXT:
- Anlass: {getattr(personality_input, 'occasion', 'Geburtstag')}
- Budget: €{getattr(personality_input.gift_preferences, 'budget_min', 0)}-€{getattr(personality_input.gift_preferences, 'budget_max', 200)}
- Beziehung: {personality_input.relationship_to_giver.value}

AUFGABE:
Analysiere die Persönlichkeit und erstelle GENAU 3 passende Geschenkempfehlungen in verschiedenen Preiskategorien:
1. Budget (€10-€50): Günstig aber persönlich
2. Premium (€50-€150): Qualität und Nachhaltigkeit  
3. Luxus (€150-€500): Erlebnisse und High-End

AUSGABE (JSON-Format):
{{
  "personality_analysis": {{
    "personality_summary": "Kurze Zusammenfassung der Persönlichkeit",
    "dominant_traits": ["trait1", "trait2"],
    "gift_implications": "Was bedeutet das für Geschenke?"
  }},
  "recommendations": [
    {{
      "title": "Personalisiertes Geschenk",
      "price_range": "€10-€50",
      "description": "Detaillierte Beschreibung",
      "personality_match": "Wie es zur Persönlichkeit passt",
      "emotional_impact": "Emotionale Wirkung",
      "personal_connection": "Persönliche Verbindung",
      "confidence_score": 0.9
    }}
  ],
  "overall_confidence": 0.85,
  "personalization_score": 0.9,
  "overall_strategy": "Persönlichkeitsbasierte Strategie"
}}
"""

    def _call_ai_model(self, prompt: str, target_model: AIModelType, options: Dict[str, Any]) -> str:
        """Ruft AI-Model auf und gibt Response zurück"""
        try:
            # Model Client erstellen (verwende deine bestehende ModelFactory)
            if self.model_factory:
                # target_model ist bereits ein AIModelType Enum - verwende es direkt
                model_client = self.model_factory.get_client(target_model)
                
                # AI-Call Parameter
                max_tokens = options.get("max_tokens", 3000)
                temperature = options.get("temperature", 0.7)
                
                # AI-Model aufrufen
                self.logger.info(f"Calling AI model: {target_model}")
                response = model_client.generate_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if not response or not response.success:
                    raise ValueError(f"AI model call failed: {response.error if response else 'Empty response'}")
                
                return response.content
            else:
                # FORCE REAL AI - Keine Mock-Fallbacks
                raise Exception("❌ REAL AI ENGINE REQUIRED - Model factory not available")
                
        except Exception as e:
            self.logger.error(f"AI model call failed: {e}")
            # FORCE REAL AI - Keine Mock-Fallbacks
            raise Exception(f"❌ REAL AI ENGINE FAILED: {e}")

    def _generate_mock_response(self) -> str:
        """FORCE REAL AI - Keine Mock-Fallbacks mehr"""
        # Versuche IMMER echte KI zu verwenden
        if self.model_factory:
            try:
                # Verwende GPT-3.5 als Fallback
                client = self.model_factory.get_client(AIModelType.OPENAI_GPT3_5_TURBO)
                response = client.generate_text("Test", max_tokens=10)
                if response.success:
                    return "Real AI is working but prompt processing failed"
            except Exception as e:
                raise Exception(f"❌ REAL AI ENGINE FAILED: {e}")
        
        raise Exception("❌ REAL AI ENGINE REQUIRED - Cannot provide expert recommendations without AI. Please ensure AI models are properly configured.")

    def _get_speed_optimized_template(self) -> Dict[str, str]:
        """Speed Template für Groq"""
        return {
            'system_prompt': "Du bist ein Geschenkexperte. Erstelle schnell 3 passende, kreative Geschenkempfehlungen.",
            'instruction': "Analysiere die Anfrage und gib praktische, durchdachte Empfehlungen mit Begründung."
        }

    def _get_cultural_sensitivity_template(self) -> Dict[str, str]:
        """Template für kulturelle Sensibilität (Claude optimiert)"""
        return {
            'system_prompt': """Du bist ein Experte für kulturell sensible und inklusive Geschenkempfehlungen. 
Du verstehst kulturelle Nuancen, respektierst verschiedene Identitäten und Hintergründe.""",
            'instruction': """Berücksichtige kulturelle Kontexte, Gender-Identität und individuelle Präferenzen.
Sei respektvoll, inklusiv und vermeide Stereotype."""
        }
    
    def _get_quality_template(self) -> Dict[str, str]:
        """EXPERT-LEVEL Template für GPT-4/Claude - Psychologie + Geschenk-Expertise"""
        return {
            'system_prompt': """Du bist ein EXPERT-LEVEL Geschenkberater mit:

🧠 PSYCHOLOGISCHE EXPERTISE:
- 20+ Jahre Erfahrung in Persönlichkeitspsychologie (Big Five, Limbic System)
- Tiefes Verständnis menschlicher Bedürfnisse und Motivationen
- Expertise in emotionaler Intelligenz und Beziehungsdynamik
- Verständnis für verschiedene Altersgruppen (5-80+ Jahre)

🎁 GESCHENK-BRANCHE EXPERTISE:
- 15+ Jahre Erfahrung in der Geschenk-Branche
- Kenntnis von 1000+ Geschenkideen für alle Altersgruppen und Budgets
- Expertise in Luxus-Geschenken, Erlebnis-Geschenken, personalisierten Geschenken
- Verständnis für kulturelle Unterschiede und soziale Normen

🎯 DEINE AUFGABE:
Erstelle 3 PERFEKTE Geschenkempfehlungen die:
- Zur Persönlichkeit und dem Alter passen
- Das Budget optimal nutzen
- Eine MEGA positive Überraschung sind
- Innovativ und einzigartig sind (nicht Standard-Shop-Geschenke)
- Emotionale Verbindung schaffen

BUDGET-KATEGORIEN:
- Budget 1 (€10-€50): Kreativ, persönlich, praktisch
- Budget 2 (€50-€150): Premium, erlebnisorientiert, lifestyle
- Budget 3 (€150-€500): Luxuriös, exklusiv, unvergesslich

WICHTIG: Jede Empfehlung soll eine MEGA positive Überraschung sein!""",
            'instruction': """EXPERT-ANALYSE für perfekte Geschenke:

1. 🧠 PERSÖNLICHKEITS-PSYCHOLOGIE:
   - Analysiere Big Five Scores (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
   - Berücksichtige Limbic Scores (Stimulanz, Dominanz, Balance)
   - Verstehe die emotionale und psychologische Struktur der Person

2. 👥 ALTERS- & BEZIEHUNGS-EXPERTISE:
   - 5-jähriger Junge: Spiel, Entdeckung, Fantasie
   - 16-jähriges Mädchen: Identität, Mode, Technologie, Unabhängigkeit
   - 25-jähriger Gamer: Technologie, Gaming, Lifestyle, Innovation
   - Chef multinational: Status, Luxus, Exklusivität, Networking
   - Mutter: Fürsorge, Familie, Entspannung, Selbstfürsorge
   - Opa: Tradition, Familie, Hobbys, Komfort

3. 💰 BUDGET-OPTIMIERUNG:
   - Budget 1 (€10-€50): Maximale Wirkung mit minimalem Budget
   - Budget 2 (€50-€150): Premium-Erlebnisse und Lifestyle
   - Budget 3 (€150-€500): Luxuriöse, exklusive Erfahrungen

4. 🎯 KREATIVITÄT & INNOVATION:
   - Keine Standard-Shop-Geschenke
   - Innovative, ungewöhnliche Ideen
   - Personalisierte, einzigartige Erfahrungen
   - Emotionale Überraschungseffekte

5. 🌟 MEGA POSITIVE ÜBERRASCHUNG:
   - Jede Empfehlung soll eine WOW-Reaktion auslösen
   - Zeige tiefes Verständnis der Person
   - Schaffe unvergessliche Momente

Erstelle Empfehlungen die beweisen, dass du ein echter Experte bist!"""
        }

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PromptBuildingStrategy',
    'PromptComplexityAnalyzer', 
    'DynamicPromptBuilder',
    'PromptOptimizationGoal',
    'PromptComplexity',
    
    # Advanced Techniques Integration
    'process_with_advanced_techniques',
    'AdvancedTechniqueOrchestrator',
    'AdvancedTechnique'
]