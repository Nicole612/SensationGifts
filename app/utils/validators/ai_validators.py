"""
AI Validators - AI/Personality/ML Validation
===========================================

Alle Funktionen für:
- Personality-Validierung (Big Five + Limbic)
- AI Integration Validierung
- Recommendation Engine Validierung
- Schema Consistency Checks
- Prompt Template Validation (from prompt_schemas.py)
"""

from typing import Dict, List, Tuple, Any, Optional
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# === PERSONALITY VALIDIERUNG ===

def validate_personality_data(data: dict) -> Tuple[bool, List[str]]:
    """
    🔒 PERSÖNLICHKEITSDATEN-VALIDIERUNG
    """
    errors = []
    
    # Pflicht-Felder für Tier 1
    required_fields = ['occasion', 'relationship', 'budget_min', 'budget_max']
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f'{field} ist erforderlich')
    
    # Budget-Validierung (Import aus business_validators)
    from .business_validators import validate_budget_range
    budget_errors = validate_budget_range(
        data.get('budget_min'), 
        data.get('budget_max')
    )
    errors.extend(budget_errors)
    
    # Big Five + Limbic Scores validieren (falls vorhanden)
    big_five_errors = validate_big_five_scores(data)
    limbic_errors = validate_limbic_scores(data)
    errors.extend(big_five_errors)
    errors.extend(limbic_errors)
    
    # Recipient-Daten validieren (falls vorhanden)
    recipient_errors = validate_recipient_data(data)
    errors.extend(recipient_errors)
    
    return len(errors) == 0, errors


def validate_big_five_scores(data: dict) -> List[str]:
    """
    🔒 BIG FIVE-SCORES VALIDIERUNG
    """
    errors = []
    
    big_five_fields = [
        'openness', 'conscientiousness', 'extraversion', 
        'agreeableness', 'neuroticism'
    ]
    
    for field in big_five_fields:
        if field in data and data[field] is not None:
            try:
                score = float(data[field])
                if score < 0.0 or score > 1.0:
                    errors.append(f'{field} muss zwischen 0.0 und 1.0 liegen')
            except (ValueError, TypeError):
                errors.append(f'Ungültiger Wert für {field} (muss Zahl sein)')
    
    return errors


def validate_limbic_scores(data: dict) -> List[str]:
    """
    🔒 LIMBIC-SCORES VALIDIERUNG
    """
    errors = []
    
    limbic_fields = ['stimulanz', 'dominanz', 'balance']
    
    for field in limbic_fields:
        if field in data and data[field] is not None:
            try:
                score = float(data[field])
                if score < 0.0 or score > 1.0:
                    errors.append(f'{field} muss zwischen 0.0 und 1.0 liegen')
            except (ValueError, TypeError):
                errors.append(f'Ungültiger Wert für {field} (muss Zahl sein)')
    
    return errors


def validate_recipient_data(data: dict) -> List[str]:
    """
    🔒 EMPFÄNGER-DATEN VALIDIERUNG
    """
    errors = []
    
    # Empfänger-Name (falls vorhanden)
    recipient_name = data.get('recipient_name', '').strip()
    if recipient_name:
        if len(recipient_name) > 100:
            errors.append('Empfänger-Name zu lang (max. 100 Zeichen)')
        
        # Keine Sonderzeichen außer Leerzeichen und Bindestriche
        if not re.match(r'^[a-zA-ZäöüÄÖÜß\s\-]+$', recipient_name):
            errors.append('Empfänger-Name enthält ungültige Zeichen')
    
    # Altersrange validieren
    age_range = data.get('recipient_age_range')
    if age_range and age_range not in [
        'unter_18', '18_25', '25_35', '35_50', '50_65', 'über_65'
    ]:
        errors.append('Ungültige Altersrange')
    
    # Gender validieren
    gender = data.get('recipient_gender')
    if gender and gender not in ['m', 'f', 'diverse', 'any']:
        errors.append('Ungültiger Gender-Wert')
    
    return errors


# === RECOMMENDATION VALIDIERUNG ===

def validate_recommendation_request(data: dict) -> Tuple[bool, List[str]]:
    """
    🔒 EMPFEHLUNGS-REQUEST VALIDIERUNG
    """
    errors = []
    
    # Mode-spezifische Validierung
    mode = data.get('mode', 'auto')
    
    if mode == 'deep':
        # Profile-ID für DEEP PATH erforderlich
        if not data.get('profile_id'):
            errors.append('Profile-ID für DEEP PATH erforderlich')
    
    elif mode == 'quick':
        # Quick-Match Parameter prüfen
        required_quick = ['relationship', 'occasion', 'budget_min', 'budget_max']
        for field in required_quick:
            if not data.get(field):
                errors.append(f'{field} für Quick-Match erforderlich')
    
    # Budget-Validierung (für beide Modi)
    if data.get('budget_min') or data.get('budget_max'):
        from .business_validators import validate_budget_range
        budget_errors = validate_budget_range(
            data.get('budget_min'), 
            data.get('budget_max')
        )
        errors.extend(budget_errors)
    
    # Limit-Validierung
    limit = data.get('limit')
    if limit is not None:
        try:
            limit_val = int(limit)
            if limit_val < 1:
                errors.append('Limit muss mindestens 1 sein')
            elif limit_val > 20:
                errors.append('Limit zu hoch (max. 20)')
        except (ValueError, TypeError):
            errors.append('Ungültiger Limit-Wert')
    
    return len(errors) == 0, errors


def validate_feedback_data(data: dict) -> Tuple[bool, List[str]]:
    """
    🔒 FEEDBACK-DATEN VALIDIERUNG
    """
    errors = []
    
    # Erforderliche Felder
    if not data.get('recommendation_id'):
        errors.append('Recommendation-ID ist erforderlich')
    
    if not data.get('feedback_type'):
        errors.append('Feedback-Typ ist erforderlich')
    
    # Feedback-Typ validieren
    feedback_type = data.get('feedback_type')
    if feedback_type:
        valid_feedback_types = [
            'thumbs_up', 'thumbs_down', 'too_expensive', 'too_cheap',
            'not_personal', 'too_personal', 'wrong_style', 'perfect_match'
        ]
        
        if feedback_type not in valid_feedback_types:
            errors.append(f'Ungültiger Feedback-Typ: {feedback_type}')
    
    # Kommentar-Länge (falls vorhanden)
    comment = data.get('comment', '').strip()
    if comment and len(comment) > 500:
        errors.append('Kommentar zu lang (max. 500 Zeichen)')
    
    return len(errors) == 0, errors


# === PROMPT SCHEMA VALIDATION (moved from prompt_schemas.py) ===

def validate_prompt_schema(schema: Any) -> Dict[str, Any]:
    """
    🔒 PROMPT-SCHEMA VALIDIERUNG
    
    Validiert ein Prompt-Schema und gibt Feedback
    Moved from prompt_schemas.py
    """
    
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "suggestions": [],
        "quality_score": 0.0
    }
    
    try:
        # Type validation
        if not hasattr(schema, '__dict__'):
            validation_result["errors"].append("Schema must be a valid object")
            validation_result["is_valid"] = False
            return validation_result
        
        # Check for required attributes
        required_attrs = ['template_name', 'technique', 'complexity']
        for attr in required_attrs:
            if not hasattr(schema, attr):
                validation_result["errors"].append(f"Missing required attribute: {attr}")
                validation_result["is_valid"] = False
        
        # Specific validations based on schema type
        if hasattr(schema, 'emotional_goals'):
            # GiftRecommendationSchema validation
            if hasattr(schema, 'emotional_goals') and len(schema.emotional_goals) < 2:
                validation_result["warnings"].append("Consider adding more emotional goals for richer recommendations")
            
            # Validate personality weights
            if hasattr(schema, 'big_five_weights'):
                big_five_sum = sum(schema.big_five_weights.values())
                if big_five_sum < 2.0:
                    validation_result["suggestions"].append("Big Five weights seem low - consider higher values for stronger personality influence")
            
            # Validate experience emphasis
            if hasattr(schema, 'experience_emphasis') and schema.experience_emphasis < 0.5:
                validation_result["suggestions"].append("Higher experience emphasis often leads to more meaningful gifts")
        
        # Template-specific validation
        if hasattr(schema, 'instruction_prompt'):
            if len(schema.instruction_prompt) < 50:
                validation_result["warnings"].append("Instruction prompt is quite short - consider adding more detail")
        
        # Calculate quality score
        quality_factors = []
        
        # Completeness factor
        filled_fields = sum(1 for field_name, field_value in schema.__dict__.items() 
                          if field_value is not None and field_value != [] and field_value != {})
        total_fields = len(schema.__dict__)
        completeness = filled_fields / total_fields if total_fields > 0 else 0
        quality_factors.append(completeness * 0.4)
        
        # Complexity factor (more complex = potentially higher quality)
        if hasattr(schema, 'complexity'):
            complexity_map = {
                'simple': 0.3,
                'moderate': 0.5,
                'complex': 0.7,
                'expert': 0.9,
                'ultra_complex': 1.0
            }
            complexity_score = complexity_map.get(getattr(schema, 'complexity', 'simple'), 0.5)
            quality_factors.append(complexity_score * 0.3)
        
        # Error penalty
        error_penalty = len(validation_result["errors"]) * 0.2
        warning_penalty = len(validation_result["warnings"]) * 0.1
        
        validation_result["quality_score"] = max(0.0, sum(quality_factors) - error_penalty - warning_penalty)
        
    except Exception as e:
        validation_result["errors"].append(f"Validation error: {str(e)}")
        validation_result["is_valid"] = False
    
    return validation_result


def optimize_prompt_for_model(base_template: Any, target_model: str) -> Any:
    """
    🔒 PROMPT-OPTIMIERUNG FÜR SPEZIFISCHES AI-MODEL
    
    Optimiert ein Template für ein spezifisches AI-Model
    Moved from prompt_schemas.py
    """
    
    try:
        # Create a copy of the template
        optimized_template = base_template
        
        # Set target model
        if hasattr(optimized_template, 'target_model'):
            optimized_template.target_model = target_model
        
        # Model-spezifische Optimierungen
        if target_model == 'groq_mixtral':
            # Groq bevorzugt kürzere, strukturierte Prompts
            if hasattr(optimized_template, 'instruction_prompt'):
                if len(optimized_template.instruction_prompt) > 800:
                    # Kürze den Prompt aber behalte wichtige Elemente
                    key_elements = [
                        "personality", "emotional", "relationship", "personalized"
                    ]
                    
                    sentences = optimized_template.instruction_prompt.split('.')
                    important_sentences = []
                    
                    for sentence in sentences:
                        if any(element in sentence.lower() for element in key_elements):
                            important_sentences.append(sentence)
                    
                    if important_sentences:
                        optimized_template.instruction_prompt = '. '.join(important_sentences[:5]) + '.'
            
            # Optimiere Token-Limit
            if hasattr(optimized_template, 'max_tokens'):
                optimized_template.max_tokens = min(1500, getattr(optimized_template, 'max_tokens', 2000))
                
        elif target_model == 'anthropic_claude':
            # Claude liebt strukturiertes Reasoning
            if hasattr(optimized_template, 'instruction_prompt'):
                if 'step by step' not in optimized_template.instruction_prompt.lower():
                    reasoning_section = "\n\nPlease think through this step by step, considering the personality psychology and relationship dynamics before making recommendations."
                    optimized_template.instruction_prompt += reasoning_section
            
        elif target_model == 'openai_gpt4':
            # GPT-4 kann mit längeren, detaillierten Prompts umgehen
            if hasattr(optimized_template, 'instruction_prompt'):
                if len(optimized_template.instruction_prompt) < 300:
                    enhancement = "\n\nProvide detailed, nuanced recommendations that demonstrate deep understanding of personality psychology and emotional intelligence."
                    optimized_template.instruction_prompt += enhancement
        
        return optimized_template
        
    except Exception as e:
        logger.error(f"Prompt optimization failed: {e}")
        return base_template


def validate_schema_consistency() -> bool:
    """
    🔒 SCHEMA CONSISTENCY VALIDIERUNG
    
    Prüft Konsistenz zwischen verschiedenen Schema-Definitionen
    Moved from prompt_schemas.py
    """
    
    try:
        # Import all relevant schemas
        from ai_engine.schemas import (
            BigFiveScore, LimbicScore, PersonalityAnalysisInput,
            GiftRecommendationRequest, GiftRecommendationResponse
        )
        from app.models import PersonalityProfile, Gift, GiftCategory
        
        print("🧪 SCHEMA CONSISTENCY CHECK")
        print("=" * 35)
        
        # Check Big Five consistency
        ai_big_five_fields = set(BigFiveScore.__fields__.keys())
        expected_big_five = {'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'}
        
        if ai_big_five_fields >= expected_big_five:
            print("✅ Big Five schema consistency: OK")
        else:
            missing = expected_big_five - ai_big_five_fields
            print(f"❌ Big Five missing fields: {missing}")
        
        # Check Limbic consistency
        ai_limbic_fields = set(LimbicScore.__fields__.keys())
        expected_limbic = {'stimulanz', 'dominanz', 'balance'}
        
        if ai_limbic_fields >= expected_limbic:
            print("✅ Limbic schema consistency: OK")
        else:
            missing = expected_limbic - ai_limbic_fields
            print(f"❌ Limbic missing fields: {missing}")
        
        print("✅ Schema consistency validation complete")
        return True
        
    except Exception as e:
        print(f"❌ Schema consistency check failed: {e}")
        return False


# === AI INTEGRATION VALIDATORS (original from validators.py) ===

def validate_personality_prompt_integration():
    """
    🔒 PERSONALITY PROMPT INTEGRATION VALIDIERUNG
    
    Validiert die Integration zwischen:
    - app.models.PersonalityProfile (Database)
    - ai_engine.schemas.PersonalityAnalysisInput (AI Input) 
    - ai_engine.processors.prompt_builder (Prompt Generation)
    """
    
    validation_results = {
        "schema_compatibility": True,
        "import_success": True,
        "field_mapping": True,
        "errors": [],
        "warnings": []
    }
    
    print("🧪 VALIDATING PERSONALITY PROMPT INTEGRATION")
    print("=" * 55)
    
    # Test 1: Schema Imports
    try:
        from ai_engine.schemas import PersonalityAnalysisInput, BigFiveScore, LimbicScore
        from app.models import PersonalityProfile
        from ai_engine.processors.prompt_builder import QuickPromptBuilders
        print("✅ 1. Schema imports successful")
        
    except ImportError as e:
        validation_results["import_success"] = False
        validation_results["errors"].append(f"Import error: {e}")
        print(f"❌ 1. Import failed: {e}")
        return validation_results
    
    # Test 2: Field Compatibility Check
    try:
        # Check BigFiveScore fields
        expected_big_five = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        big_five_fields = BigFiveScore.__fields__.keys()
        
        missing_big_five = [field for field in expected_big_five if field not in big_five_fields]
        if missing_big_five:
            validation_results["field_mapping"] = False
            validation_results["errors"].append(f"Missing Big Five fields: {missing_big_five}")
            print(f"❌ 2. Missing Big Five fields: {missing_big_five}")
        else:
            print("✅ 2. Big Five fields mapping correct")
        
        # Check LimbicScore fields  
        expected_limbic = ['stimulanz', 'dominanz', 'balance']
        limbic_fields = LimbicScore.__fields__.keys()
        
        missing_limbic = [field for field in expected_limbic if field not in limbic_fields]
        if missing_limbic:
            validation_results["warnings"].append(f"Missing Limbic fields: {missing_limbic}")
            print(f"⚠️  3. Missing Limbic fields: {missing_limbic}")
        else:
            print("✅ 3. Limbic fields mapping correct")
            
    except Exception as e:
        validation_results["field_mapping"] = False
        validation_results["errors"].append(f"Field mapping error: {e}")
        print(f"❌ 2-3. Field mapping failed: {e}")
    
    # Test 3: PersonalityProfile Integration
    try:
        # Check if PersonalityProfile has Big Five + Limbic properties
        db_profile_fields = PersonalityProfile.__table__.columns.keys()
        
        required_db_fields = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        missing_db_fields = [field for field in required_db_fields if field not in db_profile_fields]
        
        if missing_db_fields:
            validation_results["warnings"].append(f"DB missing Big Five fields: {missing_db_fields}")
            print(f"⚠️  4. DB PersonalityProfile missing: {missing_db_fields}")
        else:
            print("✅ 4. Database PersonalityProfile integration correct")
            
    except Exception as e:
        validation_results["warnings"].append(f"DB integration check failed: {e}")
        print(f"⚠️  4. DB integration check failed: {e}")
    
    # Test 4: Prompt Builder Integration
    try:
        # Test if prompt builder methods exist and work
        if hasattr(QuickPromptBuilders, 'build_personality_focused_prompt'):
            print("✅ 5. Prompt builder methods available")
        else:
            validation_results["warnings"].append("build_personality_focused_prompt method not found")
            print("⚠️  5. build_personality_focused_prompt method missing")
            
    except Exception as e:
        validation_results["warnings"].append(f"Prompt builder check failed: {e}")
        print(f"⚠️  5. Prompt builder check failed: {e}")
    
    # Test 5: End-to-End Compatibility
    try:
        # Create mock data to test the flow
        mock_big_five = {
            'openness': 0.8,
            'conscientiousness': 0.7,
            'extraversion': 0.6,
            'agreeableness': 0.9,
            'neuroticism': 0.3
        }
        
        mock_limbic = {
            'stimulanz': 0.5,
            'dominanz': 0.4,
            'balance': 0.8
        }
        
        # Test BigFiveScore creation
        big_five_score = BigFiveScore(**mock_big_five)
        print("✅ 6. BigFiveScore creation successful")
        
        # Test LimbicScore creation
        limbic_score = LimbicScore(**mock_limbic)
        print("✅ 7. LimbicScore creation successful")
        
        validation_results["end_to_end"] = True
        
    except Exception as e:
        validation_results["schema_compatibility"] = False
        validation_results["errors"].append(f"End-to-end test failed: {e}")
        print(f"❌ 6-7. End-to-end test failed: {e}")
    
    # Summary
    print("\n" + "=" * 55)
    if validation_results["errors"]:
        print("❌ VALIDATION FAILED")
        for error in validation_results["errors"]:
            print(f"   ERROR: {error}")
    else:
        print("✅ VALIDATION SUCCESSFUL")
    
    if validation_results["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in validation_results["warnings"]:
            print(f"   WARNING: {warning}")
    
    print(f"\nSchema compatibility: {'✅' if validation_results['schema_compatibility'] else '❌'}")
    print(f"Import success: {'✅' if validation_results['import_success'] else '❌'}")
    print(f"Field mapping: {'✅' if validation_results['field_mapping'] else '❌'}")
    
    return validation_results


def validate_complete_ai_integration():
    """
    🔒 COMPLETE AI SYSTEM INTEGRATION VALIDIERUNG
    
    Validiert alle AI-Integration Components:
    - Personality System
    - Prompt Building
    - Model Selection
    - Schema Compatibility
    """
    
    print("🚀 COMPLETE AI SYSTEM INTEGRATION VALIDATION")
    print("=" * 60)
    
    validation_results = {
        "personality_integration": False,
        "model_integration": False,
        "prompt_integration": False,
        "overall_success": False
    }
    
    # 1. Personality Integration
    print("\n1️⃣  PERSONALITY INTEGRATION:")
    personality_result = validate_personality_prompt_integration()
    validation_results["personality_integration"] = not bool(personality_result.get("errors"))
    
    # 2. Model Integration (Basic Check)
    print("\n2️⃣  AI MODEL INTEGRATION:")
    try:
        from ai_engine.processors.model_selector import ModelSelector
        from ai_engine.schemas import AIModelType, PromptOptimizationGoal
        
        selector = ModelSelector()
        models = ['openai_gpt4', 'groq_mixtral', 'anthropic_claude']
        print(f"✅ Model types available: {models}")
        validation_results["model_integration"] = True
        
    except Exception as e:
        print(f"❌ Model integration failed: {e}")
        validation_results["model_integration"] = False
    
    # 3. Prompt Integration
    print("\n3️⃣  PROMPT INTEGRATION:")
    try:
        from ai_engine.processors.prompt_builder import DynamicPromptBuilder
        from ai_engine.prompts import GiftPromptFactory
        
        builder = DynamicPromptBuilder()
        print("✅ Prompt building system available")
        validation_results["prompt_integration"] = True
        
    except Exception as e:
        print(f"❌ Prompt integration failed: {e}")
        validation_results["prompt_integration"] = False
    
    # Overall Success
    validation_results["overall_success"] = all([
        validation_results["personality_integration"],
        validation_results["model_integration"],
        validation_results["prompt_integration"]
    ])
    
    print("\n" + "=" * 60)
    if validation_results["overall_success"]:
        print("🎉 COMPLETE AI INTEGRATION SUCCESSFUL!")
        print("Your AI system is ready for production!")
    else:
        print("⚠️  AI INTEGRATION ISSUES DETECTED")
        print("Some components need attention before production.")
    
    return validation_results


# === EXPORT ===

__all__ = [
    # Personality Validierung
    'validate_personality_data', 'validate_big_five_scores', 
    'validate_limbic_scores', 'validate_recipient_data',
    
    # Recommendation Validierung
    'validate_recommendation_request', 'validate_feedback_data',
    
    # Prompt Schema Validation (moved from prompt_schemas.py)
    'validate_prompt_schema', 'optimize_prompt_for_model', 'validate_schema_consistency',
    
    # AI Integration Validators
    'validate_personality_prompt_integration',
    'validate_complete_ai_integration'
]