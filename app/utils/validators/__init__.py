"""
Validators Package - Zentrale Validierungs-Utilities für SensationGifts
=======================================================================

🔄 REFACTORED: Aufgeteilte Validator-Struktur für bessere Wartbarkeit
✅ BACKWARD COMPATIBLE: Alle ursprünglichen Imports funktionieren weiterhin

Struktur:
- form_validators.py: User Input & API Request Validierung
- ai_validators.py: AI, Personality & Recommendation Validierung  
- business_validators.py: Business Logic & Data Integrity Validierung

Usage:
    # Backward compatible imports (funktionieren weiterhin)
    from app.utils.validators import validate_email_format
    from app.utils.validators import validate_personality_data
    from app.utils.validators import validate_user_permissions
    
    # Neue spezifische Imports (empfohlen für neuen Code)
    from app.utils.validators.form_validators import validate_login_data
    from app.utils.validators.ai_validators import validate_big_five_scores
    from app.utils.validators.business_validators import validate_budget_range
"""

# === FORM VALIDATORS IMPORTS ===
from .form_validators import (
    # Decorator Validators
    validate_json_request,
    validate_required_fields,
    
    # Form Validierung
    validate_form_data,
    validate_registration_data,
    validate_login_data,
    
    # Basic Field Validators
    validate_email_format,
    validate_password_strength,
    validate_strong_password,
    validate_phone_number,
    validate_url,
    
    # Gift Search Validierung
    validate_gift_search_data,
    validate_search_filters,
)

# === AI VALIDATORS IMPORTS ===
from .ai_validators import (
    # Personality Validierung
    validate_personality_data,
    validate_big_five_scores,
    validate_limbic_scores,
    validate_recipient_data,
    
    # Recommendation Validierung
    validate_recommendation_request,
    validate_feedback_data,
    
    # Schema Integration Validators
    validate_personality_prompt_integration,
    validate_complete_ai_integration,
    validate_schema_consistency,
    
    # Prompt Schema Functions (moved from prompt_schemas.py)
    validate_prompt_schema,
    optimize_prompt_for_model,
)

# === BUSINESS VALIDATORS IMPORTS ===
from .business_validators import (
    # Budget & Range Validation
    validate_budget_range,
    
    # Business Logic Validators
    validate_user_permissions,
    validate_profile_ownership,
    validate_session_timeout,
    
    # Data Integrity Validators
    validate_database_constraints,
    validate_json_structure,
    
    # Template Quality & Performance (moved from prompt_schemas.py)
    validate_template_quality,
    get_optimization_suggestions,
    validate_performance_metrics,
    validate_cost_efficiency,
    #calculate_overall_performance_score,
    
    # Business Rules
    #validate_business_rules,
    #validate_rate_limits,
)

# === BACKWARD COMPATIBILITY EXPORTS ===
# Alle ursprünglichen Exports für nahtlose Migration

__all__ = [
    # === FORM VALIDATORS ===
    # Decorator Validators
    'validate_json_request',
    'validate_required_fields',
    
    # Form Validierung
    'validate_form_data',
    'validate_registration_data',
    'validate_login_data',
    
    # Basic Field Validators
    'validate_email_format',
    'validate_password_strength',
    'validate_strong_password',
    'validate_phone_number',
    'validate_url',
    
    # Gift Search Validierung
    'validate_gift_search_data',
    'validate_search_filters',
    
    # === AI VALIDATORS ===
    # Personality Validierung
    'validate_personality_data',
    'validate_big_five_scores',
    'validate_limbic_scores',
    'validate_recipient_data',
    
    # Recommendation Validierung
    'validate_recommendation_request',
    'validate_feedback_data',
    
    # Schema Integration Validators
    'validate_personality_prompt_integration',
    'validate_complete_ai_integration',
    'validate_schema_consistency',
    
    # Prompt Schema Functions (moved from prompt_schemas.py)
    'validate_prompt_schema',
    'optimize_prompt_for_model',
    
    # === BUSINESS VALIDATORS ===
    # Budget & Range Validation
    'validate_budget_range',
    
    # Business Logic Validators
    'validate_user_permissions',
    'validate_profile_ownership',
    'validate_session_timeout',
    
    # Data Integrity Validators
    'validate_database_constraints',
    'validate_json_structure',
    
    # Template Quality & Performance (moved from prompt_schemas.py)
    'validate_template_quality',
    'get_optimization_suggestions',
    'validate_performance_metrics',
    'validate_cost_efficiency',
    #'calculate_overall_performance_score',
    
    # Business Rules
    #'validate_business_rules',
    #'validate_rate_limits',
]