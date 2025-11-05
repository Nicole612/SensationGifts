"""
Utils Package - SensationGifts Utilities
=======================================

Zentraler Import für alle Utility-Funktionen
Ermöglicht saubere Imports aus app.utils
"""

# === FORMATTERS IMPORTS ===
from .formatters import (
    # API Response
    api_response,
    format_error_response, 
    format_success_response,
    
    # Request Formatierung
    is_api_request,
    get_safe_redirect_url,
    parse_filter_params,
    
    # Template Formatierung
    format_datetime_for_template,
    format_date_for_template,
    format_currency,
    format_percentage,
    
    # Score Berechnungen
    calculate_completion_score,
    calculate_answer_quality,
    calculate_confidence_scores,
    
    # Datenschutz
    sanitize_user_data,
    format_public_profile
)

# === VALIDATORS IMPORTS ===
from .validators import (
    # Form Validierung
    validate_registration_data,
    validate_login_data,
    validate_email_format,
    validate_password_strength,
    validate_form_data,
    
    # Gift Search Validierung
    validate_gift_search_data,
    validate_search_filters,
    
    # Personality Validierung
    validate_personality_data,
    validate_big_five_scores,
    validate_limbic_scores,
    validate_recipient_data,
    
    # Business Validierung
    validate_budget_range,
    validate_user_permissions,
    validate_profile_ownership,
    
    # AI Integration Validierung
    validate_recommendation_request,
    validate_feedback_data
)

# === BACKWARD COMPATIBILITY EXPORTS ===
__all__ = [
    # === FORMATTERS ===
    # API Response
    'api_response',
    'format_error_response', 
    'format_success_response',
    
    # Request Formatierung
    'is_api_request',
    'get_safe_redirect_url',
    'parse_filter_params',
    
    # Template Formatierung  
    'format_datetime_for_template',
    'format_date_for_template',
    'format_currency',
    'format_percentage',
    
    # Score Berechnungen
    'calculate_completion_score',
    'calculate_answer_quality', 
    'calculate_confidence_scores',
    
    # Datenschutz
    'sanitize_user_data',
    'format_public_profile',
    
    # === VALIDATORS ===
    # Form Validierung
    'validate_registration_data',
    'validate_login_data',
    'validate_email_format',
    'validate_password_strength',
    'validate_form_data',
    
    # Gift Search Validierung
    'validate_gift_search_data',
    'validate_search_filters',
    
    # Personality Validierung
    'validate_personality_data',
    'validate_big_five_scores',
    'validate_limbic_scores',
    'validate_recipient_data',
    
    # Business Validierung
    'validate_budget_range',
    'validate_user_permissions',
    'validate_profile_ownership',
    
    # AI Integration Validierung
    'validate_recommendation_request',
    'validate_feedback_data'
]