"""
Form Validators - User Input & API Validation
=============================================

Alle Funktionen für:
- Form-Validierung
- API-Request-Validierung  
- Basic Field Format Checks
- Gift Search Validation
"""

from flask import request
from functools import wraps
from typing import Dict, List, Tuple, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)

# === REGEX PATTERNS ===

EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
PASSWORD_PATTERN = re.compile(r'^.{6,}$')  # Mindestens 6 Zeichen
PHONE_PATTERN = re.compile(r'^[\+]?[1-9][\d]{0,15}$')


# === DECORATOR VALIDATORS ===

def validate_json_request():
    """
    🔒 DECORATOR: JSON-Request-Validierung
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json and request.method in ['POST', 'PUT', 'PATCH']:
                from app.utils.formatters import format_error_response
                return format_error_response(
                    'Content-Type muss application/json sein',
                    status=400,
                    error_code='INVALID_CONTENT_TYPE'
                )
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def validate_required_fields(required_fields: List[str]):
    """
    🔒 DECORATOR: Erforderliche Felder validieren
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            data = request.get_json() or {}
            
            missing_fields = [field for field in required_fields if not data.get(field)]
            if missing_fields:
                from app.utils.formatters import format_error_response
                return format_error_response(
                    f'Fehlende erforderliche Felder: {", ".join(missing_fields)}',
                    status=400,
                    error_code='MISSING_REQUIRED_FIELDS'
                )
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# === FORM VALIDIERUNG ===

def validate_form_data(form_data: dict, required_fields: list) -> Tuple[bool, List[str]]:
    """
    🔒 ALLGEMEINE FORM-VALIDIERUNG
    
    Args:
        form_data: Dictionary mit Form-Daten
        required_fields: Liste erforderlicher Felder
        
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Erforderliche Felder prüfen
    for field in required_fields:
        if not form_data.get(field) or not str(form_data[field]).strip():
            errors.append(f'{field} ist erforderlich')
    
    # Email-Format prüfen (falls Email-Feld vorhanden)
    email = form_data.get('email', '').strip()
    if email and not validate_email_format(email):
        errors.append('Ungültiges Email-Format')
    
    # Password-Stärke prüfen (falls Password-Feld vorhanden)
    password = form_data.get('password', '')
    if password and not validate_password_strength(password):
        errors.append('Passwort muss mindestens 6 Zeichen haben')
    
    return len(errors) == 0, errors


def validate_registration_data(form_data: dict) -> Tuple[bool, List[str]]:
    """
    🔒 REGISTRIERUNGS-VALIDIERUNG
    """
    errors = []
    
    # Basis-Validierung
    is_valid, basic_errors = validate_form_data(form_data, ['email', 'password'])
    errors.extend(basic_errors)
    
    # Password-Bestätigung
    if 'password_confirm' in form_data:
        if form_data.get('password') != form_data.get('password_confirm'):
            errors.append('Passwörter stimmen nicht überein')
    
    # Email-Eindeutigkeit (würde in Service geprüft)
    email = form_data.get('email', '').lower().strip()
    if email:
        # Zusätzliche Email-Validierung
        if len(email) > 120:
            errors.append('Email-Adresse zu lang (max. 120 Zeichen)')
    
    # Namen-Validierung (falls vorhanden)
    for name_field in ['first_name', 'last_name']:
        name = form_data.get(name_field, '').strip()
        if name and len(name) > 50:
            errors.append(f'{name_field} zu lang (max. 50 Zeichen)')
    
    return len(errors) == 0, errors


def validate_login_data(form_data: dict) -> Tuple[bool, List[str]]:
    """
    🔒 LOGIN-VALIDIERUNG
    """
    errors = []
    
    # Erforderliche Felder
    if not form_data.get('email'):
        errors.append('Email ist erforderlich')
    if not form_data.get('password'):
        errors.append('Passwort ist erforderlich')
    
    # Email-Format (basic)
    email = form_data.get('email', '').strip()
    if email and not validate_email_format(email):
        errors.append('Ungültiges Email-Format')
    
    return len(errors) == 0, errors


# === GIFT SEARCH VALIDIERUNG ===

def validate_gift_search_data(data: dict) -> Tuple[bool, List[str]]:
    """
    🔒 GIFT-SUCHE VALIDIERUNG
    """
    errors = []
    
    # Suchterm-Validierung
    query = data.get('query', '').strip()
    if not query:
        errors.append('Suchterm ist erforderlich')
    elif len(query) < 2:
        errors.append('Suchterm zu kurz (mindestens 2 Zeichen)')
    elif len(query) > 200:
        errors.append('Suchterm zu lang (max. 200 Zeichen)')
    
    # Limit-Validierung
    limit = data.get('limit')
    if limit is not None:
        try:
            limit_val = int(limit)
            if limit_val < 1:
                errors.append('Limit muss mindestens 1 sein')
            elif limit_val > 50:
                errors.append('Such-Limit zu hoch (max. 50)')
        except (ValueError, TypeError):
            errors.append('Ungültiger Limit-Wert')
    
    # Filter-Validierung
    filters = data.get('filters', {})
    if filters:
        filter_errors = validate_search_filters(filters)
        errors.extend(filter_errors)
    
    return len(errors) == 0, errors


def validate_search_filters(filters: dict) -> List[str]:
    """
    🔒 SUCH-FILTER VALIDIERUNG
    """
    errors = []
    
    # Preis-Filter (Import Budget-Validierung aus business_validators)
    if filters.get('price_min') or filters.get('price_max'):
        from .business_validators import validate_budget_range
        budget_errors = validate_budget_range(
            filters.get('price_min'),
            filters.get('price_max')
        )
        errors.extend(budget_errors)
    
    # Kategorien-Filter
    categories = filters.get('categories', [])
    if categories:
        if not isinstance(categories, list):
            errors.append('Kategorien müssen als Liste übergeben werden')
        elif len(categories) > 10:
            errors.append('Zu viele Kategorien ausgewählt (max. 10)')
    
    # Anlass-Filter
    occasion = filters.get('occasion')
    if occasion:
        valid_occasions = [
            'geburtstag', 'weihnachten', 'jahrestag', 'valentinstag',
            'muttertag', 'vatertag', 'ostern', 'abschluss', 'hochzeit'
        ]
        if occasion not in valid_occasions:
            errors.append(f'Ungültiger Anlass: {occasion}')
    
    return errors


# === BASIC FIELD VALIDATORS ===

def validate_email_format(email: str) -> bool:
    """
    🔒 EMAIL-FORMAT VALIDIERUNG
    """
    if not email or not isinstance(email, str):
        return False
    
    email = email.strip().lower()
    
    # Längen-Check
    if len(email) < 5 or len(email) > 120:
        return False
    
    # Regex-Check
    return bool(EMAIL_PATTERN.match(email))


def validate_password_strength(password: str) -> bool:
    """
    🔒 PASSWORT-STÄRKE VALIDIERUNG
    """
    if not password or not isinstance(password, str):
        return False
    
    # Mindestlänge
    if len(password) < 6:
        return False
    
    # Maximallänge (Sicherheit vor DoS)
    if len(password) > 128:
        return False
    
    return True


def validate_strong_password(password: str) -> Tuple[bool, List[str]]:
    """
    🔒 STARKE PASSWORT-VALIDIERUNG (optional für höhere Sicherheit)
    """
    errors = []
    
    if not validate_password_strength(password):
        errors.append('Passwort muss mindestens 6 Zeichen haben')
        return False, errors
    
    # Erweiterte Checks
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in '!@#$%^&*(),.?":{}|<>' for c in password)
    
    if not has_upper:
        errors.append('Passwort muss mindestens einen Großbuchstaben enthalten')
    if not has_lower:
        errors.append('Passwort muss mindestens einen Kleinbuchstaben enthalten')
    if not has_digit:
        errors.append('Passwort muss mindestens eine Zahl enthalten')
    if not has_special:
        errors.append('Passwort muss mindestens ein Sonderzeichen enthalten')
    
    return len(errors) == 0, errors


def validate_phone_number(phone: str) -> bool:
    """
    🔒 TELEFONNUMMER VALIDIERUNG
    """
    if not phone or not isinstance(phone, str):
        return False
    
    # Leerzeichen und Bindestriche entfernen
    cleaned = re.sub(r'[\s\-\(\)]', '', phone)
    
    return bool(PHONE_PATTERN.match(cleaned))


def validate_url(url: str) -> bool:
    """
    🔒 URL VALIDIERUNG
    """
    if not url or not isinstance(url, str):
        return False
    
    url = url.strip()
    
    # Basic URL format
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return bool(url_pattern.match(url))


# === EXPORT ===

__all__ = [
    # Decorator Validators
    'validate_json_request', 'validate_required_fields',
    
    # Form Validierung
    'validate_form_data', 'validate_registration_data', 'validate_login_data',
    
    # Gift Search Validierung
    'validate_gift_search_data', 'validate_search_filters',
    
    # Basic Field Validators
    'validate_email_format', 'validate_password_strength', 
    'validate_strong_password', 'validate_phone_number', 'validate_url',
]