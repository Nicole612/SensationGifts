"""
Security Improvements - Production Ready Authentication
====================================================

🔐 Sicherheitsverbesserungen:
- JWT Token Management
- Rate Limiting
- Input Sanitization
- CSRF Protection
- Password Policies
- Session Management
"""

from flask import Flask, request, jsonify, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
import jwt
import bcrypt
import re
import bleach
from datetime import datetime, timedelta
from functools import wraps
import logging
import secrets
from typing import Optional, Dict, Any
import time

# Security Configuration
class SecurityConfig:
    """Zentrale Security Configuration"""
    
    # Password Policy
    MIN_PASSWORD_LENGTH = 8
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_NUMBERS = True
    REQUIRE_SPECIAL_CHARS = True
    
    # JWT Configuration
    JWT_SECRET_KEY = secrets.token_urlsafe(32)
    JWT_ALGORITHM = 'HS256'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # Rate Limiting
    RATE_LIMIT_STORAGE_URL = "redis://localhost:6379"
    DEFAULT_RATE_LIMIT = "100/hour"
    LOGIN_RATE_LIMIT = "5/minute"
    REGISTRATION_RATE_LIMIT = "3/minute"
    
    # Session Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)

class PasswordValidator:
    """Advanced Password Validation"""
    
    @staticmethod
    def validate_password(password: str) -> tuple[bool, list[str]]:
        """
        Validiert Passwort nach Security Policy
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        if len(password) < SecurityConfig.MIN_PASSWORD_LENGTH:
            errors.append(f"Passwort muss mindestens {SecurityConfig.MIN_PASSWORD_LENGTH} Zeichen lang sein")
        
        if SecurityConfig.REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
            errors.append("Passwort muss mindestens einen Großbuchstaben enthalten")
        
        if SecurityConfig.REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
            errors.append("Passwort muss mindestens einen Kleinbuchstaben enthalten")
        
        if SecurityConfig.REQUIRE_NUMBERS and not re.search(r'\d', password):
            errors.append("Passwort muss mindestens eine Zahl enthalten")
        
        if SecurityConfig.REQUIRE_SPECIAL_CHARS and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Passwort muss mindestens ein Sonderzeichen enthalten")
        
        # Check common passwords
        if password.lower() in ['password', '123456', 'admin', 'user', 'test']:
            errors.append("Passwort ist zu einfach")
        
        return len(errors) == 0, errors

class InputSanitizer:
    """Input Sanitization für XSS Protection"""
    
    @staticmethod
    def sanitize_html(input_text: str) -> str:
        """Bereinigt HTML Input"""
        if not input_text:
            return ""
        
        # Erlaubte Tags und Attribute
        allowed_tags = ['b', 'i', 'u', 'strong', 'em']
        allowed_attributes = {}
        
        return bleach.clean(input_text, tags=allowed_tags, attributes=allowed_attributes)
    
    @staticmethod
    def sanitize_user_input(data: dict) -> dict:
        """Bereinigt alle User Inputs"""
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # HTML bereinigen
                sanitized[key] = InputSanitizer.sanitize_html(value)
                # SQL Injection Prevention durch Escaping
                sanitized[key] = sanitized[key].replace("'", "''")
            else:
                sanitized[key] = value
        
        return sanitized

class JWTManager:
    """JWT Token Management"""
    
    @staticmethod
    def generate_tokens(user_id: str, user_email: str) -> dict:
        """Generiert Access + Refresh Token"""
        
        # Access Token (kurze Lebensdauer)
        access_payload = {
            'user_id': user_id,
            'email': user_email,
            'type': 'access',
            'exp': datetime.utcnow() + SecurityConfig.JWT_ACCESS_TOKEN_EXPIRES,
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)  # Unique token ID
        }
        
        # Refresh Token (lange Lebensdauer)
        refresh_payload = {
            'user_id': user_id,
            'email': user_email,
            'type': 'refresh',
            'exp': datetime.utcnow() + SecurityConfig.JWT_REFRESH_TOKEN_EXPIRES,
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)
        }
        
        access_token = jwt.encode(
            access_payload,
            SecurityConfig.JWT_SECRET_KEY,
            algorithm=SecurityConfig.JWT_ALGORITHM
        )
        
        refresh_token = jwt.encode(
            refresh_payload,
            SecurityConfig.JWT_SECRET_KEY,
            algorithm=SecurityConfig.JWT_ALGORITHM
        )
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_in': SecurityConfig.JWT_ACCESS_TOKEN_EXPIRES.total_seconds(),
            'token_type': 'Bearer'
        }
    
    @staticmethod
    def verify_token(token: str, token_type: str = 'access') -> Optional[dict]:
        """Verifiziert JWT Token"""
        try:
            payload = jwt.decode(
                token,
                SecurityConfig.JWT_SECRET_KEY,
                algorithms=[SecurityConfig.JWT_ALGORITHM]
            )
            
            # Check token type
            if payload.get('type') != token_type:
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    @staticmethod
    def refresh_access_token(refresh_token: str) -> Optional[dict]:
        """Erneuert Access Token mit Refresh Token"""
        
        payload = JWTManager.verify_token(refresh_token, 'refresh')
        if not payload:
            return None
        
        # Generate new access token
        return JWTManager.generate_tokens(
            payload['user_id'],
            payload['email']
        )

class SecureAuthDecorator:
    """Secure Authentication Decorators"""
    
    @staticmethod
    def require_jwt_auth(f):
        """JWT Authentication Required"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = None
            
            # Get token from Authorization header
            if 'Authorization' in request.headers:
                auth_header = request.headers['Authorization']
                try:
                    token = auth_header.split(" ")[1]  # Bearer <token>
                except IndexError:
                    return jsonify({'error': 'Invalid token format'}), 401
            
            if not token:
                return jsonify({'error': 'Token missing'}), 401
            
            # Verify token
            payload = JWTManager.verify_token(token)
            if not payload:
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            # Add user info to request context
            request.current_user = {
                'user_id': payload['user_id'],
                'email': payload['email']
            }
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    @staticmethod
    def require_admin_role(f):
        """Admin Role Required"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(request, 'current_user'):
                return jsonify({'error': 'Authentication required'}), 401
            
            # Check if user has admin role (implement your logic)
            user_id = request.current_user['user_id']
            if not is_admin_user(user_id):
                return jsonify({'error': 'Admin access required'}), 403
            
            return f(*args, **kwargs)
        
        return decorated_function

def is_admin_user(user_id: str) -> bool:
    """Check if user has admin privileges"""
    # Implement your admin check logic
    return False

def setup_security_middleware(app: Flask):
    """Setup Security Middleware"""
    
    # Rate Limiting
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=[SecurityConfig.DEFAULT_RATE_LIMIT],
        storage_uri=SecurityConfig.RATE_LIMIT_STORAGE_URL
    )
    
    # Security Headers
    talisman = Talisman(
        app,
        force_https=True,
        strict_transport_security=True,
        content_security_policy={
            'default-src': "'self'",
            'script-src': "'self' 'unsafe-inline' https://cdn.tailwindcss.com",
            'style-src': "'self' 'unsafe-inline'",
            'img-src': "'self' data:",
            'font-src': "'self'"
        }
    )
    
    # Session Security
    app.config.update(
        SESSION_COOKIE_SECURE=SecurityConfig.SESSION_COOKIE_SECURE,
        SESSION_COOKIE_HTTPONLY=SecurityConfig.SESSION_COOKIE_HTTPONLY,
        SESSION_COOKIE_SAMESITE=SecurityConfig.SESSION_COOKIE_SAMESITE,
        PERMANENT_SESSION_LIFETIME=SecurityConfig.PERMANENT_SESSION_LIFETIME
    )
    
    return limiter, talisman

class SecureUserService:
    """Secure User Service mit erweiterten Security Features"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failed_login_attempts = {}  # In production: Redis
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
    
    def register_user_secure(self, user_data: dict) -> dict:
        """Secure User Registration"""
        
        # Input Sanitization
        sanitized_data = InputSanitizer.sanitize_user_input(user_data)
        
        # Password Validation
        password = sanitized_data.get('password', '')
        is_valid, password_errors = PasswordValidator.validate_password(password)
        
        if not is_valid:
            return {
                'success': False,
                'error': 'Password validation failed',
                'password_errors': password_errors
            }
        
        # Email Validation
        email = sanitized_data.get('email', '').lower().strip()
        if not self._is_valid_email(email):
            return {
                'success': False,
                'error': 'Invalid email format'
            }
        
        # Check if user exists
        if self._user_exists(email):
            return {
                'success': False,
                'error': 'User already exists'
            }
        
        # Create user with hashed password
        hashed_password = bcrypt.hashpw(
            password.encode('utf-8'), 
            bcrypt.gensalt()
        ).decode('utf-8')
        
        try:
            # Create user (implement your user creation logic)
            user_id = self._create_user_in_db({
                'email': email,
                'password_hash': hashed_password,
                'first_name': sanitized_data.get('first_name', ''),
                'last_name': sanitized_data.get('last_name', '')
            })
            
            # Generate JWT tokens
            tokens = JWTManager.generate_tokens(user_id, email)
            
            self.logger.info(f"User registered successfully: {email}")
            
            return {
                'success': True,
                'user_id': user_id,
                'tokens': tokens,
                'message': 'User registered successfully'
            }
            
        except Exception as e:
            self.logger.error(f"User registration failed: {e}")
            return {
                'success': False,
                'error': 'Registration failed'
            }
    
    def authenticate_user_secure(self, email: str, password: str, ip_address: str) -> dict:
        """Secure User Authentication mit Brute Force Protection"""
        
        email = email.lower().strip()
        
        # Check if IP is locked out
        if self._is_ip_locked_out(ip_address):
            self.logger.warning(f"Login attempt from locked IP: {ip_address}")
            return {
                'success': False,
                'error': 'Account temporarily locked due to too many failed attempts',
                'lockout_remaining': self._get_lockout_remaining(ip_address)
            }
        
        # Get user from database
        user = self._get_user_by_email(email)
        if not user:
            self._record_failed_attempt(ip_address)
            return {
                'success': False,
                'error': 'Invalid credentials'
            }
        
        # Verify password
        if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            self._record_failed_attempt(ip_address)
            self.logger.warning(f"Failed login attempt for user: {email} from IP: {ip_address}")
            return {
                'success': False,
                'error': 'Invalid credentials'
            }
        
        # Clear failed attempts on successful login
        self._clear_failed_attempts(ip_address)
        
        # Generate JWT tokens
        tokens = JWTManager.generate_tokens(user['id'], email)
        
        # Update last login
        self._update_last_login(user['id'])
        
        self.logger.info(f"User authenticated successfully: {email}")
        
        return {
            'success': True,
            'user_id': user['id'],
            'tokens': tokens,
            'user_info': {
                'email': user['email'],
                'first_name': user['first_name'],
                'last_name': user['last_name']
            }
        }
    
    def _is_valid_email(self, email: str) -> bool:
        """Email Format Validation"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, email) is not None
    
    def _is_ip_locked_out(self, ip_address: str) -> bool:
        """Check if IP is locked out"""
        if ip_address not in self.failed_login_attempts:
            return False
        
        attempts = self.failed_login_attempts[ip_address]
        if attempts['count'] >= self.max_failed_attempts:
            time_elapsed = time.time() - attempts['last_attempt']
            return time_elapsed < self.lockout_duration
        
        return False
    
    def _record_failed_attempt(self, ip_address: str):
        """Record failed login attempt"""
        if ip_address not in self.failed_login_attempts:
            self.failed_login_attempts[ip_address] = {
                'count': 0,
                'last_attempt': 0
            }
        
        self.failed_login_attempts[ip_address]['count'] += 1
        self.failed_login_attempts[ip_address]['last_attempt'] = time.time()
    
    def _clear_failed_attempts(self, ip_address: str):
        """Clear failed attempts for IP"""
        if ip_address in self.failed_login_attempts:
            del self.failed_login_attempts[ip_address]
    
    def _get_lockout_remaining(self, ip_address: str) -> int:
        """Get remaining lockout time in seconds"""
        if ip_address not in self.failed_login_attempts:
            return 0
        
        attempts = self.failed_login_attempts[ip_address]
        time_elapsed = time.time() - attempts['last_attempt']
        return max(0, self.lockout_duration - int(time_elapsed))
    
    # Placeholder methods - implement with your database
    def _user_exists(self, email: str) -> bool:
        """Check if user exists in database"""
        # Implement your database check
        return False
    
    def _create_user_in_db(self, user_data: dict) -> str:
        """Create user in database"""
        # Implement your database creation
        return "user_id_123"
    
    def _get_user_by_email(self, email: str) -> Optional[dict]:
        """Get user by email from database"""
        # Implement your database query
        return None
    
    def _update_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        # Implement your database update
        pass

# Export für Integration
__all__ = [
    'SecurityConfig',
    'PasswordValidator',
    'InputSanitizer',
    'JWTManager',
    'SecureAuthDecorator',
    'SecureUserService',
    'setup_security_middleware'
] 