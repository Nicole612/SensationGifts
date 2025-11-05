"""
User Service - Business Logic für User Operations (Session 5)
============================================================

Clean Architecture: Business Logic Schicht zwischen Routes und Models
Behandelt alle User-bezogenen Operationen mit Validation und Error Handling

Features:
- User Registration & Authentication
- Profile Management  
- Session Management
- Password Security
- Input Validation mit Pydantic
- Comprehensive Error Handling
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import re

from werkzeug.security import generate_password_hash, check_password_hash
from pydantic import BaseModel, EmailStr, Field, validator
from flask import current_app

# Import Models
from app.models import User
from app import db
from config.settings import get_settings


# =============================================================================
# PYDANTIC VALIDATION SCHEMAS
# =============================================================================

class UserRegistrationRequest(BaseModel):
    """Validation Schema für User Registration"""
    
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="User password")
    confirm_password: str = Field(..., description="Password confirmation")
    first_name: Optional[str] = Field(None, max_length=50, description="User first name")
    last_name: Optional[str] = Field(None, max_length=50, description="User last name")
    preferred_language: str = Field("de", description="Preferred language")
    preferred_currency: str = Field("EUR", description="Preferred currency")
    accept_terms: bool = Field(..., description="Terms and conditions acceptance")
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Validates password meets security requirements"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        
        return v
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        """Validates password confirmation matches"""
        if 'password' in values and v != values['password']:
            raise ValueError('Password confirmation does not match')
        return v
    
    @validator('email')
    def validate_email_format(cls, v):
        """Additional email validation"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v.lower()):
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('accept_terms')
    def terms_must_be_accepted(cls, v):
        """Validates terms acceptance"""
        if not v:
            raise ValueError('Terms and conditions must be accepted')
        return v


class UserLoginRequest(BaseModel):
    """Validation Schema für User Login"""
    
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=1, description="User password")
    remember_me: bool = Field(False, description="Remember user session")
    
    @validator('email')
    def normalize_email(cls, v):
        """Normalizes email to lowercase"""
        return v.lower()


class UserProfileUpdateRequest(BaseModel):
    """Validation Schema für Profile Updates"""
    
    first_name: Optional[str] = Field(None, max_length=50)
    last_name: Optional[str] = Field(None, max_length=50)
    preferred_language: Optional[str] = Field(None, pattern=r'^(de|en|fr|es)$')
    preferred_currency: Optional[str] = Field(None, pattern=r'^(EUR|USD|GBP)$')
    default_budget_min: Optional[float] = Field(None, ge=0, le=10000)
    default_budget_max: Optional[float] = Field(None, ge=0, le=10000)
    
    @validator('default_budget_max')
    def budget_max_greater_than_min(cls, v, values):
        """Validates max budget is greater than min budget"""
        if v and 'default_budget_min' in values and values['default_budget_min']:
            if v <= values['default_budget_min']:
                raise ValueError('Maximum budget must be greater than minimum budget')
        return v


class PasswordChangeRequest(BaseModel):
    """Validation Schema für Password Changes"""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")
    confirm_new_password: str = Field(..., description="New password confirmation")
    
    @validator('new_password')
    def validate_new_password_strength(cls, v):
        """Validates new password meets security requirements"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        
        return v
    
    @validator('confirm_new_password')
    def new_passwords_match(cls, v, values):
        """Validates new password confirmation matches"""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('New password confirmation does not match')
        return v


# =============================================================================
# SERVICE RESPONSE MODELS
# =============================================================================

class ServiceResponse(BaseModel):
    """Standard Service Response Format"""
    
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    
    @classmethod
    def success_response(cls, message: str, data: Optional[Dict[str, Any]] = None):
        """Creates successful response"""
        return cls(success=True, message=message, data=data)
    
    @classmethod
    def error_response(cls, message: str, errors: Optional[List[str]] = None):
        """Creates error response"""
        return cls(success=False, message=message, errors=errors or [])


# =============================================================================
# CORE USER SERVICE
# =============================================================================

class UserService:
    """
    Core User Service - Business Logic für alle User-Operationen
    
    Responsibilities:
    - User Registration & Authentication
    - Profile Management
    - Session Management
    - Security & Validation
    - Error Handling & Logging
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
    
    # =============================================================================
    # REGISTRATION & AUTHENTICATION
    # =============================================================================
    
    def register_user(self, registration_data: UserRegistrationRequest) -> ServiceResponse:
        """
        Registers new user with comprehensive validation
        
        Args:
            registration_data: Validated registration request
            
        Returns:
            ServiceResponse with user data or error details
        """
        try:
            self.logger.info(f"Attempting user registration for email: {registration_data.email}")
            
            # Check if user already exists
            existing_user = User.query.filter_by(email=registration_data.email).first()
            if existing_user:
                self.logger.warning(f"Registration failed: email already exists: {registration_data.email}")
                return ServiceResponse.error_response(
                    message="Ein Account mit dieser E-Mail-Adresse existiert bereits",
                    errors=["EMAIL_ALREADY_EXISTS"]
                )
            
            # Create new user
            new_user = User(
                email=registration_data.email,
                password=registration_data.password,  # Will be hashed in User.__init__
                first_name=registration_data.first_name,
                last_name=registration_data.last_name
            )
            
            # Set preferences
            new_user.preferred_language = registration_data.preferred_language
            new_user.preferred_currency = registration_data.preferred_currency
            
            # Save to database
            success = new_user.save()
            if success:
                self.logger.info(f"User successfully registered: {new_user.id}")
                
                # Prepare user data for response (without sensitive info)
                user_data = {
                    "user_id": new_user.id,
                    "email": new_user.email,
                    "full_name": new_user.full_name,
                    "display_name": new_user.display_name,
                    "preferred_language": new_user.preferred_language,
                    "preferred_currency": new_user.preferred_currency,
                    "created_at": new_user.created_at.isoformat(),
                    "is_verified": new_user.is_verified
                }
                
                return ServiceResponse.success_response(
                    message="Account erfolgreich erstellt! Sie können sich jetzt anmelden.",
                    data={"user": user_data}
                )
            else:
                self.logger.error(f"Database error during user registration: {registration_data.email}")
                return ServiceResponse.error_response(
                    message="Fehler beim Erstellen des Accounts. Bitte versuchen Sie es erneut.",
                    errors=["DATABASE_ERROR"]
                )
                
        except Exception as e:
            self.logger.error(f"Unexpected error during user registration: {str(e)}")
            return ServiceResponse.error_response(
                message="Ein unerwarteter Fehler ist aufgetreten. Bitte versuchen Sie es erneut.",
                errors=["UNEXPECTED_ERROR"]
            )
    
    def authenticate_user(self, login_data: UserLoginRequest) -> ServiceResponse:
        """
        Authenticates user login
        
        Args:
            login_data: Validated login request
            
        Returns:
            ServiceResponse with user session data or error details
        """
        try:
            self.logger.info(f"Authentication attempt for email: {login_data.email}")
            
            # Find user by email
            user = User.query.filter_by(email=login_data.email).first()
            if not user:
                self.logger.warning(f"Authentication failed: user not found: {login_data.email}")
                return ServiceResponse.error_response(
                    message="Ungültige E-Mail-Adresse oder Passwort",
                    errors=["INVALID_CREDENTIALS"]
                )
            
            # Check if account is active
            if not user.is_active:
                self.logger.warning(f"Authentication failed: account disabled: {login_data.email}")
                return ServiceResponse.error_response(
                    message="Dieser Account wurde deaktiviert. Bitte kontaktieren Sie den Support.",
                    errors=["ACCOUNT_DISABLED"]
                )
            
            # Verify password
            if not user.check_password(login_data.password):
                self.logger.warning(f"Authentication failed: wrong password: {login_data.email}")
                return ServiceResponse.error_response(
                    message="Ungültige E-Mail-Adresse oder Passwort",
                    errors=["INVALID_CREDENTIALS"]
                )
            
            # Update last login
            user.mark_login()
            
            self.logger.info(f"User successfully authenticated: {user.id}")
            
            # Prepare session data
            session_data = {
                "user_id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "display_name": user.display_name,
                "preferred_language": user.preferred_language,
                "preferred_currency": user.preferred_currency,
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "is_verified": user.is_verified,
                "total_profiles": len(user.personality_profiles),
                "default_budget": user.default_budget,
                "remember_me": login_data.remember_me
            }
            
            return ServiceResponse.success_response(
                message=f"Willkommen zurück, {user.display_name}!",
                data={"session": session_data}
            )
            
        except Exception as e:
            self.logger.error(f"Unexpected error during authentication: {str(e)}")
            return ServiceResponse.error_response(
                message="Ein Fehler ist beim Anmelden aufgetreten. Bitte versuchen Sie es erneut.",
                errors=["AUTHENTICATION_ERROR"]
            )
    
    # =============================================================================
    # PROFILE MANAGEMENT
    # =============================================================================
    
    def get_user_profile(self, user_id: str) -> ServiceResponse:
        """
        Retrieves complete user profile
        
        Args:
            user_id: User identifier
            
        Returns:
            ServiceResponse with complete user profile data
        """
        try:
            user = User.get_by_id(user_id)
            if not user:
                return ServiceResponse.error_response(
                    message="Benutzer nicht gefunden",
                    errors=["USER_NOT_FOUND"]
                )
            
            # Get comprehensive profile data
            profile_data = {
                "user": user.to_dict(),
                "recent_recipients": user.get_recent_recipients(limit=5),
                "popular_occasions": user.get_popular_occasions(limit=5),
                "recommendation_stats": {
                    "total_recommendations": user.get_total_recommendations(),
                    "recent_recommendations": len(user.get_recommendations(limit=10))
                }
            }
            
            return ServiceResponse.success_response(
                message="Profil erfolgreich geladen",
                data={"profile": profile_data}
            )
            
        except Exception as e:
            self.logger.error(f"Error retrieving user profile: {str(e)}")
            return ServiceResponse.error_response(
                message="Fehler beim Laden des Profils",
                errors=["PROFILE_LOAD_ERROR"]
            )
    
    def update_user_profile(self, user_id: str, update_data: UserProfileUpdateRequest) -> ServiceResponse:
        """
        Updates user profile information
        
        Args:
            user_id: User identifier
            update_data: Validated update request
            
        Returns:
            ServiceResponse with updated profile data
        """
        try:
            user = User.get_by_id(user_id)
            if not user:
                return ServiceResponse.error_response(
                    message="Benutzer nicht gefunden",
                    errors=["USER_NOT_FOUND"]
                )
            
            # Update fields that were provided
            updates_made = []
            
            if update_data.first_name is not None:
                user.first_name = update_data.first_name
                updates_made.append("first_name")
            
            if update_data.last_name is not None:
                user.last_name = update_data.last_name
                updates_made.append("last_name")
            
            if update_data.preferred_language is not None:
                user.preferred_language = update_data.preferred_language
                updates_made.append("preferred_language")
            
            if update_data.preferred_currency is not None:
                user.preferred_currency = update_data.preferred_currency
                updates_made.append("preferred_currency")
            
            if update_data.default_budget_min is not None and update_data.default_budget_max is not None:
                user.set_default_budget(update_data.default_budget_min, update_data.default_budget_max)
                updates_made.append("default_budget")
            
            # Save changes
            if updates_made:
                success = user.save()
                if success:
                    self.logger.info(f"User profile updated: {user_id}, fields: {updates_made}")
                    
                    return ServiceResponse.success_response(
                        message="Profil erfolgreich aktualisiert",
                        data={
                            "updated_fields": updates_made,
                            "user": user.to_dict()
                        }
                    )
                else:
                    return ServiceResponse.error_response(
                        message="Fehler beim Speichern der Änderungen",
                        errors=["DATABASE_SAVE_ERROR"]
                    )
            else:
                return ServiceResponse.success_response(
                    message="Keine Änderungen vorgenommen",
                    data={"user": user.to_dict()}
                )
                
        except Exception as e:
            self.logger.error(f"Error updating user profile: {str(e)}")
            return ServiceResponse.error_response(
                message="Fehler beim Aktualisieren des Profils",
                errors=["PROFILE_UPDATE_ERROR"]
            )
    
    def change_password(self, user_id: str, password_data: PasswordChangeRequest) -> ServiceResponse:
        """
        Changes user password with validation
        
        Args:
            user_id: User identifier
            password_data: Validated password change request
            
        Returns:
            ServiceResponse indicating success or failure
        """
        try:
            user = User.get_by_id(user_id)
            if not user:
                return ServiceResponse.error_response(
                    message="Benutzer nicht gefunden",
                    errors=["USER_NOT_FOUND"]
                )
            
            # Verify current password
            if not user.check_password(password_data.current_password):
                self.logger.warning(f"Password change failed: wrong current password for user {user_id}")
                return ServiceResponse.error_response(
                    message="Das aktuelle Passwort ist falsch",
                    errors=["WRONG_CURRENT_PASSWORD"]
                )
            
            # Check if new password is different
            if user.check_password(password_data.new_password):
                return ServiceResponse.error_response(
                    message="Das neue Passwort muss sich vom aktuellen Passwort unterscheiden",
                    errors=["SAME_PASSWORD"]
                )
            
            # Set new password
            user.set_password(password_data.new_password)
            
            # Save changes
            success = user.save()
            if success:
                self.logger.info(f"Password successfully changed for user: {user_id}")
                return ServiceResponse.success_response(
                    message="Passwort erfolgreich geändert"
                )
            else:
                return ServiceResponse.error_response(
                    message="Fehler beim Speichern des neuen Passworts",
                    errors=["DATABASE_SAVE_ERROR"]
                )
                
        except Exception as e:
            self.logger.error(f"Error changing password: {str(e)}")
            return ServiceResponse.error_response(
                message="Fehler beim Ändern des Passworts",
                errors=["PASSWORD_CHANGE_ERROR"]
            )
    
    # =============================================================================
    # USER MANAGEMENT & ADMIN FUNCTIONS
    # =============================================================================
    
    def get_user_statistics(self, user_id: str) -> ServiceResponse:
        """
        Gets comprehensive user statistics for dashboard
        
        Args:
            user_id: User identifier
            
        Returns:
            ServiceResponse with user statistics
        """
        try:
            user = User.get_by_id(user_id)
            if not user:
                return ServiceResponse.error_response(
                    message="Benutzer nicht gefunden",
                    errors=["USER_NOT_FOUND"]
                )
            
            # Gather statistics
            statistics = {
                "user_info": {
                    "member_since": user.created_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "total_logins": 1,  # Would need session tracking to get accurate count
                    "is_verified": user.is_verified
                },
                "personality_profiles": {
                    "total_profiles": len(user.personality_profiles),
                    "recent_recipients": user.get_recent_recipients(limit=5),
                    "popular_occasions": user.get_popular_occasions(limit=5)
                },
                "recommendations": {
                    "total_recommendations": user.get_total_recommendations(),
                    "recent_recommendations": len(user.get_recommendations(limit=10))
                },
                "preferences": {
                    "preferred_language": user.preferred_language,
                    "preferred_currency": user.preferred_currency,
                    "default_budget": user.default_budget
                }
            }
            
            return ServiceResponse.success_response(
                message="Statistiken erfolgreich geladen",
                data={"statistics": statistics}
            )
            
        except Exception as e:
            self.logger.error(f"Error retrieving user statistics: {str(e)}")
            return ServiceResponse.error_response(
                message="Fehler beim Laden der Statistiken",
                errors=["STATISTICS_ERROR"]
            )
    
    def delete_user_account(self, user_id: str, password: str) -> ServiceResponse:
        """
        Deletes user account with password confirmation
        
        Args:
            user_id: User identifier
            password: Password confirmation
            
        Returns:
            ServiceResponse indicating success or failure
        """
        try:
            user = User.get_by_id(user_id)
            if not user:
                return ServiceResponse.error_response(
                    message="Benutzer nicht gefunden",
                    errors=["USER_NOT_FOUND"]
                )
            
            # Verify password
            if not user.check_password(password):
                self.logger.warning(f"Account deletion failed: wrong password for user {user_id}")
                return ServiceResponse.error_response(
                    message="Passwort ist falsch",
                    errors=["WRONG_PASSWORD"]
                )
            
            # Store email for logging before deletion
            user_email = user.email
            
            # Delete user (will cascade delete personality profiles and recommendations)
            success = user.delete()
            if success:
                self.logger.info(f"User account successfully deleted: {user_email}")
                return ServiceResponse.success_response(
                    message="Account erfolgreich gelöscht"
                )
            else:
                return ServiceResponse.error_response(
                    message="Fehler beim Löschen des Accounts",
                    errors=["DELETE_ERROR"]
                )
                
        except Exception as e:
            self.logger.error(f"Error deleting user account: {str(e)}")
            return ServiceResponse.error_response(
                message="Fehler beim Löschen des Accounts",
                errors=["DELETE_ERROR"]
            )
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def search_users(self, query: str, limit: int = 20) -> ServiceResponse:
        """
        Searches users by email or name (admin function)
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            ServiceResponse with search results
        """
        try:
            search_term = f"%{query}%"
            
            users = User.query.filter(
                db.or_(
                    User.email.ilike(search_term),
                    User.first_name.ilike(search_term),
                    User.last_name.ilike(search_term)
                )
            ).limit(limit).all()
            
            user_results = [
                {
                    "user_id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "created_at": user.created_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "is_active": user.is_active,
                    "total_profiles": len(user.personality_profiles)
                }
                for user in users
            ]
            
            return ServiceResponse.success_response(
                message=f"{len(user_results)} Benutzer gefunden",
                data={"users": user_results, "query": query}
            )
            
        except Exception as e:
            self.logger.error(f"Error searching users: {str(e)}")
            return ServiceResponse.error_response(
                message="Fehler bei der Benutzersuche",
                errors=["SEARCH_ERROR"]
            )
    
    def validate_user_session(self, user_id: str) -> ServiceResponse:
        """
        Validates if user session is still valid
        
        Args:
            user_id: User identifier
            
        Returns:
            ServiceResponse with session validity
        """
        try:
            user = User.get_by_id(user_id)
            if not user:
                return ServiceResponse.error_response(
                    message="Sitzung ungültig",
                    errors=["INVALID_SESSION"]
                )
            
            if not user.is_active:
                return ServiceResponse.error_response(
                    message="Account wurde deaktiviert",
                    errors=["ACCOUNT_DISABLED"]
                )
            
            # Session is valid
            session_data = {
                "user_id": user.id,
                "display_name": user.display_name,
                "last_activity": datetime.utcnow().isoformat(),
                "is_verified": user.is_verified
            }
            
            return ServiceResponse.success_response(
                message="Sitzung gültig",
                data={"session": session_data}
            )
            
        except Exception as e:
            self.logger.error(f"Error validating user session: {str(e)}")
            return ServiceResponse.error_response(
                message="Fehler bei der Sitzungsvalidierung",
                errors=["SESSION_VALIDATION_ERROR"]
            )


# =============================================================================
# SERVICE FACTORY & SINGLETON
# =============================================================================

_user_service_instance = None

def get_user_service() -> UserService:
    """
    Returns singleton UserService instance
    
    Returns:
        UserService instance
    """
    global _user_service_instance
    if _user_service_instance is None:
        _user_service_instance = UserService()
    return _user_service_instance


# =============================================================================
# CONVENIENCE FUNCTIONS FOR ROUTES
# =============================================================================

def register_new_user(registration_data: dict) -> ServiceResponse:
    """
    Convenience function for user registration
    
    Args:
        registration_data: Raw registration data from form
        
    Returns:
        ServiceResponse
    """
    try:
        # Validate data with Pydantic
        validated_data = UserRegistrationRequest(**registration_data)
        
        # Register user
        service = get_user_service()
        return service.register_user(validated_data)
        
    except Exception as e:
        return ServiceResponse.error_response(
            message="Validierungsfehler in den Registrierungsdaten",
            errors=[str(e)]
        )

def authenticate_login(login_data: dict) -> ServiceResponse:
    """
    Convenience function for user authentication
    
    Args:
        login_data: Raw login data from form
        
    Returns:
        ServiceResponse
    """
    try:
        # Validate data with Pydantic
        validated_data = UserLoginRequest(**login_data)
        
        # Authenticate user
        service = get_user_service()
        return service.authenticate_user(validated_data)
        
    except Exception as e:
        return ServiceResponse.error_response(
            message="Validierungsfehler in den Login-Daten",
            errors=[str(e)]
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Schemas
    'UserRegistrationRequest',
    'UserLoginRequest', 
    'UserProfileUpdateRequest',
    'PasswordChangeRequest',
    'ServiceResponse',
    
    # Service
    'UserService',
    'get_user_service',
    
    # Convenience Functions
    'register_new_user',
    'authenticate_login'
]