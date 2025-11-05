"""
Authentication Routes - AI Gift Shop (Production Ready)
=======================================================

Vollständige Authentifizierung mit:
- Login/Register
- JWT Token Management
- Password Hashing
- Session Management
- User Profile Management
"""

from flask import Blueprint, request, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta
import os
from app.extensions import db
from app.utils.formatters import api_response
from app.models import User, PersonalityProfile
import logging

# Setup
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
logger = logging.getLogger(__name__)

# JWT Secret (in production sollte das in .env sein)
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')


@auth_bp.route('/register', methods=['POST'])
def register():
    """Registriert einen neuen Benutzer"""
    try:
        data = request.get_json()
        
        if not data:
            return api_response(error='No data provided', status=400)
        
        email = data.get('email')
        password = data.get('password')
        name = data.get('name', '')
        
        # Validierung
        if not email or not password:
            return api_response(error='Email and password are required', status=400)
        
        if len(password) < 6:
            return api_response(error='Password must be at least 6 characters', status=400)
        
        # Prüfe ob User bereits existiert
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return api_response(error='User with this email already exists', status=400)
        
        # Erstelle neuen User
        hashed_password = generate_password_hash(password)
        new_user = User(
            email=email,
            password_hash=hashed_password,
            name=name,
            created_at=datetime.utcnow()
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        # Login nach Registrierung
        login_user(new_user)
        
        # Generiere JWT Token
        token = generate_jwt_token(new_user.id)
        
        logger.info(f"✅ New user registered: {email}")
        
        return api_response(
            data={
                'user': {
                    'id': new_user.id,
                    'email': new_user.email,
                    'name': new_user.name,
                    'created_at': new_user.created_at.isoformat()
                },
                'token': token
            },
            message='Registration successful!',
            status=201
        )
        
    except Exception as e:
        logger.error(f"❌ Registration failed: {e}")
        db.session.rollback()
        return api_response(error='Registration failed', status=500)


@auth_bp.route('/login', methods=['POST'])
def login():
    """Loggt einen Benutzer ein"""
    try:
        data = request.get_json()
        
        if not data:
            return api_response(error='No data provided', status=400)
        
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return api_response(error='Email and password are required', status=400)
        
        # Finde User
        user = User.query.filter_by(email=email).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            return api_response(error='Invalid email or password', status=401)
        
        # Login
        login_user(user)
        
        # Generiere JWT Token
        token = generate_jwt_token(user.id)
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"✅ User logged in: {email}")
        
        return api_response(
            data={
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'name': user.name,
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None
                },
                'token': token
            },
            message='Login successful!',
            status=200
        )
        
    except Exception as e:
        logger.error(f"❌ Login failed: {e}")
        return api_response(error='Login failed', status=500)


@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    """Loggt einen Benutzer aus"""
    try:
        logout_user()
        logger.info(f"✅ User logged out: {current_user.email if current_user else 'Unknown'}")
        
        return api_response(
            message='Logout successful!',
            status=200
        )
        
    except Exception as e:
        logger.error(f"❌ Logout failed: {e}")
        return api_response(error='Logout failed', status=500)


@auth_bp.route('/profile', methods=['GET'])
@login_required
def get_profile():
    """Holt das Benutzerprofil"""
    try:
        # Hole User mit Profildaten
        user = User.query.get(current_user.id)
        
        # Hole Persönlichkeitsprofile
        personality_profiles = PersonalityProfile.query.filter_by(user_id=user.id).all()
        
        profile_data = {
            'id': user.id,
            'email': user.email,
            'name': user.name,
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'personality_profiles': [
                {
                    'id': profile.id,
                    'limbic_type': profile.limbic_type,
                    'created_at': profile.created_at.isoformat()
                }
                for profile in personality_profiles
            ]
        }
        
        return api_response(
            data={'profile': profile_data},
            message='Profile loaded successfully!',
            status=200
        )
        
    except Exception as e:
        logger.error(f"❌ Get profile failed: {e}")
        return api_response(error='Failed to load profile', status=500)


@auth_bp.route('/profile', methods=['PUT'])
@login_required
def update_profile():
    """Aktualisiert das Benutzerprofil"""
    try:
        data = request.get_json()
        
        if not data:
            return api_response(error='No data provided', status=400)
        
        user = User.query.get(current_user.id)
        
        # Update erlaubte Felder
        if 'name' in data:
            user.name = data['name']
        
        if 'email' in data:
            # Prüfe ob Email bereits existiert
            existing_user = User.query.filter_by(email=data['email']).first()
            if existing_user and existing_user.id != user.id:
                return api_response(error='Email already exists', status=400)
            user.email = data['email']
        
        db.session.commit()
        
        logger.info(f"✅ Profile updated: {user.email}")
        
        return api_response(
            data={
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'name': user.name,
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None
                }
            },
            message='Profile updated successfully!',
            status=200
        )
        
    except Exception as e:
        logger.error(f"❌ Update profile failed: {e}")
        db.session.rollback()
        return api_response(error='Failed to update profile', status=500)


@auth_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """Ändert das Passwort"""
    try:
        data = request.get_json()
        
        if not data:
            return api_response(error='No data provided', status=400)
        
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not current_password or not new_password:
            return api_response(error='Current and new password are required', status=400)
        
        if len(new_password) < 6:
            return api_response(error='New password must be at least 6 characters', status=400)
        
        user = User.query.get(current_user.id)
        
        # Prüfe aktuelles Passwort
        if not check_password_hash(user.password_hash, current_password):
            return api_response(error='Current password is incorrect', status=401)
        
        # Update Passwort
        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        
        logger.info(f"✅ Password changed: {user.email}")
        
        return api_response(
            message='Password changed successfully!',
            status=200
        )
        
    except Exception as e:
        logger.error(f"❌ Change password failed: {e}")
        db.session.rollback()
        return api_response(error='Failed to change password', status=500)


def generate_jwt_token(user_id: int) -> str:
    """Generiert einen JWT Token"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(days=30),  # 30 Tage gültig
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')


def verify_jwt_token(token: str) -> dict:
    """Verifiziert einen JWT Token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError('Token has expired')
    except jwt.InvalidTokenError:
        raise ValueError('Invalid token')


@auth_bp.route('/verify-token', methods=['POST'])
def verify_token():
    """Verifiziert einen JWT Token"""
    try:
        data = request.get_json()
        token = data.get('token')
        
        if not token:
            return api_response(error='No token provided', status=400)
        
        payload = verify_jwt_token(token)
        user = User.query.get(payload['user_id'])
        
        if not user:
            return api_response(error='User not found', status=404)
        
        return api_response(
            data={
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'name': user.name
                },
                'valid': True
            },
            message='Token is valid!',
            status=200
        )
        
    except ValueError as e:
        return api_response(error=str(e), status=401)
    except Exception as e:
        logger.error(f"❌ Token verification failed: {e}")
        return api_response(error='Token verification failed', status=500)