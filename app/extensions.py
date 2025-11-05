"""
Extensions Management - AI Gift Shop (Production Ready)
=======================================================

Zentrale Extension Management für Flask
- Löst Circular Imports
- AI-Engine Integration Ready
- Performance Optimierung
- Monitoring & Logging
- Kompatibel mit der bestehenden Struktur
"""

from flask import request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_cors import CORS
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect
from flask_caching import Cache
import logging
from typing import Optional
from datetime import datetime


# Initialize all extensions
db = SQLAlchemy()
login_manager = LoginManager()
cors = CORS()
migrate = Migrate()
csrf = CSRFProtect()
cache = Cache()


# AI Engine Integration
ai_engine = None

# Logger setup
logger = logging.getLogger(__name__)

def init_extensions(app):
    """
    Initialize all extensions with Flask app
    Kompatibel mit der bestehenden Struktur!
    """
    logger.info("🔧 Initializing extensions...")
    
    # Database (nutzt deine settings.py)
    db.init_app(app) 
    migrate.init_app(app, db)
    logger.info("✅ Database extensions initialized")
    
    # Authentication (nutzt deine User-Models)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Bitte melden Sie sich an, um auf diese Seite zuzugreifen.'
    login_manager.login_message_category = 'info'
    
    # User Loader (verwendet deine bestehende Funktion)
    @login_manager.user_loader
    def load_user(user_id):
        from app.models import get_user_by_id
        return get_user_by_id(user_id)
    
    logger.info("✅ Authentication initialized")
    
    # CORS (erweitert deine bestehende Konfiguration)
    cors.init_app(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://localhost:8080"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        },
        r"/auth/*": {
            "origins": ["http://localhost:3000", "http://localhost:8080"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    logger.info("✅ CORS initialized")
    
    # CSRF Protection (deaktiviert für API)
    csrf.init_app(app)
    
    # CSRF komplett für API-Routen deaktivieren
    @app.before_request
    def skip_csrf_for_api():
        if request.path.startswith("/api/"):
            # CSRF für API-Routen deaktivieren
            app.config['WTF_CSRF_ENABLED'] = False



    logger.info("✅ CSRF protection initialized")
    
    # Caching (neu für Performance)
    cache.init_app(app, config={
        'CACHE_TYPE': 'simple',
        'CACHE_DEFAULT_TIMEOUT': 300
    })
    logger.info("✅ Caching initialized")

     # 6. AI ENGINE INTEGRATION (für später)
    init_ai_engine(app)
    
    
    logger.info("🚀 All extensions initialized successfully!")

def init_ai_engine(app):
    """
    Initialize AI Engine Integration
    BEREIT FÜR DEINE AI-ENGINE!
    """
    global ai_engine
        
    try:
        # Hier würde deine AI-Engine initialisiert werden
        # from ai_engine import AIEngine
        # ai_engine = AIEngine(app.config)
            
        logger.info("🤖 AI Engine initialized (placeholder)")
            
    except Exception as e:
        logger.warning(f"AI Engine initialization failed: {e}")
        ai_engine = None



def get_ai_engine():
    """
    Get AI Engine Instance
    READY FÜR DEINE AI-ENGINE INTEGRATION!
    """
    global ai_engine
    
    if ai_engine is None:
        logger.warning("AI Engine not initialized")
        return None
    
    return ai_engine


def get_cache():
    """Get Cache Instance"""
    return cache


def get_db():
    """Get Database Instance"""
    return db