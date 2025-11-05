"""
Database Configuration - SensationGifts
======================================

SQLAlchemy Database Configuration mit Support für:
- SQLite (Development) 
- PostgreSQL (Production)
- Performance Optimierung
- Connection Pooling
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Base für alle Models
Base = declarative_base()

class DatabaseConfig:
    """Database Configuration Management"""
    
    @staticmethod
    def get_database_uri(settings) -> str:
        """
        Generiert Database URI basierend auf Settings
        
        Args:
            settings: Settings-Objekt von get_settings()
            
        Returns:
            Database URI String
        """
        try:
            database_url = settings.database_url
            
            # SQLite Path correction
            if database_url.startswith('sqlite:///'):
                # Stelle sicher dass data/ Verzeichnis existiert
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                data_dir = os.path.join(project_root, 'data')
                os.makedirs(data_dir, exist_ok=True)
                
                # Absolute path für SQLite
                if 'data/database.db' in database_url:
                    db_path = os.path.join(data_dir, 'database.db')
                    return f'sqlite:///{db_path}'
            
            return database_url
            
        except Exception as e:
            logger.error(f"Database URI generation failed: {e}")
            # Fallback
            return 'sqlite:///data/database.db'
    
    @staticmethod
    def get_engine_options(settings) -> Dict[str, Any]:
        """
        Engine Options basierend auf Database Type
        
        Args:
            settings: Settings-Objekt
            
        Returns:
            Engine Options Dictionary
        """
        database_url = settings.database_url
        
        if database_url.startswith('sqlite:'):
            return {
                'echo': settings.debug,
                'poolclass': StaticPool,
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 20
                },
                'pool_pre_ping': True
            }
        
        elif database_url.startswith('postgresql:'):
            return {
                'echo': settings.debug,
                'pool_size': 10,
                'max_overflow': 20,
                'pool_recycle': 3600,
                'pool_pre_ping': True,
                'pool_timeout': 30
            }
        
        else:
            # Default options
            return {
                'echo': settings.debug,
                'pool_pre_ping': True
            }
    
    @staticmethod
    def create_database_engine(settings):
        """
        Erstellt konfigurierten Database Engine
        
        Args:
            settings: Settings-Objekt
            
        Returns:
            SQLAlchemy Engine
        """
        try:
            database_uri = DatabaseConfig.get_database_uri(settings)
            engine_options = DatabaseConfig.get_engine_options(settings)
            
            logger.info(f"Creating database engine for: {database_uri}")
            
            engine = create_engine(database_uri, **engine_options)
            
            # Test connection
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            
            logger.info("Database engine created successfully")
            return engine
            
        except Exception as e:
            logger.error(f"Database engine creation failed: {e}")
            raise
    
    @staticmethod
    def create_session_factory(engine):
        """
        Erstellt Session Factory
        
        Args:
            engine: SQLAlchemy Engine
            
        Returns:
            Session Factory
        """
        return sessionmaker(bind=engine, expire_on_commit=False)


def init_database(app):
    """
    Initialisiert Database für Flask App
    
    Args:
        app: Flask Application
    """
    try:
        from config.settings import get_settings
        settings = get_settings()
        
        # Database URI für Flask-SQLAlchemy
        database_uri = DatabaseConfig.get_database_uri(settings)
        app.config['SQLALCHEMY_DATABASE_URI'] = database_uri
        
        # Engine Options für Flask-SQLAlchemy
        engine_options = DatabaseConfig.get_engine_options(settings)
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = engine_options
        
        # Additional SQLAlchemy Settings
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        app.config['SQLALCHEMY_RECORD_QUERIES'] = settings.debug
        
        logger.info(f"Database initialized: {settings.database_type}")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def get_database_health() -> Dict[str, Any]:
    """
    Database Health Check
    
    Returns:
        Health Status Dictionary
    """
    try:
        from config.settings import get_settings
        from app.extensions import db
        
        settings = get_settings()
        
        # Test query
        result = db.session.execute("SELECT 1").scalar()
        
        health = {
            'status': 'healthy',
            'database_type': settings.database_type,
            'connection': 'active',
            'test_query_result': result,
            'timestamp': os.environ.get('CURRENT_TIME', 'unknown')
        }
        
        return health
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': os.environ.get('CURRENT_TIME', 'unknown')
        }


# Export
__all__ = [
    'Base',
    'DatabaseConfig', 
    'init_database',
    'get_database_health'
]