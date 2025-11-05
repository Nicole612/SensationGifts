"""
Models Package - Production Ready Database Models für AI Gift Shop

Diese finale Version kombiniert das Beste aus der bestehenden Struktur
mit allen Verbesserungen aus dem Implementierungsplan.

Features:
- Clean Architecture: Models sind pure Data-Layer
- BaseModel mit Standard-Funktionalität
- Complete Model Registry
- Database Utilities & Helper Functions
- Sample Data Creation
- Performance-optimierte Queries
"""

from app.extensions import db
from datetime import datetime
from typing import Dict, Any, Optional, List
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy import func




class BaseModel:
    """
    Enhanced Base Model mit Standard-Feldern für alle Database Models
    
    Bietet:
    - Automatische UUID Primary Keys
    - Created/Updated Timestamps
    - Standard Save/Delete Methods
    - JSON Serialization Helpers
    - Query Utilities
    """
    
    @declared_attr
    def __tablename__(cls):
        """Automatische Table-Namen basierend auf Class-Namen"""
        return cls.__name__.lower()
    
    # === STANDARD FELDER ===
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # === STANDARD METHODEN ===
    
    def save(self):
        """Speichert das Model in der Database"""
        try:
            db.session.add(self)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            print(f"❌ Save error: {e}")
            return False
    
    def delete(self):
        """Löscht das Model aus der Database"""
        try:
            db.session.delete(self)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            print(f"❌ Delete error: {e}")
            return False
    
    def update(self, **kwargs):
        """Updated das Model mit neuen Werten"""
        try:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            self.updated_at = datetime.utcnow()
            db.session.commit()
            return self
        except Exception as e:
            db.session.rollback()
            print(f"❌ Update error: {e}")
            raise e
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert Model zu Dictionary für JSON-Serialization"""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            
            # Handle verschiedene Datentypen
            if isinstance(value, datetime):
                result[column.name] = value.isoformat()
            elif hasattr(value, 'value'):  # Enum values
                result[column.name] = value.value
            else:
                result[column.name] = value
                
        return result
    
    @classmethod
    def get_by_id(cls, id_value):
        """Holt Model by ID"""
        return cls.query.filter_by(id=id_value).first()
    
    @classmethod
    def get_all_active(cls):
        """Holt alle aktiven Models (falls is_active Feld vorhanden)"""
        if hasattr(cls, 'is_active'):
            return cls.query.filter_by(is_active=True).all()
        return cls.query.all()
    
    @classmethod
    def count_all(cls):
        """Zählt alle Records"""
        return cls.query.count()
    
    @classmethod
    def find_by(cls, **kwargs):
        """Flexible Suche nach Kriterien"""
        return cls.query.filter_by(**kwargs).first()
    
    @classmethod
    def find_all_by(cls, **kwargs):
        """Flexible Suche nach Kriterien (alle Ergebnisse)"""
        return cls.query.filter_by(**kwargs).all()


# === IMPORT ALL MODELS ===

# Core Models - User & Authentication
from .user import User, create_user, authenticate_user, get_user_by_id, get_user_by_email

# Personality Models mit Big Five + Limbic System
from .personality import (
    PersonalityProfile, EmotionalTrigger, LifestyleType, LimbicType, # Helper Functions
    get_personality_insights,
    calculate_limbic_type,
    get_gift_preferences  # Bonus function
)

# Gift System Models mit AI-Optimization
from .gift import (
    Gift, GiftCategory, GiftTag, GiftTagAssociation,
    PriceRange, GiftType, PersonalizationLevel,
    get_gifts_by_personality, get_gifts_by_category, 
    get_gifts_by_tags, search_gifts, get_featured_gifts
)

# Recommendation System Models mit ML-Tracking
from .recommendation import (
    Recommendation, RecommendationSession,
    RecommendationStatus, FeedbackType, AIModelType,
    create_recommendation_session, get_user_recommendation_history,
    get_successful_recommendations_for_gift, analyze_recommendation_performance
)

# Enhanced Database Models - Production Ready
from .enhanced_models import (
    EnhancedBase,
    EnhancedUser,
    EnhancedPersonalityProfile,
    EnhancedGiftCategory,
    EnhancedGift,
    EnhancedRecommendationSession,
    EnhancedRecommendation,
    EnhancedUserActivity,
    LimbicType as EnhancedLimbicType,
    RecommendationStatus as EnhancedRecommendationStatus,
    create_enhanced_sample_data
)


# === MODEL REGISTRY FÜR FLASK-SQLALCHEMY ===

# Alle Models die von Flask-SQLAlchemy registriert werden sollen
ALL_MODELS = [
    # Core Models
    User,
    PersonalityProfile,
    
    # Gift System Models
    GiftCategory,
    GiftTag,
    GiftTagAssociation,
    Gift,
    
    # Recommendation System Models
    RecommendationSession,
    Recommendation,
    
    # Enhanced Database Models - Production Ready
    EnhancedUser,
    EnhancedPersonalityProfile,
    EnhancedGiftCategory,
    EnhancedGift,
    EnhancedRecommendationSession,
    EnhancedRecommendation,
    EnhancedUserActivity
]


# === HELPER FUNCTIONS ===

def init_database():
    """
    Initialisiert die Database mit allen Tables
    
    Usage:
        from app.models import init_database
        init_database()
    """
    try:
        db.create_all()
        print(f"✅ Database initialized with {len(ALL_MODELS)} models")
        
        # Erstelle Sample Data wenn Database leer ist
        if User.query.count() == 0:
            print("📊 Creating sample data...")
            create_sample_data()
        
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


def drop_all_tables():
    """
    ACHTUNG: Löscht alle Tables! Nur für Development!
    """
    try:
        db.drop_all()
        print("⚠️ All tables dropped!")
        return True
    except Exception as e:
        print(f"❌ Failed to drop tables: {e}")
        return False


def get_model_by_name(model_name: str):
    """
    Holt Model-Class by Name
    
    Args:
        model_name: Name des Models (z.B. "User", "Gift")
    
    Returns:
        Model Class oder None
    """
    model_mapping = {model.__name__: model for model in ALL_MODELS}
    return model_mapping.get(model_name)


def get_all_model_names() -> List[str]:
    """Gibt Liste aller verfügbaren Model-Namen zurück"""
    return [model.__name__ for model in ALL_MODELS]


def validate_model_relationships():
    """
    Validiert alle Model-Relationships
    Nützlich für Debugging
    """
    print("🔍 Validating model relationships...")
    
    validation_results = []
    
    for model in ALL_MODELS:
        model_name = model.__name__
        relationships = []
        
        # Finde alle Relationships in dem Model
        try:
            for attr_name in dir(model):
                attr = getattr(model, attr_name)
                if hasattr(attr, 'property') and hasattr(attr.property, 'mapper'):
                    relationships.append(attr_name)
        except Exception as e:
            relationships.append(f"Error reading relationships: {e}")
        
        validation_results.append({
            'model': model_name,
            'relationships': relationships,
            'table_name': getattr(model, '__tablename__', 'unknown')
        })
    
    return validation_results


def count_all_records() -> Dict[str, int]:
    """Zählt alle Records in allen Tables"""
    counts = {}
    
    for model in ALL_MODELS:
        try:
            count = model.query.count()
            counts[model.__name__] = count
        except Exception as e:
            counts[model.__name__] = f"Error: {str(e)}"
    
    return counts


def get_database_stats() -> Dict[str, Any]:
    """Gibt umfassende Database-Statistiken zurück"""
    return {
        'total_models': len(ALL_MODELS),
        'model_names': get_all_model_names(),
        'record_counts': count_all_records(),
        'relationship_validation': validate_model_relationships(),
        'database_size_mb': get_database_size(),
        'last_updated': datetime.utcnow().isoformat()
    }


def get_database_size() -> float:
    """Schätzt Database-Größe in MB"""
    try:
        # Für SQLite - einfache Schätzung
        total_records = sum(
            model.query.count() for model in ALL_MODELS
        )
        estimated_size_mb = total_records * 0.001  # Grobe Schätzung
        return round(estimated_size_mb, 2)
    except Exception:
        return 0.0


def create_sample_data():
    """
    Erstellt umfassende Sample-Daten für Development/Testing
    """
    try:
        print("🌱 Creating comprehensive sample data...")
        
        # 1. Sample User
        sample_user, success, error = create_user(
            email="test@sensationgifts.de",
            password="testpass123",
            first_name="Max",
            last_name="Mustermann"
        )
        
        if not success:
            print(f"❌ Failed to create sample user: {error}")
            return False
        
        print(f"✅ Sample user created: {sample_user.email}")
        
        # 2. Sample Gift Categories
        categories_data = [
            {
                "name": "Für Kreative Köpfe",
                "slug": "kreative-koepfe",
                "description": "Geschenke für Menschen die gerne kreativ sind",
                "icon": "🎨",
                "color": "#FF6B6B"
            },
            {
                "name": "Tech-Enthusiasten",
                "slug": "tech-enthusiasten",
                "description": "Für alle die neue Technologien lieben",
                "icon": "💻",
                "color": "#4ECDC4"
            },
            {
                "name": "Wellness & Entspannung",
                "slug": "wellness-entspannung",
                "description": "Für erholsame Momente",
                "icon": "🧘",
                "color": "#95E1D3"
            }
        ]
        
        sample_categories = []
        for cat_data in categories_data:
            category = GiftCategory(**cat_data)
            if category.save():
                sample_categories.append(category)
                print(f"✅ Category created: {category.name}")
        
        # 3. Sample Tags
        tags_data = [
            {"name": "creative", "display_name": "Kreativ", "tag_type": "personality", "weight": 1.5},
            {"name": "tech", "display_name": "Technologie", "tag_type": "interest", "weight": 1.2},
            {"name": "relaxing", "display_name": "Entspannend", "tag_type": "mood", "weight": 1.0},
            {"name": "personalized", "display_name": "Personalisiert", "tag_type": "feature", "weight": 1.3}
        ]
        
        sample_tags = []
        for tag_data in tags_data:
            tag = GiftTag(**tag_data)
            if tag.save():
                sample_tags.append(tag)
                print(f"✅ Tag created: {tag.display_name}")
        
        # 4. Sample Gifts
        gifts_data = [
            {
                "name": "Profi Aquarell-Set",
                "short_description": "Hochwertiges Aquarell-Set für kreative Projekte",
                "price": 89.99,
                "price_range": PriceRange.AFFORDABLE,
                "gift_type": GiftType.PHYSICAL,
                "personalization_level": PersonalizationLevel.SIMPLE,
                "category": sample_categories[0] if sample_categories else None
            },
            {
                "name": "Smart Home Starter Kit",
                "short_description": "Einstieg in das Smart Home mit intelligenter Steuerung",
                "price": 149.99,
                "price_range": PriceRange.AFFORDABLE,
                "gift_type": GiftType.PHYSICAL,
                "personalization_level": PersonalizationLevel.MODERATE,
                "category": sample_categories[1] if len(sample_categories) > 1 else None
            },
            {
                "name": "Wellness-Massage Gutschein",
                "short_description": "Entspannende Massage für stressige Zeiten",
                "price": 79.99,
                "price_range": PriceRange.AFFORDABLE,
                "gift_type": GiftType.SERVICE,
                "personalization_level": PersonalizationLevel.NONE,
                "category": sample_categories[2] if len(sample_categories) > 2 else None
            }
        ]
        
        sample_gifts = []
        for gift_data in gifts_data:
            gift = Gift(**gift_data)
            if gift.save():
                sample_gifts.append(gift)
                print(f"✅ Gift created: {gift.name}")
                
                # Füge Tags hinzu
                if sample_tags:
                    gift.add_tag(sample_tags[0], relevance_score=0.8)
        
        # 5. Sample PersonalityProfile
        sample_profile = PersonalityProfile(
            buyer_user_id=sample_user.id,
            recipient_name="Meine Schwester Anna",
            occasion="geburtstag",
            relationship="familie",
            budget_min=50.0,
            budget_max=150.0,
            
            # Big Five Scores
            openness=0.8,
            conscientiousness=0.6,
            extraversion=0.7,
            agreeableness=0.8,
            neuroticism=0.3,
            
            # Limbic Scores
            stimulanz=0.6,
            dominanz=0.4,
            balance=0.7,
            
            # Preferences
            creative_type=True,
            tech_savvy=False,
            practical_type=False,
            
            # Details
            hobbies="Fotografie, Kunst, Reisen",
            interests="Nachhaltigkeit, Kultur"
        )
        
        if sample_profile.save():
            print(f"✅ Sample personality profile created")
            
            # 6. Sample Recommendation Session
            sample_session = create_recommendation_session(
                sample_profile, 
                AIModelType.OPENAI_GPT4
            )
            
            if sample_session:
                print(f"✅ Sample recommendation session created")
                
                # 7. Sample Recommendations
                for i, gift in enumerate(sample_gifts[:3]):
                    recommendation = Recommendation(
                        session_id=sample_session.id,
                        personality_profile_id=sample_profile.id,
                        gift_id=gift.id,
                        rank_position=i + 1,
                        match_score=0.85 - (i * 0.1),
                        confidence_score=0.9 - (i * 0.05),
                        reasoning=f"Passt perfekt zu {sample_profile.recipient_name}s kreativer Persönlichkeit"
                    )
                    
                    if recommendation.save():
                        print(f"✅ Recommendation {i+1} created")
        
        print("✅ Comprehensive sample data creation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def reset_database():
    """
    Komplett Database Reset - nur für Development!
    """
    try:
        print("🔄 Resetting database...")
        drop_all_tables()
        init_database()
        print("✅ Database reset completed")
        return True
    except Exception as e:
        print(f"❌ Database reset failed: {e}")
        return False


def backup_database(backup_name: str = None):
    """
    Erstellt ein Database-Backup (für SQLite)
    """
    try:
        import shutil
        import os
        
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        # Für SQLite - einfach File kopieren
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(project_root, 'data', 'database.db')
        backup_path = os.path.join(project_root, 'data', 'backups', backup_name)
        
        # Backup-Ordner erstellen
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        # Database kopieren
        shutil.copy2(db_path, backup_path)
        print(f"✅ Database backup created: {backup_name}")
        return backup_path
        
    except Exception as e:
        print(f"❌ Database backup failed: {e}")
        return None


# === EXPORT FÜR CLEAN IMPORTS ===

__all__ = [
    # Base Classes
    'BaseModel',
    
    # Core Models
    'User',
    
    # Personality Models
    'PersonalityProfile', 
    'EmotionalTrigger', 
    'LifestyleType',
    'LimbicType',
    
    # Gift Models
    'Gift', 
    'GiftCategory', 
    'GiftTag', 
    'GiftTagAssociation',
    'PriceRange', 
    'GiftType', 
    'PersonalizationLevel',
    
    # Recommendation Models
    'Recommendation', 
    'RecommendationSession',
    'RecommendationStatus', 
    'FeedbackType', 
    'AIModelType',
    
    # User Helper Functions
    'create_user',
    'authenticate_user',
    'get_user_by_id',
    'get_user_by_email',
    
    # Gift Helper Functions
    'get_gifts_by_personality',
    'get_gifts_by_category', 
    'get_gifts_by_tags', 
    'search_gifts', 
    'get_featured_gifts',
    
    # Recommendation Helper Functions
    'create_recommendation_session',
    'get_user_recommendation_history',
    'get_successful_recommendations_for_gift',
    'analyze_recommendation_performance',
    
    # Personality Helper Functions
    'get_personality_insights',
    'calculate_limbic_type',
    'get_gift_preferences',  # Bonus function
    
    # Database Utilities
    'init_database',
    'drop_all_tables',
    'get_model_by_name',
    'get_all_model_names',
    'validate_model_relationships',
    'count_all_records',
    'get_database_stats',
    'get_database_size',
    'create_sample_data',
    'reset_database',
    'backup_database',
    
    # Constants
    'ALL_MODELS'
]