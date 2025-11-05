"""
Complete Database Models - Production Ready
==========================================

🗄️ Vollständige Database Models die direkt in dein Projekt können
- Optimized für Performance
- Relationships korrekt definiert
- Indexing für schnelle Queries
- JSON Fields für flexible Daten
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import UUID
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any

# Create separate base for enhanced models to avoid conflicts
EnhancedBase = declarative_base()

class LimbicType(Enum):
    """Limbic Types für Personality Classification"""
    DISCIPLINED = "disciplined"
    TRADITIONALIST = "traditionalist"
    PERFORMER = "performer"
    ADVENTURER = "adventurer"
    HARMONIZER = "harmonizer"
    HEDONIST = "hedonist"
    PIONEER = "pioneer"

class RecommendationStatus(Enum):
    """Status für Recommendation Tracking"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# ===== USER MODELS =====

class User(EnhancedBase):
    """
    User Model mit erweiterten Features
    
    Features:
    - Secure Password Hashing
    - Preferences Storage
    - Activity Tracking
    - Soft Delete
    """
    __tablename__ = 'users'
    
    # Primary Key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Authentication
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_verified = Column(Boolean, default=False)
    
    # Profile Information
    first_name = Column(String(100))
    last_name = Column(String(100))
    display_name = Column(String(150))
    
    # Preferences
    preferred_language = Column(String(10), default='de')
    preferred_currency = Column(String(3), default='EUR')
    timezone = Column(String(50), default='Europe/Berlin')
    
    # Budget Defaults
    default_budget_min = Column(Float, default=20.0)
    default_budget_max = Column(Float, default=100.0)
    
    # Account Status
    is_active = Column(Boolean, default=True)
    is_premium = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    last_activity = Column(DateTime)
    
    # Soft Delete
    deleted_at = Column(DateTime)
    
    # JSON Fields für flexible Daten
    user_preferences = Column(JSON, default=dict)
    notification_settings = Column(JSON, default=dict)
    
    # Relationships
    personality_profiles = relationship(
        "PersonalityProfile", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )
    recommendation_sessions = relationship(
        "RecommendationSession", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )
    user_activities = relationship(
        "UserActivity", 
        back_populates="user", 
        cascade="all, delete-orphan"
    )
    
    # Indexes für Performance
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_created_at', 'created_at'),
        Index('idx_user_last_activity', 'last_activity'),
    )
    
    # ===== METHODS =====
    
    def set_password(self, password: str) -> None:
        """Hash und set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password: str) -> bool:
        """Verify password"""
        return check_password_hash(self.password_hash, password)
    
    @property
    def full_name(self) -> str:
        """Get full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.display_name or self.email.split('@')[0]
    
    def get_display_name(self) -> str:
        """Get display name with fallback"""
        return self.display_name or self.first_name or self.email.split('@')[0]
    
    def update_last_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
    
    def mark_login(self) -> None:
        """Mark user login"""
        self.last_login = datetime.utcnow()
        self.update_last_activity()
    
    def soft_delete(self) -> None:
        """Soft delete user"""
        self.deleted_at = datetime.utcnow()
        self.is_active = False
    
    def get_default_budget(self) -> Dict[str, float]:
        """Get default budget range"""
        return {
            'min': self.default_budget_min or 20.0,
            'max': self.default_budget_max or 100.0
        }
    
    def set_preference(self, key: str, value: Any) -> None:
        """Set user preference"""
        if self.user_preferences is None:
            self.user_preferences = {}
        self.user_preferences[key] = value
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference"""
        if self.user_preferences is None:
            return default
        return self.user_preferences.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'display_name': self.get_display_name(),
            'full_name': self.full_name,
            'preferred_language': self.preferred_language,
            'preferred_currency': self.preferred_currency,
            'timezone': self.timezone,
            'default_budget': self.get_default_budget(),
            'is_active': self.is_active,
            'is_premium': self.is_premium,
            'is_verified': self.is_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }

# ===== PERSONALITY MODELS =====

class PersonalityProfile(EnhancedBase):
    """
    Enhanced Personality Profile mit Big Five + Limbic System
    
    Features:
    - Big Five Traits (wissenschaftlich fundiert)
    - Limbic System Analysis (emotional drivers)
    - Auto-computed Limbic Type
    - Gift Preferences Integration
    - Versioning Support
    """
    __tablename__ = 'personality_profiles'
    
    # Primary Key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # User Reference
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    
    # Big Five Personality Traits (1.0 - 5.0)
    openness = Column(Float, nullable=False)
    conscientiousness = Column(Float, nullable=False)
    extraversion = Column(Float, nullable=False)
    agreeableness = Column(Float, nullable=False)
    neuroticism = Column(Float, nullable=False)
    
    # Limbic System Scores (1.0 - 5.0)
    stimulanz = Column(Float, nullable=False)  # Stimulation-seeking
    dominanz = Column(Float, nullable=False)   # Dominance-seeking
    balance = Column(Float, nullable=False)    # Balance-seeking
    
    # Computed Fields
    limbic_type_auto = Column(String(50))  # Auto-computed limbic type
    emotional_stability = Column(Float)    # Computed from neuroticism
    personality_summary = Column(Text)     # AI-generated summary
    
    # Gift Preferences
    budget_min = Column(Float)
    budget_max = Column(Float)
    allergies = Column(Text)              # Comma-separated
    dislikes = Column(Text)               # Comma-separated
    preferred_categories = Column(Text)   # Comma-separated
    
    # Enhanced Preferences
    prefers_experiences = Column(Boolean, default=False)
    likes_personalization = Column(Boolean, default=True)
    luxury_appreciation = Column(Float, default=3.0)  # 1-5 scale
    sustainability_focus = Column(Float, default=3.0)  # 1-5 scale
    
    # Assessment Information
    assessment_completed = Column(Boolean, default=False)
    assessment_version = Column(String(20), default='1.0')
    assessment_duration_seconds = Column(Integer)
    
    # Context Information
    age_range = Column(String(20))
    additional_context = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # JSON Fields
    raw_answers = Column(JSON)           # Raw quiz answers
    trait_details = Column(JSON)         # Detailed trait analysis
    suggested_categories = Column(JSON)   # AI-suggested gift categories
    emotional_triggers = Column(JSON)    # Emotional purchase triggers
    
    # Relationships
    user = relationship("User", back_populates="personality_profiles")
    
    # Indexes
    __table_args__ = (
        Index('idx_personality_user_id', 'user_id'),
        Index('idx_personality_limbic_type', 'limbic_type_auto'),
        Index('idx_personality_created_at', 'created_at'),
    )
    
    # ===== METHODS =====
    
    @property
    def big_five_vector(self) -> Dict[str, float]:
        """Get Big Five as vector"""
        return {
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism
        }
    
    @property
    def limbic_vector(self) -> Dict[str, float]:
        """Get Limbic scores as vector"""
        return {
            'stimulanz': self.stimulanz,
            'dominanz': self.dominanz,
            'balance': self.balance
        }
    
    def compute_limbic_type(self) -> LimbicType:
        """
        Compute Limbic Type based on scores
        
        Algorithm:
        - Highest scoring dimension determines base type
        - Secondary dimensions modify the type
        """
        scores = self.limbic_vector
        
        # Find dominant dimension
        dominant = max(scores, key=scores.get)
        dominant_score = scores[dominant]
        
        # Get secondary dimensions
        secondary_scores = {k: v for k, v in scores.items() if k != dominant}
        secondary = max(secondary_scores, key=secondary_scores.get)
        secondary_score = secondary_scores[secondary]
        
        # Determine type based on dominant + secondary
        if dominant == 'stimulanz':
            if secondary == 'dominanz' and secondary_score >= 3.5:
                return LimbicType.PERFORMER
            elif secondary == 'balance' and secondary_score >= 3.5:
                return LimbicType.ADVENTURER
            else:
                return LimbicType.HEDONIST
        
        elif dominant == 'dominanz':
            if secondary == 'stimulanz' and secondary_score >= 3.5:
                return LimbicType.PERFORMER
            elif secondary == 'balance' and secondary_score >= 3.5:
                return LimbicType.PIONEER
            else:
                return LimbicType.PERFORMER
        
        elif dominant == 'balance':
            if secondary == 'stimulanz' and secondary_score >= 3.5:
                return LimbicType.PIONEER
            elif secondary == 'dominanz' and secondary_score >= 3.5:
                return LimbicType.DISCIPLINED
            else:
                return LimbicType.HARMONIZER
        
        # Default fallback
        return LimbicType.HARMONIZER
    
    def update_computed_fields(self) -> None:
        """Update all computed fields"""
        # Limbic Type
        self.limbic_type_auto = self.compute_limbic_type().value
        
        # Emotional Stability (inverse of neuroticism)
        self.emotional_stability = 6.0 - self.neuroticism
        
        # Update timestamp
        self.updated_at = datetime.utcnow()
    
    def _get_dominant_big_five_traits(self) -> List[str]:
        """Get dominant Big Five traits (score >= 3.5)"""
        traits = []
        for trait, score in self.big_five_vector.items():
            if score >= 3.5:
                traits.append(trait)
        return traits
    
    @property
    def allergies_list(self) -> List[str]:
        """Get allergies as list"""
        if not self.allergies:
            return []
        return [a.strip() for a in self.allergies.split(',') if a.strip()]
    
    @property
    def dislikes_list(self) -> List[str]:
        """Get dislikes as list"""
        if not self.dislikes:
            return []
        return [d.strip() for d in self.dislikes.split(',') if d.strip()]
    
    @property
    def preferred_categories_list(self) -> List[str]:
        """Get preferred categories as list"""
        if not self.preferred_categories:
            return []
        return [c.strip() for c in self.preferred_categories.split(',') if c.strip()]
    
    @property
    def emotional_triggers_list(self) -> List[str]:
        """Get emotional triggers as list"""
        if not self.emotional_triggers:
            return []
        return self.emotional_triggers if isinstance(self.emotional_triggers, list) else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'big_five_scores': self.big_five_vector,
            'limbic_scores': self.limbic_vector,
            'limbic_type': self.limbic_type_auto,
            'emotional_stability': self.emotional_stability,
            'gift_preferences': {
                'budget_min': self.budget_min,
                'budget_max': self.budget_max,
                'allergies': self.allergies_list,
                'dislikes': self.dislikes_list,
                'preferred_categories': self.preferred_categories_list,
                'prefers_experiences': self.prefers_experiences,
                'likes_personalization': self.likes_personalization,
                'luxury_appreciation': self.luxury_appreciation,
                'sustainability_focus': self.sustainability_focus
            },
            'assessment_info': {
                'completed': self.assessment_completed,
                'version': self.assessment_version,
                'duration_seconds': self.assessment_duration_seconds,
                'age_range': self.age_range
            },
            'insights': {
                'personality_summary': self.personality_summary,
                'dominant_traits': self._get_dominant_big_five_traits(),
                'suggested_categories': self.suggested_categories or [],
                'emotional_triggers': self.emotional_triggers_list
            },
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

# ===== GIFT MODELS =====

class GiftCategory(EnhancedBase):
    """
    Gift Categories für Organisation
    """
    __tablename__ = 'gift_categories'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    parent_category_id = Column(Integer, ForeignKey('gift_categories.id'))
    
    # Metadata
    is_active = Column(Boolean, default=True)
    sort_order = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    parent_category = relationship("GiftCategory", remote_side=[id])
    gifts = relationship("Gift", back_populates="category")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'parent_category_id': self.parent_category_id,
            'is_active': self.is_active,
            'sort_order': self.sort_order
        }

class Gift(EnhancedBase):
    """
    Gift Database Model
    """
    __tablename__ = 'gifts'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Pricing
    price = Column(Float, nullable=False)
    price_currency = Column(String(3), default='EUR')
    
    # Classification
    category_id = Column(Integer, ForeignKey('gift_categories.id'))
    
    # Attributes
    brand = Column(String(100))
    material = Column(String(100))
    dimensions = Column(String(100))
    weight = Column(String(50))
    
    # Suitability
    occasions = Column(Text)  # Comma-separated
    age_ranges = Column(Text)  # Comma-separated
    relationships = Column(Text)  # Comma-separated
    
    # Metadata
    tags = Column(Text)  # Comma-separated
    allergen_info = Column(Text)
    care_instructions = Column(Text)
    
    # Ratings & Popularity
    rating = Column(Float, default=0.0)
    popularity_score = Column(Float, default=0.0)
    view_count = Column(Integer, default=0)
    recommendation_count = Column(Integer, default=0)
    
    # Availability
    is_available = Column(Boolean, default=True)
    is_seasonal = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # JSON Fields
    personality_match_data = Column(JSON)  # Personality matching metadata
    vendor_data = Column(JSON)             # Vendor information
    
    # Relationships
    category = relationship("GiftCategory", back_populates="gifts")
    
    # Indexes
    __table_args__ = (
        Index('idx_gift_category_price', 'category_id', 'price'),
        Index('idx_gift_popularity', 'popularity_score'),
        Index('idx_gift_rating', 'rating'),
        Index('idx_gift_available', 'is_available'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'price': self.price,
            'price_currency': self.price_currency,
            'category': self.category.to_dict() if self.category else None,
            'brand': self.brand,
            'material': self.material,
            'dimensions': self.dimensions,
            'weight': self.weight,
            'occasions': self.occasions.split(',') if self.occasions else [],
            'age_ranges': self.age_ranges.split(',') if self.age_ranges else [],
            'relationships': self.relationships.split(',') if self.relationships else [],
            'tags': self.tags.split(',') if self.tags else [],
            'allergen_info': self.allergen_info,
            'rating': self.rating,
            'popularity_score': self.popularity_score,
            'is_available': self.is_available,
            'is_seasonal': self.is_seasonal,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

# ===== RECOMMENDATION MODELS =====

class RecommendationSession(EnhancedBase):
    """
    Recommendation Session Tracking
    """
    __tablename__ = 'recommendation_sessions'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    
    # Request Parameters
    occasion = Column(String(100), nullable=False)
    relationship = Column(String(100), nullable=False)
    budget_min = Column(Float)
    budget_max = Column(Float)
    number_of_recommendations = Column(Integer, default=5)
    
    # AI Configuration
    optimization_goal = Column(String(50), default='balance')
    ai_model_used = Column(String(100))
    
    # Status
    status = Column(String(20), default=RecommendationStatus.PENDING.value)
    
    # Performance Metrics
    processing_time_ms = Column(Integer)
    cache_hit = Column(Boolean, default=False)
    
    # Results
    recommendations_generated = Column(Integer, default=0)
    avg_confidence_score = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # JSON Fields
    request_metadata = Column(JSON)
    ai_response_metadata = Column(JSON)
    
    # Relationships
    user = relationship("User", back_populates="recommendation_sessions")
    recommendations = relationship("Recommendation", back_populates="session")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'occasion': self.occasion,
            'relationship': self.relationship,
            'budget_min': self.budget_min,
            'budget_max': self.budget_max,
            'number_of_recommendations': self.number_of_recommendations,
            'optimization_goal': self.optimization_goal,
            'ai_model_used': self.ai_model_used,
            'status': self.status,
            'processing_time_ms': self.processing_time_ms,
            'cache_hit': self.cache_hit,
            'recommendations_generated': self.recommendations_generated,
            'avg_confidence_score': self.avg_confidence_score,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class Recommendation(EnhancedBase):
    """
    Individual Recommendation Results
    """
    __tablename__ = 'recommendations'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey('recommendation_sessions.id'), nullable=False)
    
    # Recommendation Data
    gift_name = Column(String(200), nullable=False)
    description = Column(Text)
    price_estimate = Column(Float)
    
    # AI Analysis
    confidence_score = Column(Float, nullable=False)
    reasoning = Column(Text)
    personality_match_score = Column(Float)
    
    # Classification
    category = Column(String(100))
    source = Column(String(50))  # 'ai_quick', 'ai_detailed', 'fallback'
    
    # User Interaction
    user_rating = Column(Integer)  # 1-5 stars
    user_feedback = Column(Text)
    clicked = Column(Boolean, default=False)
    purchased = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # JSON Fields
    ai_metadata = Column(JSON)
    
    # Relationships
    session = relationship("RecommendationSession", back_populates="recommendations")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'session_id': self.session_id,
            'gift_name': self.gift_name,
            'description': self.description,
            'price_estimate': self.price_estimate,
            'confidence_score': self.confidence_score,
            'reasoning': self.reasoning,
            'personality_match_score': self.personality_match_score,
            'category': self.category,
            'source': self.source,
            'user_rating': self.user_rating,
            'user_feedback': self.user_feedback,
            'clicked': self.clicked,
            'purchased': self.purchased,
            'created_at': self.created_at.isoformat()
        }

# ===== ACTIVITY TRACKING =====

class UserActivity(EnhancedBase):
    """
    User Activity Tracking für Analytics
    """
    __tablename__ = 'user_activities'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    
    # Activity Information
    activity_type = Column(String(50), nullable=False)  # 'login', 'quiz', 'recommendation', etc.
    activity_description = Column(String(200))
    
    # Context
    ip_address = Column(String(45))
    user_agent = Column(Text)
    referrer = Column(String(500))
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # JSON Fields
    activity_data = Column(JSON)
    
    # Relationships
    user = relationship("User", back_populates="user_activities")
    
    # Indexes
    __table_args__ = (
        Index('idx_activity_user_type', 'user_id', 'activity_type'),
        Index('idx_activity_created_at', 'created_at'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'activity_type': self.activity_type,
            'activity_description': self.activity_description,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'referrer': self.referrer,
            'created_at': self.created_at.isoformat(),
            'activity_data': self.activity_data
        }

# ===== HELPER FUNCTIONS =====

def create_sample_data():
    """Create sample data for testing"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # This would be called in your data seeding script
    # engine = create_engine('sqlite:///test.db')
    # Base.metadata.create_all(engine)
    # Session = sessionmaker(bind=engine)
    # session = Session()
    
    # Sample User
    sample_user = User(
        email='test@sensationgifts.com',
        first_name='Test',
        last_name='User',
        preferred_language='de',
        preferred_currency='EUR'
    )
    sample_user.set_password('TestPassword123!')
    
    # Sample Personality Profile
    sample_personality = PersonalityProfile(
        user=sample_user,
        openness=4.2,
        conscientiousness=3.8,
        extraversion=3.5,
        agreeableness=4.1,
        neuroticism=2.3,
        stimulanz=3.7,
        dominanz=2.9,
        balance=4.0,
        assessment_completed=True
    )
    sample_personality.update_computed_fields()
    
    # Sample Gift Category
    sample_category = GiftCategory(
        name='Kunst & Kreativität',
        description='Kreative Geschenke für künstlerische Menschen'
    )
    
    # Sample Gift
    sample_gift = Gift(
        name='Aquarell-Set Professional',
        description='Hochwertiges Aquarell-Set für Künstler',
        price=45.99,
        category=sample_category,
        occasions='birthday,christmas',
        tags='kreativ,kunst,malen',
        rating=4.5,
        popularity_score=0.8
    )
    
    return [sample_user, sample_personality, sample_category, sample_gift]

# Export all models
__all__ = [
    'EnhancedBase',
    'User',
    'PersonalityProfile',
    'GiftCategory',
    'Gift',
    'RecommendationSession',
    'Recommendation',
    'UserActivity',
    'LimbicType',
    'RecommendationStatus',
    'create_sample_data'
] 