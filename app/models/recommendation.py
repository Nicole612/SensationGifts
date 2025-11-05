"""
Recommendation Models - E-Commerce optimiertes Empfehlungs-System (Session 2)

INNOVATION: Lernende AI-Empfehlungen mit vollständigem E-Commerce Tracking
- RecommendationSession für komplette Shopping-Journey
- User-Feedback Integration für kontinuierliches Lernen
- E-Commerce Metriken (CTR, Conversion Rate, etc.)
- Explainable AI: Warum wurde etwas empfohlen?

Clean Architecture: Pure Data-Models ohne Business-Logic
Business-Logic kommt später in Services (Session 4)
"""

from app.extensions import db
from app.models import BaseModel
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import uuid
from enum import Enum
from sqlalchemy import and_, or_, desc


class RecommendationStatus(Enum):
    """Status einer Empfehlung im User-Journey"""
    GENERATED = "generated"       # Gerade von AI erstellt
    PRESENTED = "presented"       # User gezeigt
    CLICKED = "clicked"          # User hat draufgeklickt  
    PURCHASED = "purchased"      # User hat gekauft
    REJECTED = "rejected"        # User explizit abgelehnt
    EXPIRED = "expired"          # Zu alt / nicht mehr relevant


class FeedbackType(Enum):
    """Arten von User-Feedback"""
    THUMBS_UP = "thumbs_up"       # 👍 Gute Empfehlung
    THUMBS_DOWN = "thumbs_down"   # 👎 Schlechte Empfehlung
    TOO_EXPENSIVE = "too_expensive"   # 💰 Zu teuer
    TOO_CHEAP = "too_cheap"       # 💸 Zu billig
    NOT_PERSONAL = "not_personal" # 😐 Zu unpersönlich
    TOO_PERSONAL = "too_personal" # 😳 Zu persönlich
    WRONG_STYLE = "wrong_style"   # 🎨 Falscher Stil
    PERFECT_MATCH = "perfect_match"   # 💯 Perfekt!


class AIModelType(Enum):
    """Welches AI-Model für die Empfehlung verwendet wurde"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    GROQ_MIXTRAL = "groq_mixtral"
    GEMINI_PRO = "gemini_pro"
    ANTHROPIC_CLAUDE = "anthropic_claude"


class RecommendationSession(db.Model, BaseModel):
    """
    Eine Session = Eine Empfehlungs-Anfrage mit mehreren Geschenken
    
    User fragt: "Geschenk für Schwester zum Geburtstag"
    → Session wird erstellt
    → 3-5 Recommendations werden generiert
    → User gibt Feedback
    → Session wird abgeschlossen
    """
    
    __tablename__ = 'recommendation_sessions'
    
    # === PRIMARY KEY ===
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # === FOREIGN KEYS ===
    personality_profile_id = db.Column(db.String(36), db.ForeignKey('personality_profiles.id'), nullable=False)
    
    # === SESSION-METADATEN ===
    session_started_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    session_completed_at = db.Column(db.DateTime, nullable=True)
    
    # === AI-GENERATION-INFO ===
    ai_model_used = db.Column(db.Enum(AIModelType), nullable=False)
    prompt_version = db.Column(db.String(20), default="1.0", nullable=False)  # Für A/B Testing
    generation_time_seconds = db.Column(db.Float, nullable=True)
    total_ai_cost = db.Column(db.Float, default=0.0)  # Kosten in EUR
    
    # === AI-INPUT & OUTPUT ===
    ai_prompt_context = db.Column(db.Text, nullable=True)  # JSON: Der komplette Prompt-Kontext
    ai_raw_response = db.Column(db.Text, nullable=True)    # Rohe AI-Antwort
    ai_reasoning = db.Column(db.Text, nullable=True)       # AI-Begründung für Empfehlungen
    
    # === EMPFEHLUNGS-QUALITÄT ===
    overall_confidence = db.Column(db.Float, nullable=True)  # 0-1: AI-Confidence
    diversity_score = db.Column(db.Float, nullable=True)     # 0-1: Wie divers sind die Empfehlungen?
    personalization_score = db.Column(db.Float, nullable=True)  # 0-1: Wie personalisiert?
    
    # === USER-ENGAGEMENT ===
    recommendations_count = db.Column(db.Integer, default=0)
    clicks_count = db.Column(db.Integer, default=0)
    purchases_count = db.Column(db.Integer, default=0)
    
    # === STATUS ===
    is_completed = db.Column(db.Boolean, default=False)
    success_rate = db.Column(db.Float, nullable=True)  # 0-1: Wurde gekauft?
    
    # === BEZIEHUNGEN ===
    personality_profile = db.relationship('PersonalityProfile', back_populates='recommendation_sessions')
    recommendations = db.relationship('Recommendation', back_populates='session', cascade='all, delete-orphan')
    
    # === PROPERTIES ===
    
    @property
    def buyer_user(self):
        """Shortcut zum Käufer"""
        return self.personality_profile.buyer if self.personality_profile else None
    
    @property
    def session_duration_minutes(self) -> Optional[float]:
        """Wie lange hat die Session gedauert?"""
        if not self.session_completed_at:
            return None
        duration = self.session_completed_at - self.session_started_at
        return duration.total_seconds() / 60
    
    @property
    def click_through_rate(self) -> float:
        """Prozentsatz der angeklickten Empfehlungen"""
        if self.recommendations_count == 0:
            return 0.0
        return self.clicks_count / self.recommendations_count
    
    @property
    def conversion_rate(self) -> float:
        """Prozentsatz der gekauften Empfehlungen"""
        if self.recommendations_count == 0:
            return 0.0
        return self.purchases_count / self.recommendations_count
    
    def complete_session(self):
        """Markiert Session als abgeschlossen und berechnet Metriken"""
        self.session_completed_at = datetime.utcnow()
        self.is_completed = True
        
        # Berechne Erfolgsmetriken
        if self.recommendations_count > 0:
            self.success_rate = self.purchases_count / self.recommendations_count
        
        self.save()
    
    def add_feedback_summary(self, feedback_data: Dict):
        """Fügt zusammengefasstes Feedback zur Session hinzu"""
        # Hier könnte aggregiertes Feedback gespeichert werden
        pass
    
    def to_dict(self) -> Dict:
        """API-Export"""
        return {
            'id': self.id,
            'personality_profile_id': self.personality_profile_id,
            'session_started_at': self.session_started_at.isoformat(),
            'session_completed_at': self.session_completed_at.isoformat() if self.session_completed_at else None,
            'ai_model_used': self.ai_model_used.value,
            'prompt_version': self.prompt_version,
            'generation_time_seconds': self.generation_time_seconds,
            'total_ai_cost': self.total_ai_cost,
            'overall_confidence': self.overall_confidence,
            'diversity_score': self.diversity_score,
            'personalization_score': self.personalization_score,
            'recommendations_count': self.recommendations_count,
            'clicks_count': self.clicks_count,
            'purchases_count': self.purchases_count,
            'click_through_rate': self.click_through_rate,
            'conversion_rate': self.conversion_rate,
            'is_completed': self.is_completed,
            'success_rate': self.success_rate,
            'session_duration_minutes': self.session_duration_minutes
        }
    
    def __repr__(self):
        return f'<RecommendationSession {self.ai_model_used.value} - {self.recommendations_count} recs>'


class Recommendation(db.Model, BaseModel):
    """
    Eine einzelne AI-Empfehlung
    
    Verbindet PersonalityProfile mit Gift und speichert das "Warum"
    für kontinuierliche Verbesserung des AI-Systems.
    """
    
    __tablename__ = 'recommendations'
    
    # === PRIMARY KEY ===
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # === FOREIGN KEYS ===
    session_id = db.Column(db.String(36), db.ForeignKey('recommendation_sessions.id'), nullable=False)
    personality_profile_id = db.Column(db.String(36), db.ForeignKey('personality_profiles.id'), nullable=False)
    gift_id = db.Column(db.String(36), db.ForeignKey('gifts.id'), nullable=False)
    
    # === EMPFEHLUNGS-POSITION ===
    rank_position = db.Column(db.Integer, nullable=False)  # 1, 2, 3, 4, 5 (Reihenfolge)
    
    # === AI-REASONING ===
    match_score = db.Column(db.Float, nullable=False)  # 0-1: Wie gut passt es?
    confidence_score = db.Column(db.Float, nullable=False)  # 0-1: Wie sicher ist AI?
    reasoning = db.Column(db.Text, nullable=True)  # "Passt weil sie kreativ ist und Fotografie liebt"
    
    # === MATCH-DETAILS ===
    personality_match_factors = db.Column(db.Text, nullable=True)  # JSON: Welche Persönlichkeits-Faktoren
    emotional_trigger_match = db.Column(db.Text, nullable=True)    # JSON: Welche emotionalen Trigger
    occasion_relevance = db.Column(db.Float, nullable=True)        # 0-1: Wie gut passt der Anlass?
    budget_fit = db.Column(db.Float, nullable=True)                # 0-1: Wie gut passt das Budget?
    
    # === USER-INTERACTION ===
    status = db.Column(db.Enum(RecommendationStatus), default=RecommendationStatus.GENERATED)
    
    # Timestamps für User-Journey
    presented_at = db.Column(db.DateTime, nullable=True)
    clicked_at = db.Column(db.DateTime, nullable=True)
    purchased_at = db.Column(db.DateTime, nullable=True)
    
    # User-Feedback
    user_feedback = db.Column(db.Enum(FeedbackType), nullable=True)
    feedback_comment = db.Column(db.Text, nullable=True)  # Freitext-Feedback
    feedback_given_at = db.Column(db.DateTime, nullable=True)
    
    # === PERSONALISIERUNG ===
    personalization_suggestions = db.Column(db.Text, nullable=True)  # JSON: Wie personalisieren?
    alternative_gifts = db.Column(db.Text, nullable=True)             # JSON: Alternative IDs
    
    # === PERFORMANCE TRACKING ===
    view_duration_seconds = db.Column(db.Float, nullable=True)  # Wie lange angeschaut?
    
    # === BEZIEHUNGEN ===
    session = db.relationship('RecommendationSession', back_populates='recommendations')
    personality_profile = db.relationship('PersonalityProfile', back_populates='recommendations')
    gift = db.relationship('Gift', back_populates='recommendations')
    
    # === PROPERTIES ===
    
    @property
    def is_successful(self) -> bool:
        """Wurde die Empfehlung erfolgreich gekauft?"""
        return self.status == RecommendationStatus.PURCHASED
    
    @property
    def engagement_score(self) -> float:
        """0-1 Score basierend auf User-Engagement"""
        score = 0.0
        
        if self.status == RecommendationStatus.PRESENTED:
            score += 0.2
        if self.status == RecommendationStatus.CLICKED:
            score += 0.4
        if self.status == RecommendationStatus.PURCHASED:
            score += 1.0
        
        # Bonus für positive Feedback
        if self.user_feedback in [FeedbackType.THUMBS_UP, FeedbackType.PERFECT_MATCH]:
            score += 0.3
        
        # Malus für negative Feedback
        if self.user_feedback == FeedbackType.THUMBS_DOWN:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    @property
    def time_to_decision_minutes(self) -> Optional[float]:
        """Wie lange hat User gebraucht für Entscheidung?"""
        if not self.presented_at:
            return None
        
        decision_time = None
        if self.purchased_at:
            decision_time = self.purchased_at
        elif self.clicked_at:
            decision_time = self.clicked_at
        elif self.feedback_given_at:
            decision_time = self.feedback_given_at
        
        if decision_time:
            duration = decision_time - self.presented_at
            return duration.total_seconds() / 60
        
        return None
    
    def get_match_factors(self) -> Dict:
        """Holt Persönlichkeits-Match-Faktoren als Dict"""
        if not self.personality_match_factors:
            return {}
        try:
            return json.loads(self.personality_match_factors)
        except:
            return {}
    
    def get_emotional_triggers(self) -> Dict:
        """Holt emotionale Trigger-Matches als Dict"""
        if not self.emotional_trigger_match:
            return {}
        try:
            return json.loads(self.emotional_trigger_match)
        except:
            return {}
    
    def get_personalization_options(self) -> List[Dict]:
        """Holt Personalisierungs-Vorschläge"""
        if not self.personalization_suggestions:
            return []
        try:
            return json.loads(self.personalization_suggestions)
        except:
            return []
    
    def get_alternatives(self) -> List[str]:
        """Holt alternative Geschenk-IDs"""
        if not self.alternative_gifts:
            return []
        try:
            return json.loads(self.alternative_gifts)
        except:
            return []
    
    # === USER-INTERACTION METHODEN ===
    
    def mark_presented(self):
        """Markiert als dem User gezeigt"""
        self.status = RecommendationStatus.PRESENTED
        self.presented_at = datetime.utcnow()
        self.save()
    
    def mark_clicked(self):
        """Markiert als angeklickt"""
        if self.status == RecommendationStatus.PRESENTED:
            self.status = RecommendationStatus.CLICKED
            self.clicked_at = datetime.utcnow()
            
            # Update Session Stats
            if self.session:
                self.session.clicks_count += 1
                self.session.save()
        
        self.save()
    
    def mark_purchased(self, purchase_details: Dict = None):
        """Markiert als gekauft - DER ERFOLG!"""
        self.status = RecommendationStatus.PURCHASED
        self.purchased_at = datetime.utcnow()
        
        # Update Session Stats
        if self.session:
            self.session.purchases_count += 1
            self.session.save()
        
        # Update Gift Popularity
        if self.gift:
            self.gift.success_rate = (self.gift.success_rate or 0.5) * 0.9 + 0.1  # Leichter Boost
            self.gift.popularity_score += 0.01  # Popularity steigt
            self.gift.save()
        
        self.save()
    
    def add_feedback(self, feedback_type: FeedbackType, comment: str = None):
        """Fügt User-Feedback hinzu"""
        self.user_feedback = feedback_type
        self.feedback_comment = comment
        self.feedback_given_at = datetime.utcnow()
        
        # Bei negativem Feedback Geschenk-Score anpassen
        if feedback_type in [FeedbackType.THUMBS_DOWN, FeedbackType.WRONG_STYLE]:
            if self.gift:
                self.gift.success_rate = max(0.1, (self.gift.success_rate or 0.5) - 0.05)
                self.gift.save()
        
        # Bei positivem Feedback Geschenk-Score erhöhen
        elif feedback_type in [FeedbackType.THUMBS_UP, FeedbackType.PERFECT_MATCH]:
            if self.gift:
                self.gift.success_rate = min(1.0, (self.gift.success_rate or 0.5) + 0.02)
                self.gift.popularity_score += 0.005
                self.gift.save()
        
        self.save()
    
    def set_match_factors(self, factors: Dict):
        """Setzt Persönlichkeits-Match-Faktoren"""
        self.personality_match_factors = json.dumps(factors)
    
    def set_emotional_triggers(self, triggers: Dict):
        """Setzt emotionale Trigger-Matches"""
        self.emotional_trigger_match = json.dumps(triggers)
    
    def set_personalization_options(self, options: List[Dict]):
        """Setzt Personalisierungs-Vorschläge"""
        self.personalization_suggestions = json.dumps(options)
    
    def set_alternatives(self, gift_ids: List[str]):
        """Setzt alternative Geschenk-IDs"""
        self.alternative_gifts = json.dumps(gift_ids)
    
    def to_dict(self, include_detailed: bool = False) -> Dict:
        """API-Export für Frontend"""
        basic_dict = {
            'id': self.id,
            'gift': self.gift.to_dict() if self.gift else None,
            'rank_position': self.rank_position,
            'match_score': self.match_score,
            'confidence_score': self.confidence_score,
            'reasoning': self.reasoning,
            'status': self.status.value,
            'user_feedback': self.user_feedback.value if self.user_feedback else None,
            'engagement_score': self.engagement_score,
            'personalization_options': self.get_personalization_options(),
            'is_successful': self.is_successful
        }
        
        if include_detailed:
            basic_dict.update({
                'session_id': self.session_id,
                'personality_profile_id': self.personality_profile_id,
                'match_factors': self.get_match_factors(),
                'emotional_triggers': self.get_emotional_triggers(),
                'occasion_relevance': self.occasion_relevance,
                'budget_fit': self.budget_fit,
                'alternatives': self.get_alternatives(),
                'feedback_comment': self.feedback_comment,
                'time_to_decision_minutes': self.time_to_decision_minutes,
                'view_duration_seconds': self.view_duration_seconds,
                'presented_at': self.presented_at.isoformat() if self.presented_at else None,
                'clicked_at': self.clicked_at.isoformat() if self.clicked_at else None,
                'purchased_at': self.purchased_at.isoformat() if self.purchased_at else None,
                'created_at': self.created_at.isoformat()
            })
        
        return basic_dict
    
    def to_ai_training_data(self) -> Dict:
        """Strukturierte Daten für AI-Training und Verbesserung"""
        return {
            'input': {
                'personality_profile': self.personality_profile.to_ai_context() if self.personality_profile else None,
                'gift': self.gift.to_dict() if self.gift else None
            },
            'prediction': {
                'match_score': self.match_score,
                'confidence_score': self.confidence_score,
                'reasoning': self.reasoning
            },
            'actual_outcome': {
                'status': self.status.value,
                'engagement_score': self.engagement_score,
                'user_feedback': self.user_feedback.value if self.user_feedback else None,
                'was_successful': self.is_successful
            },
            'metadata': {
                'ai_model': self.session.ai_model_used.value if self.session else None,
                'prompt_version': self.session.prompt_version if self.session else None,
                'rank_position': self.rank_position,
                'time_to_decision_minutes': self.time_to_decision_minutes
            }
        }
    
    def __repr__(self):
        return f'<Recommendation #{self.rank_position}: {self.gift.name if self.gift else "Unknown"} ({self.match_score:.2f})>'


# === UTILITY FUNCTIONS ===

def create_recommendation_session(personality_profile, ai_model: AIModelType, prompt_version: str = "1.0") -> RecommendationSession:
    """
    Erstellt eine neue Empfehlungs-Session
    
    Args:
        personality_profile: PersonalityProfile instance
        ai_model: Welches AI-Model verwendet wird
        prompt_version: Version des Prompt-Templates
        
    Returns:
        RecommendationSession instance
    """
    session = RecommendationSession(
        personality_profile_id=personality_profile.id,
        ai_model_used=ai_model,
        prompt_version=prompt_version
    )
    session.save()
    return session


def get_user_recommendation_history(user_id: str, limit: int = 10) -> List[Recommendation]:
    """
    Holt Empfehlungs-Historie für einen User
    """
    from app.models.personality import PersonalityProfile
    
    return (
        Recommendation.query
        .join(PersonalityProfile)
        .filter(PersonalityProfile.buyer_user_id == user_id)
        .order_by(Recommendation.created_at.desc())
        .limit(limit)
        .all()
    )


def get_successful_recommendations_for_gift(gift_id: str) -> List[Recommendation]:
    """
    Holt alle erfolgreichen Empfehlungen für ein bestimmtes Geschenk
    Nützlich für: "Andere kauften auch..."
    """
    return (
        Recommendation.query
        .filter(
            Recommendation.gift_id == gift_id,
            Recommendation.status == RecommendationStatus.PURCHASED
        )
        .order_by(Recommendation.created_at.desc())
        .all()
    )


def analyze_recommendation_performance(ai_model: AIModelType = None, days: int = 30) -> Dict:
    """
    Analysiert die Performance von Empfehlungen
    
    Returns:
        Dict mit Metriken wie Success Rate, CTR, etc.
    """
    from datetime import timedelta
    
    # Basis-Query
    query = Recommendation.query.filter(
        Recommendation.created_at >= datetime.utcnow() - timedelta(days=days)
    )
    
    if ai_model:
        query = query.join(RecommendationSession).filter(
            RecommendationSession.ai_model_used == ai_model
        )
    
    recommendations = query.all()
    
    if not recommendations:
        return {"error": "No recommendations found"}
    
    total_count = len(recommendations)
    presented_count = len([r for r in recommendations if r.status != RecommendationStatus.GENERATED])
    clicked_count = len([r for r in recommendations if r.status in [RecommendationStatus.CLICKED, RecommendationStatus.PURCHASED]])
    purchased_count = len([r for r in recommendations if r.status == RecommendationStatus.PURCHASED])
    
    return {
        "total_recommendations": total_count,
        "presentation_rate": presented_count / total_count if total_count > 0 else 0,
        "click_through_rate": clicked_count / presented_count if presented_count > 0 else 0,
        "conversion_rate": purchased_count / presented_count if presented_count > 0 else 0,
        "overall_success_rate": purchased_count / total_count if total_count > 0 else 0,
        "average_match_score": sum(r.match_score for r in recommendations) / total_count,
        "average_confidence": sum(r.confidence_score for r in recommendations) / total_count,
        "ai_model": ai_model.value if ai_model else "all_models",
        "time_period_days": days
    }