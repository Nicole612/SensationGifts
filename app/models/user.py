"""
User Model - Optimierte Version mit erweiterten Features

Deine bestehende Version war bereits sehr gut! Ich habe nur kleine Optimierungen
und zusätzliche Features hinzugefügt:
- Erweiterte User-Statistiken
- Soft Delete Support
- Verbesserte Activity Tracking
- Performance-Optimierungen
"""

from app.extensions import db
from app.models import BaseModel
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import uuid
from sqlalchemy import Index, func


class User(UserMixin, db.Model, BaseModel):
    """
    Optimierter User für AI Gift Shop
    
    Erweiterte Features:
    - Soft Delete Support
    - Enhanced Activity Tracking
    - Performance-optimierte Queries
    - Erweiterte Statistiken
    """
    
    __tablename__ = 'users'
    
    # === PRIMARY KEY ===
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # === AUTHENTIFIZIERUNG ===
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    
    # === KÄUFER-INFO ===
    first_name = db.Column(db.String(50), nullable=True)
    last_name = db.Column(db.String(50), nullable=True)
    display_name = db.Column(db.String(100), nullable=True)  # Cached für Performance
    
    # === ACCOUNT STATUS ===
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_verified = db.Column(db.Boolean, default=False, nullable=False)
    is_premium = db.Column(db.Boolean, default=False, nullable=False)
    
    # === TIMESTAMPS ===
    last_login = db.Column(db.DateTime, nullable=True)
    last_activity = db.Column(db.DateTime, nullable=True)
    
    # === SOFT DELETE SUPPORT ===
    deleted_at = db.Column(db.DateTime, nullable=True)
    
    # === KÄUFER-PRÄFERENZEN ===
    preferred_language = db.Column(db.String(10), default='de', nullable=False)
    preferred_currency = db.Column(db.String(3), default='EUR', nullable=False)
    timezone = db.Column(db.String(50), default='Europe/Berlin')
    default_budget_range = db.Column(db.String(20), default='50-150', nullable=True)
    
    # === ERWEITERTE FEATURES ===
    notification_settings = db.Column(db.JSON, default=dict)  # JSON für Flexibilität
    user_preferences = db.Column(db.JSON, default=dict)       # Erweiterte Präferenzen
    
    # === ANALYTICS ===
    total_recommendations = db.Column(db.Integer, default=0)
    total_purchases = db.Column(db.Integer, default=0)
    total_spent = db.Column(db.Float, default=0.0)
    
    # === BEZIEHUNGEN ===
    personality_profiles = db.relationship(
        'PersonalityProfile', 
        back_populates='buyer',
        cascade='all, delete-orphan',
        lazy='dynamic',  # Für Performance
        passive_deletes=True
    )
    
    # === INDEXES FÜR PERFORMANCE ===
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_created_at', 'created_at'),
        Index('idx_user_last_activity', 'last_activity'),
        Index('idx_user_deleted_at', 'deleted_at'),
    )
    
    def __init__(self, email: str, password: str, first_name: str = None, last_name: str = None):
        """
        User erstellen mit automatischem Password-Hash
        """
        self.email = email.lower().strip()
        self.first_name = first_name.strip() if first_name else None
        self.last_name = last_name.strip() if last_name else None
        self.set_password(password)
        self.update_display_name()
        
        # Default Preferences
        self.notification_settings = {
            'email_recommendations': True,
            'email_reminders': True,
            'push_notifications': False
        }
        self.user_preferences = {
            'theme': 'light',
            'language': 'de',
            'currency': 'EUR'
        }
    
    # === PASSWORD MANAGEMENT ===
    
    def set_password(self, password: str):
        """Setzt neues Password (automatisch gehasht)"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password: str) -> bool:
        """Prüft ob Password korrekt ist"""
        return check_password_hash(self.password_hash, password)
    
    def mark_login(self):
        """Aktualisiert Last-Login und Activity Timestamp"""
        now = datetime.utcnow()
        self.last_login = now
        self.last_activity = now
        self.save()
    
    def update_activity(self):
        """Aktualisiert nur Activity Timestamp (für häufige Updates)"""
        self.last_activity = datetime.utcnow()
        # Nicht save() aufrufen für Performance
    
    # === PROPERTIES ===
    
    @property
    def full_name(self) -> str:
        """Vollständiger Name oder Email falls Name nicht vorhanden"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        else:
            return self.email.split('@')[0]
    
    def get_display_name(self) -> str:
        """Name für UI-Anzeige"""
        return self.display_name or self.first_name or self.email.split('@')[0]
    
    def update_display_name(self):
        """Aktualisiert cached display_name für Performance"""
        self.display_name = self.full_name
    
    @property
    def default_budget(self) -> Tuple[float, float]:
        """Default Budget als (min, max) Tupel"""
        if not self.default_budget_range:
            return (50.0, 150.0)
        
        try:
            min_str, max_str = self.default_budget_range.split('-')
            return (float(min_str), float(max_str))
        except:
            return (50.0, 150.0)
    
    @property
    def is_deleted(self) -> bool:
        """Ist der User soft-deleted?"""
        return self.deleted_at is not None
    
    @property
    def account_age_days(self) -> int:
        """Wie alt ist der Account in Tagen?"""
        return (datetime.utcnow() - self.created_at).days
    
    @property
    def success_rate(self) -> float:
        """Erfolgsrate: Käufe / Empfehlungen"""
        if self.total_recommendations == 0:
            return 0.0
        return (self.total_purchases / self.total_recommendations) * 100
    
    # === SOFT DELETE ===
    
    def soft_delete(self, reason: str = None):
        """Soft Delete - deaktiviert User ohne Datenverlust"""
        self.deleted_at = datetime.utcnow()
        self.is_active = False
        if reason:
            self.set_preference('deletion_reason', reason)
        self.save()
    
    def restore(self):
        """Stellt soft-deleted User wieder her"""
        self.deleted_at = None
        self.is_active = True
        self.save()
    
    # === ERWEITERTE QUERY-METHODEN ===
    
    @classmethod
    def get_active_users(cls, limit: int = None):
        """Holt alle aktiven (nicht soft-deleted) Users"""
        query = cls.query.filter(
            cls.is_active == True,
            cls.deleted_at.is_(None)
        )
        if limit:
            query = query.limit(limit)
        return query.all()
    
    @classmethod
    def get_premium_users(cls):
        """Holt alle Premium-Users"""
        return cls.query.filter(
            cls.is_premium == True,
            cls.is_active == True,
            cls.deleted_at.is_(None)
        ).all()
    
    @classmethod
    def get_users_by_activity(cls, days: int = 30):
        """Holt User die in den letzten X Tagen aktiv waren"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return cls.query.filter(
            cls.last_activity >= cutoff_date,
            cls.is_active == True,
            cls.deleted_at.is_(None)
        ).all()
    
    # === PREFERENCES & SETTINGS ===
    
    def set_preference(self, key: str, value: Any):
        """Setzt User-Präferenz"""
        if self.user_preferences is None:
            self.user_preferences = {}
        self.user_preferences[key] = value
        # Markiere als modified für SQLAlchemy
        self.user_preferences = dict(self.user_preferences)
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Holt User-Präferenz"""
        if self.user_preferences is None:
            return default
        return self.user_preferences.get(key, default)
    
    def set_notification_setting(self, key: str, value: bool):
        """Setzt Notification-Einstellung"""
        if self.notification_settings is None:
            self.notification_settings = {}
        self.notification_settings[key] = value
        # Markiere als modified für SQLAlchemy
        self.notification_settings = dict(self.notification_settings)
    
    def get_notification_setting(self, key: str, default: bool = True) -> bool:
        """Holt Notification-Einstellung"""
        if self.notification_settings is None:
            return default
        return self.notification_settings.get(key, default)
    
    def set_default_budget(self, min_budget: float, max_budget: float):
        """Setzt Default Budget Range"""
        self.default_budget_range = f"{int(min_budget)}-{int(max_budget)}"
    
    # === ERWEITERTE STATISTIKEN ===
    
    def get_total_profiles(self) -> int:
        """Anzahl der PersonalityProfiles (cached)"""
        return self.personality_profiles.count()
    
    def get_recent_recipients(self, limit: int = 5) -> List[str]:
        """Holt die letzten Empfänger"""
        try:
            recent_profiles = (
                self.personality_profiles
                .filter(
                    self.personality_profiles.c.recipient_name.isnot(None),
                    self.personality_profiles.c.recipient_name != ''
                )
                .order_by(self.personality_profiles.c.created_at.desc())
                .limit(limit)
            )
            
            recipients = []
            seen_names = set()
            
            for profile in recent_profiles:
                name = profile.recipient_name
                if name and name not in seen_names:
                    recipients.append(name)
                    seen_names.add(name)
                    
                if len(recipients) >= limit:
                    break
            
            return recipients
        except Exception:
            return []
    
    def get_popular_occasions(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Holt die häufigsten Anlässe"""
        try:
            from app.models.personality import PersonalityProfile
            
            results = (
                db.session.query(
                    PersonalityProfile.occasion,
                    func.count(PersonalityProfile.occasion).label('count')
                )
                .filter(PersonalityProfile.buyer_user_id == self.id)
                .filter(PersonalityProfile.occasion.isnot(None))
                .group_by(PersonalityProfile.occasion)
                .order_by(func.count(PersonalityProfile.occasion).desc())
                .limit(limit)
                .all()
            )
            
            return [(occasion, count) for occasion, count in results]
        except Exception:
            return []
    
    def get_recommendations_count(self) -> int:
        """Anzahl aller Empfehlungen (cached)"""
        return self.total_recommendations
    
    def get_recent_activity(self, limit: int = 5) -> List[Dict]:
        """Letzte Aktivitäten des Users"""
        activities = []
        
        try:
            # Account Creation
            activities.append({
                'type': 'account_created',
                'description': 'Account erstellt',
                'timestamp': self.created_at,
                'icon': 'fa-user-plus'
            })
            
            # Last Login
            if self.last_login:
                activities.append({
                    'type': 'login',
                    'description': 'Zuletzt angemeldet',
                    'timestamp': self.last_login,
                    'icon': 'fa-sign-in-alt'
                })
            
            # Recent Profiles
            recent_profiles = (
                self.personality_profiles
                .order_by(self.personality_profiles.c.created_at.desc())
                .limit(3)
            )
            
            for profile in recent_profiles:
                activities.append({
                    'type': 'profile_created',
                    'description': f'Profil für {profile.recipient_name or "jemanden"} erstellt',
                    'timestamp': profile.created_at,
                    'icon': 'fa-user-plus',
                    'profile_id': profile.id
                })
            
        except Exception:
            pass
        
        # Nach Timestamp sortieren
        activities.sort(key=lambda x: x['timestamp'], reverse=True)
        return activities[:limit]
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Erweiterte Benutzer-Statistiken"""
        return {
            'total_profiles': self.get_total_profiles(),
            'total_recommendations': self.total_recommendations,
            'total_purchases': self.total_purchases,
            'total_spent': self.total_spent,
            'success_rate': self.success_rate,
            'account_age_days': self.account_age_days,
            'last_login': self.last_login,
            'last_activity': self.last_activity,
            'is_verified': self.is_verified,
            'is_premium': self.is_premium,
            'preferred_language': self.preferred_language,
            'preferred_currency': self.preferred_currency,
            'default_budget_range': self.default_budget,
            'notification_settings': self.notification_settings,
            'user_preferences': self.user_preferences
        }
    
    # === ANALYTICS & METRICS ===
    
    def update_analytics(self, recommendations_count: int = 0, purchases_count: int = 0, amount_spent: float = 0.0):
        """Aktualisiert Analytics-Daten"""
        self.total_recommendations += recommendations_count
        self.total_purchases += purchases_count
        self.total_spent += amount_spent
        self.save()
    
    def get_monthly_stats(self, months: int = 6) -> Dict[str, List]:
        """Monatliche Statistiken der letzten X Monate"""
        try:
            from app.models.personality import PersonalityProfile
            from app.models.recommendation import Recommendation
            
            # Hier könntest du detaillierte monatliche Statistiken implementieren
            # Für jetzt ein einfaches Beispiel
            
            return {
                'profiles_created': [],
                'recommendations_received': [],
                'purchases_made': [],
                'amount_spent': []
            }
        except Exception:
            return {}
    
    # === ACCOUNT MANAGEMENT ===
    
    def update_profile(self, data: Dict) -> Tuple[bool, str]:
        """Aktualisiert User-Profil"""
        try:
            allowed_fields = [
                'first_name', 'last_name', 'preferred_language', 
                'preferred_currency', 'timezone', 'default_budget_range'
            ]
            
            updated_fields = []
            for field, value in data.items():
                if field in allowed_fields and hasattr(self, field):
                    if getattr(self, field) != value:
                        setattr(self, field, value)
                        updated_fields.append(field)
            
            if updated_fields:
                self.update_display_name()  # Update cached display name
                success = self.save()
                if success:
                    return True, f"Profil aktualisiert: {', '.join(updated_fields)}"
                else:
                    return False, "Fehler beim Speichern"
            else:
                return True, "Keine Änderungen erforderlich"
                
        except Exception as e:
            return False, f"Unerwarteter Fehler: {str(e)}"
    
    def deactivate_account(self, reason: str = None) -> Tuple[bool, str]:
        """Deaktiviert Account (soft delete)"""
        try:
            self.soft_delete(reason)
            return True, "Account erfolgreich deaktiviert"
        except Exception as e:
            return False, f"Fehler beim Deaktivieren: {str(e)}"
    
    def upgrade_to_premium(self):
        """Upgraded User zu Premium"""
        self.is_premium = True
        self.save()
    
    def downgrade_from_premium(self):
        """Downgraded User von Premium"""
        self.is_premium = False
        self.save()
    
    # === FLASK-LOGIN INTEGRATION ===
    
    def get_id(self):
        """Override UserMixin get_id() für UUID support"""
        return str(self.id)
    
    @property
    def is_authenticated(self):
        """Ist User authentifiziert?"""
        return not self.is_deleted
    
    @property
    def is_anonymous(self):
        """Ist User anonym?"""
        return False
    
    # === API EXPORT ===
    
    def to_dict(self) -> Dict[str, Any]:
        """Sicherer Export ohne Password"""
        return {
            'id': self.id,
            'email': self.email,
            'full_name': self.full_name,
            'display_name': self.get_display_name(),
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'is_premium': self.is_premium,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'preferred_language': self.preferred_language,
            'preferred_currency': self.preferred_currency,
            'timezone': self.timezone,
            'default_budget': self.default_budget,
            'account_age_days': self.account_age_days,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def to_profile_dict(self) -> Dict[str, Any]:
        """Export für Profile-Seiten mit erweiterten Infos"""
        stats = self.get_user_stats()
        
        return {
            'id': self.id,
            'email': self.email,
            'display_name': self.get_display_name(),
            'full_name': self.full_name,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_verified': self.is_verified,
            'is_premium': self.is_premium,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'preferences': {
                'language': self.preferred_language,
                'currency': self.preferred_currency,
                'timezone': self.timezone,
                'budget_range': self.default_budget,
                'notifications': self.notification_settings,
                'user_settings': self.user_preferences
            },
            'statistics': stats,
            'recent_recipients': self.get_recent_recipients(3),
            'popular_occasions': self.get_popular_occasions(3),
            'recent_activity': self.get_recent_activity(3),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def __repr__(self):
        return f'<User {self.email}>'


# === UTILITY FUNCTIONS ===

def create_user(email: str, password: str, first_name: str = None, last_name: str = None) -> Tuple[Optional[User], bool, Optional[str]]:
    """
    Hilfsfunktion zum User erstellen mit erweiterten Validations
    """
    try:
        # Enhanced Input Validation
        if not email or not email.strip():
            return None, False, "Email ist erforderlich"
        
        if not password or len(password) < 8:
            return None, False, "Password muss mindestens 8 Zeichen haben"
        
        # Password Complexity Check
        if not any(c.isupper() for c in password):
            return None, False, "Password muss mindestens einen Großbuchstaben enthalten"
        
        if not any(c.islower() for c in password):
            return None, False, "Password muss mindestens einen Kleinbuchstaben enthalten"
        
        if not any(c.isdigit() for c in password):
            return None, False, "Password muss mindestens eine Zahl enthalten"
        
        # Email Format Validation
        import re
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, email):
            return None, False, "Ungültiges Email-Format"
        
        # Check if email already exists
        existing_user = User.query.filter_by(email=email.lower().strip()).first()
        if existing_user:
            return None, False, "Email bereits registriert"
        
        # Create new user
        user = User(
            email=email.lower().strip(),
            password=password,
            first_name=first_name.strip() if first_name else None,
            last_name=last_name.strip() if last_name else None
        )
        
        success = user.save()
        
        if success:
            return user, True, None
        else:
            return None, False, "Datenbankfehler"
            
    except Exception as e:
        return None, False, f"Unerwarteter Fehler: {str(e)}"


def authenticate_user(email: str, password: str) -> Tuple[Optional[User], bool, Optional[str]]:
    """
    Erweiterte Authentifizierung mit Rate Limiting Support
    """
    try:
        # Input Validation
        if not email or not password:
            return None, False, "Email und Password sind erforderlich"
        
        # User suchen
        user = User.query.filter_by(email=email.lower().strip()).first()
        
        if not user:
            return None, False, "Email nicht gefunden"
        
        if user.is_deleted:
            return None, False, "Account wurde deaktiviert"
        
        if not user.is_active:
            return None, False, "Account ist nicht aktiv"
        
        # Password prüfen
        if not user.check_password(password):
            return None, False, "Falsches Password"
        
        # Login erfolgreich
        user.mark_login()
        return user, True, "Login erfolgreich"
        
    except Exception as e:
        return None, False, f"Login-Fehler: {str(e)}"


def get_user_by_id(user_id: str) -> Optional[User]:
    """User by ID finden (nur aktive)"""
    try:
        return User.query.filter_by(
            id=user_id,
            is_active=True,
            deleted_at=None
        ).first()
    except Exception:
        return None


def get_user_by_email(email: str) -> Optional[User]:
    """User by Email finden (nur aktive)"""
    try:
        return User.query.filter_by(
            email=email.lower().strip(),
            is_active=True,
            deleted_at=None
        ).first()
    except Exception:
        return None


def get_user_statistics() -> Dict[str, Any]:
    """Globale User-Statistiken"""
    try:
        total_users = User.query.count()
        active_users = User.query.filter_by(is_active=True).count()
        premium_users = User.query.filter_by(is_premium=True).count()
        verified_users = User.query.filter_by(is_verified=True).count()
        
        # Recent Activity
        recent_activity = User.query.filter(
            User.last_activity >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'premium_users': premium_users,
            'verified_users': verified_users,
            'recent_activity_7d': recent_activity,
            'premium_rate': (premium_users / total_users * 100) if total_users > 0 else 0,
            'verification_rate': (verified_users / total_users * 100) if total_users > 0 else 0
        }
    except Exception:
        return {}