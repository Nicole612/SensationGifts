"""
Gift Models - Intelligenter Geschenke-Katalog mit AI-Features (Session 2)

INNOVATION: Kategorien basierend auf PERSÖNLICHKEIT statt Produkttyp
- Psychologie-basierte Kategorisierung (nicht "Elektronik" sondern "Für Tech-Enthusiasten")
- Multi-dimensionale Tag-Systeme für flexible Filterung
- AI-optimierte Match-Scores für personalisierte Empfehlungen
- Dynamische Preis-Kategorien und Personalisierungsebenen

Clean Architecture: Models sind pure Data-Layer ohne Business-Logic
Business-Logic kommt später in Services (Session 4)
"""

from app.extensions import db
from app.models import BaseModel
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import uuid
from enum import Enum
from sqlalchemy import Index, func
from sqlalchemy.ext.hybrid import hybrid_property


# === ENUMS FÜR TYPE-SAFE OPERATIONS ===

class PriceRange(Enum):
    """
    Dynamische Preiskategorien - nicht fixe Beträge!
    AI kann je nach Zielgruppe und Inflation anpassen
    """
    BUDGET = "budget"           # Unter 50€ - für jeden erschwinglich
    AFFORDABLE = "affordable"   # 50-150€ - normale Geschenke
    MODERATE = "moderate"       # 150-300€ - besondere Anlässe
    PREMIUM = "premium"         # 300-800€ - hochwertige Geschenke
    LUXURY = "luxury"          # Über 800€ - Luxus-Segment


class GiftType(Enum):
    """
    Grundlegende Gift-Typen - bestimmt Lieferung und Erwartungen
    """
    PHYSICAL = "physical"       # Physische Gegenstände
    DIGITAL = "digital"         # Sofort verfügbar (Kurse, E-Books, etc.)
    EXPERIENCE = "experience"   # Erlebnisse (Konzerte, Reisen, etc.)
    SERVICE = "service"         # Dienstleistungen (Massage, Coaching, etc.)
    SUBSCRIPTION = "subscription" # Abonnements (Netflix, Spotify, etc.)


class PersonalizationLevel(Enum):
    """
    Wie stark kann das Geschenk personalisiert werden?
    Wichtig für AI-Empfehlungen basierend auf Beziehungstiefe
    """
    NONE = "none"               # Keine Personalisierung möglich
    SIMPLE = "simple"           # Farbe, Größe auswählbar
    MODERATE = "moderate"       # Text, Name hinzufügbar
    HIGHLY = "highly"           # Vollständig anpassbar
    UNIQUE = "unique"           # Jedes Stück ist Unikat


# === GIFT CATEGORY MODEL ===

class GiftCategory(db.Model, BaseModel):
    """
    INNOVATION: Psychologie-basierte Kategorien statt Produktkategorien
    
    Traditionell: "Elektronik", "Kleidung", "Bücher"
    Neu: "Für Kreative Köpfe", "Entspannung & Wellness", "Tech-Enthusiasten"
    
    Vorteil: AI kann direkt Persönlichkeit → Kategorie matchen
    """
    
    __tablename__ = 'gift_categories'
    
    # === PRIMARY FIELDS ===
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False, unique=True)
    slug = db.Column(db.String(100), nullable=False, unique=True)  # URL-friendly
    description = db.Column(db.Text, nullable=False)
    
    # === VISUAL BRANDING ===
    icon = db.Column(db.String(20), nullable=True)     # Emoji oder Icon-Name
    color = db.Column(db.String(7), nullable=True)     # Hex Color Code
    image_url = db.Column(db.String(500), nullable=True)
    
    # === AI MATCHING ATTRIBUTES ===
    # JSON-encoded Listen für flexible AI-Zuordnung
    target_traits = db.Column(db.Text, nullable=True)        # ["high_openness", "creative_type"]
    target_occasions = db.Column(db.Text, nullable=True)     # ["geburtstag", "weihnachten"]
    target_relationships = db.Column(db.Text, nullable=True) # ["partner", "freund"]
    target_age_groups = db.Column(db.Text, nullable=True)    # ["25_35", "35_50"]
    
    # === CATEGORY METADATA ===
    sort_order = db.Column(db.Integer, default=0)      # Display-Reihenfolge
    is_active = db.Column(db.Boolean, default=True)    # Kann deaktiviert werden
    seasonal_weight = db.Column(db.Float, default=1.0) # Saison-Gewichtung
    
    # === RELATIONSHIPS ===
    gifts = db.relationship('Gift', back_populates='category', lazy='dynamic')
    
    # === COMPUTED PROPERTIES ===
    
    @property
    def target_traits_list(self) -> List[str]:
        """Target Personality Traits als Python Liste"""
        if not self.target_traits:
            return []
        try:
            return json.loads(self.target_traits)
        except:
            return []
    
    @property
    def target_occasions_list(self) -> List[str]:
        """Target Occasions als Python Liste"""
        if not self.target_occasions:
            return []
        try:
            return json.loads(self.target_occasions)
        except:
            return []
    
    @property
    def target_relationships_list(self) -> List[str]:
        """Target Relationships als Python Liste"""
        if not self.target_relationships:
            return []
        try:
            return json.loads(self.target_relationships)
        except:
            return []
    
    @hybrid_property
    def gift_count(self):
        """Anzahl aktiver Geschenke in dieser Kategorie"""
        return self.gifts.filter_by(is_active=True).count()
    
    @property
    def average_price(self) -> float:
        """Durchschnittspreis der Geschenke in dieser Kategorie"""
        gifts = self.gifts.filter_by(is_active=True).all()
        if not gifts:
            return 0.0
        return sum(gift.price for gift in gifts) / len(gifts)
    
    # === SETTER METHODS FOR JSON FIELDS ===
    
    def set_target_traits(self, traits: List[str]):
        """Setzt die Ziel-Persönlichkeitsmerkmale"""
        self.target_traits = json.dumps(traits) if traits else None
    
    def set_occasions(self, occasions: List[str]):
        """Setzt die passenden Anlässe"""
        self.target_occasions = json.dumps(occasions) if occasions else None
    
    def set_relationships(self, relationships: List[str]):
        """Setzt die passenden Beziehungstypen"""
        self.target_relationships = json.dumps(relationships) if relationships else None
    
    # === AI MATCHING METHODS ===
    
    def calculate_personality_match(self, personality_profile) -> float:
        """
        Berechnet wie gut diese Kategorie zur Persönlichkeit passt
        
        Returns: 0.0 - 1.0 (0 = gar nicht, 1 = perfekt)
        """
        if not self.target_traits_list:
            return 0.5  # Neutral wenn keine Traits definiert
        
        total_score = 0.0
        trait_count = 0
        
        for trait in self.target_traits_list:
            score = self._get_trait_score(personality_profile, trait)
            if score is not None:
                total_score += score
                trait_count += 1
        
        return total_score / trait_count if trait_count > 0 else 0.5
    
    def _get_trait_score(self, profile, trait: str) -> Optional[float]:
        """
        Holt den Score für ein spezifisches Trait aus dem PersonalityProfile
        """
        try:
            trait_mapping = {
                # Persönlichkeits-Dimensionen
                "high_openness": lambda p: p.openness if p.openness and p.openness > 0.6 else 0.0,
                "high_extraversion": lambda p: p.extraversion if p.extraversion and p.extraversion > 0.6 else 0.0,
                "high_conscientiousness": lambda p: p.conscientiousness if p.conscientiousness and p.conscientiousness > 0.6 else 0.0,
                
                # Präferenz-basierte Traits
                "creative_type": lambda p: 1.0 if p.creative_type else 0.0,
                "tech_savvy": lambda p: 1.0 if p.tech_savvy else 0.0,
                "practical_type": lambda p: 1.0 if p.practical_type else 0.0,
                "health_conscious": lambda p: 1.0 if p.health_conscious else 0.0,
            }
            
            if trait in trait_mapping:
                return trait_mapping[trait](profile)
            
        except Exception:
            pass
        
        return None
    
    def to_dict(self) -> Dict:
        """API Export"""
        return {
            'id': self.id,
            'name': self.name,
            'slug': self.slug,
            'description': self.description,
            'icon': self.icon,
            'color': self.color,
            'image_url': self.image_url,
            'target_traits': self.target_traits_list,
            'target_occasions': self.target_occasions_list,
            'target_relationships': self.target_relationships_list,
            'sort_order': self.sort_order,
            'is_active': self.is_active,
            'gift_count': self.gift_count,
            'average_price': self.average_price,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def __repr__(self):
        return f'<GiftCategory {self.name} ({self.gift_count} gifts)>'


# === GIFT TAG MODEL ===

class GiftTag(db.Model, BaseModel):
    """
    Flexible Tags für Multi-Dimensional-Filtering
    
    INNOVATION: Gewichtete Tags mit Typen für intelligente Filterung
    - Verschiedene Tag-Typen (personality, occasion, feature, etc.)
    - Gewichtungen für AI-Relevanz
    - Farb-Codierung für UI
    """
    
    __tablename__ = 'gift_tags'
    
    # === PRIMARY FIELDS ===
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(50), nullable=False, unique=True)        # tech_name (für Code)
    display_name = db.Column(db.String(100), nullable=False)            # Human-readable
    description = db.Column(db.Text, nullable=True)
    
    # === TAG CLASSIFICATION ===
    tag_type = db.Column(db.String(30), nullable=False, default='general')  # personality, occasion, feature, etc.
    weight = db.Column(db.Float, default=1.0)                               # Gewichtung für AI (0.1 - 3.0)
    color = db.Column(db.String(7), nullable=True)                          # Hex Color für UI
    
    # === RELATIONSHIPS ===
    # Many-to-Many mit Gifts über Association Table
    gift_associations = db.relationship('GiftTagAssociation', back_populates='tag')
    
    @property
    def gifts(self):
        """Alle Gifts mit diesem Tag"""
        return [assoc.gift for assoc in self.gift_associations if assoc.gift and assoc.gift.is_active]
    
    @property
    def usage_count(self) -> int:
        """Wie oft wird dieser Tag verwendet?"""
        return len([assoc for assoc in self.gift_associations if assoc.gift and assoc.gift.is_active])
    
    def to_dict(self) -> Dict:
        """API Export"""
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'tag_type': self.tag_type,
            'weight': self.weight,
            'color': self.color,
            'usage_count': self.usage_count,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<GiftTag {self.display_name} ({self.usage_count} uses)>'


# === GIFT TAG ASSOCIATION (Many-to-Many) ===

class GiftTagAssociation(db.Model, BaseModel):
    """
    Association Table für Gift ↔ Tag Many-to-Many
    
    INNOVATION: Mit Relevance-Score statt nur Boolean-Relation
    Erlaubt gewichtete Tag-Zuordnungen für bessere AI-Matches
    """
    
    __tablename__ = 'gift_tag_associations'
    
    # === FOREIGN KEYS ===
    gift_id = db.Column(db.String(36), db.ForeignKey('gifts.id'), primary_key=True)
    tag_id = db.Column(db.String(36), db.ForeignKey('gift_tags.id'), primary_key=True)
    
    # === ASSOCIATION METADATA ===
    relevance_score = db.Column(db.Float, default=1.0)  # Wie relevant ist der Tag für dieses Gift? (0.1 - 1.0)
    auto_assigned = db.Column(db.Boolean, default=False)  # Wurde automatisch durch AI zugewiesen?
    
    # === RELATIONSHIPS ===
    gift = db.relationship('Gift', back_populates='tag_associations')
    tag = db.relationship('GiftTag', back_populates='gift_associations')
    
    def __repr__(self):
        gift_name = self.gift.name if self.gift else "Unknown"
        tag_name = self.tag.display_name if self.tag else "Unknown" 
        return f'<GiftTagAssociation {gift_name} ↔ {tag_name} (relevance: {self.relevance_score})>'


# === MAIN GIFT MODEL ===

class Gift(db.Model, BaseModel):
    """
    Hauptmodel für Geschenke mit AI-optimierten Attributen
    
    INNOVATION: Eingebaute AI-Matching-Scores und flexible Attribute
    - Personality-Match-Scores für verschiedene Traits
    - Occasion & Relationship Suitability Scores
    - Dynamische Preisklassen-Zuordnung
    - Strukturierte Purchase Links und Metadata
    """
    
    __tablename__ = 'gifts'
    
    # === PRIMARY FIELDS ===
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(200), nullable=False)
    short_description = db.Column(db.String(500), nullable=False)  # Für Listen/Previews
    long_description = db.Column(db.Text, nullable=True)           # Detailansicht
    
    # === PRICING ===
    price = db.Column(db.Float, nullable=False)
    price_range = db.Column(db.Enum(PriceRange), nullable=False)
    currency = db.Column(db.String(3), default='EUR')
    
    # === CLASSIFICATION ===
    gift_type = db.Column(db.Enum(GiftType), nullable=False)
    personalization_level = db.Column(db.Enum(PersonalizationLevel), nullable=False)
    category_id = db.Column(db.String(36), db.ForeignKey('gift_categories.id'), nullable=False)
    
    # === TARGETING ===
    target_age_min = db.Column(db.Integer, nullable=True)  # Minimum Alter
    target_age_max = db.Column(db.Integer, nullable=True)  # Maximum Alter
    target_gender = db.Column(db.String(20), nullable=True)  # Optional: "m", "f", "any"
    
    # === MEDIA ===
    image_url = db.Column(db.String(500), nullable=True)
    additional_images = db.Column(db.Text, nullable=True)  # JSON Array von URLs
    video_url = db.Column(db.String(500), nullable=True)
    
    # === AI-OPTIMIZED ATTRIBUTES ===
    # JSON-encoded Dictionaries für flexible AI-Zuordnung
    personality_match_scores = db.Column(db.Text, nullable=True)     # {"high_openness": 0.9, "creative_type": 0.8}
    occasion_suitability = db.Column(db.Text, nullable=True)         # {"geburtstag": 0.9, "weihnachten": 0.7}
    relationship_suitability = db.Column(db.Text, nullable=True)     # {"partner": 0.8, "freund": 0.9}
    
    # === PURCHASE INFORMATION ===
    purchase_links = db.Column(db.Text, nullable=True)              # JSON Array von Shop-Links
    availability_status = db.Column(db.String(20), default='available')  # available, limited, out_of_stock
    delivery_time_days = db.Column(db.Integer, nullable=True)       # Geschätzte Lieferzeit
    
    # === QUALITY METRICS ===
    quality_score = db.Column(db.Float, default=0.5)               # 0.0 - 1.0 (basierend auf Reviews, etc.)
    popularity_score = db.Column(db.Float, default=0.5)            # 0.0 - 1.0 (wie oft wird es empfohlen?)
    success_rate = db.Column(db.Float, default=0.5)                # 0.0 - 1.0 (Kauf-Erfolgsrate bei Empfehlungen)
    
    # === STATUS ===
    is_active = db.Column(db.Boolean, default=True)
    is_featured = db.Column(db.Boolean, default=False)             # Featured/Promoted Gifts
    is_seasonal = db.Column(db.Boolean, default=False)             # Saison-spezifisch?
    
    # === RELATIONSHIPS ===
    category = db.relationship('GiftCategory', back_populates='gifts')
    tag_associations = db.relationship('GiftTagAssociation', back_populates='gift', cascade='all, delete-orphan')
    recommendations = db.relationship('Recommendation', back_populates='gift', cascade='all, delete-orphan')
    

    # NEU für Generator:
    age_categories = db.Column(db.Text, nullable=True)  # JSON: ["4-7", "8-12"]
    emotional_story = db.Column(db.Text, nullable=True)  # Emotional Story aus Katalog
    emotional_impact = db.Column(db.Text, nullable=True)  # JSON: emotional impact data
    is_generated = db.Column(db.Boolean, default=False)  # Kommt aus Katalog-Generator
    template_name = db.Column(db.String(100), nullable=True)  # Template-ID für Referenz
    # === COMPUTED PROPERTIES ===
    
    @property
    def tags(self):
        """Alle Tags dieses Gifts"""
        return [assoc.tag for assoc in self.tag_associations if assoc.tag]
    
    @property
    def weighted_tags(self) -> List[Tuple[str, float]]:
        """Tags mit ihren Gewichtungen für AI"""
        return [(assoc.tag.name, assoc.relevance_score * assoc.tag.weight) 
                for assoc in self.tag_associations if assoc.tag]
    
    @property
    def personality_match_dict(self) -> Dict[str, float]:
        """Personality Match Scores als Python Dict"""
        if not self.personality_match_scores:
            return {}
        try:
            return json.loads(self.personality_match_scores)
        except:
            return {}
    
    @property
    def occasion_suitability_dict(self) -> Dict[str, float]:
        """Occasion Suitability als Python Dict"""
        if not self.occasion_suitability:
            return {}
        try:
            return json.loads(self.occasion_suitability)
        except:
            return {}
    
    @property
    def relationship_suitability_dict(self) -> Dict[str, float]:
        """Relationship Suitability als Python Dict"""
        if not self.relationship_suitability:
            return {}
        try:
            return json.loads(self.relationship_suitability)
        except:
            return {}
    
    @property
    def purchase_links_list(self) -> List[Dict]:
        """Purchase Links als Python Liste"""
        if not self.purchase_links:
            return []
        try:
            return json.loads(self.purchase_links)
        except:
            return []
    
    @property
    def display_price(self) -> str:
        """Formatierter Preis für UI"""
        return f"{self.price:.2f} {self.currency}"
    
    # === SETTER METHODS FOR JSON FIELDS ===
    
    def set_personality_match(self, scores: Dict[str, float]):
        """Setzt Personality Match Scores"""
        self.personality_match_scores = json.dumps(scores) if scores else None
    
    def set_occasion_suitability(self, scores: Dict[str, float]):
        """Setzt Occasion Suitability Scores"""
        self.occasion_suitability = json.dumps(scores) if scores else None
    
    def set_relationship_suitability(self, scores: Dict[str, float]):
        """Setzt Relationship Suitability Scores"""
        self.relationship_suitability = json.dumps(scores) if scores else None
    
    def set_purchase_links(self, links: List[Dict]):
        """Setzt Purchase Links (Format: [{"name": "Amazon", "url": "...", "price": 99.99}])"""
        self.purchase_links = json.dumps(links) if links else None
    
    # === TAG MANAGEMENT ===
    
    def add_tag(self, tag: GiftTag, relevance_score: float = 1.0, auto_assigned: bool = False):
        """Fügt einen Tag zu diesem Gift hinzu"""
        try:
            # Prüfe ob bereits vorhanden
            existing = GiftTagAssociation.query.filter_by(gift_id=self.id, tag_id=tag.id).first()
            if existing:
                # Update Relevance Score
                existing.relevance_score = relevance_score
                existing.auto_assigned = auto_assigned
            else:
                # Erstelle neue Association
                association = GiftTagAssociation(
                    gift_id=self.id,
                    tag_id=tag.id,
                    relevance_score=relevance_score,
                    auto_assigned=auto_assigned
                )
                db.session.add(association)
            
            db.session.commit()
        except Exception:
            db.session.rollback()
    
    def remove_tag(self, tag: GiftTag):
        """Entfernt einen Tag von diesem Gift"""
        try:
            association = GiftTagAssociation.query.filter_by(gift_id=self.id, tag_id=tag.id).first()
            if association:
                db.session.delete(association)
                db.session.commit()
        except Exception:
            db.session.rollback()
    
    # === AI MATCHING METHODS ===
    
    def calculate_match_score(self, personality_profile) -> float:
        """
        HAUPTFUNKTION: Berechnet AI-Match-Score für PersonalityProfile
        
        Kombiniert verschiedene Faktoren:
        - Persönlichkeits-Match (40%)
        - Kategorie-Match (25%) 
        - Anlass-Match (20%)
        - Beziehungs-Match (15%)
        
        Returns: 0.0 - 1.0 (höher = bessere Übereinstimmung)
        """
        try:
            weights = {
                'personality': 0.40,
                'category': 0.25,
                'occasion': 0.20,
                'relationship': 0.15
            }
            
            # Personality Match Score
            personality_score = self._calculate_personality_score(personality_profile)
            
            # Category Match Score
            category_score = 0.5
            if self.category:
                category_score = self.category.calculate_personality_match(personality_profile)
            
            # Occasion Match Score
            occasion_score = self._calculate_occasion_score(personality_profile.occasion)
            
            # Relationship Match Score
            relationship_score = self._calculate_relationship_score(personality_profile.relationship)
            
            # Gewichtete Summe
            final_score = (
                personality_score * weights['personality'] +
                category_score * weights['category'] +
                occasion_score * weights['occasion'] +
                relationship_score * weights['relationship']
            )
            
            # Bonus für perfekte Preisrange
            if personality_profile.budget_min <= self.price <= personality_profile.budget_max:
                final_score *= 1.1  # 10% Bonus für Budget-Match
            elif self.price < personality_profile.budget_min * 0.8:
                final_score *= 0.8  # Penalty für zu günstig
            elif self.price > personality_profile.budget_max * 1.2:
                final_score *= 0.6  # Stärkere Penalty für zu teuer
            
            # Bonus für hohe Qualität
            final_score *= (0.8 + 0.2 * self.quality_score)
            
            return min(final_score, 1.0)  # Cap bei 1.0
            
        except Exception:
            return 0.5  # Fallback bei Fehlern
    
    def _calculate_personality_score(self, profile) -> float:
        """Berechnet Personality Match Score"""
        try:
            personality_scores = self.personality_match_dict
            if not personality_scores:
                return 0.5  # Neutral wenn keine Scores definiert
            
            total_score = 0.0
            trait_count = 0
            
            for trait, gift_score in personality_scores.items():
                profile_score = self._get_trait_score_from_profile(profile, trait)
                if profile_score is not None:
                    # Multiply gift's trait suitability with profile's trait strength
                    total_score += gift_score * profile_score
                    trait_count += 1
            
            return total_score / trait_count if trait_count > 0 else 0.5
            
        except Exception:
            return 0.5
    
    def _get_trait_score_from_profile(self, profile, trait: str) -> Optional[float]:
        """Holt den Score für ein spezifisches Trait aus dem PersonalityProfile"""
        try:
            if hasattr(profile, 'category') and profile.category:
                return profile.category._get_trait_score(profile, trait)
            
            # Direct trait mapping as fallback
            trait_mapping = {
                "high_openness": lambda p: p.openness if p.openness and p.openness > 0.6 else 0.0,
                "high_extraversion": lambda p: p.extraversion if p.extraversion and p.extraversion > 0.6 else 0.0,
                "high_conscientiousness": lambda p: p.conscientiousness if p.conscientiousness and p.conscientiousness > 0.6 else 0.0,
                "creative_type": lambda p: 1.0 if p.creative_type else 0.0,
                "tech_savvy": lambda p: 1.0 if p.tech_savvy else 0.0,
                "practical_type": lambda p: 1.0 if p.practical_type else 0.0,
                "health_conscious": lambda p: 1.0 if p.health_conscious else 0.0,
            }
            
            if trait in trait_mapping:
                return trait_mapping[trait](profile)
                
        except Exception:
            pass
        
        return None
    
    def _calculate_occasion_score(self, occasion: str) -> float:
        """Berechnet Occasion Match Score"""
        try:
            occasion_scores = self.occasion_suitability_dict
            if not occasion_scores:
                return 0.7  # Neutral-positiv wenn keine Scores definiert
            
            return occasion_scores.get(occasion, 0.3)  # Default 0.3 für unbekannte Anlässe
        except Exception:
            return 0.5
    
    def _calculate_relationship_score(self, relationship: str) -> float:
        """Berechnet Relationship Match Score"""
        try:
            relationship_scores = self.relationship_suitability_dict
            if not relationship_scores:
                return 0.7  # Neutral-positiv wenn keine Scores definiert
            
            return relationship_scores.get(relationship, 0.4)  # Default 0.4 für unbekannte Beziehungen
        except Exception:
            return 0.5
    
    def update_price_range(self):
        """Aktualisiert automatisch die Preisklasse basierend auf dem Preis"""
        if self.price < 50:
            self.price_range = PriceRange.BUDGET
        elif self.price < 150:
            self.price_range = PriceRange.AFFORDABLE
        elif self.price < 300:
            self.price_range = PriceRange.MODERATE
        elif self.price < 800:
            self.price_range = PriceRange.PREMIUM
        else:
            self.price_range = PriceRange.LUXURY
    
    def to_dict(self) -> Dict:
        """Vollständiger API Export"""
        return {
            'id': self.id,
            'name': self.name,
            'short_description': self.short_description,
            'long_description': self.long_description,
            'price': self.price,
            'display_price': self.display_price,
            'price_range': self.price_range.value,
            'currency': self.currency,
            'gift_type': self.gift_type.value,
            'personalization_level': self.personalization_level.value,
            'category': self.category.to_dict() if self.category else None,
            'tags': [{'tag': assoc.tag.to_dict(), 'relevance': assoc.relevance_score} 
                    for assoc in self.tag_associations if assoc.tag],
            'target_age_min': self.target_age_min,
            'target_age_max': self.target_age_max,
            'target_gender': self.target_gender,
            'image_url': self.image_url,
            'video_url': self.video_url,
            'personality_match_scores': self.personality_match_dict,
            'occasion_suitability': self.occasion_suitability_dict,
            'relationship_suitability': self.relationship_suitability_dict,
            'purchase_links': self.purchase_links_list,
            'availability_status': self.availability_status,
            'delivery_time_days': self.delivery_time_days,
            'quality_score': self.quality_score,
            'popularity_score': self.popularity_score,
            'success_rate': self.success_rate,
            'is_active': self.is_active,
            'is_featured': self.is_featured,
            'is_seasonal': self.is_seasonal,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    # ERWEITUNGEN FÜR DEINE BESTEHENDE app/models/gift.py


# === EMOTIONAL METHODS (In deine Gift class einfügen) ===

    def get_emotional_tags(self) -> List[str]:
        """Gibt emotionale Tags basierend auf Kategorie und Content zurück"""
        tags = []
        
        if not self.category:
            return tags
        
        # Category-based emotional tags
        category_emotion_map = {
            "romantic": ["love", "intimacy", "romance", "passion", "tenderness"],
            "adventurous": ["excitement", "adventure", "discovery", "freedom", "courage"],
            "thoughtful": ["gratitude", "appreciation", "mindfulness", "care", "understanding"],
            "creative": ["creativity", "inspiration", "expression", "innovation", "artistry"],
            "luxury": ["luxury", "prestige", "exclusivity", "sophistication", "elegance"],
            "wellness": ["peace", "relaxation", "healing", "balance", "renewal"],
            "connection": ["belonging", "unity", "togetherness", "family", "friendship"]
        }
        
        category_slug = self.category.slug
        if category_slug in category_emotion_map:
            tags.extend(category_emotion_map[category_slug])
        
        # Content-based emotional tags
        name_desc = (self.name + " " + self.short_description).lower()
        
        emotion_keywords = {
            "surprise": ["überraschung", "unexpected", "mystery", "secret"],
            "nostalgia": ["erinnerung", "memory", "vintage", "classic", "traditional"],
            "achievement": ["erfolg", "milestone", "accomplishment", "victory"],
            "comfort": ["comfort", "cozy", "warm", "safe", "secure"],
            "energy": ["energy", "power", "dynamic", "active", "vibrant"]
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in name_desc for keyword in keywords):
                tags.append(emotion)
        
        return list(set(tags))  # Remove duplicates

    def get_emotional_story(self) -> str:
        """Generiert emotionale Geschichte für das Gift"""
        if not self.category:
            return "Ein besonderes Geschenk, das von Herzen kommt."
        
        story_templates = {
            "romantic": f"Ein Geschenk, das 'Ich liebe dich' sagt ohne Worte zu brauchen. {self.name} schafft einen unvergesslichen Moment der Zweisamkeit.",
            "adventurous": f"Für alle, die das Leben voll auskosten wollen. {self.name} öffnet neue Horizonte und weckt den Entdeckergeist.",
            "thoughtful": f"Zeigt echte Wertschätzung und tiefe Verbundenheit. {self.name} sagt: 'Ich kenne dich und schätze dich.'",
            "creative": f"Weckt die Künstlerseele und beflügelt die Fantasie. {self.name} ist Inspiration zum Anfassen.",
            "luxury": f"Luxus, der das Leben bereichert und unvergesslich macht. {self.name} verwandelt gewöhnliche Momente in außergewöhnliche.",
            "wellness": f"Schenkt Momente der Ruhe in einer hektischen Welt. {self.name} ist eine Oase der Entspannung.",
            "connection": f"Bringt Menschen zusammen und schafft bleibende Erinnerungen. {self.name} stärkt die Bande zwischen euch."
        }
        
        return story_templates.get(self.category.slug, f"Ein durchdachtes Geschenk, das Freude bereitet. {self.name} zeigt, wie sehr du dir Gedanken gemacht hast.")

    def get_perfect_occasions(self) -> List[Dict[str, str]]:
        """Gibt perfekte Anlässe für dieses Geschenk zurück"""
        occasions = []
        
        if not self.category:
            return [{"occasion": "geburtstag", "reason": "Ein durchdachtes Geschenk"}]
        
        occasion_map = {
            "romantic": [
                {"occasion": "valentine", "reason": "Romantik pur für den Tag der Liebe"},
                {"occasion": "anniversary", "reason": "Feiert eure gemeinsame Zeit"},
                {"occasion": "proposal", "reason": "Der perfekte Moment für den Antrag"}
            ],
            "adventurous": [
                {"occasion": "graduation", "reason": "Start in ein neues Abenteuer"},
                {"occasion": "birthday", "reason": "Ein Jahr voller neuer Entdeckungen"},
                {"occasion": "sabbatical", "reason": "Zeit für große Träume"}
            ],
            "thoughtful": [
                {"occasion": "appreciation", "reason": "Zeigt echte Dankbarkeit"},
                {"occasion": "friendship", "reason": "Für besondere Freundschaften"},
                {"occasion": "support", "reason": "Unterstützung in schweren Zeiten"}
            ],
            "creative": [
                {"occasion": "birthday", "reason": "Inspiration für ein kreatives Jahr"},
                {"occasion": "new_hobby", "reason": "Start einer künstlerischen Reise"},
                {"occasion": "retirement", "reason": "Zeit für kreative Entfaltung"}
            ],
            "luxury": [
                {"occasion": "milestone_birthday", "reason": "Besondere Jahre verdienen Luxus"},
                {"occasion": "promotion", "reason": "Erfolg soll gefeiert werden"},
                {"occasion": "achievement", "reason": "Belohnung für harte Arbeit"}
            ],
            "wellness": [
                {"occasion": "stress_relief", "reason": "Entspannung wenn sie gebraucht wird"},
                {"occasion": "new_year", "reason": "Gesunde Vorsätze unterstützen"},
                {"occasion": "recovery", "reason": "Heilung für Körper und Seele"}
            ],
            "connection": [
                {"occasion": "family_reunion", "reason": "Gemeinsame Zeit wird noch schöner"},
                {"occasion": "friendship", "reason": "Stärkt die Verbindung zwischen euch"},
                {"occasion": "team_building", "reason": "Schweißt Gruppen zusammen"}
            ]
        }
        
        return occasion_map.get(self.category.slug, [{"occasion": "geburtstag", "reason": "Ein besonderes Geschenk"}])

    def calculate_emotional_match_score(self, target_emotions: List[str]) -> float:
        """Berechnet emotionalen Match-Score basierend auf Ziel-Emotionen"""
        if not target_emotions:
            return 0.0
        
        gift_emotions = self.get_emotional_tags()
        if not gift_emotions:
            return 0.0
        
        matches = 0
        for emotion in target_emotions:
            if emotion in gift_emotions:
                matches += 1
            else:
                # Semantic similarity check
                similar_emotions = self._get_similar_emotions(emotion)
                if any(sim_emotion in gift_emotions for sim_emotion in similar_emotions):
                    matches += 0.5  # Partial match for similar emotions
        
        return min(matches / len(target_emotions), 1.0)

    def _get_similar_emotions(self, emotion: str) -> List[str]:
        """Gibt ähnliche Emotionen zurück (semantische Ähnlichkeit)"""
        emotion_clusters = {
            "love": ["romance", "affection", "intimacy", "passion"],
            "joy": ["happiness", "delight", "cheerfulness", "celebration"],
            "excitement": ["thrill", "adventure", "energy", "enthusiasm"],
            "peace": ["calm", "tranquility", "serenity", "relaxation"],
            "creativity": ["inspiration", "innovation", "artistry", "expression"],
            "gratitude": ["appreciation", "thankfulness", "recognition"],
            "luxury": ["prestige", "elegance", "sophistication", "exclusivity"]
        }
        
        return emotion_clusters.get(emotion, [])

    def __repr__(self):
        return f'<Gift {self.name} ({self.display_price})>'


# === DATABASE INDEXES FÜR PERFORMANCE ===

# Index für häufige Queries
Index('idx_gifts_category_price', Gift.category_id, Gift.price)
Index('idx_gifts_active_featured', Gift.is_active, Gift.is_featured)
Index('idx_gifts_price_range_type', Gift.price_range, Gift.gift_type)
Index('idx_categories_active_sort', GiftCategory.is_active, GiftCategory.sort_order)
Index('idx_tags_type_weight', GiftTag.tag_type, GiftTag.weight)


# === QUERY HELPER FUNCTIONS ===

def get_gifts_by_personality(personality_profile, limit: int = 10, min_score: float = 0.3) -> List[Gift]:
    """
    Holt die besten Geschenke für ein PersonalityProfile
    
    Args:
        personality_profile: PersonalityProfile instance
        limit: Maximale Anzahl Ergebnisse
        min_score: Minimaler Match-Score (0.0 - 1.0)
    
    Returns:
        Liste von Gifts, sortiert nach Match-Score (beste zuerst)
    """
    try:
        # Alle aktiven Gifts im Budget
        gifts = Gift.query.filter(
            Gift.is_active == True,
            Gift.price >= personality_profile.budget_min,
            Gift.price <= personality_profile.budget_max
        ).all()
        
        # Berechne Match-Scores und filtere
        scored_gifts = []
        for gift in gifts:
            score = gift.calculate_match_score(personality_profile)
            if score >= min_score:
                scored_gifts.append((gift, score))
        
        # Sortiere nach Score (höchste zuerst)
        scored_gifts.sort(key=lambda x: x[1], reverse=True)
        
        # Gib nur die Gifts zurück (ohne Scores)
        return [gift for gift, score in scored_gifts[:limit]]
        
    except Exception:
        return []


def get_gifts_by_category(category_slug: str, limit: int = 20) -> List[Gift]:
    """Holt alle Gifts einer Kategorie"""
    try:
        category = GiftCategory.query.filter_by(slug=category_slug, is_active=True).first()
        if not category:
            return []
        
        return Gift.query.filter_by(
            category_id=category.id,
            is_active=True
        ).order_by(Gift.popularity_score.desc()).limit(limit).all()
        
    except Exception:
        return []


def get_gifts_by_tags(tag_names: List[str], limit: int = 20) -> List[Gift]:
    """Holt Gifts die bestimmte Tags haben"""
    try:
        if not tag_names:
            return []
        
        # Query für Gifts die mindestens einen der Tags haben
        return db.session.query(Gift).join(GiftTagAssociation).join(GiftTag).filter(
            Gift.is_active == True,
            GiftTag.name.in_(tag_names)
        ).order_by(Gift.popularity_score.desc()).limit(limit).all()
        
    except Exception:
        return []


def search_gifts(query: str, limit: int = 20) -> List[Gift]:
    """Einfache Text-Suche in Gifts"""
    try:
        search_term = f"%{query}%"
        
        return Gift.query.filter(
            Gift.is_active == True,
            db.or_(
                Gift.name.ilike(search_term),
                Gift.short_description.ilike(search_term),
                Gift.long_description.ilike(search_term)
            )
        ).order_by(Gift.popularity_score.desc()).limit(limit).all()
        
    except Exception:
        return []


def get_featured_gifts(limit: int = 6) -> List[Gift]:
    """Holt Featured/Promoted Gifts"""
    try:
        return Gift.query.filter_by(
            is_active=True,
            is_featured=True
        ).order_by(Gift.popularity_score.desc()).limit(limit).all()
        
    except Exception:
        return []