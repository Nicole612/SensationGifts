"""
Cart Models - E-Commerce Warenkorb-System für SensationGifts
===========================================================

🛒 LOCATION: app/models/cart.py

FEATURES:
- Multi-Item Cart Management
- Gift-spezifische Features (Wrapping, Messages, Delivery)
- Session & Persistent Carts
- Multiple Recipients Support
- Price Calculation with Taxes & Shipping
- Cart Analytics & Abandonment Tracking

CORRECTED LOGIC:
- PersonalityProfile beschreibt EMPFÄNGER (nicht User/Käufer)
- Cart gehört dem USER (Käufer)
- CartItems sind FÜR Empfänger basierend auf deren Persönlichkeit
"""

from app.extensions import db
from app.models.base import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from decimal import Decimal
from enum import Enum
import uuid
import json


class CartStatus(Enum):
    """Status des Warenkorbs"""
    ACTIVE = "active"                    # Aktiv in Bearbeitung
    ABANDONED = "abandoned"              # Verlassen (> 24h inaktiv)
    CHECKOUT_STARTED = "checkout_started" # Checkout begonnen
    CONVERTED = "converted"              # In Bestellung umgewandelt
    EXPIRED = "expired"                  # Abgelaufen


class CartItemType(Enum):
    """Typ des Cart-Items"""
    PHYSICAL_GIFT = "physical_gift"      # Physisches Geschenk
    DIGITAL_GIFT = "digital_gift"        # Digitales Geschenk
    EXPERIENCE_GIFT = "experience_gift"   # Erlebnis-Geschenk
    GIFT_CARD = "gift_card"              # Geschenkgutschein
    SUBSCRIPTION = "subscription"         # Abo-Geschenk
    CUSTOM_GIFT = "custom_gift"          # Individuelles Geschenk


class DeliveryMethod(Enum):
    """Lieferungsmethoden"""
    STANDARD_SHIPPING = "standard_shipping"  # Standard Post
    EXPRESS_SHIPPING = "express_shipping"    # Express Versand
    PICKUP = "pickup"                        # Selbstabholung
    EMAIL_DELIVERY = "email_delivery"        # E-Mail (Digital)
    SCHEDULED_DELIVERY = "scheduled_delivery" # Terminierte Lieferung


class GiftWrappingType(Enum):
    """Geschenkverpackungs-Optionen"""
    NONE = "none"                        # Keine Verpackung
    BASIC = "basic"                      # Basis Geschenkpapier
    PREMIUM = "premium"                  # Premium Verpackung
    LUXURY = "luxury"                    # Luxus Verpackung
    ECO_FRIENDLY = "eco_friendly"        # Nachhaltige Verpackung
    CUSTOM = "custom"                    # Individuelle Verpackung


class Cart(BaseModel):
    """
    Haupt-Warenkorb Model - gehört dem USER (Käufer)
    
    🛒 Der User (Käufer) hat Carts und kauft Geschenke für Empfänger
    """
    
    __tablename__ = 'carts'
    
    # === PRIMARY KEY ===
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # === USER ASSOCIATION (KÄUFER) ===
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=True)  # Käufer
    session_id = db.Column(db.String(100), nullable=True, index=True)  # Für anonyme Käufer
    
    # === CART METADATEN ===
    cart_status = db.Column(db.Enum(CartStatus), nullable=False, default=CartStatus.ACTIVE)
    cart_name = db.Column(db.String(100), nullable=True)  # "Weihnachtsgeschenke 2024"
    
    # === TOTALS & PRICING ===
    subtotal = db.Column(db.Numeric(10, 2), nullable=False, default=0.00)
    tax_amount = db.Column(db.Numeric(10, 2), nullable=False, default=0.00)
    shipping_cost = db.Column(db.Numeric(10, 2), nullable=False, default=0.00)
    discount_amount = db.Column(db.Numeric(10, 2), nullable=False, default=0.00)
    total_amount = db.Column(db.Numeric(10, 2), nullable=False, default=0.00)
    
    # === APPLIED DISCOUNTS ===
    discount_codes = db.Column(db.Text, nullable=True)  # JSON: ["CODE1", "CODE2"]
    applied_discounts = db.Column(db.Text, nullable=True)  # JSON: Discount Details
    
    # === CURRENCY & REGION ===
    currency_code = db.Column(db.String(3), nullable=False, default='EUR')
    country_code = db.Column(db.String(2), nullable=True)  # Für Steuern/Versand
    tax_rate = db.Column(db.Numeric(5, 4), nullable=False, default=0.19)  # 19% MwSt
    
    # === DELIVERY INFORMATION ===
    preferred_delivery_date = db.Column(db.DateTime, nullable=True)
    delivery_method = db.Column(db.Enum(DeliveryMethod), nullable=True)
    delivery_notes = db.Column(db.Text, nullable=True)
    
    # === GIFT-SPEZIFISCHE INFORMATIONEN ===
    is_gift = db.Column(db.Boolean, nullable=False, default=True)
    global_gift_message = db.Column(db.Text, nullable=True)  # Nachricht für alle Items
    gift_wrapping_preference = db.Column(db.Enum(GiftWrappingType), nullable=True)
    
    # === TRACKING & ANALYTICS ===
    items_count = db.Column(db.Integer, nullable=False, default=0)
    last_activity = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    abandonment_email_sent = db.Column(db.Boolean, nullable=False, default=False)
    
    # === CONVERSION TRACKING ===
    utm_source = db.Column(db.String(100), nullable=True)
    utm_campaign = db.Column(db.String(100), nullable=True)
    
    # === RELATIONSHIPS ===
    user = db.relationship('User', back_populates='carts')
    cart_items = db.relationship('CartItem', back_populates='cart', cascade='all, delete-orphan')
    
    # === COMPUTED PROPERTIES ===
    @property
    def is_empty(self) -> bool:
        """Ist der Cart leer?"""
        return self.items_count == 0
    
    @property
    def is_abandoned(self) -> bool:
        """Ist der Cart verlassen? (> 24h inaktiv)"""
        if self.cart_status == CartStatus.ABANDONED:
            return True
        
        time_threshold = datetime.utcnow() - timedelta(hours=24)
        return self.last_activity < time_threshold
    
    @property
    def abandonment_risk(self) -> float:
        """Risiko-Score für Cart-Abandonment (0.0-1.0)"""
        if self.is_empty:
            return 0.0
        
        hours_inactive = (datetime.utcnow() - self.last_activity).total_seconds() / 3600
        
        # Risk increases over time
        if hours_inactive < 1:
            return 0.1
        elif hours_inactive < 6:
            return 0.3
        elif hours_inactive < 12:
            return 0.6
        elif hours_inactive < 24:
            return 0.8
        else:
            return 1.0
    
    @property
    def discount_codes_list(self) -> List[str]:
        """Angewendete Discount-Codes als Liste"""
        if not self.discount_codes:
            return []
        try:
            return json.loads(self.discount_codes)
        except (json.JSONDecodeError, TypeError):
            return []
    
    @property
    def applied_discounts_data(self) -> List[Dict[str, Any]]:
        """Angewendete Discounts als strukturierte Daten"""
        if not self.applied_discounts:
            return []
        try:
            return json.loads(self.applied_discounts)
        except (json.JSONDecodeError, TypeError):
            return []
    
    def update_activity(self):
        """Aktualisiert letzte Aktivität"""
        self.last_activity = datetime.utcnow()
        if self.cart_status == CartStatus.ABANDONED:
            self.cart_status = CartStatus.ACTIVE
    
    def calculate_totals(self):
        """Berechnet alle Cart-Totals neu"""
        # Subtotal von allen Items
        self.subtotal = sum(item.line_total for item in self.cart_items)
        
        # Tax calculation
        self.tax_amount = self.subtotal * self.tax_rate
        
        # Shipping (vereinfacht - könnte komplexer sein)
        self.shipping_cost = self._calculate_shipping()
        
        # Total
        self.total_amount = self.subtotal + self.tax_amount + self.shipping_cost - self.discount_amount
        
        # Items count
        self.items_count = sum(item.quantity for item in self.cart_items)
    
    def _calculate_shipping(self) -> Decimal:
        """Berechnet Versandkosten (vereinfacht)"""
        if self.delivery_method == DeliveryMethod.PICKUP:
            return Decimal('0.00')
        elif self.delivery_method == DeliveryMethod.EXPRESS_SHIPPING:
            return Decimal('9.99')
        elif self.subtotal > 50:  # Gratis Versand ab 50€
            return Decimal('0.00')
        else:
            return Decimal('4.99')
    
    def apply_discount_code(self, code: str, discount_amount: Decimal, discount_info: Dict[str, Any]):
        """Wendet Discount-Code an"""
        codes = self.discount_codes_list
        if code not in codes:
            codes.append(code)
            self.discount_codes = json.dumps(codes)
        
        discounts = self.applied_discounts_data
        discounts.append({
            'code': code,
            'amount': float(discount_amount),
            'applied_at': datetime.utcnow().isoformat(),
            **discount_info
        })
        self.applied_discounts = json.dumps(discounts)
        
        self.discount_amount += discount_amount
        self.calculate_totals()
    
    def mark_as_abandoned(self):
        """Markiert Cart als verlassen"""
        self.cart_status = CartStatus.ABANDONED
    
    def start_checkout(self):
        """Startet Checkout-Prozess"""
        self.cart_status = CartStatus.CHECKOUT_STARTED
        self.update_activity()
    
    def convert_to_order(self):
        """Markiert Cart als konvertiert"""
        self.cart_status = CartStatus.CONVERTED


class CartItem(BaseModel):
    """
    Einzelnes Item im Warenkorb
    
    🎁 Gift für EMPFÄNGER basierend auf deren PersonalityProfile
    """
    
    __tablename__ = 'cart_items'
    
    # === PRIMARY KEY ===
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # === RELATIONSHIPS ===
    cart_id = db.Column(db.String(36), db.ForeignKey('carts.id'), nullable=False)
    gift_id = db.Column(db.String(36), nullable=True)  # Referenz auf Gift (falls vorhanden)
    
    # === ITEM DETAILS ===
    item_type = db.Column(db.Enum(CartItemType), nullable=False, default=CartItemType.PHYSICAL_GIFT)
    item_name = db.Column(db.String(200), nullable=False)
    item_description = db.Column(db.Text, nullable=True)
    item_image_url = db.Column(db.String(500), nullable=True)
    item_url = db.Column(db.String(500), nullable=True)  # Link zum Produkt
    
    # === PRICING ===
    unit_price = db.Column(db.Numeric(10, 2), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=1)
    line_total = db.Column(db.Numeric(10, 2), nullable=False)
    
    # === EMPFÄNGER-INFORMATIONEN (KORRIGIERT!) ===
    recipient_name = db.Column(db.String(100), nullable=True)
    recipient_email = db.Column(db.String(100), nullable=True)
    recipient_phone = db.Column(db.String(50), nullable=True)
    
    # === PERSONALISIERUNG ===
    personal_message = db.Column(db.Text, nullable=True)
    gift_wrapping = db.Column(db.Enum(GiftWrappingType), nullable=True, default=GiftWrappingType.BASIC)
    gift_wrapping_cost = db.Column(db.Numeric(10, 2), nullable=False, default=0.00)
    
    # === DELIVERY ===
    preferred_delivery_date = db.Column(db.DateTime, nullable=True)
    delivery_address = db.Column(db.Text, nullable=True)  # JSON mit Adress-Details
    delivery_method = db.Column(db.Enum(DeliveryMethod), nullable=True)
    
    # === AI & PERSONALIZATION CONTEXT (KORRIGIERT!) ===
    source_recommendation_id = db.Column(db.String(36), nullable=True)  # Von welcher AI-Empfehlung?
    source_personality_profile_id = db.Column(db.String(36), nullable=True)  # EMPFÄNGER's PersonalityProfile!
    personality_match_score = db.Column(db.Float, nullable=True)  # Wie gut passt es zur EMPFÄNGER-Persönlichkeit?
    recommendation_reasoning = db.Column(db.Text, nullable=True)  # Warum wurde es empfohlen?
    
    # === CUSTOMIZATION ===
    customization_options = db.Column(db.Text, nullable=True)  # JSON: Customization Details
    custom_engraving = db.Column(db.String(200), nullable=True)
    custom_color = db.Column(db.String(50), nullable=True)
    custom_size = db.Column(db.String(50), nullable=True)
    
    # === RELATIONSHIPS ===
    cart = db.relationship('Cart', back_populates='cart_items')
    
    # === COMPUTED PROPERTIES ===
    @property
    def total_price(self) -> Decimal:
        """Gesamtpreis inkl. Gift Wrapping"""
        return self.line_total + self.gift_wrapping_cost
    
    @property
    def delivery_address_data(self) -> Dict[str, Any]:
        """Lieferadresse als strukturierte Daten"""
        if not self.delivery_address:
            return {}
        try:
            return json.loads(self.delivery_address)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    @property
    def customization_data(self) -> Dict[str, Any]:
        """Customization-Optionen als strukturierte Daten"""
        if not self.customization_options:
            return {}
        try:
            return json.loads(self.customization_options)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    @property
    def is_personalized(self) -> bool:
        """Hat dieses Item Personalisierung?"""
        return bool(
            self.personal_message or 
            self.custom_engraving or 
            self.customization_options or
            self.gift_wrapping != GiftWrappingType.NONE
        )
    
    @property
    def requires_delivery_date(self) -> bool:
        """Benötigt dieses Item ein Lieferdatum?"""
        return self.item_type in [
            CartItemType.PHYSICAL_GIFT,
            CartItemType.EXPERIENCE_GIFT
        ]
    
    def calculate_line_total(self):
        """Berechnet Line Total neu"""
        self.line_total = self.unit_price * self.quantity
    
    def update_delivery_address(self, address_data: Dict[str, Any]):
        """Aktualisiert Lieferadresse"""
        self.delivery_address = json.dumps(address_data)
    
    def update_customization(self, customization_data: Dict[str, Any]):
        """Aktualisiert Customization-Optionen"""
        self.customization_options = json.dumps(customization_data)
    
    def set_gift_wrapping(self, wrapping_type: GiftWrappingType):
        """Setzt Gift Wrapping und berechnet Kosten"""
        self.gift_wrapping = wrapping_type
        
        # Wrapping costs (vereinfacht)
        wrapping_costs = {
            GiftWrappingType.NONE: Decimal('0.00'),
            GiftWrappingType.BASIC: Decimal('2.99'),
            GiftWrappingType.PREMIUM: Decimal('5.99'),
            GiftWrappingType.LUXURY: Decimal('9.99'),
            GiftWrappingType.ECO_FRIENDLY: Decimal('3.99'),
            GiftWrappingType.CUSTOM: Decimal('12.99')
        }
        
        self.gift_wrapping_cost = wrapping_costs.get(wrapping_type, Decimal('0.00'))


class CartActivity(BaseModel):
    """
    Cart-Aktivitäten für Analytics und Abandonment-Recovery
    
    📊 Trackt alle Cart-Interaktionen für Business Intelligence
    """
    
    __tablename__ = 'cart_activities'
    
    # === PRIMARY KEY ===
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # === RELATIONSHIPS ===
    cart_id = db.Column(db.String(36), db.ForeignKey('carts.id'), nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=True)
    
    # === ACTIVITY DETAILS ===
    activity_type = db.Column(db.String(50), nullable=False)  # 'add_item', 'remove_item', 'update_quantity', etc.
    activity_description = db.Column(db.Text, nullable=True)
    
    # === CONTEXT DATA ===
    item_id = db.Column(db.String(36), nullable=True)  # Betroffenes Item (falls relevant)
    old_value = db.Column(db.Text, nullable=True)  # Alter Wert (JSON)
    new_value = db.Column(db.Text, nullable=True)  # Neuer Wert (JSON)
    
    # === SESSION INFO ===
    session_id = db.Column(db.String(100), nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)
    user_agent = db.Column(db.String(500), nullable=True)
    referrer = db.Column(db.String(500), nullable=True)
    
    # === TIMING ===
    activity_timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # === RELATIONSHIPS ===
    cart = db.relationship('Cart')
    user = db.relationship('User')


class SavedCart(BaseModel):
    """
    Gespeicherte Carts für Wishlist-Funktionalität
    
    💾 Erlaubt es Usern, Carts für später zu speichern
    """
    
    __tablename__ = 'saved_carts'
    
    # === PRIMARY KEY ===
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # === RELATIONSHIPS ===
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    original_cart_id = db.Column(db.String(36), db.ForeignKey('carts.id'), nullable=True)
    
    # === SAVED CART INFO ===
    saved_cart_name = db.Column(db.String(100), nullable=False)  # "Weihnachtsgeschenke für Familie"
    saved_cart_description = db.Column(db.Text, nullable=True)
    
    # === CART DATA (Snapshot) ===
    cart_data = db.Column(db.Text, nullable=False)  # JSON: Complete cart snapshot
    items_snapshot = db.Column(db.Text, nullable=False)  # JSON: Items at time of saving
    
    # === METADATA ===
    is_public = db.Column(db.Boolean, nullable=False, default=False)  # Öffentliche Wishlist?
    share_code = db.Column(db.String(20), nullable=True, unique=True)  # Code zum Teilen
    
    # === TIMING ===
    saved_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    last_accessed = db.Column(db.DateTime, nullable=True)
    
    # === RELATIONSHIPS ===
    user = db.relationship('User')
    original_cart = db.relationship('Cart')
    
    def generate_share_code(self):
        """Generiert Share-Code für öffentliche Wishlists"""
        import random
        import string
        
        self.share_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    
    def restore_to_cart(self) -> 'Cart':
        """Stellt gespeicherten Cart wieder her"""
        try:
            cart_data = json.loads(self.cart_data)
            items_data = json.loads(self.items_snapshot)
            
            # Erstelle neuen Cart
            new_cart = Cart(
                user_id=self.user_id,
                cart_name=f"Restored: {self.saved_cart_name}",
                currency_code=cart_data.get('currency_code', 'EUR'),
                is_gift=cart_data.get('is_gift', True)
            )
            
            db.session.add(new_cart)
            db.session.flush()  # Get cart ID
            
            # Erstelle Items
            for item_data in items_data:
                cart_item = CartItem(
                    cart_id=new_cart.id,
                    item_name=item_data['item_name'],
                    item_description=item_data.get('item_description'),
                    unit_price=Decimal(str(item_data['unit_price'])),
                    quantity=item_data['quantity'],
                    recipient_name=item_data.get('recipient_name'),
                    personal_message=item_data.get('personal_message')
                )
                cart_item.calculate_line_total()
                db.session.add(cart_item)
            
            new_cart.calculate_totals()
            return new_cart
            
        except Exception as e:
            db.session.rollback()
            raise e


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_cart_for_user(user_id: str, session_id: str = None) -> Cart:
    """Erstellt neuen Cart für User"""
    cart = Cart(
        user_id=user_id,
        session_id=session_id,
        cart_status=CartStatus.ACTIVE
    )
    db.session.add(cart)
    return cart


def get_or_create_cart(user_id: str = None, session_id: str = None) -> Cart:
    """Holt existierenden Cart oder erstellt neuen"""
    
    # Für eingeloggte User
    if user_id:
        cart = Cart.query.filter_by(
            user_id=user_id,
            cart_status=CartStatus.ACTIVE
        ).first()
        
        if cart:
            cart.update_activity()
            return cart
        
        return create_cart_for_user(user_id, session_id)
    
    # Für anonyme User
    elif session_id:
        cart = Cart.query.filter_by(
            session_id=session_id,
            cart_status=CartStatus.ACTIVE
        ).first()
        
        if cart:
            cart.update_activity()
            return cart
        
        cart = Cart(session_id=session_id, cart_status=CartStatus.ACTIVE)
        db.session.add(cart)
        return cart
    
    else:
        raise ValueError("Either user_id or session_id must be provided")


def merge_carts(user_cart: Cart, session_cart: Cart) -> Cart:
    """Mergt Session-Cart in User-Cart bei Login"""
    
    # Übertrage alle Items von session_cart zu user_cart
    for item in session_cart.cart_items:
        item.cart_id = user_cart.id
    
    # Übertrage relevante Cart-Properties
    if session_cart.global_gift_message and not user_cart.global_gift_message:
        user_cart.global_gift_message = session_cart.global_gift_message
    
    if session_cart.preferred_delivery_date and not user_cart.preferred_delivery_date:
        user_cart.preferred_delivery_date = session_cart.preferred_delivery_date
    
    # Berechne Totals neu
    user_cart.calculate_totals()
    user_cart.update_activity()
    
    # Markiere Session-Cart als konvertiert
    session_cart.cart_status = CartStatus.CONVERTED
    
    return user_cart


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'CartStatus', 'CartItemType', 'DeliveryMethod', 'GiftWrappingType',
    
    # Models
    'Cart', 'CartItem', 'CartActivity', 'SavedCart',
    
    # Utility Functions
    'create_cart_for_user', 'get_or_create_cart', 'merge_carts'
]