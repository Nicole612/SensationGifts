# 📁 app/services/email_service.py
# ================================================================
# EMAIL & NOTIFICATION SERVICE - Complete Customer Engagement
# ================================================================

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from jinja2 import Environment, FileSystemLoader
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import json
import requests
from celery import Celery
from flask import current_app, url_for
import boto3
from dataclasses import dataclass
import uuid
from pathlib import Path

@dataclass
class EmailTemplate:
    subject: str
    template_name: str
    variables: Dict
    recipient_type: str = 'customer'  # customer, admin, system
    priority: str = 'normal'  # high, normal, low
    category: str = 'transactional'  # marketing, transactional, notification

@dataclass
class NotificationMessage:
    recipient_id: str
    title: str
    message: str
    type: str  # success, info, warning, error
    action_url: Optional[str] = None
    image_url: Optional[str] = None
    expires_at: Optional[datetime] = None

class EmailService:
    """
    Comprehensive Email & Notification Service
    - Transactional Emails (Orders, Password Reset, etc.)
    - Marketing Emails (Newsletters, Promotions)
    - Push Notifications
    - SMS Notifications
    - In-App Notifications
    - Emotional Email Templates
    """
    
    def __init__(self):
        self.smtp_server = current_app.config.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = current_app.config.get('SMTP_PORT', 587)
        self.smtp_username = current_app.config.get('SMTP_USERNAME')
        self.smtp_password = current_app.config.get('SMTP_PASSWORD')
        self.from_email = current_app.config.get('FROM_EMAIL', 'noreply@sensationgifts.com')
        self.from_name = current_app.config.get('FROM_NAME', 'SensationGifts')
        
        # Template environment
        template_dir = Path(current_app.root_path) / 'templates' / 'emails'
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # AWS SES (if configured)
        self.ses_client = None
        if current_app.config.get('AWS_SES_REGION'):
            self.ses_client = boto3.client(
                'ses',
                region_name=current_app.config.get('AWS_SES_REGION'),
                aws_access_key_id=current_app.config.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=current_app.config.get('AWS_SECRET_ACCESS_KEY')
            )
        
        # Push notification service
        self.fcm_server_key = current_app.config.get('FCM_SERVER_KEY')
        
        # SMS service (Twilio)
        self.twilio_sid = current_app.config.get('TWILIO_ACCOUNT_SID')
        self.twilio_token = current_app.config.get('TWILIO_AUTH_TOKEN')
        self.twilio_from = current_app.config.get('TWILIO_FROM_NUMBER')
    
    # ================================================================
    # TRANSACTIONAL EMAILS
    # ================================================================
    
    def send_order_confirmation(self, order_data: Dict) -> bool:
        """Send order confirmation email with emotional touch"""
        
        template = EmailTemplate(
            subject=f"🎉 Deine Bestellung #{order_data['order_number']} ist bestätigt!",
            template_name='order_confirmation.html',
            variables={
                'customer_name': order_data['customer_name'],
                'order_number': order_data['order_number'],
                'order_date': order_data['created_at'].strftime('%d.%m.%Y'),
                'items': order_data['items'],
                'total_amount': order_data['total_amount'],
                'shipping_address': order_data['shipping_address'],
                'estimated_delivery': order_data['estimated_delivery'],
                'tracking_url': f"{current_app.config.get('BASE_URL')}/bestellung/{order_data['id']}/tracking",
                'emotional_message': self._generate_emotional_order_message(order_data),
                'gift_suggestions': self._get_related_gift_suggestions(order_data)
            },
            category='transactional',
            priority='high'
        )
        
        return self._send_email(order_data['customer_email'], template)
    
    def send_shipping_notification(self, order_data: Dict) -> bool:
        """Send shipping notification with tracking info"""
        
        template = EmailTemplate(
            subject=f"📦 Deine Geschenke sind unterwegs! (#{order_data['order_number']})",
            template_name='shipping_notification.html',
            variables={
                'customer_name': order_data['customer_name'],
                'order_number': order_data['order_number'],
                'tracking_number': order_data['tracking_number'],
                'carrier': order_data['carrier'],
                'estimated_delivery': order_data['estimated_delivery'],
                'tracking_url': order_data['tracking_url'],
                'items': order_data['items'],
                'excitement_message': "Die Vorfreude steigt! 🎁✨ Deine sorgfältig ausgewählten Geschenke sind jetzt auf dem Weg zu dir.",
                'delivery_tips': [
                    "📱 Verfolge deine Sendung mit der Tracking-Nummer",
                    "🏠 Stelle sicher, dass jemand für die Annahme da ist", 
                    "📦 Bewahre die Verpackung für eventuelle Rückgaben auf",
                    "⭐ Wir freuen uns auf dein Feedback nach der Lieferung!"
                ]
            }
        )
        
        return self._send_email(order_data['customer_email'], template)
    
    def send_delivery_confirmation(self, order_data: Dict) -> bool:
        """Send delivery confirmation and request feedback"""
        
        template = EmailTemplate(
            subject=f"🎊 Deine Geschenke sind angekommen! Wie war die Reaktion?",
            template_name='delivery_confirmation.html',
            variables={
                'customer_name': order_data['customer_name'],
                'order_number': order_data['order_number'],
                'delivery_date': datetime.now().strftime('%d.%m.%Y'),
                'items': order_data['items'],
                'review_url': f"{current_app.config.get('BASE_URL')}/bewertung/{order_data['id']}",
                'emotional_survey_url': f"{current_app.config.get('BASE_URL')}/emotional-feedback/{order_data['id']}",
                'celebration_message': "🎉 Das perfekte Geschenk ist angekommen! Wir sind gespannt auf die Reaktion des Beschenkten.",
                'review_incentive': "📝 Teile deine Erfahrung und erhalte 10% Rabatt auf deine nächste Bestellung!"
            }
        )
        
        return self._send_email(order_data['customer_email'], template)
    
    def send_password_reset(self, user_data: Dict, reset_token: str) -> bool:
        """Send password reset email"""
        
        reset_url = f"{current_app.config.get('BASE_URL')}/auth/reset-password?token={reset_token}"
        
        template = EmailTemplate(
            subject="🔐 Passwort zurücksetzen - SensationGifts",
            template_name='password_reset.html',
            variables={
                'user_name': user_data['first_name'] or 'Lieber Kunde',
                'reset_url': reset_url,
                'expiry_time': '24 Stunden',
                'security_message': "Falls du diese E-Mail nicht angefordert hast, ignoriere sie einfach. Dein Passwort bleibt unverändert."
            }
        )
        
        return self._send_email(user_data['email'], template)
    
    def send_welcome_email(self, user_data: Dict) -> bool:
        """Send welcome email to new users"""
        
        template = EmailTemplate(
            subject="🎁 Willkommen bei SensationGifts! Deine Geschenk-Reise beginnt",
            template_name='welcome_email.html',
            variables={
                'user_name': user_data['first_name'] or 'Lieber Geschenkfinder',
                'personality_quiz_url': f"{current_app.config.get('BASE_URL')}/geschenke-finden/quiz",
                'welcome_discount': 'WELCOME15',
                'discount_amount': '15%',
                'getting_started_tips': [
                    "🧠 Mache den Persönlichkeitstest für beste Empfehlungen",
                    "💝 Entdecke über 1000 emotionale Geschenkideen", 
                    "🎨 Personalisiere Geschenke mit deiner individuellen Note",
                    "📱 Installiere unsere App für die beste Erfahrung"
                ],
                'featured_categories': self._get_featured_categories()
            }
        )
        
        return self._send_email(user_data['email'], template)
    
    # ================================================================
    # MARKETING EMAILS
    # ================================================================
    
    def send_personalized_newsletter(self, user_data: Dict, content: Dict) -> bool:
        """Send personalized newsletter based on user preferences"""
        
        template = EmailTemplate(
            subject=f"✨ {user_data['first_name']}, neue Geschenkideen nur für dich!",
            template_name='personalized_newsletter.html',
            variables={
                'user_name': user_data['first_name'],
                'personality_type': user_data.get('personality_type', 'Individualist'),
                'recommended_products': content['recommended_products'],
                'trending_gifts': content['trending_gifts'],
                'seasonal_suggestions': content['seasonal_suggestions'],
                'personal_message': self._generate_personal_newsletter_message(user_data),
                'unsubscribe_url': f"{current_app.config.get('BASE_URL')}/newsletter/unsubscribe?token={user_data['unsubscribe_token']}"
            },
            category='marketing'
        )
        
        return self._send_email(user_data['email'], template)
    
    def send_abandoned_cart_email(self, user_data: Dict, cart_items: List[Dict]) -> bool:
        """Send abandoned cart recovery email"""
        
        total_value = sum(item['price'] * item['quantity'] for item in cart_items)
        
        template = EmailTemplate(
            subject="💔 Du hast etwas Wunderbares zurückgelassen...",
            template_name='abandoned_cart.html',
            variables={
                'user_name': user_data['first_name'],
                'cart_items': cart_items,
                'total_value': total_value,
                'cart_url': f"{current_app.config.get('BASE_URL')}/warenkorb",
                'incentive_discount': 'COMEBACK10',
                'discount_amount': '10%',
                'urgency_message': "⏰ Deine Geschenkauswahl wartet noch 48 Stunden auf dich!",
                'emotional_hook': "Diese Geschenke wurden speziell für deine Persönlichkeit ausgewählt. Lass sie nicht entkommen! 💝"
            }
        )
        
        return self._send_email(user_data['email'], template)
    
    def send_birthday_reminder(self, recipient_data: Dict, gift_suggestions: List[Dict]) -> bool:
        """Send birthday reminder with gift suggestions"""
        
        template = EmailTemplate(
            subject="🎂 Geburtstag in Sicht! Perfekte Geschenkideen warten",
            template_name='birthday_reminder.html',
            variables={
                'user_name': recipient_data['user_name'],
                'birthday_person': recipient_data['birthday_person'],
                'days_until_birthday': recipient_data['days_until_birthday'],
                'relationship': recipient_data['relationship'],
                'gift_suggestions': gift_suggestions,
                'quick_order_url': f"{current_app.config.get('BASE_URL')}/schnellbestellung",
                'personality_hint': recipient_data.get('personality_hint', ''),
                'reminder_message': f"🎁 In {recipient_data['days_until_birthday']} Tagen ist es soweit! Wir haben die perfekten Geschenke für {recipient_data['birthday_person']} gefunden."
            }
        )
        
        return self._send_email(recipient_data['user_email'], template)
    
    # ================================================================
    # PUSH NOTIFICATIONS
    # ================================================================
    
    def send_push_notification(self, user_tokens: List[str], title: str, body: str, 
                               data: Optional[Dict] = None, image_url: Optional[str] = None) -> bool:
        """Send push notification via Firebase Cloud Messaging"""
        
        if not self.fcm_server_key or not user_tokens:
            return False
        
        headers = {
            'Authorization': f'key={self.fcm_server_key}',
            'Content-Type': 'application/json'
        }
        
        notification_data = {
            'registration_ids': user_tokens,
            'notification': {
                'title': title,
                'body': body,
                'icon': '/icons/icon-192x192.png',
                'badge': '/icons/badge-72x72.png',
                'click_action': data.get('url', '/') if data else '/',
                'tag': data.get('tag', 'default') if data else 'default'
            },
            'data': data or {}
        }
        
        if image_url:
            notification_data['notification']['image'] = image_url
        
        try:
            response = requests.post(
                'https://fcm.googleapis.com/fcm/send',
                headers=headers,
                json=notification_data,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            current_app.logger.error(f"Push notification failed: {str(e)}")
            return False
    
    def send_order_status_push(self, user_tokens: List[str], order_data: Dict, status: str) -> bool:
        """Send order status push notification"""
        
        status_messages = {
            'confirmed': {
                'title': '🎉 Bestellung bestätigt!',
                'body': f'Deine Bestellung #{order_data["order_number"]} wurde bestätigt.',
                'icon': '✅'
            },
            'processing': {
                'title': '📦 Bestellung wird bearbeitet',
                'body': f'Wir bereiten deine Geschenke liebevoll vor.',
                'icon': '⚙️'
            },
            'shipped': {
                'title': '🚚 Geschenke sind unterwegs!',
                'body': f'Tracking: {order_data.get("tracking_number", "Siehe E-Mail")}',
                'icon': '📱'
            },
            'delivered': {
                'title': '🎊 Geschenke angekommen!',
                'body': 'Wie war die Reaktion? Teile deine Erfahrung!',
                'icon': '⭐'
            }
        }
        
        message = status_messages.get(status, status_messages['confirmed'])
        
        return self.send_push_notification(
            user_tokens=user_tokens,
            title=message['title'],
            body=message['body'],
            data={
                'url': f'/bestellung/{order_data["id"]}/tracking',
                'order_id': order_data['id'],
                'status': status
            }
        )
    
    def send_personalized_recommendation_push(self, user_tokens: List[str], 
                                              user_data: Dict, products: List[Dict]) -> bool:
        """Send personalized product recommendation push"""
        
        if not products:
            return False
        
        featured_product = products[0]
        
        return self.send_push_notification(
            user_tokens=user_tokens,
            title=f"💝 {user_data['first_name']}, perfekte Geschenke für dich!",
            body=f"{featured_product['name']} - {featured_product['match_score']}% Match",
            data={
                'url': f'/produkt/{featured_product["slug"]}',
                'product_id': featured_product['id'],
                'match_score': featured_product['match_score']
            },
            image_url=featured_product.get('image_url')
        )
    
    # ================================================================
    # SMS NOTIFICATIONS
    # ================================================================
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS notification via Twilio"""
        
        if not all([self.twilio_sid, self.twilio_token, self.twilio_from]):
            return False
        
        try:
            from twilio.rest import Client
            
            client = Client(self.twilio_sid, self.twilio_token)
            
            message = client.messages.create(
                body=message,
                from_=self.twilio_from,
                to=phone_number
            )
            
            return message.sid is not None
            
        except Exception as e:
            current_app.logger.error(f"SMS sending failed: {str(e)}")
            return False
    
    def send_delivery_sms(self, phone_number: str, order_data: Dict) -> bool:
        """Send delivery notification SMS"""
        
        message = f"""
🎁 SensationGifts: Deine Bestellung #{order_data['order_number']} wird heute geliefert! 
📦 Tracking: {order_data.get('tracking_number', 'Siehe E-Mail')}
🏠 Bitte sei für die Annahme verfügbar.
        """.strip()
        
        return self.send_sms(phone_number, message)
    
    def send_urgent_sms(self, phone_number: str, message: str) -> bool:
        """Send urgent notification SMS"""
        
        urgent_message = f"🚨 DRINGEND - SensationGifts: {message}"
        return self.send_sms(phone_number, urgent_message)
    
    # ================================================================
    # IN-APP NOTIFICATIONS
    # ================================================================
    
    def create_in_app_notification(self, notification: NotificationMessage) -> str:
        """Create in-app notification in database"""
        
        from app.models.notification import InAppNotification
        from app.models import db
        
        db_notification = InAppNotification(
            id=str(uuid.uuid4()),
            recipient_id=notification.recipient_id,
            title=notification.title,
            message=notification.message,
            type=notification.type,
            action_url=notification.action_url,
            image_url=notification.image_url,
            expires_at=notification.expires_at or datetime.utcnow() + timedelta(days=30),
            is_read=False,
            created_at=datetime.utcnow()
        )
        
        db.session.add(db_notification)
        db.session.commit()
        
        return db_notification.id
    
    def send_real_time_notification(self, user_id: str, notification: NotificationMessage) -> bool:
        """Send real-time notification via WebSocket"""
        
        # Store in database
        notification_id = self.create_in_app_notification(notification)
        
        # Send via WebSocket (implementation depends on your WebSocket setup)
        notification_data = {
            'id': notification_id,
            'title': notification.title,
            'message': notification.message,
            'type': notification.type,
            'action_url': notification.action_url,
            'image_url': notification.image_url,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Using Socket.IO or similar WebSocket implementation
            from app.websockets import emit_to_user
            emit_to_user(user_id, 'notification', notification_data)
            return True
        except Exception as e:
            current_app.logger.error(f"Real-time notification failed: {str(e)}")
            return False
    
    # ================================================================
    # HELPER METHODS
    # ================================================================
    
    def _send_email(self, recipient: str, template: EmailTemplate) -> bool:
        """Send email using configured method (SMTP or SES)"""
        
        try:
            # Render template
            template_obj = self.jinja_env.get_template(template.template_name)
            html_content = template_obj.render(**template.variables)
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = template.subject
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = recipient
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            # Send via SES if available, otherwise SMTP
            if self.ses_client:
                return self._send_via_ses(recipient, msg)
            else:
                return self._send_via_smtp(recipient, msg)
                
        except Exception as e:
            current_app.logger.error(f"Email sending failed: {str(e)}")
            return False
    
    def _send_via_smtp(self, recipient: str, msg: MIMEMultipart) -> bool:
        """Send email via SMTP"""
        
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            return True
        except Exception as e:
            current_app.logger.error(f"SMTP sending failed: {str(e)}")
            return False
    
    def _send_via_ses(self, recipient: str, msg: MIMEMultipart) -> bool:
        """Send email via AWS SES"""
        
        try:
            response = self.ses_client.send_raw_email(
                Source=self.from_email,
                Destinations=[recipient],
                RawMessage={'Data': msg.as_string()}
            )
            return 'MessageId' in response
        except Exception as e:
            current_app.logger.error(f"SES sending failed: {str(e)}")
            return False
    
    def _generate_emotional_order_message(self, order_data: Dict) -> str:
        """Generate personalized emotional message for order"""
        
        messages = [
            "🎉 Was für eine wunderbare Auswahl! Diese Geschenke werden garantiert für strahlende Gesichter sorgen.",
            "✨ Du hast den perfekten Weg gefunden, jemandem eine Freude zu machen. Das ist etwas ganz Besonderes!",
            "💝 Deine durchdachte Geschenkauswahl zeigt, wie sehr du dich um andere kümmerst. Das ist wirklich schön.",
            "🌟 Diese Geschenke tragen deine Persönlichkeit in sich - authentisch, durchdacht und von Herzen kommend."
        ]
        
        # Select message based on order value or items
        return messages[hash(order_data['id']) % len(messages)]
    
    def _get_related_gift_suggestions(self, order_data: Dict) -> List[Dict]:
        """Get related gift suggestions for order confirmation"""
        
        # This would typically query your recommendation engine
        return [
            {
                'name': 'Geschenkverpackung Premium',
                'price': 4.99,
                'image': '/images/gift-wrap-premium.jpg',
                'description': 'Mache dein Geschenk noch spezieller'
            },
            {
                'name': 'Persönliche Grußkarte',
                'price': 2.99,
                'image': '/images/greeting-card.jpg', 
                'description': 'Mit deiner individuellen Nachricht'
            }
        ]
    
    def _generate_personal_newsletter_message(self, user_data: Dict) -> str:
        """Generate personalized newsletter message"""
        
        personality_type = user_data.get('personality_type', 'Individualist')
        
        messages = {
            'Romantiker': "💕 Als romantische Seele hast du ein Gespür für bedeutungsvolle Gesten...",
            'Abenteurer': "🚀 Dein Abenteuergeist verdient Geschenke, die genauso einzigartig sind wie du...",
            'Kreativer': "🎨 Deine kreative Energie inspiriert uns täglich zu neuen Geschenkideen...",
            'Traditionalist': "🏛️ Du schätzt zeitlose Werte und Geschenke mit Geschichte...",
            'Individualist': "✨ Du gehst deinen eigenen Weg - und deine Geschenke sollten das auch tun..."
        }
        
        return messages.get(personality_type, messages['Individualist'])
    
    def _get_featured_categories(self) -> List[Dict]:
        """Get featured categories for welcome email"""
        
        return [
            {'name': 'Romantische Geschenke', 'emoji': '💕', 'url': '/kategorie/romantisch'},
            {'name': 'Personalisierte Geschenke', 'emoji': '🎨', 'url': '/kategorie/personalisiert'},
            {'name': 'Erlebnisgeschenke', 'emoji': '🎭', 'url': '/kategorie/erlebnisse'},
            {'name': 'Wellness & Entspannung', 'emoji': '🧘‍♀️', 'url': '/kategorie/wellness'}
        ]

# ================================================================
# CELERY TASKS - For background email processing
# ================================================================

from app.extensions import celery

@celery.task(bind=True, max_retries=3)
def send_async_email(self, recipient: str, template_data: Dict):
    """Send email asynchronously"""
    
    try:
        email_service = EmailService()
        template = EmailTemplate(**template_data)
        success = email_service._send_email(recipient, template)
        
        if not success:
            raise Exception("Email sending failed")
            
        return {'status': 'sent', 'recipient': recipient}
        
    except Exception as e:
        current_app.logger.error(f"Async email task failed: {str(e)}")
        raise self.retry(countdown=60 * (self.request.retries + 1))

@celery.task
def send_bulk_newsletter(user_list: List[Dict], content: Dict):
    """Send newsletter to multiple users"""
    
    email_service = EmailService()
    results = {'sent': 0, 'failed': 0}
    
    for user_data in user_list:
        try:
            success = email_service.send_personalized_newsletter(user_data, content)
            if success:
                results['sent'] += 1
            else:
                results['failed'] += 1
        except Exception as e:
            current_app.logger.error(f"Newsletter failed for {user_data['email']}: {str(e)}")
            results['failed'] += 1
    
    return results

@celery.task
def process_abandoned_carts():
    """Process abandoned carts and send recovery emails"""
    
    from app.models.cart import CartItem
    from app.models.user import User
    from datetime import datetime, timedelta
    
    # Find abandoned carts (inactive for 2+ hours)
    cutoff_time = datetime.utcnow() - timedelta(hours=2)
    
    abandoned_carts = db.session.query(CartItem.user_id, func.count(CartItem.id).label('item_count')).filter(
        CartItem.updated_at < cutoff_time,
        CartItem.abandoned_email_sent == False
    ).group_by(CartItem.user_id).all()
    
    email_service = EmailService()
    
    for user_id, item_count in abandoned_carts:
        user = User.query.get(user_id)
        if not user:
            continue
            
        cart_items = CartItem.query.filter_by(user_id=user_id).all()
        cart_data = [
            {
                'name': item.product.name,
                'price': item.product.price_basic,
                'quantity': item.quantity,
                'image': item.product.primary_image
            }
            for item in cart_items
        ]
        
        success = email_service.send_abandoned_cart_email(user.to_dict(), cart_data)
        
        if success:
            # Mark as sent to avoid duplicate emails
            for item in cart_items:
                item.abandoned_email_sent = True
            db.session.commit()

# ================================================================
# FLASK ROUTES - Notification API
# ================================================================

from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

notifications_bp = Blueprint('notifications', __name__)

@notifications_bp.route('/api/notifications/subscribe', methods=['POST'])
@login_required
def subscribe_push_notifications():
    """Subscribe user to push notifications"""
    
    data = request.get_json()
    subscription_data = data.get('subscription')
    
    if not subscription_data:
        return jsonify({'error': 'Subscription data required'}), 400
    
    # Save subscription to database
    from app.models.push_subscription import PushSubscription
    from app.models import db
    
    existing = PushSubscription.query.filter_by(
        user_id=current_user.id,
        endpoint=subscription_data['endpoint']
    ).first()
    
    if not existing:
        subscription = PushSubscription(
            user_id=current_user.id,
            endpoint=subscription_data['endpoint'],
            p256dh_key=subscription_data['keys']['p256dh'],
            auth_key=subscription_data['keys']['auth'],
            is_active=True
        )
        db.session.add(subscription)
        db.session.commit()
    
    return jsonify({'success': True})

@notifications_bp.route('/api/notifications/unread', methods=['GET'])
@login_required
def get_unread_notifications():
    """Get unread in-app notifications for user"""
    
    from app.models.notification import InAppNotification
    
    notifications = InAppNotification.query.filter_by(
        recipient_id=current_user.id,
        is_read=False
    ).filter(
        InAppNotification.expires_at > datetime.utcnow()
    ).order_by(InAppNotification.created_at.desc()).limit(20).all()
    
    return jsonify({
        'notifications': [notif.to_dict() for notif in notifications],
        'unread_count': len(notifications)
    })

@notifications_bp.route('/api/notifications/<notification_id>/read', methods=['PUT'])
@login_required
def mark_notification_read(notification_id):
    """Mark notification as read"""
    
    from app.models.notification import InAppNotification
    from app.models import db
    
    notification = InAppNotification.query.filter_by(
        id=notification_id,
        recipient_id=current_user.id
    ).first_or_404()
    
    notification.is_read = True
    notification.read_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'success': True})

@notifications_bp.route('/api/notifications/test', methods=['POST'])
@login_required
def send_test_notification():
    """Send test notification (admin only)"""
    
    if not current_user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    data = request.get_json()
    notification_type = data.get('type', 'push')
    
    email_service = EmailService()
    
    if notification_type == 'push':
        # Get user's push subscriptions
        from app.models.push_subscription import PushSubscription
        subscriptions = PushSubscription.query.filter_by(
            user_id=current_user.id,
            is_active=True
        ).all()
        
        tokens = [sub.endpoint for sub in subscriptions]
        
        success = email_service.send_push_notification(
            user_tokens=tokens,
            title="🧪 Test Notification",
            body="Dies ist eine Test-Benachrichtigung von SensationGifts!",
            data={'url': '/'}
        )
        
    elif notification_type == 'email':
        template = EmailTemplate(
            subject="🧪 Test E-Mail von SensationGifts",
            template_name='test_email.html',
            variables={
                'user_name': current_user.first_name or 'Test User',
                'test_message': 'Dies ist eine Test-E-Mail. Alles funktioniert perfekt! 🎉'
            }
        )
        
        success = email_service._send_email(current_user.email, template)
    
    elif notification_type == 'sms':
        if current_user.phone:
            success = email_service.send_sms(
                current_user.phone,
                "🧪 Test SMS von SensationGifts: Alles funktioniert! 🎉"
            )
        else:
            return jsonify({'error': 'No phone number available'}), 400
    
    else:
        return jsonify({'error': 'Invalid notification type'}), 400
    
    return jsonify({'success': success})

@notifications_bp.route('/api/notifications/preferences', methods=['GET', 'PUT'])
@login_required
def notification_preferences():
    """Get or update user notification preferences"""
    
    from app.models.user import User
    from app.models import db
    
    if request.method == 'GET':
        preferences = current_user.notification_preferences or {
            'email_marketing': True,
            'email_transactional': True,
            'push_notifications': True,
            'sms_notifications': False,
            'order_updates': True,
            'recommendation_alerts': True,
            'birthday_reminders': True
        }
        
        return jsonify({'preferences': preferences})
    
    elif request.method == 'PUT':
        data = request.get_json()
        preferences = data.get('preferences', {})
        
        current_user.notification_preferences = preferences
        db.session.commit()
        
        return jsonify({'success': True, 'preferences': preferences})