#!/usr/bin/env python3
"""
AI Gift Shop - Application Starter

Diese Datei startet die Flask-Anwendung und lädt alle notwendigen
Konfigurationen und Extensions.

Verwendung:
    python run.py                    # Startet Development Server
    flask run                        # Alternative
    python -m flask run              # Weitere Alternative
"""

import os
import sys
from pathlib import Path


# Füge das Projekt-Root-Verzeichnis zum Python-Path hinzu
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app import create_app
from app.extensions import db
from config.settings import get_settings


def create_data_directories():
    """
    Erstellt notwendige Verzeichnisse falls sie nicht existieren
    """
    directories = [
        'data',           # Für SQLite Database
        'logs',           # Für Logfiles  
        'data/exports',   # Für Daten-Exports
        'static/uploads'  # Für User-Uploads (falls später benötigt)
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Verzeichnis erstellt/geprüft: {directory}")


def check_environment():
    """
    Prüft ob alle notwendigen Umgebungsvariablen gesetzt sind
    """
    try:
        from config.settings import get_settings
        settings = get_settings()
        
        # Verwende Pydantic Settings statt os.getenv()
        is_valid, missing = settings.validate_required_settings()
        
        if not is_valid:
            print(f"❌ FEHLER: Erforderliche Konfiguration fehlt: {missing}")
            print("💡 Bitte .env Datei prüfen!")
            return False
        
        # Prüfe optionale AI Keys
        if not settings.has_ai_capabilities:
            print(f"⚠️ WARNUNG: Keine AI API-Keys gefunden")
            print("💡 AI-Features sind deaktiviert")
        
        print("✅ Umgebungsvariablen-Check erfolgreich!")
        return True
        
    except Exception as e:
        print(f"❌ Fehler beim Laden der Konfiguration: {e}")
        return False


def init_database_with_sample_data():
    """
    Initialisiert die Datenbank und fügt Sample-Daten hinzu
    """
    try:
        # Erstelle alle Tabellen
        db.create_all()
        print("✅ Datenbank-Tabellen erstellt!")
        
        # Prüfe ob bereits Daten vorhanden sind
        from app.models import User, GiftCategory
        
        existing_users = User.query.count()
        existing_categories = GiftCategory.query.count()
        
        if existing_users == 0 and existing_categories == 0:
            print("📊 Keine Daten gefunden - Sample-Daten werden erstellt...")
            create_sample_data()
        else:
            print(f"📊 Datenbank bereits gefüllt: {existing_users} Users, {existing_categories} Kategorien")
    
    except Exception as e:
        print(f"❌ Datenbank-Initialisierung fehlgeschlagen: {e}")
        return False
    
    return True


def create_sample_data():
    """
    Erstellt minimale Sample-Daten zum Testen
    """
    try:
        from app.models import User, GiftCategory, GiftTag
        
        # Erstelle Test-User
        test_user = User(
            email="test@geschenkshop.de",
            password="testpass123",
            first_name="Max",
            last_name="Mustermann"
        )
        test_user.save()
        print("👤 Test-User erstellt: test@geschenkshop.de")
        
        # Erstelle erste Kategorie
        category = GiftCategory(
            name="Für Kreative Köpfe",
            slug="kreative-koepfe", 
            description="Geschenke für Menschen die gerne kreativ sind und sich ausdrücken",
            icon="🎨",
            color="#E74C3C",
            sort_order=1
        )
        category.set_target_traits(["high_openness", "creative_type"])
        category.set_occasions(["geburtstag", "weihnachten"])
        category.save()
        print("🎨 Kategorie erstellt: Für Kreative Köpfe")
        
        # Erstelle erste Tags
        tags_data = [
            {"name": "kreativ", "display": "Kreativ", "type": "personality", "color": "#E74C3C"},
            {"name": "praktisch", "display": "Praktisch", "type": "feature", "color": "#3498DB"},
            {"name": "personalisierbar", "display": "Personalisierbar", "type": "feature", "color": "#9B59B6"}
        ]
        
        for tag_data in tags_data:
            tag = GiftTag(
                name=tag_data["name"],
                display_name=tag_data["display"],
                tag_type=tag_data["type"],
                color=tag_data["color"]
            )
            tag.save()
        
        print(f"🏷️ {len(tags_data)} Tags erstellt")
        print("✅ Sample-Daten erfolgreich erstellt!")
        
    except Exception as e:
        print(f"❌ Fehler beim Erstellen der Sample-Daten: {e}")


def show_startup_info():
    """
    Zeigt nützliche Informationen beim Start
    """
    from config.settings import get_settings
    settings = get_settings()
    
    print("\n" + "="*60)
    print("🎁 AI GIFT SHOP - Erfolgreich gestartet!")
    print("="*60)
    print(f"🌍 Environment: {settings.environment}")
    print(f"🔍 Debug Mode: {settings.debug}")
    print(f"💾 Database: {settings.database_url}")
    print(f"🤖 AI Features: {'✅ Aktiviert' if settings.has_ai_capabilities else '❌ Deaktiviert (API Keys fehlen)'}")
    if settings.available_ai_models:
        print(f"🔗 AI Models: {', '.join(settings.available_ai_models)}")
    print("\n📋 Verfügbare Endpoints:")
    print("   http://localhost:5000/          - Homepage (später)")
    print("   http://localhost:5000/api/      - API Documentation (später)")
    print("\n🧪 Test-Commands:")
    print("\n🔑 Test-Login:")
    print("   Email: test@geschenkshop.de")
    print("   Password: testpass123")
    print("="*60 + "\n")


if __name__ == '__main__':
    print("🚀 Starte AI SensattionGifts mit moderner Architecture...")
    
    # 1. Erstelle notwendige Verzeichnisse
    create_data_directories()
    
    # 2. Prüfe Umgebungsvariablen
    if not check_environment():
        sys.exit(1)
    
    # 3. Erstelle Flask App
    app = create_app()
    
    # 4. Initialisiere Datenbank
    with app.app_context():
        if not init_database_with_sample_data():
            sys.exit(1)
    
    # 5. Zeige Startup-Info
    show_startup_info()
    
    # 6. Starte Development Server
    try:
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True,
            use_reloader=True
        )
    except KeyboardInterrupt:
        print("\n👋 AI Gift Shop wurde gestoppt. Bis bald!")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        sys.exit(1)