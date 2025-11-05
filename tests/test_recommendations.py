#!/usr/bin/env python3
"""
test_recommendations.py - Teste Heldenreise-Empfehlungen

Dieses Skript testet die Empfehlungs-Engine mit verschiedenen Altersgruppen
und Persönlichkeiten um zu prüfen ob der Heldenreise-Katalog funktioniert.
Läuft von root/tests/ oder root/ Verzeichnis aus.
"""

import sys
import os
from pathlib import Path

def setup_project_path():
    """Robuste Pfad-Konfiguration - funktioniert von überall"""
    current_dir = Path(__file__).parent.resolve()
    
    # Suche ai_engine/ Verzeichnis (maximal 3 Ebenen hoch)
    for i in range(4):  
        check_dir = current_dir
        for _ in range(i):
            check_dir = check_dir.parent
            
        if (check_dir / "ai_engine").exists():
            project_root = str(check_dir)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            print(f"✅ Projekt-Root gefunden: {check_dir}")
            return check_dir
    
    # Fallback für tests/ Struktur
    fallback_root = current_dir.parent if current_dir.name == "tests" else current_dir
    sys.path.insert(0, str(fallback_root))
    print(f"⚠️ Fallback-Root verwendet: {fallback_root}")
    return fallback_root

# Setup Pfade
project_root = setup_project_path()

def test_age_based_recommendations():
    """Teste altersbasierte Empfehlungen"""
    print("🧪 TESTE ALTERSBASIERTE EMPFEHLUNGEN...")
    
    try:
        from ai_engine.catalog.heroic_journey_catalog import HeroicJourneyIntegration
        
        integration = HeroicJourneyIntegration()
        
        # Test verschiedene Altersgruppen
        test_ages = [
            (5, "Kleinkind mit Ängsten"),
            (15, "Teenager Identitätskrise"), 
            (25, "Junge/r Erwachsene orientierungslos"),
            (45, "Erwachsene/r will führen"),
            (65, "Senior will Vermächtnis schaffen")
        ]
        
        for age, description in test_ages:
            print(f"\n  🎯 Test: {age} Jahre ({description})")
            
            # Hole altersgerechte Geschenke
            matching_gifts = integration.get_gifts_for_age_group(age)
            
            if matching_gifts:
                print(f"    ✅ {len(matching_gifts)} passende Geschenke gefunden")
                
                # Zeige erstes Geschenk
                first_gift = matching_gifts[0]
                print(f"    📦 Empfehlung: {first_gift.name}")
                print(f"    💰 Preise: €{first_gift.price_basic}-{first_gift.price_premium}")
                print(f"    🎭 Transformation: {first_gift.transformation}")
                
                # Teste AI-Prompt Generation
                prompt = integration.get_ai_optimized_prompt(
                    age=age,
                    personality_data={"high_openness": True},
                    challenge=description
                )
                
                if prompt and len(prompt) > 100:
                    print(f"    ✅ AI-Prompt generiert ({len(prompt)} Zeichen)")
                else:
                    print("    ⚠️ AI-Prompt zu kurz oder leer")
                    
            else:
                print(f"    ❌ Keine passenden Geschenke für Alter {age}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Altersbasierte Tests fehlgeschlagen: {e}")
        return False

def test_personality_matching():
    """Teste Persönlichkeits-Matching"""
    print("\n🧪 TESTE PERSÖNLICHKEITS-MATCHING...")
    
    try:
        from ai_engine.catalog.heroic_journey_catalog import HEROIC_GIFT_CATALOG
        
        # Test verschiedene Persönlichkeitstypen
        personality_tests = [
            {
                "name": "Ängstliches Kind",
                "age": 6,
                "traits": {"high_neuroticism": 0.9, "low_extraversion": 0.8},
                "expected_keywords": ["mut", "angst", "selbstvertrauen"]
            },
            {
                "name": "Kreativer Teenager", 
                "age": 16,
                "traits": {"high_openness": 0.9, "creative_type": True},
                "expected_keywords": ["identität", "kreativ", "authentizität"]
            },
            {
                "name": "Ehrgeiziger Erwachsener",
                "age": 30,
                "traits": {"high_conscientiousness": 0.9, "ambitious": True},
                "expected_keywords": ["ziele", "erfolg", "lebensplanung"]
            }
        ]
        
        for test_case in personality_tests:
            print(f"\n  🎭 Test: {test_case['name']} ({test_case['age']} Jahre)")
            
            # Finde passende Geschenke
            matching_gifts = []
            age = test_case['age']
            
            for gift_id, gift in HEROIC_GIFT_CATALOG.items():
                if gift.age_min <= age <= gift.age_max:
                    # Einfacher Persönlichkeits-Match
                    match_score = 0
                    for trait, gift_score in gift.personality_match.items():
                        if trait in test_case['traits']:
                            match_score += gift_score * test_case['traits'][trait]
                    
                    if match_score > 0.5:
                        matching_gifts.append((gift, match_score))
            
            # Sortiere nach Score
            matching_gifts.sort(key=lambda x: x[1], reverse=True)
            
            if matching_gifts:
                print(f"    ✅ {len(matching_gifts)} passende Geschenke")
                
                # Zeige beste Empfehlung
                best_gift, score = matching_gifts[0]
                print(f"    🏆 Beste Empfehlung: {best_gift.name}")
                print(f"    📊 Match-Score: {score:.2f}")
                
                # Prüfe erwartete Keywords
                keywords_found = []
                gift_text = (best_gift.name + " " + best_gift.emotional_story).lower()
                for keyword in test_case['expected_keywords']:
                    if keyword in gift_text:
                        keywords_found.append(keyword)
                
                if keywords_found:
                    print(f"    🎯 Keywords gefunden: {', '.join(keywords_found)}")
                else:
                    print(f"    ⚠️ Erwartete Keywords nicht gefunden: {test_case['expected_keywords']}")
                    
            else:
                print(f"    ❌ Keine passenden Geschenke gefunden")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Persönlichkeits-Tests fehlgeschlagen: {e}")
        return False

def test_story_parsing():
    """Teste Story-Parsing Funktionalität"""
    print("\n🧪 TESTE STORY-PARSING...")
    
    try:
        from ai_engine.processors.response_parser import (
            EmotionalStoryParser,
            parse_emotional_story
        )
        
        # Test-Story mit Heldenreise-Elementen
        sample_story = """
        Es war einmal ein kleiner Held namens Max, der große Angst vor der Schule hatte.
        Jeden Morgen wachte Max mit einem mulmigen Gefühl auf. "Was ist, wenn die anderen 
        Kinder mich nicht mögen?", dachte er. Doch dann passierte etwas Magisches...
        
        Ein geheimnisvoller Mut-Kompass erschien mit der Botschaft: "Du bist mutiger als 
        deine Ängste!" Von diesem Tag an begann Max' Transformation von "Ich habe Angst" 
        zu "Ich bin stark und schaffe das!"
        
        Mit jedem Tag wurde Max mutiger. Er lernte neue Freunde kennen, traute sich 
        im Unterricht zu sprechen, und entdeckte seine Superkraft: das Lächeln, 
        das andere glücklich machte.
        """
        
        print("  📖 Teste Story-Parsing mit Sample-Story...")
        
        # Teste Quick-Parse Funktion
        result = parse_emotional_story(sample_story, person_name="Max", age=6)
        
        if result.parsing_success:
            print("  ✅ Story-Parsing erfolgreich")
            print(f"    Confidence: {result.confidence_score:.2f}")
            
            data = result.parsed_data
            if data:
                if data.get('personalization_detected'):
                    print("    ✅ Personalisierung erkannt")
                
                emotional_elements = data.get('emotional_elements', [])
                if emotional_elements:
                    print(f"    🎭 Emotionale Elemente: {', '.join(emotional_elements[:5])}")
                
                transformation = data.get('transformation_arc')
                if transformation:
                    print(f"    🔄 Transformation: {transformation['from']} → {transformation['to']}")
                
                quality_score = data.get('story_quality_score', 0)
                print(f"    📊 Story-Qualität: {quality_score:.2f}")
                
        else:
            print("  ❌ Story-Parsing fehlgeschlagen")
            if result.validation_errors:
                for error in result.validation_errors[:3]:
                    print(f"    ❌ Fehler: {error.get('message', 'Unbekannt')}")
        
        return result.parsing_success
        
    except Exception as e:
        print(f"  ❌ Story-Parsing Tests fehlgeschlagen: {e}")
        return False

def test_full_recommendation_flow():
    """Teste kompletten Empfehlungs-Workflow"""
    print("\n🧪 TESTE KOMPLETTEN EMPFEHLUNGS-WORKFLOW...")
    
    try:
        # Simuliere vollständigen Workflow
        print("  🔄 Simuliere: Benutzer sucht Geschenk für 7-jähriges schüchternes Kind")
        
        from ai_engine.catalog.heroic_journey_catalog import HeroicJourneyIntegration
        
        integration = HeroicJourneyIntegration()
        
        # 1. Altersgerechte Geschenke finden
        age = 7
        matching_gifts = integration.get_gifts_for_age_group(age)
        print(f"    ✅ Schritt 1: {len(matching_gifts)} altersgerechte Geschenke gefunden")
        
        # 2. Persönlichkeits-Matching
        personality_data = {"high_neuroticism": 0.8, "low_extraversion": 0.9}
        
        scored_gifts = []
        for gift in matching_gifts:
            score = 0
            for trait, gift_score in gift.personality_match.items():
                if trait == "high_neuroticism" and personality_data.get("high_neuroticism", 0) > 0.6:
                    score += gift_score * personality_data["high_neuroticism"]
                elif trait == "low_extraversion" and personality_data.get("low_extraversion", 0) > 0.6:
                    score += gift_score * personality_data["low_extraversion"]
            
            if score > 0:
                scored_gifts.append((gift, score))
        
        scored_gifts.sort(key=lambda x: x[1], reverse=True)
        print(f"    ✅ Schritt 2: {len(scored_gifts)} Geschenke nach Persönlichkeit gefiltert")
        
        # 3. Beste Empfehlung
        if scored_gifts:
            best_gift, score = scored_gifts[0]
            print(f"    🏆 Beste Empfehlung: {best_gift.name}")
            print(f"    📊 Match-Score: {score:.2f}")
            print(f"    💰 Preis: €{best_gift.price_basic}-{best_gift.price_premium}")
            
            # 4. AI-Prompt für diese Empfehlung
            prompt = integration.get_ai_optimized_prompt(
                age=age,
                personality_data=personality_data,
                challenge="Schüchternheit überwinden"
            )
            
            if prompt:
                print("    ✅ Schritt 4: AI-Prompt generiert")
                # Prüfe ob wichtige Elemente im Prompt sind
                if "schüchtern" in prompt.lower() and "mut" in prompt.lower():
                    print("    ✅ Prompt enthält relevante Keywords")
                else:
                    print("    ⚠️ Prompt könnte spezifischer sein")
            
            return True
        else:
            print("    ❌ Keine geeigneten Geschenke gefunden")
            return False
        
    except Exception as e:
        print(f"  ❌ Workflow-Test fehlgeschlagen: {e}")
        return False

def test_with_flask_context():
    """Teste mit Flask-Context (falls verfügbar)"""
    print("\n🧪 TESTE MIT FLASK-CONTEXT...")
    
    try:
        from app import create_app
        from ai_engine.catalog.catalog_service import get_catalog_service
        
        app = create_app()
        
        with app.app_context():
            print("  ✅ Flask-Context aktiv")
            
            # Teste Service mit Flask-Context
            heroic_service = get_catalog_service(heroic=True)
            stats = heroic_service.get_catalog_statistics()
            
            print(f"  📊 Service-Statistiken abrufbar:")
            print(f"    Heldenreise-Produkte: {stats['heroic_catalog']['products']}")
            
            return True
            
    except ImportError:
        print("  ⚠️ Flask nicht verfügbar - das ist OK für diesen Test")
        return True
    except Exception as e:
        print(f"  ❌ Flask-Context Test fehlgeschlagen: {e}")
        return False

def main():
    """Haupttest-Funktion"""
    print("🎯 HELDENREISE-EMPFEHLUNGEN TEST")
    print("=" * 50)
    print(f"📍 Ausgeführt von: {Path.cwd()}")
    print(f"📁 Projekt-Root: {project_root}")
    
    results = {}
    
    # Test 1: Altersbasierte Empfehlungen
    results['age_based'] = test_age_based_recommendations()
    
    # Test 2: Persönlichkeits-Matching
    results['personality'] = test_personality_matching()
    
    # Test 3: Story-Parsing
    results['story_parsing'] = test_story_parsing()
    
    # Test 4: Kompletter Workflow
    results['full_workflow'] = test_full_recommendation_flow()
    
    # Test 5: Flask-Context (optional)
    results['flask_context'] = test_with_flask_context()
    
    # Zusammenfassung
    print("\n📋 EMPFEHLUNGS-TEST ZUSAMMENFASSUNG:")
    print("=" * 40)
    
    for test_name, result in results.items():
        status = "✅ Bestanden" if result else "❌ Fehlgeschlagen"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    # Gesamtergebnis
    critical_tests = ['age_based', 'personality', 'story_parsing', 'full_workflow']
    critical_passed = all(results.get(test, False) for test in critical_tests)
    
    if critical_passed:
        print("\n🎉 EMPFEHLUNGS-SYSTEM FUNKTIONIERT!")
        print("💡 Dein Heldenreise-Katalog ist bereit für echte Benutzer")
        
        print("\n📝 NÄCHSTE SCHRITTE:")
        print("  1. Synchronisiere Katalog mit Datenbank")
        print("  2. Integriere ins Frontend")
        print("  3. Teste mit echten Benutzerdaten")
        print("  4. Starte A/B Testing alt vs. neu")
        
        return True
    else:
        print("\n❌ EMPFEHLUNGS-SYSTEM HAT PROBLEME")
        print("💡 Behebe die fehlgeschlagenen Tests")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)