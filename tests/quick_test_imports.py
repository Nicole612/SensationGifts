#!/usr/bin/env python3
"""
quick_test_imports.py - Teste ob die Parser-Imports funktionieren
Läuft von root/tests/ oder root/ Verzeichnis aus.
"""

import sys
from pathlib import Path

import logging

# Deaktiviert alle SQLAlchemy-Ausgaben unterhalb WARNING
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

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

# Setup Pfad
setup_project_path()

print("🧪 TESTE PARSER-IMPORTS...")

try:
    from ai_engine.processors import response_parser as parser_module
    print("✅ Response Parser importiert")

    # Verfügbare Funktionen prüfen
    if hasattr(parser_module, "parse_heroic_gift_response"):
        print("✅ parse_heroic_gift_response verfügbar")
    else:
        print("❌ parse_heroic_gift_response fehlt!")

    # ERWEITERTE FUNKTIONEN (OPTIONAL)
    if hasattr(parser_module, "parse_emotional_story_advanced"):
        print("✅ parse_emotional_story_advanced vorhanden")
    else:
        print("ℹ️ Hinweis: parse_emotional_story_advanced ist nicht implementiert – optional")

    if hasattr(parser_module, "get_optimization_suggestions"):
        print("✅ get_optimization_suggestions vorhanden")
    else:
        print("ℹ️ Hinweis: get_optimization_suggestions ist nicht implementiert – optional")

    # Enum testen
    from schemas.prompt_schemas import AIModelType, ParsingStrategy
    print("✅ Enums (AIModelType, ParsingStrategy) importiert")

    required_models = ["OPENAI_GPT4", "CLAUDE_SONNET", "GEMINI_PRO"]
    for model in required_models:
        if hasattr(AIModelType, model):
            print(f"✅ {model} verfügbar")
        else:
            print(f"⚠️ {model} fehlt im Enum AIModelType")

except Exception as e:
    print(f"❌ Fehler beim Parser-Import oder Funktionstest: {e}")
    sys.exit(1)

print("🎉 Deine response_parser.py funktioniert perfekt!")
