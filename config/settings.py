"""
Application Settings - Pydantic Settings mit .env Support

Diese Datei lädt alle Konfigurationen aus .env Dateien und
stellt sie typisiert für die Anwendung zur Verfügung.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """
    Haupt-Konfiguration für AI Gift Shop
    
    Lädt automatisch aus .env Datei und Umgebungsvariablen.
    Validation durch Pydantic.
    """
    
    # === FLASK KONFIGURATION ===
    environment: str = Field(default="development", env="FLASK_ENV")
    debug: bool = Field(default=True, env="FLASK_DEBUG")
    secret_key: str = Field(..., env="SECRET_KEY")  # Required!
    
    # === DATENBANK ===
    database_url: str = Field(default="sqlite:///data/database.db", env="DATABASE_URL")
    
    # === AI API KEYS ===
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # === AI KONFIGURATION ===
    default_ai_model: str = Field(default="openai_gpt4", env="DEFAULT_AI_MODEL")
    max_ai_cost_per_session: float = Field(default=2.00, env="MAX_AI_COST_PER_SESSION")
    ai_timeout_seconds: int = Field(default=30, env="AI_TIMEOUT_SECONDS")
    
    # === RATE LIMITING ===
    recommendations_per_hour: int = Field(default=50, env="RECOMMENDATIONS_PER_HOUR")
    api_calls_per_minute: int = Field(default=20, env="API_CALLS_PER_MINUTE")
    
    # === LOGGING ===
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")

    
    
    # === FEATURES ===
    enable_ai_recommendations: bool = Field(default=True, env="ENABLE_AI_RECOMMENDATIONS")
    enable_multi_model_testing: bool = Field(default=True, env="ENABLE_MULTI_MODEL_TESTING")
    enable_cost_tracking: bool = Field(default=True, env="ENABLE_COST_TRACKING")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"  # Ignoriert unbekannte Felder in .env
    }
    
    # === COMPUTED PROPERTIES ===
    
    @property
    def is_development(self) -> bool:
        """Läuft die App im Development Mode?"""
        return self.environment.lower() in ["development", "dev"]
    
    @property
    def is_production(self) -> bool:
        """Läuft die App im Production Mode?"""
        return self.environment.lower() in ["production", "prod"]
    
    @property
    def available_ai_models(self) -> List[str]:
        """Liste der verfügbaren AI-Models basierend auf API Keys"""
        models = []
        
        if self.openai_api_key:
            models.extend(["openai_gpt4", "openai_gpt35"])
        if self.groq_api_key:
            models.extend(["groq_mixtral", "groq_llama"])
        if self.gemini_api_key:
            models.extend(["gemini_pro", "gemini_flash"])
        if self.anthropic_api_key:
            models.extend(["anthropic_claude"])
        
        return models
    
    @property
    def has_ai_capabilities(self) -> bool:
        """Sind AI-Features verfügbar?"""
        return bool(self.available_ai_models) and self.enable_ai_recommendations
    
    @property
    def database_type(self) -> str:
        """Welcher Database-Typ wird verwendet?"""
        if self.database_url.startswith("sqlite"):
            return "sqlite"
        elif self.database_url.startswith("postgresql"):
            return "postgresql"
        elif self.database_url.startswith("mysql"):
            return "mysql"
        else:
            return "unknown"
    
    def get_ai_config(self) -> dict:
        """AI-Konfiguration als Dictionary"""
        return {
            "default_model": self.default_ai_model,
            "available_models": self.available_ai_models,
            "max_cost_per_session": self.max_ai_cost_per_session,
            "timeout_seconds": self.ai_timeout_seconds,
            "enable_multi_model": self.enable_multi_model_testing,
            "enable_cost_tracking": self.enable_cost_tracking
        }
    
    def validate_required_settings(self) -> tuple[bool, List[str]]:
        """
        Validiert ob alle erforderlichen Settings vorhanden sind
        
        Returns:
            tuple: (is_valid, list_of_missing_settings)
        """
        missing = []
        
        # Secret Key ist immer erforderlich
        if not self.secret_key or len(self.secret_key) < 16:
            missing.append("SECRET_KEY (mindestens 16 Zeichen)")
        
        # Für AI-Features brauchen wir mindestens einen API Key
        if self.enable_ai_recommendations and not self.available_ai_models:
            missing.append("Mindestens ein AI API Key (OPENAI_API_KEY, GROQ_API_KEY, etc.)")
        
        return len(missing) == 0, missing
    
    def to_dict(self) -> dict:
        """Settings als Dictionary (ohne sensitive Daten)"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "database_type": self.database_type,
            "has_ai_capabilities": self.has_ai_capabilities,
            "available_ai_models": self.available_ai_models,
            "default_ai_model": self.default_ai_model,
            "log_level": self.log_level,
            "enable_ai_recommendations": self.enable_ai_recommendations,
            "enable_multi_model_testing": self.enable_multi_model_testing,
            "enable_cost_tracking": self.enable_cost_tracking
        }


# === SINGLETON PATTERN ===

@lru_cache()
def get_settings() -> Settings:
    """
    Singleton Settings Instance
    
    Lädt die Settings nur einmal und cached sie.
    Verwendung:
        from config.settings import get_settings
        settings = get_settings()
    """
    return Settings()


# === HELPER FUNCTIONS ===

def print_settings_info():
    """Gibt Settings-Info auf der Konsole aus"""
    settings = get_settings()
    
    print("\n🔧 SETTINGS OVERVIEW:")
    print("=" * 50)
    print(f"Environment: {settings.environment}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Database: {settings.database_type}")
    print(f"AI Capabilities: {settings.has_ai_capabilities}")
    if settings.available_ai_models:
        print(f"Available AI Models: {', '.join(settings.available_ai_models)}")
    print("=" * 50)


def validate_environment() -> bool:
    """
    Validiert die komplette Umgebung
    
    Returns:
        bool: True wenn alles OK, False wenn kritische Probleme
    """
    try:
        settings = get_settings()
        is_valid, missing = settings.validate_required_settings()
        
        if not is_valid:
            print("❌ KRITISCHE KONFIGURATIONSFEHLER:")
            for item in missing:
                print(f"   - {item}")
            print("\n💡 Bitte .env Datei prüfen!")
            return False
        
        print("✅ Konfiguration ist gültig!")
        return True
        
    except Exception as e:
        print(f"❌ Fehler beim Laden der Konfiguration: {e}")
        return False


# === DEVELOPMENT HELPERS ===

if __name__ == "__main__":
    """Direkte Ausführung für Testing"""
    print("🧪 Testing Settings...")
    
    # Test Settings laden
    try:
        settings = get_settings()
        print("✅ Settings erfolgreich geladen!")
        
        # Zeige Settings
        print_settings_info()
        
        # Validiere
        validate_environment()
        
        # Zeige AI Config
        print(f"\n🤖 AI Configuration:")
        ai_config = settings.get_ai_config()
        for key, value in ai_config.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Fehler: {e}")
        print("💡 Stelle sicher dass .env Datei existiert und SECRET_KEY gesetzt ist!")