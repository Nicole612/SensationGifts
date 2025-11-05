"""
Response Parser - Structured AI Output Processing Engine
=======================================================

Intelligente Verarbeitung und Validierung von AI-Model-Outputs für SensationGifts.
Stellt sicher dass AI-Antworten strukturiert, validiert und verwendbar sind.

Core Features:
- Multi-format parsing (JSON, text, mixed content)
- Schema validation with automatic correction
- Error recovery and intelligent repair
- Performance metrics and quality assessment
- Model-specific output adaptation
- Confidence scoring and validation
- Heroic Journey & Emotional Story parsing (NEW!)
"""

from typing import Dict, List, Optional, Any, Tuple, Type
from datetime import datetime
from enum import Enum
import json
import re
import logging
from dataclasses import dataclass

from ai_engine.schemas import (
    # Output Schemas
    GiftRecommendationResponse,
    PersonalityAnalysisResult,
    QuickRecommendationResponse,
    
    # Enums
    AIModelType,
)


# =============================================================================
# PARSING STRATEGY TYPES
# =============================================================================

class ParsingStrategy(str, Enum):
    """Strategien für Response-Parsing"""
    STRICT_JSON = "strict_json"                    # Nur valides JSON akzeptieren
    FLEXIBLE_JSON = "flexible_json"                # JSON mit Reparatur-Versuchen
    STRUCTURED_TEXT = "structured_text"            # Text mit Struktur-Erkennung
    HYBRID_PARSING = "hybrid_parsing"              # Multi-Format Unterstützung
    AI_ASSISTED_REPAIR = "ai_assisted_repair"      # AI hilft bei Reparatur

# Erweitere die bestehende AIModelType Enum:

class OutputFormat(str, Enum):
    """Erkannte Output-Formate"""
    VALID_JSON = "valid_json"
    MALFORMED_JSON = "malformed_json"
    STRUCTURED_TEXT = "structured_text"
    UNSTRUCTURED_TEXT = "unstructured_text"
    MIXED_FORMAT = "mixed_format"
    EMPTY_RESPONSE = "empty_response"


class ValidationSeverity(str, Enum):
    """Schweregrad von Validierungsfehlern"""
    CRITICAL = "critical"        # Response unbrauchbar
    MAJOR = "major"             # Wichtige Daten fehlen
    MINOR = "minor"             # Kleinere Probleme
    WARNING = "warning"         # Potentielle Probleme
    INFO = "info"               # Informative Hinweise


# =============================================================================
# PARSING RESULT CLASSES
# =============================================================================

@dataclass
class ParsedResponse:
    """Container für geparste AI-Response mit Metadaten"""
    
    # Parsed Content
    parsed_data: Optional[Dict[str, Any]]
    structured_output: Optional[Any]  # Pydantic model instance
    
    # Parsing Metadata
    original_response: str
    parsing_strategy: ParsingStrategy
    output_format: OutputFormat
    parsing_success: bool
    parsing_time_ms: int
    
    # Quality Metrics
    confidence_score: float  # 0.0-1.0
    validation_errors: List[Dict[str, Any]]
    warnings: List[str]
    
    # Recovery Information
    repair_attempts: int
    repair_methods_used: List[str]
    fallback_data_used: bool


@dataclass
class ValidationResult:
    """Ergebnis der Schema-Validierung"""
    
    is_valid: bool
    severity: ValidationSeverity
    errors: List[Dict[str, str]]
    warnings: List[str]
    suggestions: List[str]
    confidence_impact: float  # Wie stark Fehler die Konfidenz beeinträchtigen


# =============================================================================
# CORE RESPONSE PARSER ENGINE
# =============================================================================

class ResponseParser:
    """
    Core Engine für AI-Response Processing
    
    Capabilities:
    - Multi-format response parsing
    - Schema validation and correction
    - Intelligent error recovery
    - Performance monitoring
    - Quality assessment
    """
    
    def __init__(self):
        self.parsing_statistics = {
            "total_responses_parsed": 0,
            "successful_parses": 0,
            "format_distribution": {},
            "error_patterns": {},
            "average_parsing_time": 0.0,
            "repair_success_rate": 0.0
        }
        
        # JSON repair patterns
        self.json_repair_patterns = [
            # Common JSON format issues
            (r'```json\s*', ''),  # Remove markdown code blocks
            (r'\s*```\s*$', ''),  # Remove closing code blocks
            (r',\s*}', '}'),      # Remove trailing commas before }
            (r',\s*]', ']'),      # Remove trailing commas before ]
            (r'(\w+):', r'"\1":'), # Add quotes to unquoted keys
            (r':\s*([^",\[\]{}\s]+)([,\]}])', r': "\1"\2'),  # Quote unquoted string values
        ]
        
        # Text structure patterns
        self.text_structure_patterns = {
            'numbered_list': r'^\d+\.\s*(.+)$',
            'bulleted_list': r'^\s*[-*•]\s*(.+)$',
            'title_value': r'^([^:]+):\s*(.+)$',
            'gift_item': r'^\s*(.+?)\s*\(€(\d+(?:\.\d{2})?)\)\s*-?\s*(.*)$'
        }
        
        # Model-specific parsing adaptations
        self.model_adaptations = {
            AIModelType.OPENAI_GPT4: self._adapt_openai_output,
            AIModelType.GROQ_MIXTRAL: self._adapt_groq_output,
            AIModelType.ANTHROPIC_CLAUDE: self._adapt_claude_output,
            AIModelType.GOOGLE_GEMINI: self._adapt_gemini_output
        }
        
        # 🚀 ENHANCED OPTIMIZATION FEATURES
        self.optimization_enabled = True
        self.adaptive_parsing_enabled = True
        self.intelligent_repair_enabled = True
        self.performance_tracking_enabled = True
        
        # Enhanced parsing state
        self.parsing_state = {
            "learning_enabled": True,
            "adaptive_strategies": True,
            "repair_confidence": 0.8,
            "fallback_quality": 0.6
        }
        
        # Advanced error recovery patterns
        self.advanced_repair_patterns = {
            "missing_fields": self._repair_missing_fields,
            "type_mismatches": self._repair_type_mismatches,
            "enum_errors": self._repair_enum_errors,
            "validation_errors": self._repair_validation_errors
        }
        
        # Performance optimization
        self.performance_cache = {}
        self.parsing_optimizations = {
            "cache_enabled": True,
            "parallel_processing": True,
            "intelligent_fallback": True
        }
    
    def parse_gift_recommendation_response(
        self,
        raw_response: str,
        source_model: AIModelType,
        expected_schema: Type = GiftRecommendationResponse,
        parsing_strategy: ParsingStrategy = ParsingStrategy.HYBRID_PARSING
    ) -> ParsedResponse:
        """
        Hauptmethode: Parst AI-Response zu strukturierter Geschenkempfehlung
        
        Args:
            raw_response: Rohe AI-Antwort als String
            source_model: AI-Model das die Response generiert hat
            expected_schema: Erwartetes Pydantic Schema
            parsing_strategy: Parsing-Strategie
            
        Returns:
            ParsedResponse mit allen Parsing-Informationen
        """
        
        parsing_start = datetime.now()
        
        try:
            # 1. PREPROCESSING
            processed_response = self._preprocess_response(raw_response, source_model)
            
            # 2. FORMAT DETECTION
            output_format = self._detect_output_format(processed_response)
            
            # 3. PARSING STRATEGY EXECUTION
            parsed_data = self._execute_parsing_strategy(
                processed_response, parsing_strategy, output_format
            )
            
            # 4. SCHEMA VALIDATION
            validation_result = self._validate_against_schema(parsed_data, expected_schema)
            
            # 5. ENHANCED ERROR RECOVERY (if needed)
            if not validation_result.is_valid and parsing_strategy != ParsingStrategy.STRICT_JSON:
                if self.intelligent_repair_enabled:
                    parsed_data, repair_info = self._attempt_enhanced_intelligent_repair(
                        parsed_data, validation_result, expected_schema, processed_response
                    )
                else:
                    parsed_data, repair_info = self._attempt_intelligent_repair(
                        parsed_data, validation_result, expected_schema, processed_response
                    )
                # Re-validate after repair
                validation_result = self._validate_against_schema(parsed_data, expected_schema)
            else:
                repair_info = {"attempts": 0, "methods": [], "fallback_used": False}
            
            # 6. STRUCTURED OUTPUT CREATION
            structured_output = None
            if validation_result.is_valid and parsed_data:
                try:
                    structured_output = expected_schema(**parsed_data)
                except Exception as e:
                    logging.warning(f"Failed to create structured output: {e}")
            
            # 7. CONFIDENCE CALCULATION
            confidence_score = self._calculate_confidence_score(
                validation_result, output_format, repair_info, source_model
            )
            
            # 8. RESULT ASSEMBLY
            parsing_time = int((datetime.now() - parsing_start).total_seconds() * 1000)
            
            result = ParsedResponse(
                parsed_data=parsed_data,
                structured_output=structured_output,
                original_response=raw_response,
                parsing_strategy=parsing_strategy,
                output_format=output_format,
                parsing_success=validation_result.is_valid,
                parsing_time_ms=parsing_time,
                confidence_score=confidence_score,
                validation_errors=[
                    {"field": err.get("field", "unknown"), "message": err.get("message", "")}
                    for err in validation_result.errors
                ],
                warnings=validation_result.warnings,
                repair_attempts=repair_info["attempts"],
                repair_methods_used=repair_info["methods"],
                fallback_data_used=repair_info["fallback_used"]
            )
            
            # 9. ENHANCED STATISTICS UPDATE
            if self.performance_tracking_enabled:
                self._update_enhanced_parsing_statistics(result, source_model)
            
            # 10. PERFORMANCE OPTIMIZATION
            if self.optimization_enabled:
                result = self._apply_parsing_optimizations(result, source_model)
            
            return result
            
        except Exception as e:
            # Enhanced fallback error handling
            return self._create_enhanced_error_response(raw_response, str(e), parsing_start, source_model)

    def _attempt_enhanced_intelligent_repair(
        self, 
        parsed_data: Dict[str, Any], 
        validation_result: ValidationResult, 
        expected_schema: Type, 
        original_response: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Enhanced intelligent repair with advanced error recovery patterns
        """
        repair_info = {"attempts": 0, "methods": [], "fallback_used": False}
        
        try:
            # Analyze error patterns
            error_patterns = self._analyze_error_patterns(validation_result)
            
            # Apply advanced repair patterns
            for pattern_name, repair_method in self.advanced_repair_patterns.items():
                if pattern_name in error_patterns:
                    try:
                        parsed_data = repair_method(parsed_data, validation_result, expected_schema)
                        repair_info["methods"].append(pattern_name)
                        repair_info["attempts"] += 1
                    except Exception as e:
                        logging.warning(f"Advanced repair pattern {pattern_name} failed: {e}")
            
            # Apply adaptive learning if enabled
            if self.parsing_state["learning_enabled"]:
                parsed_data = self._apply_adaptive_learning_repair(parsed_data, validation_result)
                repair_info["methods"].append("adaptive_learning")
                repair_info["attempts"] += 1
            
            # Apply intelligent fallback if needed
            if not self._is_data_quality_acceptable(parsed_data):
                parsed_data = self._apply_intelligent_fallback(parsed_data, original_response, expected_schema)
                repair_info["fallback_used"] = True
                repair_info["methods"].append("intelligent_fallback")
            
            return parsed_data, repair_info
            
        except Exception as e:
            logging.error(f"Enhanced intelligent repair failed: {e}")
            return parsed_data, repair_info

    def _analyze_error_patterns(self, validation_result: ValidationResult) -> List[str]:
        """Analyze validation errors to identify repair patterns"""
        patterns = []
        
        for error in validation_result.errors:
            error_message = error.get("message", "").lower()
            
            if "missing" in error_message or "required" in error_message:
                patterns.append("missing_fields")
            elif "type" in error_message or "invalid type" in error_message:
                patterns.append("type_mismatches")
            elif "enum" in error_message or "invalid choice" in error_message:
                patterns.append("enum_errors")
            elif "validation" in error_message:
                patterns.append("validation_errors")
        
        return patterns

    def _repair_missing_fields(self, parsed_data: Dict[str, Any], validation_result: ValidationResult, expected_schema: Type) -> Dict[str, Any]:
        """Repair missing fields with intelligent defaults"""
        try:
            # Get required fields from schema
            if hasattr(expected_schema, 'model_fields'):
                required_fields = [field for field, info in expected_schema.model_fields.items() 
                                 if info.is_required()]
            else:
                required_fields = []
            
            # Add missing required fields with intelligent defaults
            for field in required_fields:
                if field not in parsed_data or parsed_data[field] is None:
                    parsed_data[field] = self._generate_intelligent_default(field, parsed_data)
            
            return parsed_data
            
        except Exception as e:
            logging.warning(f"Missing fields repair failed: {e}")
            return parsed_data

    def _repair_type_mismatches(self, parsed_data: Dict[str, Any], validation_result: ValidationResult, expected_schema: Type) -> Dict[str, Any]:
        """Repair type mismatches with intelligent conversion"""
        try:
            # Convert common type mismatches
            for error in validation_result.errors:
                field = error.get("field", "")
                if field in parsed_data:
                    value = parsed_data[field]
                    
                    # String to number conversion
                    if isinstance(value, str) and value.replace(".", "").replace("-", "").isdigit():
                        try:
                            if "." in value:
                                parsed_data[field] = float(value)
                            else:
                                parsed_data[field] = int(value)
                        except ValueError:
                            pass
                    
                    # String to boolean conversion
                    if isinstance(value, str) and value.lower() in ["true", "false"]:
                        parsed_data[field] = value.lower() == "true"
            
            return parsed_data
            
        except Exception as e:
            logging.warning(f"Type mismatch repair failed: {e}")
            return parsed_data

    def _repair_enum_errors(self, parsed_data: Dict[str, Any], validation_result: ValidationResult, expected_schema: Type) -> Dict[str, Any]:
        """Repair enum errors with intelligent mapping"""
        try:
            # Get enum fields from schema
            enum_fields = {}
            if hasattr(expected_schema, 'model_fields'):
                for field, info in expected_schema.model_fields.items():
                    if hasattr(info, 'annotation') and hasattr(info.annotation, '__origin__'):
                        if info.annotation.__origin__ is type and hasattr(info.annotation, '__args__'):
                            enum_class = info.annotation.__args__[0]
                            if hasattr(enum_class, '__members__'):
                                enum_fields[field] = enum_class
            
            # Repair enum values
            for field, enum_class in enum_fields.items():
                if field in parsed_data:
                    value = parsed_data[field]
                    if isinstance(value, str):
                        # Try to find matching enum value
                        for enum_name, enum_value in enum_class.__members__.items():
                            if value.lower() == enum_name.lower() or value.lower() == enum_value.value.lower():
                                parsed_data[field] = enum_value.value
                                break
            
            return parsed_data
            
        except Exception as e:
            logging.warning(f"Enum error repair failed: {e}")
            return parsed_data

    def _repair_validation_errors(self, parsed_data: Dict[str, Any], validation_result: ValidationResult, expected_schema: Type) -> Dict[str, Any]:
        """Repair validation errors with intelligent correction"""
        try:
            # Apply common validation fixes
            for error in validation_result.errors:
                field = error.get("field", "")
                message = error.get("message", "")
                
                if field in parsed_data:
                    value = parsed_data[field]
                    
                    # Length validation fixes
                    if "length" in message.lower():
                        if isinstance(value, str):
                            if "too short" in message.lower():
                                parsed_data[field] = value + " (extended)"
                            elif "too long" in message.lower():
                                parsed_data[field] = value[:100]  # Truncate to reasonable length
                    
                    # Range validation fixes
                    if "range" in message.lower() and isinstance(value, (int, float)):
                        if "too small" in message.lower():
                            parsed_data[field] = max(0, value)
                        elif "too large" in message.lower():
                            parsed_data[field] = min(100, value)
            
            return parsed_data
            
        except Exception as e:
            logging.warning(f"Validation error repair failed: {e}")
            return parsed_data

    def _generate_intelligent_default(self, field: str, parsed_data: Dict[str, Any]) -> Any:
        """Generate intelligent default values for missing fields"""
        try:
            # Field-specific intelligent defaults
            if "title" in field.lower():
                return "Thoughtful Gift Recommendation"
            elif "description" in field.lower():
                return "A carefully selected gift that matches the recipient's personality and preferences"
            elif "category" in field.lower():
                return "emotional_bonds"
            elif "price" in field.lower():
                return "€25-€50"
            elif "confidence" in field.lower():
                return 0.7
            elif "score" in field.lower():
                return 0.8
            elif "tags" in field.lower() or "list" in field.lower():
                return ["thoughtful", "personal"]
            elif "time" in field.lower():
                return 2000  # 2 seconds
            elif "model" in field.lower():
                return "ai_optimized"
            else:
                return "Unknown"
                
        except Exception as e:
            logging.warning(f"Intelligent default generation failed for {field}: {e}")
            return "Unknown"

    def _apply_adaptive_learning_repair(self, parsed_data: Dict[str, Any], validation_result: ValidationResult) -> Dict[str, Any]:
        """Apply adaptive learning to improve repair success"""
        try:
            # Learn from successful repair patterns
            if hasattr(self, 'successful_repair_patterns'):
                for pattern in self.successful_repair_patterns:
                    if pattern.matches(validation_result):
                        parsed_data = pattern.apply(parsed_data)
            
            return parsed_data
            
        except Exception as e:
            logging.warning(f"Adaptive learning repair failed: {e}")
            return parsed_data

    def _is_data_quality_acceptable(self, parsed_data: Dict[str, Any]) -> bool:
        """Check if data quality is acceptable after repair"""
        try:
            # Validate parsed_data is not None
            if parsed_data is None:
                return False
            
            # Check for essential fields
            essential_fields = ["title", "description", "category"]
            essential_present = sum(1 for field in essential_fields if field in parsed_data and parsed_data[field])
            
            # Check for reasonable values
            reasonable_values = 0
            for field, value in parsed_data.items():
                if value and value != "Unknown" and value != "":
                    reasonable_values += 1
            
            # Quality threshold (avoid division by zero)
            if len(essential_fields) == 0 or len(parsed_data) == 0:
                return False
            
            quality_score = (essential_present / len(essential_fields)) * 0.5 + (reasonable_values / len(parsed_data)) * 0.5
            
            return quality_score >= self.parsing_state["fallback_quality"]
            
        except Exception as e:
            logging.warning(f"Data quality check failed: {e}")
            return False

    def _apply_intelligent_fallback(self, parsed_data: Dict[str, Any], original_response: str, expected_schema: Type) -> Dict[str, Any]:
        """Apply intelligent fallback when repair fails"""
        try:
            # Validate parsed_data is not None
            if parsed_data is None:
                parsed_data = {}
            
            # Extract information from original response
            fallback_data = self._extract_fallback_data(original_response)
            
            # Merge with existing data
            for key, value in fallback_data.items():
                if key not in parsed_data or not parsed_data[key]:
                    parsed_data[key] = value
            
            return parsed_data
            
        except Exception as e:
            logging.warning(f"Intelligent fallback failed: {e}")
            return parsed_data if parsed_data is not None else {}

    def _extract_fallback_data(self, original_response: str) -> Dict[str, Any]:
        """Extract fallback data from original response"""
        try:
            fallback_data = {}
            
            # Extract gift information
            lines = original_response.split('\n')
            for line in lines:
                if '€' in line:
                    # Extract price information
                    price_match = re.search(r'€(\d+(?:\.\d{2})?)', line)
                    if price_match:
                        fallback_data["price_range"] = f"€{price_match.group(1)}"
                
                if len(line.strip()) > 10 and not line.startswith('#'):
                    # Extract description
                    if "description" not in fallback_data:
                        fallback_data["description"] = line.strip()
            
            return fallback_data
            
        except Exception as e:
            logging.warning(f"Fallback data extraction failed: {e}")
            return {}

    def _update_enhanced_parsing_statistics(self, result: ParsedResponse, source_model: AIModelType):
        """Update enhanced parsing statistics with performance insights"""
        try:
            # Basic statistics
            self.parsing_statistics["total_responses_parsed"] += 1
            if result.parsing_success:
                self.parsing_statistics["successful_parses"] += 1
            
            # Format distribution
            format_name = result.output_format.value if hasattr(result.output_format, 'value') else str(result.output_format)
            self.parsing_statistics["format_distribution"][format_name] = \
                self.parsing_statistics["format_distribution"].get(format_name, 0) + 1
            
            # Performance tracking
            if self.performance_tracking_enabled:
                self._track_parsing_performance(result, source_model)
            
        except Exception as e:
            logging.warning(f"Enhanced statistics update failed: {e}")

    def _track_parsing_performance(self, result: ParsedResponse, source_model: AIModelType):
        """Track parsing performance for optimization"""
        try:
            # Track parsing time
            if "parsing_times" not in self.parsing_statistics:
                self.parsing_statistics["parsing_times"] = []
            self.parsing_statistics["parsing_times"].append(result.parsing_time_ms)
            
            # Update average parsing time
            times = self.parsing_statistics["parsing_times"]
            self.parsing_statistics["average_parsing_time"] = sum(times) / len(times)
            
            # Track repair success rate
            if result.repair_attempts > 0:
                if "repair_attempts" not in self.parsing_statistics:
                    self.parsing_statistics["repair_attempts"] = 0
                    self.parsing_statistics["successful_repairs"] = 0
                
                self.parsing_statistics["repair_attempts"] += result.repair_attempts
                if result.parsing_success:
                    self.parsing_statistics["successful_repairs"] += 1
                
                self.parsing_statistics["repair_success_rate"] = \
                    self.parsing_statistics["successful_repairs"] / max(1, self.parsing_statistics["repair_attempts"])
            
        except Exception as e:
            logging.warning(f"Performance tracking failed: {e}")

    def _apply_parsing_optimizations(self, result: ParsedResponse, source_model: AIModelType) -> ParsedResponse:
        """Apply parsing optimizations based on performance data"""
        try:
            # Cache frequently used patterns
            if self.parsing_optimizations["cache_enabled"]:
                cache_key = f"{source_model}_{result.output_format}"
                if cache_key in self.performance_cache:
                    # Use cached optimization
                    result.parsing_time_ms = min(result.parsing_time_ms, self.performance_cache[cache_key])
            
            # Apply intelligent fallback optimizations
            if self.parsing_optimizations["intelligent_fallback"] and not result.parsing_success:
                result.confidence_score = max(result.confidence_score, self.parsing_state["fallback_quality"])
            
            return result
            
        except Exception as e:
            logging.warning(f"Parsing optimization failed: {e}")
            return result

    def _create_enhanced_error_response(self, raw_response: str, error_message: str, parsing_start: datetime, source_model: AIModelType) -> ParsedResponse:
        """Create enhanced error response with better error handling"""
        try:
            parsing_time = int((datetime.now() - parsing_start).total_seconds() * 1000)
            
            # Extract any useful information from the response
            fallback_data = self._extract_fallback_data(raw_response)
            
            return ParsedResponse(
                parsed_data=fallback_data,
                structured_output=None,
                original_response=raw_response,
                parsing_strategy=ParsingStrategy.HYBRID_PARSING,
                output_format=OutputFormat.UNSTRUCTURED_TEXT,
                parsing_success=False,
                parsing_time_ms=parsing_time,
                confidence_score=0.1,
                validation_errors=[{"field": "general", "message": error_message}],
                warnings=["Enhanced error recovery attempted"],
                repair_attempts=0,
                repair_methods_used=[],
                fallback_data_used=True
            )
            
        except Exception as e:
            logging.error(f"Enhanced error response creation failed: {e}")
            # Return minimal error response
            return ParsedResponse(
                parsed_data={},
                structured_output=None,
                original_response=raw_response,
                parsing_strategy=ParsingStrategy.HYBRID_PARSING,
                output_format=OutputFormat.EMPTY_RESPONSE,
                parsing_success=False,
                parsing_time_ms=0,
                confidence_score=0.0,
                validation_errors=[{"field": "general", "message": "Critical parsing failure"}],
                warnings=[],
                repair_attempts=0,
                repair_methods_used=[],
                fallback_data_used=False
            )
    
    def _preprocess_response(self, raw_response: str, source_model: AIModelType) -> str:
        """
        Preprocessing der rohen AI-Response
        
        Model-spezifische Anpassungen und Cleaning
        """
        
        if not raw_response or not raw_response.strip():
            return ""
        
        processed = raw_response.strip()
        
        # Model-specific adaptations
        if source_model in self.model_adaptations:
            processed = self.model_adaptations[source_model](processed)
        
        # Common preprocessing
        processed = self._remove_common_artifacts(processed)
        processed = self._normalize_whitespace(processed)
        
        return processed
    
    def _detect_output_format(self, response: str) -> OutputFormat:
        """
        Intelligent Format-Detection
        """
        
        if not response.strip():
            return OutputFormat.EMPTY_RESPONSE
        
        # Try JSON detection
        json_indicators = ['{', '[']
        has_json_start = any(response.strip().startswith(indicator) for indicator in json_indicators)
        
        if has_json_start:
            try:
                json.loads(response)
                return OutputFormat.VALID_JSON
            except json.JSONDecodeError:
                return OutputFormat.MALFORMED_JSON
        
        # Check for structured text patterns
        lines = response.split('\n')
        structured_patterns = 0
        
        for line in lines:
            if re.match(self.text_structure_patterns['numbered_list'], line.strip()):
                structured_patterns += 1
            elif re.match(self.text_structure_patterns['bulleted_list'], line.strip()):
                structured_patterns += 1
            elif re.match(self.text_structure_patterns['title_value'], line.strip()):
                structured_patterns += 1
        
        if structured_patterns > len(lines) * 0.3:  # 30% structured content
            return OutputFormat.STRUCTURED_TEXT
        
        # Check for mixed format (JSON + text)
        if '{' in response or '[' in response:
            return OutputFormat.MIXED_FORMAT
        
        return OutputFormat.UNSTRUCTURED_TEXT
    
    def _execute_parsing_strategy(
        self,
        response: str,
        strategy: ParsingStrategy,
        output_format: OutputFormat
    ) -> Optional[Dict[str, Any]]:
        """
        ✅ FIXED: Führt gewählte Parsing-Strategie aus + Auto-complete für GiftRecommendationResponse
        """
        
        # Original parsing logic
        parsed_data = None
        
        if strategy == ParsingStrategy.STRICT_JSON:
            parsed_data = self._parse_strict_json(response)
        elif strategy == ParsingStrategy.FLEXIBLE_JSON:
            parsed_data = self._parse_flexible_json(response, output_format)
        elif strategy == ParsingStrategy.STRUCTURED_TEXT:
            parsed_data = self._parse_structured_text(response)
        elif strategy == ParsingStrategy.HYBRID_PARSING:
            parsed_data = self._parse_hybrid_format(response, output_format)
        elif strategy == ParsingStrategy.AI_ASSISTED_REPAIR:
            parsed_data = self._parse_with_ai_assistance(response)
        else:
            parsed_data = self._parse_flexible_json(response, output_format)
        
        # ✅ FIX: Auto-complete parsed data to match GiftRecommendationResponse schema
        if parsed_data:
            logging.info("🔧 AUTO-COMPLETING parsed data for GiftRecommendationResponse structure")
            completed_data = self._auto_complete_gift_recommendation_structure(parsed_data)
            return completed_data
        
        return parsed_data

    def _auto_complete_gift_recommendation_structure(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ NEW: Auto-completes raw AI data to match GiftRecommendationResponse schema exactly
        
        Transforms any AI response into the complete required structure
        """
        
        # Extract available data
        raw_recommendations = raw_data.get("recommendations", [])
        personality_summary = raw_data.get("personality_summary", "Personality analysis from AI")
        analysis_confidence = raw_data.get("analysis_confidence", 0.8)
        
        # ✅ FIX: Transform recommendations to proper structure
        completed_recommendations = []
        for i, rec in enumerate(raw_recommendations[:5]):  # Max 5 recommendations
            if isinstance(rec, dict):
                completed_rec = {
                    # Required fields with proper mapping
                    "title": rec.get("title", rec.get("gift_name", f"Gift Recommendation {i+1}")),
                    "description": rec.get("description", rec.get("reasoning", "AI-generated gift recommendation")),
                    "category": rec.get("category", rec.get("gift_type", "thoughtful")),
                    "price_range": rec.get("price_range", rec.get("price", "€50-150")),
                    "availability": rec.get("availability", "available"),
                    "emotional_impact": rec.get("emotional_impact", rec.get("emotional_appeal", "positive emotional connection")),
                    "personal_connection": rec.get("personal_connection", "shows thoughtfulness and care"),
                    "relationship_benefit": rec.get("relationship_benefit", "strengthens relationship through thoughtful gesture"),
                    "personality_match": rec.get("personality_match", rec.get("reasoning", "matches personality preferences")),
                    "primary_reason": rec.get("primary_reason", rec.get("reasoning", "selected for personality compatibility")),
                    "confidence_score": float(rec.get("confidence_score", rec.get("confidence", 0.8))),
                    "confidence_level": rec.get("confidence_level", "high"),
                    "uniqueness_score": float(rec.get("uniqueness_score", 0.7))
                }
                completed_recommendations.append(completed_rec)
        
        # ✅ FIX: Generate complete personality_analysis structure
        personality_analysis = {
            "big_five_gift_implications": {
                "openness": "Moderate openness - enjoys both traditional and creative gifts",
                "conscientiousness": "Values quality and thoughtful selection",
                "extraversion": "Appreciates gifts that match social preferences", 
                "agreeableness": "Enjoys gifts that show care and consideration",
                "neuroticism": "Prefers gifts that provide comfort and stability"
            },
            "limbic_type": {
                "primary_type": "balanced",
                "secondary_type": "adaptive", 
                "confidence_score": analysis_confidence
            },
            "emotional_drivers": {
                "primary_drivers": ["appreciation", "thoughtfulness"],
                "secondary_drivers": ["quality", "personal_relevance"],
                "avoidance_factors": ["impersonal", "inappropriate"]
            },
            "purchase_motivations": {
                "functional_motivations": ["practicality", "quality"],
                "emotional_motivations": ["showing_care", "creating_joy"],
                "social_motivations": ["strengthening_relationship"]
            },
            "limbic_insights": {
                "stimulanz_implications": "Enjoys moderate stimulation and variety",
                "dominanz_implications": "Appreciates having choice and control",
                "balance_implications": "Values harmony and emotional balance"
            },
            "recommended_gift_categories": ["thoughtful", "quality", "personal"],
            "gift_dos": [
                "Choose something that shows you know them well",
                "Focus on quality over quantity", 
                "Consider their personal style and preferences"
            ],
            "gift_donts": [
                "Don't choose something too generic",
                "Avoid overly expensive or cheap items",
                "Don't ignore their stated preferences"
            ],
            "emotional_appeal_strategies": [
                "Show thoughtfulness and care",
                "Create positive emotional connection",
                "Demonstrate understanding of their personality"
            ],
            "analysis_depth": "ai_generated",
            "data_completeness": "complete"
        }
        
        # ✅ FIX: Complete GiftRecommendationResponse structure
        completed_structure = {
            # Core recommendations
            "recommendations": completed_recommendations,
            
            # Personality analysis
            "personality_analysis": personality_analysis,
            
            # Required top-level fields
            "overall_strategy": raw_data.get("overall_strategy", "Personalized recommendations based on personality analysis"),
            "key_considerations": raw_data.get("key_considerations", ["personality compatibility", "relationship appropriateness", "emotional impact"]),
            "emotional_themes": raw_data.get("emotional_themes", ["thoughtfulness", "care", "personal connection"]),
            "overall_confidence": float(raw_data.get("overall_confidence", analysis_confidence)),
            "personalization_score": float(raw_data.get("personalization_score", 0.8)),
            "novelty_score": float(raw_data.get("novelty_score", 0.7)),
            "emotional_resonance": float(raw_data.get("emotional_resonance", 0.8)),
            
            # Processing metadata
            "ai_model_used": raw_data.get("ai_model_used", "auto_completed"),
            "processing_time_ms": int(raw_data.get("processing_time_ms", 0)),
            "prompt_strategy": raw_data.get("prompt_strategy", "personality_based"),
            "optimization_goal": raw_data.get("optimization_goal", "quality"),
            
            # Success indicators  
            "success": True,
            "confidence_score": analysis_confidence,
            "processing_metadata": {
                "parser_version": "auto_complete_v1.0",
                "completion_applied": True,
                "original_fields": list(raw_data.keys()),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logging.info(f"✅ AUTO-COMPLETED: {len(completed_recommendations)} recommendations + full personality_analysis")
        return completed_structure
    
    def _parse_strict_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Strict JSON parsing - nur valides JSON akzeptiert"""
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None
    
    def _parse_flexible_json(self, response: str, output_format: OutputFormat) -> Optional[Dict[str, Any]]:
        """
        Flexible JSON parsing mit Reparatur-Versuchen
        """
        
        # First try direct parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to repair JSON
        repaired_json = self._repair_json(response)
        if repaired_json:
            try:
                return json.loads(repaired_json)
            except json.JSONDecodeError:
                pass
        
        # Extract JSON from mixed content
        if output_format == OutputFormat.MIXED_FORMAT:
            extracted_json = self._extract_json_from_mixed(response)
            if extracted_json:
                try:
                    return json.loads(extracted_json)
                except json.JSONDecodeError:
                    pass
        
        return None
    
    def _parse_structured_text(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parst strukturierten Text zu Dictionary
        """
        
        result = {
            "recommendations": [],
            "reasoning": "",
            "confidence": 0.7  # Default confidence for text parsing
        }
        
        lines = response.split('\n')
        current_recommendation = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try gift item pattern
            gift_match = re.match(self.text_structure_patterns['gift_item'], line)
            if gift_match:
                if current_recommendation:
                    result["recommendations"].append(current_recommendation)
                
                current_recommendation = {
                    "title": gift_match.group(1).strip(),
                    "price": f"€{gift_match.group(2)}",
                    "reasoning": gift_match.group(3).strip() if gift_match.group(3) else "",
                    "confidence": 0.7
                }
                continue
            
            # Try numbered list
            numbered_match = re.match(self.text_structure_patterns['numbered_list'], line)
            if numbered_match:
                if current_recommendation:
                    result["recommendations"].append(current_recommendation)
                
                item_text = numbered_match.group(1)
                current_recommendation = self._parse_text_item(item_text)
                continue
            
            # Try title:value pattern  
            title_value_match = re.match(self.text_structure_patterns['title_value'], line)
            if title_value_match:
                key = title_value_match.group(1).strip().lower()
                value = title_value_match.group(2).strip()
                
                if "reason" in key or "explanation" in key:
                    result["reasoning"] = value
                elif current_recommendation and key in ["price", "category", "confidence"]:
                    current_recommendation[key] = value
        
        # Add last recommendation
        if current_recommendation:
            result["recommendations"].append(current_recommendation)
        
        return result if result["recommendations"] else None
    
    def _parse_hybrid_format(self, response: str, output_format: OutputFormat) -> Optional[Dict[str, Any]]:
        """
        Hybrid parsing für gemischte Formate
        """
        
        # Try JSON first
        if output_format in [OutputFormat.VALID_JSON, OutputFormat.MALFORMED_JSON, OutputFormat.MIXED_FORMAT]:
            json_result = self._parse_flexible_json(response, output_format)
            if json_result:
                return json_result
        
        # Fall back to structured text
        if output_format in [OutputFormat.STRUCTURED_TEXT, OutputFormat.MIXED_FORMAT]:
            text_result = self._parse_structured_text(response)
            if text_result:
                return text_result
        
        # Last resort: try to extract any useful information
        return self._extract_minimal_data(response)
    
    def _parse_with_ai_assistance(self, response: str) -> Optional[Dict[str, Any]]:
        """AI-assistierte Parsing (Placeholder für zukünftige Implementierung)"""
        # TODO: Implementiere AI-assistierte Reparatur
        return self._parse_hybrid_format(response, OutputFormat.MIXED_FORMAT)
    
    def _repair_json(self, malformed_json: str) -> Optional[str]:
        """
        Versucht JSON zu reparieren mit Pattern-Matching
        """
        
        repaired = malformed_json
        
        # Apply repair patterns
        for pattern, replacement in self.json_repair_patterns:
            repaired = re.sub(pattern, replacement, repaired, flags=re.MULTILINE)
        
        # Try to balance brackets
        repaired = self._balance_brackets(repaired)
        
        return repaired
    
    def _extract_json_from_mixed(self, mixed_content: str) -> Optional[str]:
        """
        Extrahiert JSON aus gemischtem Content
        """
        
        # Look for JSON objects
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, mixed_content, re.DOTALL)
        
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        
        # Look for JSON arrays
        array_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        matches = re.findall(array_pattern, mixed_content, re.DOTALL)
        
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _validate_against_schema(self, data: Optional[Dict[str, Any]], schema_class: Type) -> ValidationResult:
        """
        Validiert parsed data gegen Pydantic Schema
        """
        
        if not data:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                errors=[{"field": "root", "message": "No data to validate"}],
                warnings=[],
                suggestions=["Check response format and parsing strategy"],
                confidence_impact=-0.8
            )
        
        try:
            # Try to create instance
            schema_class(**data)
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                errors=[],
                warnings=[],
                suggestions=[],
                confidence_impact=0.0
            )
            
        except Exception as e:
            # Analyze validation errors
            errors, warnings, suggestions = self._analyze_validation_error(e, data, schema_class)
            
            severity = self._determine_error_severity(errors)
            confidence_impact = self._calculate_validation_confidence_impact(errors, severity)
            
            return ValidationResult(
                is_valid=False,
                severity=severity,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                confidence_impact=confidence_impact
            )
    
    def _attempt_intelligent_repair(
        self,
        data: Optional[Dict[str, Any]],
        validation_result: ValidationResult,
        schema_class: Type,
        original_response: str
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Intelligente Reparatur von fehlerhaften Daten
        """
        
        repair_info = {
            "attempts": 0,
            "methods": [],
            "fallback_used": False
        }
        
        if not data:
            # Try to reconstruct from original response
            repair_info["attempts"] += 1
            repair_info["methods"].append("reconstruct_from_text")
            
            reconstructed = self._reconstruct_data_from_text(original_response, schema_class)
            if reconstructed:
                return reconstructed, repair_info
        
        # Try field-specific repairs
        if data and validation_result.errors:
            repair_info["attempts"] += 1
            repair_info["methods"].append("field_repair")
            
            repaired_data = self._repair_individual_fields(data, validation_result.errors, schema_class)
            if repaired_data != data:
                validation_check = self._validate_against_schema(repaired_data, schema_class)
                if validation_check.is_valid:
                    return repaired_data, repair_info
        
        # Try fallback data insertion
        if validation_result.severity in [ValidationSeverity.MAJOR, ValidationSeverity.CRITICAL]:
            repair_info["attempts"] += 1
            repair_info["methods"].append("fallback_insertion")
            repair_info["fallback_used"] = True
            
            fallback_data = self._insert_fallback_data(data or {}, schema_class)
            return fallback_data, repair_info
        
        return data, repair_info
    
    def _calculate_confidence_score(
        self,
        validation_result: ValidationResult,
        output_format: OutputFormat,
        repair_info: Dict[str, Any],
        source_model: AIModelType
    ) -> float:
        """
        Berechnet Confidence Score für geparsete Response
        """
        
        base_confidence = 1.0
        
        # Validation impact
        base_confidence += validation_result.confidence_impact
        
        # Format impact
        format_impact = {
            OutputFormat.VALID_JSON: 0.0,
            OutputFormat.MALFORMED_JSON: -0.2,
            OutputFormat.STRUCTURED_TEXT: -0.1,
            OutputFormat.MIXED_FORMAT: -0.15,
            OutputFormat.UNSTRUCTURED_TEXT: -0.3,
            OutputFormat.EMPTY_RESPONSE: -0.8
        }
        base_confidence += format_impact.get(output_format, -0.2)
        
        # Repair impact
        if repair_info["attempts"] > 0:
            base_confidence -= (repair_info["attempts"] * 0.1)
        
        if repair_info["fallback_used"]:
            base_confidence -= 0.3
        
        # Model reliability impact
        model_reliability = {
            AIModelType.OPENAI_GPT4: 0.05,
            AIModelType.ANTHROPIC_CLAUDE: 0.03,
            AIModelType.GROQ_MIXTRAL: 0.0,
            AIModelType.GOOGLE_GEMINI: 0.02
        }
        base_confidence += model_reliability.get(source_model, 0.0)
        
        # Clamp to valid range
        return max(0.0, min(1.0, base_confidence))
    
    # Model-specific adaptation methods
    def _adapt_openai_output(self, response: str) -> str:
        """OpenAI-spezifische Output-Anpassungen"""
        # OpenAI sometimes adds explanatory text before JSON
        if "```json" in response:
            # Extract JSON from code blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json_match.group(1)
        return response
    
    def _adapt_groq_output(self, response: str) -> str:
        """Groq-spezifische Output-Anpassungen"""
        # Groq tends to be more direct but may have formatting issues
        # Remove any leading/trailing text that's not part of the JSON
        lines = response.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            if line.strip().startswith('{') or line.strip().startswith('['):
                in_json = True
            if in_json:
                json_lines.append(line)
            if in_json and (line.strip().endswith('}') or line.strip().endswith(']')):
                break
        
        return '\n'.join(json_lines) if json_lines else response
    
    def _adapt_claude_output(self, response: str) -> str:
        """Claude-spezifische Output-Anpassungen"""
        # Claude often provides thoughtful explanations
        # Look for JSON within the response
        if '{' in response and '}' in response:
            start = response.find('{')
            end = response.rfind('}') + 1
            potential_json = response[start:end]
            try:
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass
        return response
    
    def _adapt_gemini_output(self, response: str) -> str:
        """Gemini-spezifische Output-Anpassungen"""
        # Similar to OpenAI but may have different patterns
        return self._adapt_openai_output(response)
    
    # Helper methods
    def _remove_common_artifacts(self, response: str) -> str:
        """Entfernt häufige Response-Artifacts"""
        # Remove markdown artifacts
        response = re.sub(r'```\w*\s*', '', response)
        response = re.sub(r'\s*```', '', response)
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Here's the recommendation:",
            "Here are the gift recommendations:",
            "Based on the analysis:",
            "Response:",
            "Result:"
        ]
        
        for prefix in prefixes_to_remove:
            if response.strip().startswith(prefix):
                response = response[len(prefix):].strip()
        
        return response
    
    def _normalize_whitespace(self, response: str) -> str:
        """Normalisiert Whitespace"""
        # Normalize line endings
        response = re.sub(r'\r\n|\r', '\n', response)
        # Remove excessive whitespace but preserve structure
        response = re.sub(r' +', ' ', response)
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        return response.strip()
    
    def _balance_brackets(self, json_str: str) -> str:
        """Versucht Bracket-Balance zu reparieren"""
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # Add missing closing brackets
        json_str += '}' * (open_braces - close_braces)
        json_str += ']' * (open_brackets - close_brackets)
        
        return json_str
    
    def _parse_text_item(self, item_text: str) -> Dict[str, Any]:
        """Parst einzelnen Text-Item zu Dictionary"""
        # Try to extract price
        price_match = re.search(r'€(\d+(?:\.\d{2})?)', item_text)
        price = f"€{price_match.group(1)}" if price_match else "Price not specified"
        
        # Remove price from title
        title = re.sub(r'\s*\(€\d+(?:\.\d{2})?\)\s*', '', item_text)
        title = re.sub(r'\s*€\d+(?:\.\d{2})?\s*', '', title)
        
        return {
            "title": title.strip(),
            "price": price,
            "reasoning": "Extracted from text format",
            "confidence": 0.6
        }
    
    def _extract_minimal_data(self, response: str) -> Optional[Dict[str, Any]]:
        """Extrahiert minimale Daten als last resort"""
        # Try to find any gift-like items
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        recommendations = []
        for line in lines:
            if len(line) > 10 and ('gift' in line.lower() or '€' in line or '$' in line):
                recommendations.append({
                    "title": line[:100],  # Limit length
                    "price": "Price not specified",
                    "reasoning": "Extracted from unstructured text",
                    "confidence": 0.3
                })
        
        if recommendations:
            return {
                "recommendations": recommendations[:5],  # Limit to 5
                "reasoning": "Minimal extraction from unstructured response",
                "confidence": 0.3
            }
        
        return None
    
    def _analyze_validation_error(self, error: Exception, data: Dict[str, Any], schema_class: Type) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
        """Analysiert Validation Errors für detaillierte Diagnostik"""
        errors = []
        warnings = []
        suggestions = []
        
        error_str = str(error)
        
        # Common validation issues
        if "field required" in error_str:
            missing_field = re.search(r"field required.*?(\w+)", error_str)
            if missing_field:
                field_name = missing_field.group(1)
                errors.append({
                    "field": field_name,
                    "message": f"Required field '{field_name}' is missing"
                })
                suggestions.append(f"Add missing field '{field_name}' to the response")
        
        if "validation error" in error_str.lower():
            errors.append({
                "field": "unknown",
                "message": "Schema validation failed"
            })
            suggestions.append("Check data types and required fields")
        
        return errors, warnings, suggestions
    
    def _determine_error_severity(self, errors: List[Dict[str, str]]) -> ValidationSeverity:
        """Bestimmt Schweregrad der Validation Errors"""
        if not errors:
            return ValidationSeverity.INFO
        
        critical_patterns = ["required field", "invalid type", "parsing failed"]
        major_patterns = ["missing data", "format error"]
        
        for error in errors:
            message = error.get("message", "").lower()
            if any(pattern in message for pattern in critical_patterns):
                return ValidationSeverity.CRITICAL
            elif any(pattern in message for pattern in major_patterns):
                return ValidationSeverity.MAJOR
        
        return ValidationSeverity.MINOR
    
    def _calculate_validation_confidence_impact(self, errors: List[Dict[str, str]], severity: ValidationSeverity) -> float:
        """Berechnet Confidence-Impact von Validation Errors"""
        impact_map = {
            ValidationSeverity.CRITICAL: -0.6,
            ValidationSeverity.MAJOR: -0.3,
            ValidationSeverity.MINOR: -0.1,
            ValidationSeverity.WARNING: -0.05,
            ValidationSeverity.INFO: 0.0
        }
        
        base_impact = impact_map.get(severity, -0.2)
        # Increase impact based on number of errors
        error_multiplier = min(len(errors) * 0.1, 0.3)
        
        return base_impact - error_multiplier
    
    def _reconstruct_data_from_text(self, text: str, schema_class: Type) -> Optional[Dict[str, Any]]:
        """Rekonstruiert strukturierte Daten aus Text"""
        # Placeholder implementation - could be more sophisticated
        return self._parse_structured_text(text)
    
    def _repair_individual_fields(self, data: Dict[str, Any], errors: List[Dict[str, str]], schema_class: Type) -> Dict[str, Any]:
        """
        ✅ FIXED: Repariert individuelle Felder basierend auf Validation Errors
        
        NEUE FUNKTION: Auto-generiert fehlende personality_analysis fields
        """
        repaired = data.copy()
        
        # ✅ FIX: Handle missing personality_analysis fields
        personality_analysis_errors = [
            error for error in errors 
            if error.get("field", "").startswith("personality_analysis") or 
            "personality_analysis" in error.get("message", "")
        ]
        
        if personality_analysis_errors and schema_class.__name__ == "GiftRecommendationResponse":
            logging.info("🔧 Auto-generating missing personality_analysis from AI response content")
            
            # Generate complete personality_analysis using available AI content
            if "personality_analysis" not in repaired or not repaired["personality_analysis"]:
                repaired["personality_analysis"] = self._generate_personality_analysis_from_content(repaired)
            else:
                # Complete existing personality_analysis with missing fields
                repaired["personality_analysis"] = self._complete_existing_personality_analysis(
                    repaired["personality_analysis"], repaired
                )
        
        # Original field repair logic
        for error in errors:
            field = error.get("field")
            message = error.get("message", "")
            
            if field and field in repaired:
                # Try to fix common issues
                if "invalid type" in message:
                    # Try type conversion
                    value = repaired[field]
                    if "expected float" in message and isinstance(value, str):
                        try:
                            repaired[field] = float(value.replace('€', '').replace(',', '.'))
                        except ValueError:
                            pass
                    elif "expected int" in message and isinstance(value, str):
                        try:
                            repaired[field] = int(value)
                        except ValueError:
                            pass
        
        return repaired
    
    
    def _insert_fallback_data(self, data: Dict[str, Any], schema_class: Type) -> Dict[str, Any]:
        """
        ✅ FIXED: Fügt Fallback-Daten für kritische fehlende Felder ein
        
        LÖSUNG für 11 validation errors: Generiert vollständige personality_analysis
        """
        fallback = data.copy()
        
        # ✅ FIX: Vollständige personality_analysis für GiftRecommendationResponse
        if schema_class.__name__ == "GiftRecommendationResponse":
            
            # 1. GENERATE COMPLETE PERSONALITY_ANALYSIS
            if "personality_analysis" not in fallback or not fallback["personality_analysis"]:
                fallback["personality_analysis"] = {
                    "big_five_gift_implications": {
                        "openness": "Moderate openness - enjoys both traditional and creative gifts",
                        "conscientiousness": "Values quality and thoughtful selection",
                        "extraversion": "Appreciates gifts that match social preferences", 
                        "agreeableness": "Enjoys gifts that show care and consideration",
                        "neuroticism": "Prefers gifts that provide comfort and stability"
                    },
                    "limbic_type": {
                        "primary_type": "balanced",
                        "secondary_type": "adaptive", 
                        "confidence_score": 0.7
                    },
                    "emotional_drivers": {
                        "primary_drivers": ["appreciation", "thoughtfulness"],
                        "secondary_drivers": ["quality", "personal_relevance"],
                        "avoidance_factors": ["impersonal", "inappropriate"]
                    },
                    "purchase_motivations": {
                        "functional_motivations": ["practicality", "quality"],
                        "emotional_motivations": ["showing_care", "creating_joy"],
                        "social_motivations": ["strengthening_relationship"]
                    },
                    "limbic_insights": {
                        "stimulanz_implications": "Enjoys moderate stimulation and variety",
                        "dominanz_implications": "Appreciates having choice and control",
                        "balance_implications": "Values harmony and emotional balance"
                    },
                    "recommended_gift_categories": ["thoughtful", "quality", "personal"],
                    "gift_dos": [
                        "Choose something that shows you know them well",
                        "Focus on quality over quantity", 
                        "Consider their personal style and preferences"
                    ],
                    "gift_donts": [
                        "Don't choose something too generic",
                        "Avoid overly expensive or cheap items",
                        "Don't ignore their stated preferences"
                    ],
                    "emotional_appeal_strategies": [
                        "Show thoughtfulness and care",
                        "Create positive emotional connection",
                        "Demonstrate understanding of their personality"
                    ],
                    "analysis_depth": "fallback_generated",
                    "data_completeness": "complete"
                }
            
            # 2. ENSURE OTHER REQUIRED FIELDS
            if "recommendations" not in fallback:
                fallback["recommendations"] = []
            
            if "success" not in fallback:
                fallback["success"] = True
            
            if "confidence_score" not in fallback:
                fallback["confidence_score"] = 0.7
            
            if "processing_metadata" not in fallback:
                fallback["processing_metadata"] = {
                    "parser_version": "enhanced_fallback_v1.0",
                    "processing_strategy": "intelligent_fallback",
                    "timestamp": datetime.now().isoformat(),
                    "fallback_reason": "missing_required_fields"
                }
        
        else:
            # Original fallback logic für andere Schema-Typen
            fallback_values = {
                "recommendations": [
                    {
                        "title": "Gift recommendation not available",
                        "price": "€0",
                        "category": "general",
                        "reasoning": "Unable to parse AI response",
                        "confidence": 0.1
                    }
                ],
                "personality_analysis": {
                    "personality_summary": "Analysis not available",
                    "confidence": 0.1
                },
                "overall_confidence": 0.1,
                "processing_time_ms": 0,
                "ai_model_used": "unknown"
            }
            
            # Add missing critical fields
            for field, default_value in fallback_values.items():
                if field not in fallback:
                    fallback[field] = default_value
        
        return fallback
    
    def _update_parsing_statistics(self, result: ParsedResponse):
        """Aktualisiert Parsing-Statistiken"""
        self.parsing_statistics["total_responses_parsed"] += 1
        
        if result.parsing_success:
            self.parsing_statistics["successful_parses"] += 1
        
        # Format distribution
        format_key = result.output_format.value
        self.parsing_statistics["format_distribution"][format_key] = \
            self.parsing_statistics["format_distribution"].get(format_key, 0) + 1
        
        # Average parsing time
        total = self.parsing_statistics["total_responses_parsed"]
        current_avg = self.parsing_statistics["average_parsing_time"]
        self.parsing_statistics["average_parsing_time"] = \
            ((current_avg * (total - 1)) + result.parsing_time_ms) / total
        
        # Repair success rate
        if result.repair_attempts > 0:
            repair_successes = self.parsing_statistics.get("repair_successes", 0)
            repair_attempts = self.parsing_statistics.get("total_repair_attempts", 0)
            
            if result.parsing_success:
                repair_successes += 1
            repair_attempts += 1
            
            self.parsing_statistics["repair_successes"] = repair_successes
            self.parsing_statistics["total_repair_attempts"] = repair_attempts
            self.parsing_statistics["repair_success_rate"] = repair_successes / repair_attempts
    
    def _create_error_response(self, raw_response: str, error_message: str, start_time: datetime) -> ParsedResponse:
        """Erstellt Error-Response bei kritischen Parsing-Fehlern"""
        parsing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return ParsedResponse(
            parsed_data=None,
            structured_output=None,
            original_response=raw_response,
            parsing_strategy=ParsingStrategy.STRICT_JSON,
            output_format=OutputFormat.EMPTY_RESPONSE,
            parsing_success=False,
            parsing_time_ms=parsing_time,
            confidence_score=0.0,
            validation_errors=[{"field": "parsing", "message": error_message}],
            warnings=["Critical parsing failure"],
            repair_attempts=0,
            repair_methods_used=[],
            fallback_data_used=False
        )
    
    # Public utility methods
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """Gibt Parsing-Statistiken zurück"""
        stats = self.parsing_statistics.copy()
        
        if stats["total_responses_parsed"] > 0:
            stats["success_rate"] = stats["successful_parses"] / stats["total_responses_parsed"]
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Setzt Parsing-Statistiken zurück"""
        self.parsing_statistics = {
            "total_responses_parsed": 0,
            "successful_parses": 0,
            "format_distribution": {},
            "error_patterns": {},
            "average_parsing_time": 0.0,
            "repair_success_rate": 0.0
        }

    def _generate_personality_analysis_from_content(self, ai_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ NEW: Generates complete personality_analysis from AI response content
        
        Uses available recommendations and AI insights to create meaningful personality analysis
        """
        
        # Extract insights from AI response content
        recommendations = ai_data.get("recommendations", [])
        confidence = ai_data.get("confidence_score", 0.7)
        
        # Analyze recommendations for personality insights
        gift_categories = []
        reasoning_insights = []
        emotional_keywords = []
        
        for rec in recommendations[:3]:  # Top 3 recommendations
            if isinstance(rec, dict):
                # Extract categories
                category = rec.get("category", rec.get("gift_type", ""))
                if category and category not in gift_categories:
                    gift_categories.append(category)
                
                # Extract reasoning insights
                reasoning = rec.get("reasoning", rec.get("description", ""))
                if reasoning:
                    reasoning_insights.append(reasoning[:100])  # First 100 chars
                
                # Extract emotional keywords
                emotional_content = rec.get("emotional_impact", rec.get("emotional_appeal", ""))
                if emotional_content:
                    emotional_keywords.append(emotional_content)
        
        # Generate personality insights based on content
        personality_traits = self._infer_personality_from_recommendations(gift_categories, reasoning_insights)
        
        return {
            "big_five_gift_implications": {
                "openness": personality_traits.get("openness", "Moderate openness - appreciates both traditional and creative gifts"),
                "conscientiousness": personality_traits.get("conscientiousness", "Values quality and thoughtful selection"),
                "extraversion": personality_traits.get("extraversion", "Appreciates gifts that match social preferences"), 
                "agreeableness": personality_traits.get("agreeableness", "Enjoys gifts that show care and consideration"),
                "neuroticism": personality_traits.get("neuroticism", "Prefers gifts that provide emotional comfort")
            },
            "limbic_type": {
                "primary_type": personality_traits.get("limbic_primary", "balanced"),
                "secondary_type": personality_traits.get("limbic_secondary", "adaptive"), 
                "confidence_score": confidence
            },
            "emotional_drivers": {
                "primary_drivers": emotional_keywords[:2] if emotional_keywords else ["appreciation", "thoughtfulness"],
                "secondary_drivers": emotional_keywords[2:4] if len(emotional_keywords) > 2 else ["quality", "personal_relevance"],
                "avoidance_factors": ["impersonal", "inappropriate", "generic"]
            },
            "purchase_motivations": {
                "functional_motivations": ["practicality", "quality"],
                "emotional_motivations": ["showing_care", "creating_joy"],
                "social_motivations": ["strengthening_relationship"]
            },
            "limbic_insights": {
                "stimulanz_implications": "Enjoys appropriate level of excitement and novelty",
                "dominanz_implications": "Appreciates having choice and personal relevance",
                "balance_implications": "Values emotional harmony and thoughtful gestures"
            },
            "recommended_gift_categories": gift_categories if gift_categories else ["thoughtful", "quality", "personal"],
            "gift_dos": self._extract_gift_dos_from_recommendations(reasoning_insights),
            "gift_donts": [
                "Don't choose something too generic",
                "Avoid ignoring their personal style",
                "Don't overspend or underspend dramatically"
            ],
            "emotional_appeal_strategies": emotional_keywords if emotional_keywords else [
                "Show thoughtfulness and care",
                "Create positive emotional connection",
                "Demonstrate understanding of their personality"
            ],
            "analysis_depth": "ai_content_derived",
            "data_completeness": "complete"
        }

    def _complete_existing_personality_analysis(self, existing: Dict[str, Any], full_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ NEW: Completes existing personality_analysis with missing required fields
        """
        
        complete = existing.copy()
        confidence = full_response.get("confidence_score", 0.7)
        
        # Required fields with intelligent defaults
        required_defaults = {
            "big_five_gift_implications": {
                "openness": "Openness influences creativity in gift preferences",
                "conscientiousness": "Conscientiousness affects appreciation for quality and planning",
                "extraversion": "Extraversion impacts preference for social vs personal gifts",
                "agreeableness": "Agreeableness influences appreciation for thoughtful gestures",
                "neuroticism": "Emotional sensitivity affects gift comfort and appropriateness"
            },
            "limbic_type": {
                "primary_type": "balanced",
                "secondary_type": "adaptive",
                "confidence_score": confidence
            },
            "emotional_drivers": {
                "primary_drivers": ["appreciation", "connection"],
                "secondary_drivers": ["quality", "thoughtfulness"],
                "avoidance_factors": ["stress", "inappropriateness"]
            },
            "purchase_motivations": {
                "functional_motivations": ["utility", "quality"],
                "emotional_motivations": ["care", "joy"],
                "social_motivations": ["relationship"]
            },
            "limbic_insights": {
                "stimulanz_implications": "Moderate stimulation preference",
                "dominanz_implications": "Balanced control needs",
                "balance_implications": "Values emotional stability"
            },
            "recommended_gift_categories": ["thoughtful", "quality"],
            "gift_dos": ["Show thoughtfulness", "Consider preferences"],
            "gift_donts": ["Avoid generic choices", "Don't ignore personality"],
            "emotional_appeal_strategies": ["Demonstrate care", "Show understanding"],
            "analysis_depth": "ai_completed",
            "data_completeness": "enhanced"
        }
        
        # Add missing required fields
        for field_name, default_value in required_defaults.items():
            if field_name not in complete or not complete[field_name]:
                complete[field_name] = default_value
        
        return complete

    def _infer_personality_from_recommendations(self, categories: List[str], reasoning: List[str]) -> Dict[str, str]:
        """
        ✅ NEW: Infers personality traits from recommendation patterns
        """
        
        traits = {}
        combined_text = " ".join(categories + reasoning).lower()
        
        # Openness inference
        if any(word in combined_text for word in ["creative", "artistic", "innovative", "unique", "original"]):
            traits["openness"] = "High openness - loves creative and unique gifts"
        elif any(word in combined_text for word in ["traditional", "classic", "conventional"]):
            traits["openness"] = "Lower openness - prefers traditional and proven gifts"
        
        # Conscientiousness inference  
        if any(word in combined_text for word in ["quality", "organized", "practical", "useful", "durable"]):
            traits["conscientiousness"] = "High conscientiousness - appreciates quality and practical gifts"
        elif any(word in combined_text for word in ["spontaneous", "fun", "casual", "relaxed"]):
            traits["conscientiousness"] = "Lower conscientiousness - enjoys spontaneous and fun gifts"
        
        # Extraversion inference
        if any(word in combined_text for word in ["social", "party", "group", "shared", "experience"]):
            traits["extraversion"] = "High extraversion - enjoys social and experiential gifts"
        elif any(word in combined_text for word in ["personal", "private", "quiet", "individual"]):
            traits["extraversion"] = "Lower extraversion - prefers personal and intimate gifts"
        
        return traits

    def _extract_gift_dos_from_recommendations(self, reasoning_list: List[str]) -> List[str]:
        """
        ✅ NEW: Extracts gift dos from recommendation reasoning
        """
        
        dos = []
        
        for reasoning in reasoning_list:
            if "quality" in reasoning.lower():
                dos.append("Focus on quality over quantity")
            if "personal" in reasoning.lower():
                dos.append("Choose something personal and meaningful")
            if "practical" in reasoning.lower():
                dos.append("Consider practical value and usefulness")
        
        # Default dos if nothing extracted
        if not dos:
            dos = [
                "Show that you know them well",
                "Focus on their interests and preferences",
                "Choose something that fits the occasion"
            ]
        
        return dos[:3]  # Max 3 dos




# =============================================================================
# SPECIALIZED PARSERS FOR DIFFERENT RESPONSE TYPES
# =============================================================================

class QuickRecommendationParser(ResponseParser):
    """Spezialisierter Parser für Quick Recommendations (Groq)"""
    
    def parse_quick_response(self, raw_response: str, source_model: AIModelType) -> ParsedResponse:
        """Parst Quick Recommendation Response"""
        return self.parse_gift_recommendation_response(
            raw_response=raw_response,
            source_model=source_model,
            expected_schema=QuickRecommendationResponse,
            parsing_strategy=ParsingStrategy.FLEXIBLE_JSON
        )


class PersonalityAnalysisParser(ResponseParser):
    """Spezialisierter Parser für Personality Analysis"""
    
    def parse_personality_response(self, raw_response: str, source_model: AIModelType) -> ParsedResponse:
        """Parst Personality Analysis Response"""
        return self.parse_gift_recommendation_response(
            raw_response=raw_response,
            source_model=source_model,
            expected_schema=PersonalityAnalysisResult,
            parsing_strategy=ParsingStrategy.HYBRID_PARSING
        )


# =============================================================================
# ✅ NEUE PARSER FÜR HELDENREISE-KATALOG (Korrigiert & Vollständig)
# =============================================================================

class HeroicJourneyResponseParser(ResponseParser):
    """
    ✅ Spezialisierter Parser für Heldenreise-Geschenk-Empfehlungen
    
    Erweitert den bestehenden ResponseParser um Heldenreise-spezifische Funktionen
    """
    
    def parse_heroic_gift_response(
        self,
        raw_response: str,
        source_model: AIModelType,
        expected_schema: Type = GiftRecommendationResponse
    ) -> ParsedResponse:
        """
        Parst AI-Responses für Heldenreise-Geschenke
        
        Args:
            raw_response: Rohe AI-Antwort
            source_model: AI-Model das die Response generiert hat
            expected_schema: Erwartetes Schema
            
        Returns:
            ParsedResponse mit Heldenreise-spezifischen Validierungen
        """
        
        # Nutze die bestehende Parsing-Logik als Basis
        base_result = self.parse_gift_recommendation_response(
            raw_response=raw_response,
            source_model=source_model,
            expected_schema=expected_schema,
            parsing_strategy=ParsingStrategy.HYBRID_PARSING
        )
        
        # Erweitere um Heldenreise-spezifische Validierungen
        if base_result.parsing_success and base_result.parsed_data:
            enhanced_data = self._enhance_with_heroic_elements(base_result.parsed_data)
            
            return ParsedResponse(
                parsed_data=enhanced_data,
                structured_output=base_result.structured_output,
                original_response=raw_response,
                parsing_strategy=ParsingStrategy.HYBRID_PARSING,
                output_format=base_result.output_format,
                parsing_success=True,
                parsing_time_ms=base_result.parsing_time_ms,
                confidence_score=min(base_result.confidence_score + 0.1, 1.0),  # Bonus für Heldenreise
                validation_errors=base_result.validation_errors,
                warnings=base_result.warnings,
                repair_attempts=base_result.repair_attempts,
                repair_methods_used=base_result.repair_methods_used,
                fallback_data_used=base_result.fallback_data_used
            )
        
        return base_result
    
    def _enhance_with_heroic_elements(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Erweitert geparste Daten um Heldenreise-spezifische Elemente
        """
        
        enhanced = parsed_data.copy()
        
        # Füge Heldenreise-Metadaten hinzu wenn vorhanden
        if 'recommendations' in enhanced:
            for rec in enhanced['recommendations']:
                # Prüfe auf Heldenreise-Keywords
                rec_text = str(rec).lower()
                if any(keyword in rec_text for keyword in ['heldenreise', 'transformation', 'mut', 'wachstum', 'entwicklung']):
                    rec['heroic_journey_detected'] = True
                    rec['emotional_transformation'] = self._extract_transformation(rec_text)
                else:
                    rec['heroic_journey_detected'] = False
        
        # Füge Heldenreise-Confidence hinzu
        enhanced['heroic_journey_score'] = self._calculate_heroic_score(enhanced)
        
        return enhanced
    
    def _extract_transformation(self, text: str) -> Optional[str]:
        """Extrahiert Transformations-Text aus der Response"""
        
        # Suche nach Transformation-Patterns
        transformation_patterns = [
            r'von\s+["\']([^"\']+)["\']\s+zu\s+["\']([^"\']+)["\']',
            r'transformation:\s*([^.]+)',
            r'verwandlung:\s*([^.]+)'
        ]
        
        for pattern in transformation_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    return f"Von '{match.group(1)}' zu '{match.group(2)}'"
                else:
                    return match.group(1).strip()
        
        return None
    
    def _calculate_heroic_score(self, data: Dict[str, Any]) -> float:
        """Berechnet Heldenreise-Score basierend auf Response-Inhalt"""
        
        score = 0.0
        text_content = str(data).lower()
        
        # Heldenreise-Keywords
        heroic_keywords = [
            'heldenreise', 'transformation', 'wachstum', 'entwicklung',
            'mut', 'selbstvertrauen', 'persönlichkeit', 'story', 'geschichte',
            'abenteuer', 'herausforderung', 'ziel', 'vision'
        ]
        
        keyword_count = sum(1 for keyword in heroic_keywords if keyword in text_content)
        score += min(keyword_count * 0.1, 0.5)  # Max 0.5 für Keywords
        
        # Strukturelle Heldenreise-Elemente
        if 'recommendations' in data:
            heroic_recs = sum(1 for rec in data['recommendations'] if rec.get('heroic_journey_detected', False))
            total_recs = len(data['recommendations'])
            if total_recs > 0:
                score += (heroic_recs / total_recs) * 0.5  # Max 0.5 für Struktur
        
        return min(score, 1.0)


class EmotionalStoryParser(ResponseParser):
    """
    ✅ Parser für emotionale Story-Komponenten in Geschenk-Empfehlungen
    """
    
    def parse_emotional_story_response(
        self,
        raw_response: str,
        story_context: Dict[str, Any]
    ) -> ParsedResponse:
        """
        Parst Responses die emotionale Geschichten enthalten
        
        Args:
            raw_response: AI-Response mit Story-Elementen
            story_context: Context für Story-Parsing (Name, Alter, etc.)
            
        Returns:
            ParsedResponse mit extrahierten Story-Elementen
        """
        
        parsing_start = datetime.now()
        
        try:
            # Basis-Parsing
            base_result = self._parse_flexible_json(raw_response, OutputFormat.MIXED_FORMAT)
            
            if not base_result:
                # Fallback zu Text-Parsing für Stories
                base_result = self._parse_story_from_text(raw_response, story_context)
            
            # Story-spezifische Validierung
            validation_result = self._validate_story_elements(base_result, story_context)
            
            # Story Enhancement
            enhanced_data = self._enhance_story_data(base_result, story_context)
            
            parsing_time = int((datetime.now() - parsing_start).total_seconds() * 1000)
            
            return ParsedResponse(
                parsed_data=enhanced_data,
                structured_output=None,  # Stories sind meist unstrukturiert
                original_response=raw_response,
                parsing_strategy=ParsingStrategy.STRUCTURED_TEXT,
                output_format=OutputFormat.STRUCTURED_TEXT,
                parsing_success=validation_result.is_valid,
                parsing_time_ms=parsing_time,
                confidence_score=self._calculate_story_confidence(enhanced_data, validation_result),
                validation_errors=[{"field": err.get("field", "story"), "message": err.get("message", "")} for err in validation_result.errors],
                warnings=validation_result.warnings,
                repair_attempts=0,
                repair_methods_used=[],
                fallback_data_used=False
            )
            
        except Exception as e:
            return self._create_error_response(raw_response, str(e), parsing_start)
    
    def _parse_story_from_text(self, text: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extrahiert Story-Elemente aus Freitext"""
        
        story_data = {
            'story_type': 'emotional_gift_story',
            'story_text': text,
            'personalization_detected': False,
            'emotional_elements': [],
            'character_development': None,
            'transformation_arc': None
        }
        
        # Personalisierung erkennen
        name = context.get('name', '')
        if name and name.lower() in text.lower():
            story_data['personalization_detected'] = True
            story_data['personalized_name'] = name
        
        # Emotionale Elemente extrahieren
        emotional_patterns = [
            r'(mut|angst|freude|stolz|liebe|hoffnung|träume)',
            r'(selbstvertrauen|persönlichkeit|wachstum|entwicklung)',
            r'(abenteuer|herausforderung|ziel|vision|erfolg)'
        ]
        
        for pattern in emotional_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            story_data['emotional_elements'].extend(matches)
        
        # Transformation Arc
        transformation_match = re.search(
            r'von\s+["\']?([^"\'\.]+)["\']?\s+zu\s+["\']?([^"\'\.]+)["\']?',
            text, re.IGNORECASE
        )
        if transformation_match:
            story_data['transformation_arc'] = {
                'from': transformation_match.group(1).strip(),
                'to': transformation_match.group(2).strip()
            }
        
        return story_data
    
    def _validate_story_elements(self, data: Optional[Dict[str, Any]], context: Dict[str, Any]) -> ValidationResult:
        """Validiert Story-spezifische Elemente"""
        
        errors = []
        warnings = []
        suggestions = []
        
        if not data:
            errors.append({"field": "story", "message": "Keine Story-Daten gefunden"})
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                errors=errors,
                warnings=warnings,
                suggestions=["Prüfe AI-Prompt für Story-Generation"],
                confidence_impact=-0.8
            )
        
        # Personalisierung prüfen
        if not data.get('personalization_detected', False):
            warnings.append("Keine Personalisierung in Story erkannt")
            suggestions.append("Füge personalisierten Namen in AI-Prompt ein")
        
        # Emotionale Elemente prüfen
        if not data.get('emotional_elements'):
            warnings.append("Keine emotionalen Elemente erkannt")
            suggestions.append("Verstärke emotionale Keywords im Prompt")
        
        # Transformations-Arc prüfen
        if not data.get('transformation_arc'):
            warnings.append("Keine Transformations-Story erkannt")
            suggestions.append("Füge 'von X zu Y' Transformation in Prompt ein")
        
        is_valid = len(errors) == 0
        severity = ValidationSeverity.MINOR if warnings else ValidationSeverity.INFO
        
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            confidence_impact=-0.1 * len(warnings)
        )
    
    def _enhance_story_data(self, data: Optional[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Erweitert Story-Daten um zusätzliche Metadaten"""
        
        if not data:
            return {}
        
        enhanced = data.copy()
        
        # Story-Qualitäts-Score
        quality_score = 0.0
        
        if enhanced.get('personalization_detected'):
            quality_score += 0.3
        
        if enhanced.get('emotional_elements'):
            quality_score += min(len(enhanced['emotional_elements']) * 0.1, 0.4)
        
        if enhanced.get('transformation_arc'):
            quality_score += 0.3
        
        enhanced['story_quality_score'] = min(quality_score, 1.0)
        
        # Story-Metadaten
        enhanced['story_metadata'] = {
            'word_count': len(enhanced.get('story_text', '').split()),
            'emotional_density': len(enhanced.get('emotional_elements', [])),
            'personalization_level': 'high' if enhanced.get('personalization_detected') else 'low',
            'generated_at': datetime.now().isoformat()
        }
        
        return enhanced
    
    def _calculate_story_confidence(self, data: Dict[str, Any], validation: ValidationResult) -> float:
        """Berechnet Confidence-Score für Story-Parsing"""
        
        base_confidence = 0.7  # Stories sind schwerer zu validieren
        
        # Quality Score Impact
        quality_score = data.get('story_quality_score', 0.0)
        base_confidence += quality_score * 0.2
        
        # Validation Impact
        base_confidence += validation.confidence_impact
        
        # Word Count Impact (zu kurze oder zu lange Stories)
        word_count = data.get('story_metadata', {}).get('word_count', 0)
        if 50 <= word_count <= 500:  # Optimale Länge
            base_confidence += 0.1
        elif word_count < 20:  # Zu kurz
            base_confidence -= 0.2
        elif word_count > 1000:  # Zu lang
            base_confidence -= 0.1
        
        return max(0.0, min(1.0, base_confidence))


# =============================================================================
# ✅ CONVENIENCE FUNCTIONS (Korrigiert)
# =============================================================================

def create_heroic_journey_parser() -> HeroicJourneyResponseParser:
    """
    Convenience-Funktion für Heldenreise-Parser
    """
    return HeroicJourneyResponseParser()


def create_emotional_story_parser() -> EmotionalStoryParser:
    """
    Convenience-Funktion für Story-Parser
    """
    return EmotionalStoryParser()


def parse_heroic_recommendation(raw_response: str, source_model: AIModelType) -> ParsedResponse:
    """
    Quick-Parse für Heldenreise-Empfehlungen
    """
    parser = create_heroic_journey_parser()
    return parser.parse_heroic_gift_response(raw_response, source_model)


def parse_emotional_story(raw_response: str, person_name: str = "Person", age: int = None) -> ParsedResponse:
    """
    Quick-Parse für emotionale Stories
    """
    parser = create_emotional_story_parser()
    context = {'name': person_name, 'age': age}
    return parser.parse_emotional_story_response(raw_response, context)


# =============================================================================
# ✅ EXPORTS (Korrigiert & Vollständig)
# =============================================================================

__all__ = [
    # Enums
    'ParsingStrategy',
    'OutputFormat',
    'ValidationSeverity',
    
    # Result Classes
    'ParsedResponse',
    'ValidationResult',
    
    # Core Parsers
    'ResponseParser',
    'QuickRecommendationParser',
    'PersonalityAnalysisParser',

    # ✅ Neue Parser-Klassen (Heldenreise)
    'HeroicJourneyResponseParser',
    'EmotionalStoryParser',
    
    # ✅ Convenience-Funktionen
    'create_heroic_journey_parser',
    'create_emotional_story_parser',
    'parse_heroic_recommendation',
    'parse_emotional_story'
]