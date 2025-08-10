"""
Internationalization (i18n) and Localization for SNN-Fusion

Comprehensive internationalization support with dynamic language switching,
cultural adaptation, and accessibility features.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
import locale
import gettext
from babel import Locale, dates, numbers
from babel.support import Translations
import threading


class SupportedLanguage(Enum):
    """Supported languages with their codes."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh_CN"
    CHINESE_TRADITIONAL = "zh_TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ITALIAN = "it"
    DUTCH = "nl"
    SWEDISH = "sv"
    HINDI = "hi"
    ARABIC = "ar"


class DateTimeFormat(Enum):
    """Date and time formatting styles."""
    SHORT = "short"
    MEDIUM = "medium" 
    LONG = "long"
    FULL = "full"


class NumberFormat(Enum):
    """Number formatting styles."""
    DECIMAL = "decimal"
    CURRENCY = "currency"
    PERCENT = "percent"
    SCIENTIFIC = "scientific"


@dataclass
class LocalizationConfig:
    """Configuration for localization settings."""
    language: SupportedLanguage = SupportedLanguage.ENGLISH
    country_code: Optional[str] = None
    timezone: str = "UTC"
    currency_code: str = "USD"
    date_format: DateTimeFormat = DateTimeFormat.MEDIUM
    time_format: DateTimeFormat = DateTimeFormat.MEDIUM
    number_format: NumberFormat = NumberFormat.DECIMAL
    use_24hour_time: bool = True
    decimal_separator: str = "."
    thousand_separator: str = ","
    rtl_language: bool = False  # Right-to-left languages
    
    def to_locale_string(self) -> str:
        """Convert to locale string format."""
        if self.country_code:
            return f"{self.language.value}_{self.country_code}"
        return self.language.value


class InternationalizationManager:
    """
    Comprehensive internationalization manager.
    
    Handles language switching, message translation, cultural formatting,
    and accessibility features for global deployments.
    """
    
    def __init__(self, translations_dir: Optional[str] = None):
        """
        Initialize i18n manager.
        
        Args:
            translations_dir: Directory containing translation files
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.current_config = LocalizationConfig()
        self.translations_dir = Path(translations_dir) if translations_dir else Path(__file__).parent / "translations"
        
        # Translation management
        self.translations: Dict[str, Translations] = {}
        self.fallback_translations: Optional[Translations] = None
        self.message_cache: Dict[str, Dict[str, str]] = {}
        
        # Thread-local storage for per-request localization
        self.local_storage = threading.local()
        
        # Cultural formatters
        self.date_formatters: Dict[str, Any] = {}
        self.number_formatters: Dict[str, Any] = {}
        
        # Dynamic content handlers
        self.dynamic_translators: Dict[str, Callable[[str, str], str]] = {}
        
        # Initialize default translations
        self._initialize_translations()
        self._setup_formatters()
        
        self.logger.info("InternationalizationManager initialized")
    
    def set_language(self, language: SupportedLanguage, country_code: Optional[str] = None):
        """
        Set the current language.
        
        Args:
            language: Target language
            country_code: Optional country code for regional variations
        """
        old_language = self.current_config.language
        
        self.current_config.language = language
        self.current_config.country_code = country_code
        
        # Update RTL flag for right-to-left languages
        self.current_config.rtl_language = language in [
            SupportedLanguage.ARABIC, 
            # Add Hebrew when supported
        ]
        
        # Load translations for new language
        self._load_language_translations(language)
        
        # Update formatters
        self._update_formatters()
        
        self.logger.info(f"Language changed from {old_language.value} to {language.value}")
    
    def set_config(self, config: LocalizationConfig):
        """Set complete localization configuration."""
        old_config = self.current_config
        self.current_config = config
        
        # Reload translations if language changed
        if old_config.language != config.language:
            self._load_language_translations(config.language)
        
        # Update formatters
        self._update_formatters()
        
        self.logger.info(f"Localization config updated: {config.to_locale_string()}")
    
    def get_current_config(self) -> LocalizationConfig:
        """Get current localization configuration."""
        return self.current_config
    
    def translate(self, message_key: str, **kwargs) -> str:
        """
        Translate a message key to current language.
        
        Args:
            message_key: Key for the message to translate
            **kwargs: Variables for message formatting
            
        Returns:
            Translated and formatted message
        """
        # Check thread-local context first
        local_config = getattr(self.local_storage, 'config', None)
        config = local_config or self.current_config
        
        # Check cache first
        cache_key = f"{config.language.value}:{message_key}"
        if cache_key in self.message_cache.get(config.language.value, {}):
            template = self.message_cache[config.language.value][cache_key]
        else:
            # Get translation
            translations = self.translations.get(config.language.value)
            if translations:
                template = translations.gettext(message_key)
            else:
                # Fallback to English or message key
                template = self.fallback_translations.gettext(message_key) if self.fallback_translations else message_key
            
            # Cache the result
            if config.language.value not in self.message_cache:
                self.message_cache[config.language.value] = {}
            self.message_cache[config.language.value][cache_key] = template
        
        # Handle dynamic translation
        if message_key in self.dynamic_translators:
            template = self.dynamic_translators[message_key](template, config.language.value)
        
        # Format with variables
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError) as e:
            self.logger.warning(f"Failed to format message '{message_key}': {e}")
            return template
    
    def format_datetime(
        self, 
        dt: datetime, 
        date_format: Optional[DateTimeFormat] = None,
        time_format: Optional[DateTimeFormat] = None,
        include_timezone: bool = False
    ) -> str:
        """
        Format datetime according to current locale.
        
        Args:
            dt: Datetime to format
            date_format: Date formatting style
            time_format: Time formatting style
            include_timezone: Whether to include timezone info
            
        Returns:
            Formatted datetime string
        """
        config = getattr(self.local_storage, 'config', None) or self.current_config
        locale_obj = Locale(config.language.value)
        
        # Use provided formats or defaults
        df = date_format or config.date_format
        tf = time_format or config.time_format
        
        # Convert to local timezone if needed
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        try:
            # Format date and time separately
            date_str = dates.format_date(dt, format=df.value, locale=locale_obj)
            time_str = dates.format_time(dt, format=tf.value, locale=locale_obj)
            
            # Combine date and time
            if include_timezone:
                tz_str = dates.format_timezone(dt.tzinfo, locale=locale_obj)
                return f"{date_str} {time_str} {tz_str}"
            else:
                return f"{date_str} {time_str}"
                
        except Exception as e:
            self.logger.error(f"Failed to format datetime: {e}")
            return dt.isoformat()
    
    def format_number(
        self,
        number: float,
        format_type: Optional[NumberFormat] = None,
        currency: Optional[str] = None
    ) -> str:
        """
        Format number according to current locale.
        
        Args:
            number: Number to format
            format_type: Number formatting style
            currency: Currency code for currency formatting
            
        Returns:
            Formatted number string
        """
        config = getattr(self.local_storage, 'config', None) or self.current_config
        locale_obj = Locale(config.language.value)
        
        fmt = format_type or config.number_format
        
        try:
            if fmt == NumberFormat.DECIMAL:
                return numbers.format_decimal(number, locale=locale_obj)
            elif fmt == NumberFormat.CURRENCY:
                currency_code = currency or config.currency_code
                return numbers.format_currency(number, currency_code, locale=locale_obj)
            elif fmt == NumberFormat.PERCENT:
                return numbers.format_percent(number, locale=locale_obj)
            elif fmt == NumberFormat.SCIENTIFIC:
                return numbers.format_scientific(number, locale=locale_obj)
            else:
                return str(number)
                
        except Exception as e:
            self.logger.error(f"Failed to format number: {e}")
            return str(number)
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with metadata."""
        languages = []
        
        for lang in SupportedLanguage:
            try:
                locale_obj = Locale(lang.value)
                languages.append({
                    'code': lang.value,
                    'name': lang.name,
                    'display_name': locale_obj.display_name,
                    'english_name': locale_obj.english_name,
                    'rtl': lang in [SupportedLanguage.ARABIC]
                })
            except Exception as e:
                self.logger.warning(f"Failed to get info for language {lang.value}: {e}")
        
        return languages
    
    def with_language_context(self, language: SupportedLanguage, country_code: Optional[str] = None):
        """
        Context manager for temporary language switching.
        
        Args:
            language: Temporary language to use
            country_code: Optional country code
            
        Usage:
            with i18n.with_language_context(SupportedLanguage.SPANISH):
                message = i18n.translate("welcome_message")
        """
        return _LanguageContext(self, language, country_code)
    
    def register_dynamic_translator(self, message_key: str, translator_func: Callable[[str, str], str]):
        """
        Register a dynamic translator for complex translations.
        
        Args:
            message_key: Message key to handle dynamically
            translator_func: Function that takes (template, language) and returns translated text
        """
        self.dynamic_translators[message_key] = translator_func
        self.logger.debug(f"Registered dynamic translator for '{message_key}'")
    
    def add_custom_translation(self, language: SupportedLanguage, message_key: str, translation: str):
        """
        Add custom translation at runtime.
        
        Args:
            language: Target language
            message_key: Message key
            translation: Translated text
        """
        lang_code = language.value
        
        if lang_code not in self.message_cache:
            self.message_cache[lang_code] = {}
        
        cache_key = f"{lang_code}:{message_key}"
        self.message_cache[lang_code][cache_key] = translation
        
        self.logger.debug(f"Added custom translation for '{message_key}' in {lang_code}")
    
    def get_translation_coverage(self) -> Dict[str, Dict[str, Any]]:
        """Get translation coverage statistics."""
        coverage = {}
        
        # Get all message keys from English (base language)
        english_translations = self.translations.get(SupportedLanguage.ENGLISH.value)
        if not english_translations:
            return {}
        
        # Count translations for each language
        for lang in SupportedLanguage:
            lang_code = lang.value
            translations = self.translations.get(lang_code)
            
            if translations:
                # This is simplified - in practice you'd iterate through the catalog
                coverage[lang_code] = {
                    'language': lang.name,
                    'total_messages': 100,  # Placeholder
                    'translated_messages': 95,  # Placeholder  
                    'coverage_percentage': 95.0,
                    'missing_keys': ['example_missing_key']  # Placeholder
                }
        
        return coverage
    
    def _initialize_translations(self):
        """Initialize translation files."""
        # Create default translations structure
        self._create_default_translations()
        
        # Load English as fallback
        try:
            self._load_language_translations(SupportedLanguage.ENGLISH)
            self.fallback_translations = self.translations.get(SupportedLanguage.ENGLISH.value)
        except Exception as e:
            self.logger.error(f"Failed to load fallback translations: {e}")
    
    def _create_default_translations(self):
        """Create default translation files if they don't exist."""
        if not self.translations_dir.exists():
            self.translations_dir.mkdir(parents=True, exist_ok=True)
        
        # Default messages in multiple languages
        default_messages = {
            SupportedLanguage.ENGLISH.value: {
                "welcome_message": "Welcome to SNN-Fusion",
                "processing_data": "Processing neural data...",
                "training_complete": "Training completed successfully",
                "error_occurred": "An error occurred: {error}",
                "model_accuracy": "Model accuracy: {accuracy:.2%}",
                "data_loaded": "Loaded {count} data samples",
                "system_status": "System Status: {status}",
                "memory_usage": "Memory usage: {usage_mb:.1f} MB",
                "performance_metrics": "Performance: {ops_per_sec:.0f} ops/sec"
            },
            SupportedLanguage.SPANISH.value: {
                "welcome_message": "Bienvenido a SNN-Fusion",
                "processing_data": "Procesando datos neuronales...",
                "training_complete": "Entrenamiento completado exitosamente",
                "error_occurred": "Ocurrió un error: {error}",
                "model_accuracy": "Precisión del modelo: {accuracy:.2%}",
                "data_loaded": "Cargadas {count} muestras de datos",
                "system_status": "Estado del Sistema: {status}",
                "memory_usage": "Uso de memoria: {usage_mb:.1f} MB",
                "performance_metrics": "Rendimiento: {ops_per_sec:.0f} ops/seg"
            },
            SupportedLanguage.FRENCH.value: {
                "welcome_message": "Bienvenue dans SNN-Fusion",
                "processing_data": "Traitement des données neuronales...",
                "training_complete": "Entraînement terminé avec succès",
                "error_occurred": "Une erreur s'est produite: {error}",
                "model_accuracy": "Précision du modèle: {accuracy:.2%}",
                "data_loaded": "Chargé {count} échantillons de données",
                "system_status": "État du Système: {status}",
                "memory_usage": "Utilisation mémoire: {usage_mb:.1f} MB",
                "performance_metrics": "Performance: {ops_per_sec:.0f} ops/sec"
            },
            SupportedLanguage.GERMAN.value: {
                "welcome_message": "Willkommen bei SNN-Fusion",
                "processing_data": "Verarbeitung neuraler Daten...",
                "training_complete": "Training erfolgreich abgeschlossen",
                "error_occurred": "Ein Fehler ist aufgetreten: {error}",
                "model_accuracy": "Modellgenauigkeit: {accuracy:.2%}",
                "data_loaded": "{count} Datenproben geladen",
                "system_status": "Systemstatus: {status}",
                "memory_usage": "Speicherverbrauch: {usage_mb:.1f} MB",
                "performance_metrics": "Leistung: {ops_per_sec:.0f} Ops/Sek"
            },
            SupportedLanguage.JAPANESE.value: {
                "welcome_message": "SNN-Fusionへようこそ",
                "processing_data": "ニューラルデータを処理中...",
                "training_complete": "トレーニングが正常に完了しました",
                "error_occurred": "エラーが発生しました: {error}",
                "model_accuracy": "モデル精度: {accuracy:.2%}",
                "data_loaded": "{count}個のデータサンプルを読み込みました",
                "system_status": "システムステータス: {status}",
                "memory_usage": "メモリ使用量: {usage_mb:.1f} MB",
                "performance_metrics": "パフォーマンス: {ops_per_sec:.0f} ops/秒"
            },
            SupportedLanguage.CHINESE_SIMPLIFIED.value: {
                "welcome_message": "欢迎使用SNN-Fusion",
                "processing_data": "正在处理神经数据...",
                "training_complete": "训练成功完成",
                "error_occurred": "发生错误: {error}",
                "model_accuracy": "模型精度: {accuracy:.2%}",
                "data_loaded": "已加载{count}个数据样本",
                "system_status": "系统状态: {status}",
                "memory_usage": "内存使用: {usage_mb:.1f} MB",
                "performance_metrics": "性能: {ops_per_sec:.0f} 操作/秒"
            }
        }
        
        # Write translation files
        for lang_code, messages in default_messages.items():
            lang_file = self.translations_dir / f"{lang_code}.json"
            if not lang_file.exists():
                with open(lang_file, 'w', encoding='utf-8') as f:
                    json.dump(messages, f, ensure_ascii=False, indent=2)
                self.logger.debug(f"Created default translations for {lang_code}")
    
    def _load_language_translations(self, language: SupportedLanguage):
        """Load translations for a specific language."""
        lang_code = language.value
        lang_file = self.translations_dir / f"{lang_code}.json"
        
        if lang_file.exists():
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
                
                # Create a simple translations object
                self.translations[lang_code] = SimpleTranslations(messages)
                self.logger.debug(f"Loaded translations for {lang_code}")
                
            except Exception as e:
                self.logger.error(f"Failed to load translations for {lang_code}: {e}")
        else:
            self.logger.warning(f"Translation file not found for {lang_code}")
    
    def _setup_formatters(self):
        """Setup cultural formatters."""
        for lang in SupportedLanguage:
            try:
                locale_obj = Locale(lang.value)
                self.date_formatters[lang.value] = locale_obj
                self.number_formatters[lang.value] = locale_obj
            except Exception as e:
                self.logger.warning(f"Failed to setup formatter for {lang.value}: {e}")
    
    def _update_formatters(self):
        """Update formatters for current configuration."""
        # This method can be used to update formatters when configuration changes
        pass


class SimpleTranslations:
    """Simple translations implementation."""
    
    def __init__(self, messages: Dict[str, str]):
        """Initialize with messages dictionary."""
        self.messages = messages
    
    def gettext(self, message: str) -> str:
        """Get translated message."""
        return self.messages.get(message, message)


class _LanguageContext:
    """Context manager for temporary language switching."""
    
    def __init__(self, i18n_manager: InternationalizationManager, 
                 language: SupportedLanguage, country_code: Optional[str]):
        """Initialize language context."""
        self.i18n_manager = i18n_manager
        self.temp_config = LocalizationConfig(
            language=language,
            country_code=country_code
        )
        self.original_config = None
    
    def __enter__(self):
        """Enter context - set temporary language."""
        # Store original config
        self.original_config = getattr(self.i18n_manager.local_storage, 'config', None)
        
        # Set temporary config
        self.i18n_manager.local_storage.config = self.temp_config
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original language."""
        if self.original_config:
            self.i18n_manager.local_storage.config = self.original_config
        else:
            delattr(self.i18n_manager.local_storage, 'config')


# Example usage and testing
if __name__ == "__main__":
    print("Testing Internationalization System...")
    
    # Create i18n manager
    i18n = InternationalizationManager()
    
    print("\n1. Testing basic translation:")
    
    # Test translations in different languages
    languages_to_test = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.SPANISH,
        SupportedLanguage.FRENCH,
        SupportedLanguage.GERMAN,
        SupportedLanguage.JAPANESE,
        SupportedLanguage.CHINESE_SIMPLIFIED
    ]
    
    for lang in languages_to_test:
        i18n.set_language(lang)
        welcome_msg = i18n.translate("welcome_message")
        print(f"  {lang.value}: {welcome_msg}")
    
    print("\n2. Testing message formatting:")
    i18n.set_language(SupportedLanguage.ENGLISH)
    
    # Test formatted messages
    error_msg = i18n.translate("error_occurred", error="Network timeout")
    accuracy_msg = i18n.translate("model_accuracy", accuracy=0.9567)
    data_msg = i18n.translate("data_loaded", count=1000)
    
    print(f"  Error message: {error_msg}")
    print(f"  Accuracy message: {accuracy_msg}")
    print(f"  Data message: {data_msg}")
    
    print("\n3. Testing datetime formatting:")
    now = datetime.now()
    
    for lang in [SupportedLanguage.ENGLISH, SupportedLanguage.GERMAN, SupportedLanguage.JAPANESE]:
        i18n.set_language(lang)
        formatted_date = i18n.format_datetime(now, DateTimeFormat.LONG, DateTimeFormat.SHORT)
        print(f"  {lang.value}: {formatted_date}")
    
    print("\n4. Testing number formatting:")
    test_number = 1234567.89
    
    for lang in [SupportedLanguage.ENGLISH, SupportedLanguage.GERMAN, SupportedLanguage.FRENCH]:
        i18n.set_language(lang)
        formatted_number = i18n.format_number(test_number)
        formatted_currency = i18n.format_number(test_number, NumberFormat.CURRENCY, "EUR")
        formatted_percent = i18n.format_number(0.1234, NumberFormat.PERCENT)
        
        print(f"  {lang.value}:")
        print(f"    Number: {formatted_number}")
        print(f"    Currency: {formatted_currency}")
        print(f"    Percent: {formatted_percent}")
    
    print("\n5. Testing context manager:")
    i18n.set_language(SupportedLanguage.ENGLISH)
    
    print(f"  Outside context: {i18n.translate('welcome_message')}")
    
    with i18n.with_language_context(SupportedLanguage.SPANISH):
        print(f"  Inside Spanish context: {i18n.translate('welcome_message')}")
        
        with i18n.with_language_context(SupportedLanguage.FRENCH):
            print(f"  Inside French context: {i18n.translate('welcome_message')}")
    
    print(f"  Back outside context: {i18n.translate('welcome_message')}")
    
    print("\n6. Testing supported languages:")
    supported = i18n.get_supported_languages()
    print("  Supported languages:")
    for lang_info in supported[:5]:  # Show first 5
        print(f"    {lang_info['code']}: {lang_info['display_name']} ({lang_info['english_name']})")
    
    print("\n✓ Internationalization test completed!")