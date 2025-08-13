"""
Internationalization and Localization Manager for Neuromorphic System

Provides comprehensive i18n support including message localization,
data format handling, cultural adaptations, and accessibility features.
"""

import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings

# Mock gettext for systems without it
try:
    import gettext
    GETTEXT_AVAILABLE = True
except ImportError:
    class MockGettextTranslation:
        def gettext(self, message):
            return message
        def ngettext(self, singular, plural, n):
            return singular if n == 1 else plural
    
    class MockGettext:
        def translation(self, domain, localedir=None, languages=None, fallback=False):
            return MockGettextTranslation()
        def bindtextdomain(self, domain, localedir):
            pass
        def textdomain(self, domain):
            pass
    
    gettext = MockGettext()
    GETTEXT_AVAILABLE = False


class SupportedLocale(Enum):
    """Supported locales for the neuromorphic system."""
    EN_US = "en_US"  # English (United States)
    EN_GB = "en_GB"  # English (United Kingdom)
    DE_DE = "de_DE"  # German (Germany)
    FR_FR = "fr_FR"  # French (France)
    ES_ES = "es_ES"  # Spanish (Spain)
    JA_JP = "ja_JP"  # Japanese (Japan)
    ZH_CN = "zh_CN"  # Chinese (Simplified)
    KO_KR = "ko_KR"  # Korean (South Korea)
    PT_BR = "pt_BR"  # Portuguese (Brazil)
    RU_RU = "ru_RU"  # Russian (Russia)


class CulturalContext(Enum):
    """Cultural context considerations."""
    WESTERN = "western"
    EASTERN = "eastern"
    ARABIC = "arabic"
    UNIVERSAL = "universal"


@dataclass
class LocaleConfig:
    """Configuration for a specific locale."""
    locale: SupportedLocale
    language_code: str
    country_code: str
    display_name: str
    currency: str
    date_format: str
    time_format: str
    number_format: str
    decimal_separator: str
    thousands_separator: str
    rtl: bool = False  # Right-to-left reading direction
    cultural_context: CulturalContext = CulturalContext.UNIVERSAL
    encoding: str = "utf-8"


@dataclass
class TranslationMessage:
    """A translatable message with metadata."""
    key: str
    default_text: str
    translated_text: str
    context: Optional[str] = None
    plural_forms: Optional[Dict[str, str]] = None
    variables: Optional[List[str]] = None
    last_updated: Optional[float] = None


class I18nManager:
    """Comprehensive internationalization manager."""
    
    def __init__(self, default_locale: SupportedLocale = SupportedLocale.EN_US,
                 translations_dir: Optional[Path] = None):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations_dir = translations_dir or Path(__file__).parent / "translations"
        
        # Thread-local storage for locale context
        self._local_data = threading.local()
        
        # Translation caches
        self.translation_cache: Dict[str, Dict[str, TranslationMessage]] = {}
        self.locale_configs: Dict[SupportedLocale, LocaleConfig] = {}
        
        # Message extraction patterns
        self.extraction_patterns = [
            r'_\(["\']([^"\']+)["\']\)',  # _("message")
            r'gettext\(["\']([^"\']+)["\']\)',  # gettext("message")
            r'ngettext\(["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*,\s*\d+\)',  # ngettext
        ]
        
        # Initialize locale configurations
        self._initialize_locale_configs()
        
        # Load translations
        self._load_translations()
        
        # Set up gettext if available
        if GETTEXT_AVAILABLE:
            self._setup_gettext()
    
    def _initialize_locale_configs(self):
        """Initialize locale configurations."""
        locale_data = {
            SupportedLocale.EN_US: LocaleConfig(
                locale=SupportedLocale.EN_US,
                language_code="en",
                country_code="US",
                display_name="English (United States)",
                currency="USD",
                date_format="%m/%d/%Y",
                time_format="%I:%M:%S %p",
                number_format="#,##0.##",
                decimal_separator=".",
                thousands_separator=",",
                cultural_context=CulturalContext.WESTERN
            ),
            SupportedLocale.EN_GB: LocaleConfig(
                locale=SupportedLocale.EN_GB,
                language_code="en",
                country_code="GB", 
                display_name="English (United Kingdom)",
                currency="GBP",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                number_format="#,##0.##",
                decimal_separator=".",
                thousands_separator=",",
                cultural_context=CulturalContext.WESTERN
            ),
            SupportedLocale.DE_DE: LocaleConfig(
                locale=SupportedLocale.DE_DE,
                language_code="de",
                country_code="DE",
                display_name="Deutsch (Deutschland)",
                currency="EUR",
                date_format="%d.%m.%Y",
                time_format="%H:%M:%S",
                number_format="#.##0,##",
                decimal_separator=",",
                thousands_separator=".",
                cultural_context=CulturalContext.WESTERN
            ),
            SupportedLocale.FR_FR: LocaleConfig(
                locale=SupportedLocale.FR_FR,
                language_code="fr",
                country_code="FR",
                display_name="Français (France)",
                currency="EUR", 
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                number_format="# ##0,##",
                decimal_separator=",",
                thousands_separator=" ",
                cultural_context=CulturalContext.WESTERN
            ),
            SupportedLocale.ES_ES: LocaleConfig(
                locale=SupportedLocale.ES_ES,
                language_code="es",
                country_code="ES",
                display_name="Español (España)",
                currency="EUR",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                number_format="#.##0,##",
                decimal_separator=",",
                thousands_separator=".",
                cultural_context=CulturalContext.WESTERN
            ),
            SupportedLocale.JA_JP: LocaleConfig(
                locale=SupportedLocale.JA_JP,
                language_code="ja",
                country_code="JP",
                display_name="日本語 (日本)",
                currency="JPY",
                date_format="%Y/%m/%d",
                time_format="%H:%M:%S",
                number_format="#,##0",
                decimal_separator=".",
                thousands_separator=",",
                cultural_context=CulturalContext.EASTERN
            ),
            SupportedLocale.ZH_CN: LocaleConfig(
                locale=SupportedLocale.ZH_CN,
                language_code="zh",
                country_code="CN",
                display_name="中文 (简体)",
                currency="CNY",
                date_format="%Y-%m-%d",
                time_format="%H:%M:%S",
                number_format="#,##0.##",
                decimal_separator=".",
                thousands_separator=",",
                cultural_context=CulturalContext.EASTERN
            ),
            SupportedLocale.KO_KR: LocaleConfig(
                locale=SupportedLocale.KO_KR,
                language_code="ko",
                country_code="KR",
                display_name="한국어 (대한민국)",
                currency="KRW",
                date_format="%Y. %m. %d.",
                time_format="%H:%M:%S",
                number_format="#,##0",
                decimal_separator=".",
                thousands_separator=",",
                cultural_context=CulturalContext.EASTERN
            ),
            SupportedLocale.PT_BR: LocaleConfig(
                locale=SupportedLocale.PT_BR,
                language_code="pt",
                country_code="BR",
                display_name="Português (Brasil)",
                currency="BRL",
                date_format="%d/%m/%Y",
                time_format="%H:%M:%S",
                number_format="#.##0,##",
                decimal_separator=",",
                thousands_separator=".",
                cultural_context=CulturalContext.WESTERN
            ),
            SupportedLocale.RU_RU: LocaleConfig(
                locale=SupportedLocale.RU_RU,
                language_code="ru",
                country_code="RU",
                display_name="Русский (Россия)",
                currency="RUB",
                date_format="%d.%m.%Y",
                time_format="%H:%M:%S",
                number_format="# ##0,##",
                decimal_separator=",",
                thousands_separator=" ",
                cultural_context=CulturalContext.EASTERN
            )
        }
        
        self.locale_configs.update(locale_data)
    
    def _load_translations(self):
        """Load translation files from the translations directory."""
        if not self.translations_dir.exists():
            self.translations_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_translations()
        
        for locale in SupportedLocale:
            translation_file = self.translations_dir / f"{locale.value}.json"
            
            if translation_file.exists():
                try:
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        translations_data = json.load(f)
                    
                    locale_translations = {}
                    for key, data in translations_data.items():
                        if isinstance(data, str):
                            # Simple string translation
                            locale_translations[key] = TranslationMessage(
                                key=key,
                                default_text=data,
                                translated_text=data,
                                last_updated=time.time()
                            )
                        elif isinstance(data, dict):
                            # Complex translation with metadata
                            locale_translations[key] = TranslationMessage(
                                key=key,
                                default_text=data.get("default", ""),
                                translated_text=data.get("translation", data.get("default", "")),
                                context=data.get("context"),
                                plural_forms=data.get("plural_forms"),
                                variables=data.get("variables", []),
                                last_updated=data.get("last_updated", time.time())
                            )
                    
                    self.translation_cache[locale.value] = locale_translations
                    
                except Exception as e:
                    print(f"Warning: Could not load translations for {locale.value}: {e}")
                    self.translation_cache[locale.value] = {}
            else:
                self.translation_cache[locale.value] = {}
    
    def _create_default_translations(self):
        """Create default translation files with sample content."""
        sample_translations = {
            "system.startup": {
                "default": "System starting up...",
                "context": "System initialization message"
            },
            "system.ready": {
                "default": "Neuromorphic system ready",
                "context": "System ready notification"
            },
            "error.general": {
                "default": "An error occurred",
                "context": "General error message"
            },
            "error.network": {
                "default": "Network connection failed",
                "context": "Network error message"
            },
            "model.loading": {
                "default": "Loading neuromorphic model...",
                "context": "Model loading status"
            },
            "model.training": {
                "default": "Training model",
                "context": "Model training status"
            },
            "data.processing": {
                "default": "Processing {count} samples",
                "variables": ["count"],
                "context": "Data processing status with variable"
            },
            "fusion.multimodal": {
                "default": "Multi-modal data fusion in progress",
                "context": "Multi-modal fusion status"
            },
            "performance.optimization": {
                "default": "Optimizing system performance",
                "context": "Performance optimization message"
            },
            "security.validation": {
                "default": "Validating security requirements",
                "context": "Security validation message"
            }
        }
        
        # Create translations for each supported locale
        locale_specific_translations = {
            SupportedLocale.DE_DE: {
                "system.startup": "System startet...",
                "system.ready": "Neuromorphes System bereit",
                "error.general": "Ein Fehler ist aufgetreten",
                "error.network": "Netzwerkverbindung fehlgeschlagen",
                "model.loading": "Neuromorphes Modell wird geladen...",
                "model.training": "Modell wird trainiert",
                "data.processing": "Verarbeitung von {count} Proben",
                "fusion.multimodal": "Multi-modale Datenfusion läuft",
                "performance.optimization": "Systemleistung wird optimiert",
                "security.validation": "Sicherheitsanforderungen werden validiert"
            },
            SupportedLocale.FR_FR: {
                "system.startup": "Démarrage du système...",
                "system.ready": "Système neuromorphique prêt",
                "error.general": "Une erreur s'est produite",
                "error.network": "Connexion réseau échouée",
                "model.loading": "Chargement du modèle neuromorphique...",
                "model.training": "Entraînement du modèle",
                "data.processing": "Traitement de {count} échantillons",
                "fusion.multimodal": "Fusion de données multi-modale en cours",
                "performance.optimization": "Optimisation des performances du système",
                "security.validation": "Validation des exigences de sécurité"
            },
            SupportedLocale.ES_ES: {
                "system.startup": "Sistema iniciándose...",
                "system.ready": "Sistema neuromorfico listo",
                "error.general": "Ha ocurrido un error",
                "error.network": "Conexión de red fallida",
                "model.loading": "Cargando modelo neuromorfico...",
                "model.training": "Entrenando modelo",
                "data.processing": "Procesando {count} muestras",
                "fusion.multimodal": "Fusión de datos multimodal en progreso",
                "performance.optimization": "Optimizando rendimiento del sistema",
                "security.validation": "Validando requisitos de seguridad"
            },
            SupportedLocale.JA_JP: {
                "system.startup": "システム起動中...",
                "system.ready": "ニューロモルフィックシステム準備完了",
                "error.general": "エラーが発生しました",
                "error.network": "ネットワーク接続に失敗しました",
                "model.loading": "ニューロモルフィックモデル読み込み中...",
                "model.training": "モデルトレーニング中",
                "data.processing": "{count}サンプルを処理中",
                "fusion.multimodal": "マルチモーダルデータフュージョン実行中",
                "performance.optimization": "システムパフォーマンス最適化中",
                "security.validation": "セキュリティ要件検証中"
            },
            SupportedLocale.ZH_CN: {
                "system.startup": "系统启动中...",
                "system.ready": "神经形态系统就绪",
                "error.general": "发生错误",
                "error.network": "网络连接失败",
                "model.loading": "正在加载神经形态模型...",
                "model.training": "正在训练模型",
                "data.processing": "正在处理 {count} 个样本",
                "fusion.multimodal": "多模态数据融合进行中",
                "performance.optimization": "正在优化系统性能",
                "security.validation": "正在验证安全要求"
            }
        }
        
        # Write English (default) translations
        en_translations = {}
        for key, data in sample_translations.items():
            en_translations[key] = data
        
        en_file = self.translations_dir / f"{SupportedLocale.EN_US.value}.json"
        with open(en_file, 'w', encoding='utf-8') as f:
            json.dump(en_translations, f, indent=2, ensure_ascii=False)
        
        # Write localized translations
        for locale, translations in locale_specific_translations.items():
            locale_translations = {}
            for key, base_data in sample_translations.items():
                locale_translations[key] = {
                    "default": base_data["default"],
                    "translation": translations.get(key, base_data["default"]),
                    "context": base_data.get("context"),
                    "variables": base_data.get("variables", [])
                }
            
            locale_file = self.translations_dir / f"{locale.value}.json"
            with open(locale_file, 'w', encoding='utf-8') as f:
                json.dump(locale_translations, f, indent=2, ensure_ascii=False)
    
    def _setup_gettext(self):
        """Set up gettext for standard i18n support."""
        try:
            locale_dir = str(self.translations_dir)
            gettext.bindtextdomain('neuromorphic', locale_dir)
            gettext.textdomain('neuromorphic')
        except Exception:
            pass  # Fallback to our custom implementation
    
    def set_locale(self, locale: Union[SupportedLocale, str]):
        """Set the current locale for translations."""
        if isinstance(locale, str):
            try:
                locale = SupportedLocale(locale)
            except ValueError:
                print(f"Warning: Unsupported locale '{locale}', falling back to default")
                locale = self.default_locale
        
        self.current_locale = locale
        
        # Set thread-local locale if in threaded context
        if hasattr(self._local_data, 'locale'):
            self._local_data.locale = locale
    
    def get_current_locale(self) -> SupportedLocale:
        """Get the current active locale."""
        # Check thread-local first
        if hasattr(self._local_data, 'locale'):
            return self._local_data.locale
        return self.current_locale
    
    def get_locale_config(self, locale: Optional[SupportedLocale] = None) -> LocaleConfig:
        """Get configuration for a specific locale."""
        if locale is None:
            locale = self.get_current_locale()
        return self.locale_configs.get(locale, self.locale_configs[self.default_locale])
    
    def translate(self, key: str, locale: Optional[SupportedLocale] = None, 
                  variables: Optional[Dict[str, Any]] = None,
                  default: Optional[str] = None) -> str:
        """
        Translate a message key to the specified locale.
        
        Args:
            key: Translation key
            locale: Target locale (uses current if not specified)
            variables: Variables for string formatting
            default: Default text if translation not found
            
        Returns:
            Translated text
        """
        if locale is None:
            locale = self.get_current_locale()
        
        locale_key = locale.value
        
        # Check cache
        if locale_key in self.translation_cache:
            if key in self.translation_cache[locale_key]:
                message = self.translation_cache[locale_key][key]
                text = message.translated_text
            else:
                text = default or key
        else:
            text = default or key
        
        # Apply variable substitution
        if variables:
            try:
                text = text.format(**variables)
            except (KeyError, ValueError) as e:
                print(f"Warning: Variable substitution failed for '{key}': {e}")
        
        return text
    
    def translate_plural(self, singular_key: str, plural_key: str, count: int,
                        locale: Optional[SupportedLocale] = None,
                        variables: Optional[Dict[str, Any]] = None) -> str:
        """
        Translate a message with plural forms.
        
        Args:
            singular_key: Key for singular form
            plural_key: Key for plural form  
            count: Count to determine plural form
            locale: Target locale
            variables: Variables for formatting
            
        Returns:
            Translated text in appropriate plural form
        """
        if locale is None:
            locale = self.get_current_locale()
        
        # Simple plural rule (English-like)
        key = singular_key if count == 1 else plural_key
        
        # Add count to variables
        if variables is None:
            variables = {}
        variables['count'] = count
        
        return self.translate(key, locale, variables)
    
    def format_number(self, number: Union[int, float], 
                     locale: Optional[SupportedLocale] = None) -> str:
        """Format a number according to locale conventions."""
        if locale is None:
            locale = self.get_current_locale()
        
        config = self.get_locale_config(locale)
        
        # Simple number formatting
        if isinstance(number, float):
            # Handle decimal numbers
            integer_part = int(number)
            decimal_part = number - integer_part
            
            # Format integer part with thousands separator
            integer_str = f"{integer_part:,}".replace(",", config.thousands_separator)
            
            if decimal_part > 0:
                decimal_str = f"{decimal_part:.2f}".split('.')[1]
                return f"{integer_str}{config.decimal_separator}{decimal_str}"
            else:
                return integer_str
        else:
            # Handle integers
            return f"{number:,}".replace(",", config.thousands_separator)
    
    def format_date(self, date_obj, locale: Optional[SupportedLocale] = None) -> str:
        """Format a date according to locale conventions."""
        if locale is None:
            locale = self.get_current_locale()
        
        config = self.get_locale_config(locale)
        
        try:
            return date_obj.strftime(config.date_format)
        except AttributeError:
            # Fallback for non-datetime objects
            return str(date_obj)
    
    def format_time(self, time_obj, locale: Optional[SupportedLocale] = None) -> str:
        """Format a time according to locale conventions."""
        if locale is None:
            locale = self.get_current_locale()
        
        config = self.get_locale_config(locale)
        
        try:
            return time_obj.strftime(config.time_format)
        except AttributeError:
            return str(time_obj)
    
    def format_currency(self, amount: Union[int, float], 
                       locale: Optional[SupportedLocale] = None) -> str:
        """Format currency according to locale conventions."""
        if locale is None:
            locale = self.get_current_locale()
        
        config = self.get_locale_config(locale)
        formatted_amount = self.format_number(amount, locale)
        
        # Simple currency formatting
        currency_symbols = {
            "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", 
            "CNY": "¥", "KRW": "₩", "BRL": "R$", "RUB": "₽"
        }
        
        symbol = currency_symbols.get(config.currency, config.currency)
        
        # Different currency placement by locale
        if config.cultural_context == CulturalContext.EASTERN:
            return f"{symbol}{formatted_amount}"
        else:
            return f"{symbol}{formatted_amount}"
    
    def get_supported_locales(self) -> List[SupportedLocale]:
        """Get list of all supported locales."""
        return list(SupportedLocale)
    
    def get_locale_display_names(self) -> Dict[str, str]:
        """Get display names for all supported locales."""
        return {
            locale.value: config.display_name
            for locale, config in self.locale_configs.items()
        }
    
    def extract_translatable_strings(self, source_code: str) -> List[str]:
        """Extract translatable strings from source code."""
        extractable_strings = []
        
        for pattern in self.extraction_patterns:
            matches = re.finditer(pattern, source_code, re.MULTILINE)
            for match in matches:
                extractable_strings.extend(match.groups())
        
        return list(set(extractable_strings))  # Remove duplicates
    
    def validate_translations(self) -> Dict[str, List[str]]:
        """Validate translation completeness across all locales."""
        validation_report = {}
        
        # Get all keys from default locale
        default_keys = set()
        if self.default_locale.value in self.translation_cache:
            default_keys = set(self.translation_cache[self.default_locale.value].keys())
        
        # Check each locale for missing translations
        for locale in SupportedLocale:
            if locale == self.default_locale:
                continue
                
            locale_key = locale.value
            missing_keys = []
            
            if locale_key in self.translation_cache:
                locale_keys = set(self.translation_cache[locale_key].keys())
                missing_keys = list(default_keys - locale_keys)
            else:
                missing_keys = list(default_keys)
            
            if missing_keys:
                validation_report[locale_key] = missing_keys
        
        return validation_report
    
    def export_translations(self, format: str = "json") -> Dict[str, Any]:
        """Export all translations in specified format."""
        if format.lower() == "json":
            export_data = {}
            
            for locale in SupportedLocale:
                locale_key = locale.value
                if locale_key in self.translation_cache:
                    locale_data = {}
                    for key, message in self.translation_cache[locale_key].items():
                        locale_data[key] = {
                            "default": message.default_text,
                            "translation": message.translated_text,
                            "context": message.context,
                            "variables": message.variables,
                            "last_updated": message.last_updated
                        }
                    export_data[locale_key] = locale_data
            
            return export_data
        
        raise ValueError(f"Unsupported export format: {format}")
    
    def get_cultural_context(self, locale: Optional[SupportedLocale] = None) -> CulturalContext:
        """Get cultural context for a locale."""
        if locale is None:
            locale = self.get_current_locale()
        
        config = self.get_locale_config(locale)
        return config.cultural_context
    
    def is_rtl_locale(self, locale: Optional[SupportedLocale] = None) -> bool:
        """Check if locale uses right-to-left reading direction."""
        if locale is None:
            locale = self.get_current_locale()
        
        config = self.get_locale_config(locale)
        return config.rtl


# Convenience functions
_default_i18n_manager = None

def get_i18n_manager() -> I18nManager:
    """Get the default i18n manager instance."""
    global _default_i18n_manager
    if _default_i18n_manager is None:
        _default_i18n_manager = I18nManager()
    return _default_i18n_manager

def _(key: str, **kwargs) -> str:
    """Convenient translation function."""
    return get_i18n_manager().translate(key, **kwargs)

def set_locale(locale: Union[SupportedLocale, str]):
    """Set the global locale."""
    get_i18n_manager().set_locale(locale)

def format_number(number: Union[int, float], locale: Optional[SupportedLocale] = None) -> str:
    """Format number with current locale."""
    return get_i18n_manager().format_number(number, locale)


# Example usage and testing
if __name__ == "__main__":
    print("🌐 Testing Internationalization Manager")
    print("=" * 50)
    
    # Create i18n manager
    i18n = I18nManager()
    
    # Test basic translation
    print("\n1. Testing Basic Translation:")
    print(f"EN: {i18n.translate('system.ready')}")
    
    i18n.set_locale(SupportedLocale.DE_DE)
    print(f"DE: {i18n.translate('system.ready')}")
    
    i18n.set_locale(SupportedLocale.JA_JP)
    print(f"JA: {i18n.translate('system.ready')}")
    
    i18n.set_locale(SupportedLocale.ZH_CN)
    print(f"ZH: {i18n.translate('system.ready')}")
    
    # Test variable substitution
    print("\n2. Testing Variable Substitution:")
    i18n.set_locale(SupportedLocale.EN_US)
    print(f"EN: {i18n.translate('data.processing', variables={'count': 1000})}")
    
    i18n.set_locale(SupportedLocale.DE_DE)
    print(f"DE: {i18n.translate('data.processing', variables={'count': 1000})}")
    
    # Test number formatting
    print("\n3. Testing Number Formatting:")
    test_number = 1234567.89
    
    for locale in [SupportedLocale.EN_US, SupportedLocale.DE_DE, SupportedLocale.FR_FR]:
        i18n.set_locale(locale)
        formatted = i18n.format_number(test_number)
        print(f"{locale.value}: {formatted}")
    
    # Test currency formatting
    print("\n4. Testing Currency Formatting:")
    test_amount = 1234.56
    
    for locale in [SupportedLocale.EN_US, SupportedLocale.DE_DE, SupportedLocale.JA_JP]:
        i18n.set_locale(locale)
        formatted = i18n.format_currency(test_amount)
        print(f"{locale.value}: {formatted}")
    
    # Test supported locales
    print("\n5. Supported Locales:")
    display_names = i18n.get_locale_display_names()
    for locale_code, display_name in display_names.items():
        print(f"  {locale_code}: {display_name}")
    
    # Test translation validation
    print("\n6. Translation Validation:")
    validation_report = i18n.validate_translations()
    if validation_report:
        print("Missing translations found:")
        for locale, missing_keys in validation_report.items():
            print(f"  {locale}: {len(missing_keys)} missing keys")
    else:
        print("All translations complete!")
    
    print("\n✅ Internationalization testing completed!")