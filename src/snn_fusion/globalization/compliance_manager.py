"""
Global Compliance Manager for Neuromorphic System

Provides comprehensive compliance management for global regulations including
GDPR, CCPA, LGPD, PIPL, and other privacy/data protection laws worldwide.
"""

import json
import time
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings


class ComplianceRegulation(Enum):
    """Supported compliance regulations."""
    GDPR = "gdpr"           # General Data Protection Regulation (EU)
    CCPA = "ccpa"           # California Consumer Privacy Act (US)
    LGPD = "lgpd"           # Lei Geral de Proteção de Dados (Brazil)
    PIPL = "pipl"           # Personal Information Protection Law (China)
    PIPEDA = "pipeda"       # Personal Information Protection and Electronic Documents Act (Canada)
    PDPA_SG = "pdpa_sg"     # Personal Data Protection Act (Singapore)
    PDPA_TH = "pdpa_th"     # Personal Data Protection Act (Thailand)
    FADP = "fadp"           # Federal Act on Data Protection (Switzerland)
    KVKK = "kvkk"           # Kişisel Verilerin Korunması Kanunu (Turkey)
    DPA_UK = "dpa_uk"       # Data Protection Act (United Kingdom)


class DataCategory(Enum):
    """Categories of data for compliance purposes."""
    PERSONAL_IDENTIFIABLE = "pii"           # Personally Identifiable Information
    SENSITIVE_PERSONAL = "sensitive_pii"     # Sensitive PII (health, biometric, etc.)
    BEHAVIORAL = "behavioral"                # Behavioral data
    TECHNICAL = "technical"                  # Technical/system data
    BIOMETRIC = "biometric"                  # Biometric data (neuromorphic patterns)
    HEALTH = "health"                        # Health-related data
    FINANCIAL = "financial"                  # Financial information
    LOCATION = "location"                    # Location data
    COMMUNICATION = "communication"          # Communication records
    PREFERENCES = "preferences"              # User preferences


class ConsentType(Enum):
    """Types of consent for data processing."""
    EXPLICIT = "explicit"                    # Explicit consent required
    IMPLICIT = "implicit"                    # Implicit consent acceptable
    LEGITIMATE_INTEREST = "legitimate_interest"  # Legitimate interest basis
    CONTRACT = "contract"                    # Contractual necessity
    LEGAL_OBLIGATION = "legal_obligation"    # Legal obligation
    VITAL_INTERESTS = "vital_interests"      # Vital interests protection


class DataProcessingPurpose(Enum):
    """Purposes for data processing."""
    NEUROMORPHIC_TRAINING = "neuromorphic_training"    # AI/ML model training
    SYSTEM_OPTIMIZATION = "system_optimization"        # Performance optimization
    USER_PERSONALIZATION = "user_personalization"     # User experience personalization
    SECURITY_MONITORING = "security_monitoring"       # Security and fraud detection
    ANALYTICS = "analytics"                            # Analytics and insights
    RESEARCH = "research"                              # Research purposes
    SERVICE_DELIVERY = "service_delivery"              # Core service delivery
    MARKETING = "marketing"                            # Marketing communications
    COMPLIANCE = "compliance"                          # Compliance monitoring


@dataclass
class ComplianceRule:
    """A compliance rule for a specific regulation."""
    regulation: ComplianceRegulation
    rule_id: str
    title: str
    description: str
    data_categories: List[DataCategory]
    required_consent: ConsentType
    processing_purposes: List[DataProcessingPurpose]
    retention_period_days: Optional[int] = None
    cross_border_transfer_allowed: bool = False
    special_requirements: List[str] = field(default_factory=list)
    penalties: Dict[str, str] = field(default_factory=dict)


@dataclass
class DataSubject:
    """Represents a data subject for compliance tracking."""
    subject_id: str
    jurisdiction: str
    applicable_regulations: List[ComplianceRegulation]
    consent_records: Dict[str, Any] = field(default_factory=dict)
    data_categories: List[DataCategory] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


@dataclass
class ConsentRecord:
    """Records consent given by a data subject."""
    consent_id: str
    subject_id: str
    purpose: DataProcessingPurpose
    consent_type: ConsentType
    granted: bool
    timestamp: float
    expiry_timestamp: Optional[float] = None
    withdrawal_timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceViolation:
    """Represents a potential compliance violation."""
    violation_id: str
    regulation: ComplianceRegulation
    rule_id: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    detected_at: float
    affected_subjects: List[str]
    remediation_required: List[str]
    status: str = "open"  # "open", "investigating", "resolved", "false_positive"


class GlobalComplianceManager:
    """Comprehensive global compliance management system."""
    
    def __init__(self, compliance_data_dir: Optional[Path] = None):
        self.compliance_data_dir = compliance_data_dir or Path(__file__).parent / "compliance_data"
        self.compliance_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.data_subjects: Dict[str, DataSubject] = {}
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.violations: Dict[str, ComplianceViolation] = {}
        
        # Configuration
        self.retention_schedules: Dict[DataCategory, int] = {}  # days
        self.cross_border_transfer_rules: Dict[ComplianceRegulation, List[str]] = {}
        
        # Initialize compliance framework
        self._initialize_compliance_rules()
        self._initialize_retention_schedules()
        self._initialize_transfer_rules()
        
        # Load existing data
        self._load_compliance_data()
    
    def _initialize_compliance_rules(self):
        """Initialize comprehensive compliance rules for supported regulations."""
        
        # GDPR Rules (EU)
        gdpr_rules = [
            ComplianceRule(
                regulation=ComplianceRegulation.GDPR,
                rule_id="gdpr_consent",
                title="Explicit Consent Required",
                description="Processing of personal data requires explicit consent",
                data_categories=[DataCategory.PERSONAL_IDENTIFIABLE, DataCategory.SENSITIVE_PERSONAL],
                required_consent=ConsentType.EXPLICIT,
                processing_purposes=[DataProcessingPurpose.NEUROMORPHIC_TRAINING, DataProcessingPurpose.USER_PERSONALIZATION],
                retention_period_days=365,
                cross_border_transfer_allowed=False,
                special_requirements=["right_to_erasure", "data_portability", "privacy_by_design"],
                penalties={"max_fine": "4% of annual revenue or €20M"}
            ),
            ComplianceRule(
                regulation=ComplianceRegulation.GDPR,
                rule_id="gdpr_biometric",
                title="Biometric Data Protection",
                description="Biometric data requires special category protections",
                data_categories=[DataCategory.BIOMETRIC],
                required_consent=ConsentType.EXPLICIT,
                processing_purposes=[DataProcessingPurpose.NEUROMORPHIC_TRAINING],
                retention_period_days=180,  # Shorter retention for sensitive data
                cross_border_transfer_allowed=False,
                special_requirements=["encryption_at_rest", "pseudonymization", "access_logging"],
                penalties={"max_fine": "4% of annual revenue or €20M"}
            ),
            ComplianceRule(
                regulation=ComplianceRegulation.GDPR,
                rule_id="gdpr_legitimate_interest",
                title="Legitimate Interest Processing",
                description="Processing based on legitimate interests with balancing test",
                data_categories=[DataCategory.TECHNICAL, DataCategory.BEHAVIORAL],
                required_consent=ConsentType.LEGITIMATE_INTEREST,
                processing_purposes=[DataProcessingPurpose.SYSTEM_OPTIMIZATION, DataProcessingPurpose.SECURITY_MONITORING],
                retention_period_days=730,
                special_requirements=["balancing_test", "opt_out_mechanism"],
                penalties={"max_fine": "4% of annual revenue or €20M"}
            )
        ]
        
        # CCPA Rules (California)
        ccpa_rules = [
            ComplianceRule(
                regulation=ComplianceRegulation.CCPA,
                rule_id="ccpa_sale_opt_out",
                title="Right to Opt-Out of Sale",
                description="Consumers have right to opt out of personal data sale",
                data_categories=[DataCategory.PERSONAL_IDENTIFIABLE, DataCategory.BEHAVIORAL],
                required_consent=ConsentType.IMPLICIT,
                processing_purposes=[DataProcessingPurpose.MARKETING, DataProcessingPurpose.ANALYTICS],
                retention_period_days=1095,  # 3 years
                cross_border_transfer_allowed=True,
                special_requirements=["opt_out_mechanism", "do_not_sell_link"],
                penalties={"max_fine": "$7,500 per violation"}
            ),
            ComplianceRule(
                regulation=ComplianceRegulation.CCPA,
                rule_id="ccpa_data_deletion",
                title="Right to Delete Personal Information",
                description="Consumers have right to request deletion of personal data",
                data_categories=[DataCategory.PERSONAL_IDENTIFIABLE, DataCategory.SENSITIVE_PERSONAL],
                required_consent=ConsentType.IMPLICIT,
                processing_purposes=list(DataProcessingPurpose),
                special_requirements=["deletion_mechanism", "verification_process"],
                penalties={"max_fine": "$7,500 per violation"}
            )
        ]
        
        # LGPD Rules (Brazil)
        lgpd_rules = [
            ComplianceRule(
                regulation=ComplianceRegulation.LGPD,
                rule_id="lgpd_consent",
                title="Free and Informed Consent",
                description="Processing requires free, informed, and unambiguous consent",
                data_categories=[DataCategory.PERSONAL_IDENTIFIABLE, DataCategory.SENSITIVE_PERSONAL],
                required_consent=ConsentType.EXPLICIT,
                processing_purposes=[DataProcessingPurpose.NEUROMORPHIC_TRAINING, DataProcessingPurpose.USER_PERSONALIZATION],
                retention_period_days=365,
                special_requirements=["consent_granularity", "withdrawal_mechanism"],
                penalties={"max_fine": "2% of revenue up to R$50M"}
            ),
            ComplianceRule(
                regulation=ComplianceRegulation.LGPD,
                rule_id="lgpd_biometric",
                title="Sensitive Personal Data - Biometric",
                description="Biometric data is sensitive and requires specific consent",
                data_categories=[DataCategory.BIOMETRIC],
                required_consent=ConsentType.EXPLICIT,
                processing_purposes=[DataProcessingPurpose.NEUROMORPHIC_TRAINING],
                retention_period_days=180,
                special_requirements=["specific_consent", "security_measures"],
                penalties={"max_fine": "2% of revenue up to R$50M"}
            )
        ]
        
        # PIPL Rules (China)
        pipl_rules = [
            ComplianceRule(
                regulation=ComplianceRegulation.PIPL,
                rule_id="pipl_consent",
                title="Individual Consent for Processing",
                description="Processing of personal information requires individual consent",
                data_categories=[DataCategory.PERSONAL_IDENTIFIABLE],
                required_consent=ConsentType.EXPLICIT,
                processing_purposes=[DataProcessingPurpose.SERVICE_DELIVERY, DataProcessingPurpose.NEUROMORPHIC_TRAINING],
                retention_period_days=365,
                cross_border_transfer_allowed=False,  # Requires separate approval
                special_requirements=["data_localization", "security_assessment"],
                penalties={"max_fine": "RMB 50M or 5% of revenue"}
            ),
            ComplianceRule(
                regulation=ComplianceRegulation.PIPL,
                rule_id="pipl_sensitive",
                title="Sensitive Personal Information Protection",
                description="Sensitive personal information requires separate consent",
                data_categories=[DataCategory.SENSITIVE_PERSONAL, DataCategory.BIOMETRIC],
                required_consent=ConsentType.EXPLICIT,
                processing_purposes=[DataProcessingPurpose.NEUROMORPHIC_TRAINING],
                retention_period_days=90,  # Shorter retention in China
                cross_border_transfer_allowed=False,
                special_requirements=["separate_consent", "impact_assessment", "data_localization"],
                penalties={"max_fine": "RMB 50M or 5% of revenue"}
            )
        ]
        
        # Store all rules
        all_rules = gdpr_rules + ccpa_rules + lgpd_rules + pipl_rules
        
        for rule in all_rules:
            self.compliance_rules[f"{rule.regulation.value}_{rule.rule_id}"] = rule
    
    def _initialize_retention_schedules(self):
        """Initialize data retention schedules by category."""
        self.retention_schedules = {
            DataCategory.PERSONAL_IDENTIFIABLE: 1095,      # 3 years
            DataCategory.SENSITIVE_PERSONAL: 365,          # 1 year
            DataCategory.BIOMETRIC: 180,                   # 6 months
            DataCategory.HEALTH: 2555,                     # 7 years (medical requirement)
            DataCategory.FINANCIAL: 2555,                  # 7 years (regulatory requirement)
            DataCategory.BEHAVIORAL: 730,                  # 2 years
            DataCategory.TECHNICAL: 1095,                  # 3 years
            DataCategory.LOCATION: 365,                    # 1 year
            DataCategory.COMMUNICATION: 730,               # 2 years
            DataCategory.PREFERENCES: 1095,                # 3 years
        }
    
    def _initialize_transfer_rules(self):
        """Initialize cross-border transfer rules."""
        self.cross_border_transfer_rules = {
            ComplianceRegulation.GDPR: ["adequacy_decision", "standard_contractual_clauses", "binding_corporate_rules"],
            ComplianceRegulation.PIPL: ["security_assessment", "personal_consent", "government_approval"],
            ComplianceRegulation.LGPD: ["adequacy_decision", "standard_contractual_clauses"],
            ComplianceRegulation.CCPA: [],  # Generally allows transfers
            ComplianceRegulation.PIPEDA: ["adequacy_agreement", "contractual_protections"]
        }
    
    def _load_compliance_data(self):
        """Load existing compliance data from storage."""
        data_files = {
            "subjects": self.compliance_data_dir / "subjects.json",
            "consents": self.compliance_data_dir / "consents.json",
            "violations": self.compliance_data_dir / "violations.json"
        }
        
        for data_type, file_path in data_files.items():
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if data_type == "subjects":
                        for subject_id, subject_data in data.items():
                            self.data_subjects[subject_id] = DataSubject(
                                subject_id=subject_data["subject_id"],
                                jurisdiction=subject_data["jurisdiction"],
                                applicable_regulations=[ComplianceRegulation(r) for r in subject_data["applicable_regulations"]],
                                consent_records=subject_data["consent_records"],
                                data_categories=[DataCategory(c) for c in subject_data["data_categories"]],
                                preferences=subject_data["preferences"],
                                created_at=subject_data["created_at"],
                                last_updated=subject_data["last_updated"]
                            )
                    
                    elif data_type == "consents":
                        for consent_id, consent_data in data.items():
                            self.consent_records[consent_id] = ConsentRecord(
                                consent_id=consent_data["consent_id"],
                                subject_id=consent_data["subject_id"],
                                purpose=DataProcessingPurpose(consent_data["purpose"]),
                                consent_type=ConsentType(consent_data["consent_type"]),
                                granted=consent_data["granted"],
                                timestamp=consent_data["timestamp"],
                                expiry_timestamp=consent_data.get("expiry_timestamp"),
                                withdrawal_timestamp=consent_data.get("withdrawal_timestamp"),
                                metadata=consent_data["metadata"]
                            )
                    
                    elif data_type == "violations":
                        for violation_id, violation_data in data.items():
                            self.violations[violation_id] = ComplianceViolation(
                                violation_id=violation_data["violation_id"],
                                regulation=ComplianceRegulation(violation_data["regulation"]),
                                rule_id=violation_data["rule_id"],
                                severity=violation_data["severity"],
                                description=violation_data["description"],
                                detected_at=violation_data["detected_at"],
                                affected_subjects=violation_data["affected_subjects"],
                                remediation_required=violation_data["remediation_required"],
                                status=violation_data["status"]
                            )
                    
                except Exception as e:
                    print(f"Warning: Could not load {data_type} data: {e}")
    
    def _save_compliance_data(self):
        """Save compliance data to storage."""
        # Save data subjects
        subjects_data = {}
        for subject_id, subject in self.data_subjects.items():
            subjects_data[subject_id] = {
                "subject_id": subject.subject_id,
                "jurisdiction": subject.jurisdiction,
                "applicable_regulations": [r.value for r in subject.applicable_regulations],
                "consent_records": subject.consent_records,
                "data_categories": [c.value for c in subject.data_categories],
                "preferences": subject.preferences,
                "created_at": subject.created_at,
                "last_updated": subject.last_updated
            }
        
        with open(self.compliance_data_dir / "subjects.json", 'w', encoding='utf-8') as f:
            json.dump(subjects_data, f, indent=2)
        
        # Save consent records
        consents_data = {}
        for consent_id, consent in self.consent_records.items():
            consents_data[consent_id] = {
                "consent_id": consent.consent_id,
                "subject_id": consent.subject_id,
                "purpose": consent.purpose.value,
                "consent_type": consent.consent_type.value,
                "granted": consent.granted,
                "timestamp": consent.timestamp,
                "expiry_timestamp": consent.expiry_timestamp,
                "withdrawal_timestamp": consent.withdrawal_timestamp,
                "metadata": consent.metadata
            }
        
        with open(self.compliance_data_dir / "consents.json", 'w', encoding='utf-8') as f:
            json.dump(consents_data, f, indent=2)
        
        # Save violations
        violations_data = {}
        for violation_id, violation in self.violations.items():
            violations_data[violation_id] = {
                "violation_id": violation.violation_id,
                "regulation": violation.regulation.value,
                "rule_id": violation.rule_id,
                "severity": violation.severity,
                "description": violation.description,
                "detected_at": violation.detected_at,
                "affected_subjects": violation.affected_subjects,
                "remediation_required": violation.remediation_required,
                "status": violation.status
            }
        
        with open(self.compliance_data_dir / "violations.json", 'w', encoding='utf-8') as f:
            json.dump(violations_data, f, indent=2)
    
    def register_data_subject(self, subject_id: str, jurisdiction: str,
                            data_categories: List[DataCategory] = None) -> DataSubject:
        """Register a new data subject in the compliance system."""
        if data_categories is None:
            data_categories = []
        
        # Determine applicable regulations based on jurisdiction
        applicable_regulations = self._determine_applicable_regulations(jurisdiction)
        
        subject = DataSubject(
            subject_id=subject_id,
            jurisdiction=jurisdiction,
            applicable_regulations=applicable_regulations,
            data_categories=data_categories
        )
        
        self.data_subjects[subject_id] = subject
        self._save_compliance_data()
        
        return subject
    
    def _determine_applicable_regulations(self, jurisdiction: str) -> List[ComplianceRegulation]:
        """Determine applicable regulations based on jurisdiction."""
        jurisdiction_lower = jurisdiction.lower()
        regulations = []
        
        # EU jurisdictions
        eu_countries = ["eu", "germany", "france", "spain", "italy", "netherlands", "ireland", "austria", "belgium", "denmark"]
        if any(country in jurisdiction_lower for country in eu_countries):
            regulations.append(ComplianceRegulation.GDPR)
        
        # US jurisdictions
        if "us" in jurisdiction_lower or "california" in jurisdiction_lower or "united states" in jurisdiction_lower:
            regulations.append(ComplianceRegulation.CCPA)
        
        # Brazil
        if "brazil" in jurisdiction_lower or "br" in jurisdiction_lower:
            regulations.append(ComplianceRegulation.LGPD)
        
        # China
        if "china" in jurisdiction_lower or "cn" in jurisdiction_lower:
            regulations.append(ComplianceRegulation.PIPL)
        
        # Canada
        if "canada" in jurisdiction_lower or "ca" in jurisdiction_lower:
            regulations.append(ComplianceRegulation.PIPEDA)
        
        # Singapore
        if "singapore" in jurisdiction_lower or "sg" in jurisdiction_lower:
            regulations.append(ComplianceRegulation.PDPA_SG)
        
        return regulations
    
    def record_consent(self, subject_id: str, purpose: DataProcessingPurpose,
                      consent_type: ConsentType, granted: bool = True,
                      expiry_days: Optional[int] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> ConsentRecord:
        """Record consent from a data subject."""
        if subject_id not in self.data_subjects:
            raise ValueError(f"Data subject {subject_id} not registered")
        
        consent_id = f"{subject_id}_{purpose.value}_{int(time.time())}"
        
        expiry_timestamp = None
        if expiry_days:
            expiry_timestamp = time.time() + (expiry_days * 24 * 3600)
        
        consent = ConsentRecord(
            consent_id=consent_id,
            subject_id=subject_id,
            purpose=purpose,
            consent_type=consent_type,
            granted=granted,
            timestamp=time.time(),
            expiry_timestamp=expiry_timestamp,
            metadata=metadata or {}
        )
        
        self.consent_records[consent_id] = consent
        
        # Update subject's consent records
        subject = self.data_subjects[subject_id]
        subject.consent_records[purpose.value] = {
            "consent_id": consent_id,
            "granted": granted,
            "timestamp": consent.timestamp,
            "expiry": expiry_timestamp
        }
        subject.last_updated = time.time()
        
        self._save_compliance_data()
        
        return consent
    
    def withdraw_consent(self, subject_id: str, purpose: DataProcessingPurpose) -> bool:
        """Withdraw consent for a specific processing purpose."""
        if subject_id not in self.data_subjects:
            return False
        
        subject = self.data_subjects[subject_id]
        
        # Find active consent record
        for consent_id, consent in self.consent_records.items():
            if (consent.subject_id == subject_id and 
                consent.purpose == purpose and 
                consent.granted and 
                consent.withdrawal_timestamp is None):
                
                # Mark as withdrawn
                consent.withdrawal_timestamp = time.time()
                
                # Update subject record
                if purpose.value in subject.consent_records:
                    subject.consent_records[purpose.value]["granted"] = False
                    subject.consent_records[purpose.value]["withdrawn_at"] = time.time()
                
                subject.last_updated = time.time()
                self._save_compliance_data()
                
                return True
        
        return False
    
    def check_processing_lawfulness(self, subject_id: str, purpose: DataProcessingPurpose,
                                  data_categories: List[DataCategory]) -> Dict[str, Any]:
        """Check if data processing is lawful for a subject."""
        if subject_id not in self.data_subjects:
            return {"lawful": False, "reason": "Data subject not registered"}
        
        subject = self.data_subjects[subject_id]
        result = {
            "lawful": True,
            "applicable_regulations": [r.value for r in subject.applicable_regulations],
            "consent_status": {},
            "violations": [],
            "requirements": []
        }
        
        # Check each applicable regulation
        for regulation in subject.applicable_regulations:
            regulation_rules = [rule for rule in self.compliance_rules.values() 
                              if rule.regulation == regulation]
            
            for rule in regulation_rules:
                # Check if rule applies to this processing
                if (purpose in rule.processing_purposes and
                    any(cat in rule.data_categories for cat in data_categories)):
                    
                    # Check consent requirement
                    consent_valid = self._check_consent_validity(subject_id, purpose, rule.required_consent)
                    
                    if not consent_valid:
                        result["lawful"] = False
                        result["violations"].append(f"Missing or invalid consent for {regulation.value}")
                        result["requirements"].append(f"Obtain {rule.required_consent.value} consent")
                    
                    result["consent_status"][f"{regulation.value}_{rule.rule_id}"] = consent_valid
        
        return result
    
    def _check_consent_validity(self, subject_id: str, purpose: DataProcessingPurpose,
                               required_consent: ConsentType) -> bool:
        """Check if consent is valid for the required type."""
        subject = self.data_subjects[subject_id]
        
        # Check if consent is recorded and valid
        if purpose.value in subject.consent_records:
            consent_record = subject.consent_records[purpose.value]
            
            if not consent_record["granted"]:
                return False
            
            # Check expiry
            if consent_record.get("expiry") and time.time() > consent_record["expiry"]:
                return False
            
            # Check if withdrawn
            if consent_record.get("withdrawn_at"):
                return False
            
            return True
        
        # For legitimate interest, no explicit consent needed
        if required_consent == ConsentType.LEGITIMATE_INTEREST:
            return True
        
        return False
    
    def scan_for_violations(self) -> List[ComplianceViolation]:
        """Scan system for potential compliance violations."""
        violations = []
        
        for subject_id, subject in self.data_subjects.items():
            # Check consent expiry
            violations.extend(self._check_consent_expiry(subject))
            
            # Check data retention
            violations.extend(self._check_data_retention(subject))
            
            # Check cross-border transfers
            violations.extend(self._check_cross_border_transfers(subject))
        
        # Store new violations
        for violation in violations:
            self.violations[violation.violation_id] = violation
        
        self._save_compliance_data()
        
        return violations
    
    def _check_consent_expiry(self, subject: DataSubject) -> List[ComplianceViolation]:
        """Check for expired consents."""
        violations = []
        current_time = time.time()
        
        for purpose, consent_info in subject.consent_records.items():
            expiry = consent_info.get("expiry")
            if expiry and current_time > expiry and consent_info["granted"]:
                
                violation = ComplianceViolation(
                    violation_id=f"consent_expiry_{subject.subject_id}_{purpose}_{int(current_time)}",
                    regulation=subject.applicable_regulations[0],  # Primary regulation
                    rule_id="consent_expiry",
                    severity="medium",
                    description=f"Consent expired for purpose {purpose}",
                    detected_at=current_time,
                    affected_subjects=[subject.subject_id],
                    remediation_required=["obtain_fresh_consent", "stop_processing"]
                )
                violations.append(violation)
        
        return violations
    
    def _check_data_retention(self, subject: DataSubject) -> List[ComplianceViolation]:
        """Check for data retention violations."""
        violations = []
        current_time = time.time()
        
        for data_category in subject.data_categories:
            retention_days = self.retention_schedules.get(data_category, 365)
            retention_seconds = retention_days * 24 * 3600
            
            if current_time - subject.created_at > retention_seconds:
                violation = ComplianceViolation(
                    violation_id=f"retention_{subject.subject_id}_{data_category.value}_{int(current_time)}",
                    regulation=subject.applicable_regulations[0],
                    rule_id="data_retention",
                    severity="high",
                    description=f"Data retained beyond limit for category {data_category.value}",
                    detected_at=current_time,
                    affected_subjects=[subject.subject_id],
                    remediation_required=["delete_expired_data", "update_retention_policy"]
                )
                violations.append(violation)
        
        return violations
    
    def _check_cross_border_transfers(self, subject: DataSubject) -> List[ComplianceViolation]:
        """Check for cross-border transfer violations."""
        violations = []
        # This would implement more complex transfer checking logic
        return violations
    
    def handle_data_subject_request(self, subject_id: str, request_type: str,
                                   details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle data subject rights requests (access, deletion, portability, etc.)."""
        if subject_id not in self.data_subjects:
            return {"status": "error", "message": "Data subject not found"}
        
        subject = self.data_subjects[subject_id]
        current_time = time.time()
        
        if request_type == "access":
            # Right of access - provide all data
            return {
                "status": "completed",
                "request_type": "access",
                "subject_id": subject_id,
                "data": {
                    "subject_info": {
                        "subject_id": subject.subject_id,
                        "jurisdiction": subject.jurisdiction,
                        "data_categories": [c.value for c in subject.data_categories],
                        "created_at": subject.created_at,
                        "last_updated": subject.last_updated
                    },
                    "consent_records": subject.consent_records,
                    "preferences": subject.preferences
                },
                "processed_at": current_time
            }
        
        elif request_type == "deletion":
            # Right to erasure
            try:
                # Remove from all records
                del self.data_subjects[subject_id]
                
                # Remove associated consent records
                consents_to_remove = [cid for cid, consent in self.consent_records.items() 
                                     if consent.subject_id == subject_id]
                for consent_id in consents_to_remove:
                    del self.consent_records[consent_id]
                
                self._save_compliance_data()
                
                return {
                    "status": "completed",
                    "request_type": "deletion",
                    "subject_id": subject_id,
                    "processed_at": current_time,
                    "data_deleted": True
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "request_type": "deletion",
                    "subject_id": subject_id,
                    "message": f"Deletion failed: {e}"
                }
        
        elif request_type == "portability":
            # Data portability
            portable_data = {
                "subject_id": subject_id,
                "export_timestamp": current_time,
                "jurisdiction": subject.jurisdiction,
                "data_categories": [c.value for c in subject.data_categories],
                "consent_history": [],
                "preferences": subject.preferences
            }
            
            # Include consent history
            for consent_id, consent in self.consent_records.items():
                if consent.subject_id == subject_id:
                    portable_data["consent_history"].append({
                        "purpose": consent.purpose.value,
                        "granted": consent.granted,
                        "timestamp": consent.timestamp,
                        "withdrawn_at": consent.withdrawal_timestamp
                    })
            
            return {
                "status": "completed",
                "request_type": "portability",
                "subject_id": subject_id,
                "portable_data": portable_data,
                "format": "json",
                "processed_at": current_time
            }
        
        elif request_type == "rectification":
            # Right to rectification
            if details and "updates" in details:
                updates = details["updates"]
                
                if "preferences" in updates:
                    subject.preferences.update(updates["preferences"])
                
                subject.last_updated = current_time
                self._save_compliance_data()
                
                return {
                    "status": "completed",
                    "request_type": "rectification",
                    "subject_id": subject_id,
                    "updates_applied": list(updates.keys()),
                    "processed_at": current_time
                }
            else:
                return {
                    "status": "error",
                    "request_type": "rectification",
                    "message": "No updates provided"
                }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown request type: {request_type}"
            }
    
    def generate_compliance_report(self, regulation: Optional[ComplianceRegulation] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        current_time = time.time()
        
        # Filter by regulation if specified
        if regulation:
            subjects = [s for s in self.data_subjects.values() 
                       if regulation in s.applicable_regulations]
            violations = [v for v in self.violations.values() 
                         if v.regulation == regulation]
            rules = [r for r in self.compliance_rules.values() 
                    if r.regulation == regulation]
        else:
            subjects = list(self.data_subjects.values())
            violations = list(self.violations.values())
            rules = list(self.compliance_rules.values())
        
        # Calculate metrics
        total_subjects = len(subjects)
        active_consents = sum(
            1 for s in subjects 
            for consent_info in s.consent_records.values() 
            if consent_info["granted"] and not consent_info.get("withdrawn_at")
        )
        
        open_violations = [v for v in violations if v.status == "open"]
        critical_violations = [v for v in violations if v.severity == "critical"]
        
        # Consent status breakdown
        consent_breakdown = {}
        for s in subjects:
            for purpose, consent_info in s.consent_records.items():
                if purpose not in consent_breakdown:
                    consent_breakdown[purpose] = {"granted": 0, "withdrawn": 0, "expired": 0}
                
                if consent_info["granted"] and not consent_info.get("withdrawn_at"):
                    if consent_info.get("expiry") and current_time > consent_info["expiry"]:
                        consent_breakdown[purpose]["expired"] += 1
                    else:
                        consent_breakdown[purpose]["granted"] += 1
                else:
                    consent_breakdown[purpose]["withdrawn"] += 1
        
        return {
            "report_generated_at": current_time,
            "regulation": regulation.value if regulation else "all",
            "summary": {
                "total_data_subjects": total_subjects,
                "active_consents": active_consents,
                "total_violations": len(violations),
                "open_violations": len(open_violations),
                "critical_violations": len(critical_violations)
            },
            "consent_breakdown": consent_breakdown,
            "violation_breakdown": {
                "by_severity": {
                    "critical": len([v for v in violations if v.severity == "critical"]),
                    "high": len([v for v in violations if v.severity == "high"]),
                    "medium": len([v for v in violations if v.severity == "medium"]),
                    "low": len([v for v in violations if v.severity == "low"])
                },
                "by_status": {
                    "open": len([v for v in violations if v.status == "open"]),
                    "investigating": len([v for v in violations if v.status == "investigating"]),
                    "resolved": len([v for v in violations if v.status == "resolved"])
                }
            },
            "compliance_rules_count": len(rules),
            "jurisdictions": list(set(s.jurisdiction for s in subjects)),
            "data_categories": list(set(
                cat.value for s in subjects for cat in s.data_categories
            ))
        }
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data."""
        current_time = time.time()
        
        # Recent activity
        recent_consents = [
            c for c in self.consent_records.values() 
            if current_time - c.timestamp < 86400 * 30  # Last 30 days
        ]
        
        recent_violations = [
            v for v in self.violations.values() 
            if current_time - v.detected_at < 86400 * 30  # Last 30 days
        ]
        
        # Expiring consents (next 30 days)
        expiring_consents = []
        for subject in self.data_subjects.values():
            for purpose, consent_info in subject.consent_records.items():
                expiry = consent_info.get("expiry")
                if (expiry and 
                    current_time < expiry < current_time + 86400 * 30 and
                    consent_info["granted"]):
                    expiring_consents.append({
                        "subject_id": subject.subject_id,
                        "purpose": purpose,
                        "expiry": expiry
                    })
        
        return {
            "overview": {
                "total_subjects": len(self.data_subjects),
                "total_consents": len(self.consent_records),
                "active_violations": len([v for v in self.violations.values() if v.status == "open"]),
                "compliance_score": self._calculate_compliance_score()
            },
            "recent_activity": {
                "new_consents_30d": len(recent_consents),
                "new_violations_30d": len(recent_violations),
                "expiring_consents_30d": len(expiring_consents)
            },
            "alerts": {
                "critical_violations": len([v for v in self.violations.values() 
                                          if v.severity == "critical" and v.status == "open"]),
                "expiring_consents": expiring_consents[:5]  # Top 5 most urgent
            },
            "regulations": {
                reg.value: len([s for s in self.data_subjects.values() if reg in s.applicable_regulations])
                for reg in ComplianceRegulation
            }
        }
    
    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)."""
        if not self.data_subjects:
            return 100.0  # No data, no violations
        
        total_subjects = len(self.data_subjects)
        
        # Deduct points for violations
        violation_penalties = {
            "critical": 20,
            "high": 10,
            "medium": 5,
            "low": 2
        }
        
        penalty_points = 0
        for violation in self.violations.values():
            if violation.status == "open":
                penalty_points += violation_penalties.get(violation.severity, 2)
        
        # Deduct points for missing consents
        missing_consent_penalty = 0
        for subject in self.data_subjects.values():
            required_consents = len(DataProcessingPurpose)  # Simplified
            actual_consents = len([c for c in subject.consent_records.values() if c["granted"]])
            
            if actual_consents < required_consents:
                missing_consent_penalty += (required_consents - actual_consents) * 2
        
        # Calculate score
        max_score = 100.0
        total_penalty = penalty_points + missing_consent_penalty
        
        score = max(0.0, max_score - (total_penalty / max(total_subjects, 1)))
        
        return min(100.0, score)


# Example usage and testing
if __name__ == "__main__":
    print("🛡️ Testing Global Compliance Manager")
    print("=" * 60)
    
    # Create compliance manager
    compliance_manager = GlobalComplianceManager()
    
    # Test data subject registration
    print("\n1. Testing Data Subject Registration:")
    subject1 = compliance_manager.register_data_subject(
        subject_id="user_001",
        jurisdiction="Germany",
        data_categories=[DataCategory.PERSONAL_IDENTIFIABLE, DataCategory.BIOMETRIC]
    )
    print(f"  Registered: {subject1.subject_id}")
    print(f"  Jurisdiction: {subject1.jurisdiction}")
    print(f"  Applicable Regulations: {[r.value for r in subject1.applicable_regulations]}")
    
    subject2 = compliance_manager.register_data_subject(
        subject_id="user_002",
        jurisdiction="California",
        data_categories=[DataCategory.PERSONAL_IDENTIFIABLE, DataCategory.BEHAVIORAL]
    )
    print(f"  Registered: {subject2.subject_id}")
    print(f"  Applicable Regulations: {[r.value for r in subject2.applicable_regulations]}")
    
    # Test consent recording
    print("\n2. Testing Consent Recording:")
    consent1 = compliance_manager.record_consent(
        subject_id="user_001",
        purpose=DataProcessingPurpose.NEUROMORPHIC_TRAINING,
        consent_type=ConsentType.EXPLICIT,
        granted=True,
        expiry_days=365
    )
    print(f"  Recorded consent: {consent1.consent_id}")
    
    consent2 = compliance_manager.record_consent(
        subject_id="user_002",
        purpose=DataProcessingPurpose.ANALYTICS,
        consent_type=ConsentType.IMPLICIT,
        granted=True
    )
    print(f"  Recorded consent: {consent2.consent_id}")
    
    # Test processing lawfulness check
    print("\n3. Testing Processing Lawfulness:")
    lawfulness_check = compliance_manager.check_processing_lawfulness(
        subject_id="user_001",
        purpose=DataProcessingPurpose.NEUROMORPHIC_TRAINING,
        data_categories=[DataCategory.BIOMETRIC]
    )
    print(f"  Processing lawful: {lawfulness_check['lawful']}")
    print(f"  Applicable regulations: {lawfulness_check['applicable_regulations']}")
    
    # Test data subject request handling
    print("\n4. Testing Data Subject Rights:")
    access_request = compliance_manager.handle_data_subject_request(
        subject_id="user_001",
        request_type="access"
    )
    print(f"  Access request: {access_request['status']}")
    
    portability_request = compliance_manager.handle_data_subject_request(
        subject_id="user_001",
        request_type="portability"
    )
    print(f"  Portability request: {portability_request['status']}")
    
    # Test violation scanning
    print("\n5. Testing Violation Scanning:")
    violations = compliance_manager.scan_for_violations()
    print(f"  Violations found: {len(violations)}")
    
    # Test compliance report
    print("\n6. Generating Compliance Report:")
    report = compliance_manager.generate_compliance_report(ComplianceRegulation.GDPR)
    print(f"  GDPR Report:")
    print(f"    Total subjects: {report['summary']['total_data_subjects']}")
    print(f"    Active consents: {report['summary']['active_consents']}")
    print(f"    Total violations: {report['summary']['total_violations']}")
    
    # Test compliance dashboard
    print("\n7. Compliance Dashboard:")
    dashboard = compliance_manager.get_compliance_dashboard()
    print(f"  Compliance Score: {dashboard['overview']['compliance_score']:.1f}/100")
    print(f"  Total Subjects: {dashboard['overview']['total_subjects']}")
    print(f"  Active Violations: {dashboard['overview']['active_violations']}")
    print(f"  Regulations Coverage: {list(dashboard['regulations'].keys())}")
    
    print("\n✅ Global compliance management testing completed!")