"""
Global Compliance and Data Privacy Management

Comprehensive compliance framework supporting GDPR, CCPA, PDPA, 
and other international data protection regulations.
"""

import json
import hashlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from cryptography.fernet import Fernet
import base64


class ComplianceRegulation(Enum):
    """Supported compliance regulations."""
    GDPR = "gdpr"           # General Data Protection Regulation (EU)
    CCPA = "ccpa"           # California Consumer Privacy Act (US)
    PDPA = "pdpa"           # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"           # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"       # Personal Information Protection and Electronic Documents Act (Canada)
    DPA = "dpa"             # Data Protection Act (UK)
    APPI = "appi"           # Act on Protection of Personal Information (Japan)


class DataCategory(Enum):
    """Categories of personal data."""
    PERSONAL_IDENTIFIABLE = "pii"       # Names, addresses, IDs
    SENSITIVE_PERSONAL = "sensitive"     # Health, biometric, genetic data
    BEHAVIORAL = "behavioral"            # Usage patterns, preferences
    TECHNICAL = "technical"              # IP addresses, device info
    FINANCIAL = "financial"              # Payment information
    BIOMETRIC = "biometric"              # Fingerprints, neural patterns
    LOCATION = "location"                # GPS, geolocation data


class ProcessingPurpose(Enum):
    """Purposes for data processing."""
    RESEARCH = "research"                # Scientific research
    TRAINING = "training"                # Model training
    INFERENCE = "inference"              # Model inference
    ANALYTICS = "analytics"              # Data analysis
    PERSONALIZATION = "personalization" # Service personalization
    SECURITY = "security"                # Security and fraud prevention
    MARKETING = "marketing"              # Marketing communications
    LEGAL = "legal"                     # Legal compliance


class ConsentStatus(Enum):
    """Data processing consent status."""
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    EXPIRED = "expired"


@dataclass
class DataSubject:
    """Represents a data subject (person whose data is processed)."""
    subject_id: str
    email: Optional[str] = None
    jurisdiction: Optional[str] = None
    consents: Dict[str, 'ConsentRecord'] = field(default_factory=dict)
    data_categories: Set[DataCategory] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def has_valid_consent(self, purpose: ProcessingPurpose) -> bool:
        """Check if subject has valid consent for purpose."""
        consent = self.consents.get(purpose.value)
        if not consent:
            return False
        
        return (consent.status == ConsentStatus.GRANTED and 
                consent.is_valid())


@dataclass
class ConsentRecord:
    """Record of data processing consent."""
    purpose: ProcessingPurpose
    status: ConsentStatus
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    legal_basis: str = "consent"
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.status != ConsentStatus.GRANTED:
            return False
        
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        
        return True


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    processing_id: str
    subject_id: str
    purpose: ProcessingPurpose
    data_categories: Set[DataCategory]
    legal_basis: str
    timestamp: datetime
    processor: str
    retention_period: Optional[timedelta] = None
    cross_border_transfer: bool = False
    automated_decision_making: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataPrivacyManager:
    """
    Comprehensive data privacy management system.
    
    Handles consent management, data subject rights, processing records,
    and compliance with multiple regulations.
    """
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """Initialize data privacy manager."""
        self.logger = logging.getLogger(__name__)
        
        # Encryption for sensitive data
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Data storage
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_records: List[DataProcessingRecord] = []
        self.anonymization_records: Dict[str, Dict[str, Any]] = {}
        
        # Consent management
        self.consent_cache: Dict[str, Dict[str, ConsentRecord]] = {}
        self.consent_lock = threading.RLock()
        
        # Compliance tracking
        self.compliance_reports: List[Dict[str, Any]] = []
        
        self.logger.info("DataPrivacyManager initialized")
    
    def register_data_subject(
        self, 
        subject_id: str, 
        email: Optional[str] = None,
        jurisdiction: Optional[str] = None
    ) -> DataSubject:
        """Register a new data subject."""
        subject = DataSubject(
            subject_id=subject_id,
            email=email,
            jurisdiction=jurisdiction
        )
        
        self.data_subjects[subject_id] = subject
        self.logger.info(f"Registered data subject: {subject_id}")
        
        return subject
    
    def record_consent(
        self,
        subject_id: str,
        purpose: ProcessingPurpose,
        status: ConsentStatus,
        expires_at: Optional[datetime] = None,
        legal_basis: str = "consent",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConsentRecord:
        """Record consent for data processing."""
        consent = ConsentRecord(
            purpose=purpose,
            status=status,
            granted_at=datetime.now(timezone.utc) if status == ConsentStatus.GRANTED else None,
            expires_at=expires_at,
            legal_basis=legal_basis,
            metadata=metadata or {}
        )
        
        with self.consent_lock:
            # Ensure subject exists
            if subject_id not in self.data_subjects:
                self.register_data_subject(subject_id)
            
            # Record consent
            self.data_subjects[subject_id].consents[purpose.value] = consent
            
            # Update cache
            if subject_id not in self.consent_cache:
                self.consent_cache[subject_id] = {}
            self.consent_cache[subject_id][purpose.value] = consent
        
        self.logger.info(f"Recorded {status.value} consent for {subject_id}:{purpose.value}")
        return consent
    
    def withdraw_consent(self, subject_id: str, purpose: ProcessingPurpose) -> bool:
        """Withdraw consent for data processing."""
        with self.consent_lock:
            if subject_id not in self.data_subjects:
                return False
            
            subject = self.data_subjects[subject_id]
            if purpose.value not in subject.consents:
                return False
            
            # Update consent record
            consent = subject.consents[purpose.value]
            consent.status = ConsentStatus.WITHDRAWN
            consent.withdrawn_at = datetime.now(timezone.utc)
            
            # Update cache
            if subject_id in self.consent_cache:
                self.consent_cache[subject_id][purpose.value] = consent
        
        self.logger.info(f"Withdrew consent for {subject_id}:{purpose.value}")
        return True
    
    def check_processing_consent(
        self, 
        subject_id: str, 
        purpose: ProcessingPurpose,
        data_categories: Set[DataCategory]
    ) -> bool:
        """Check if processing is allowed based on consent."""
        if subject_id not in self.data_subjects:
            self.logger.warning(f"Unknown data subject: {subject_id}")
            return False
        
        subject = self.data_subjects[subject_id]
        
        # Check consent for purpose
        if not subject.has_valid_consent(purpose):
            self.logger.warning(f"No valid consent for {subject_id}:{purpose.value}")
            return False
        
        # Check data category restrictions
        consent = subject.consents[purpose.value]
        allowed_categories = consent.metadata.get('allowed_categories', set())
        
        if allowed_categories and not data_categories.issubset(allowed_categories):
            self.logger.warning(f"Data categories not covered by consent: {data_categories - allowed_categories}")
            return False
        
        return True
    
    def record_processing_activity(
        self,
        subject_id: str,
        purpose: ProcessingPurpose,
        data_categories: Set[DataCategory],
        processor: str,
        legal_basis: str = "consent",
        retention_period: Optional[timedelta] = None,
        cross_border_transfer: bool = False,
        automated_decision_making: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataProcessingRecord:
        """Record data processing activity."""
        processing_record = DataProcessingRecord(
            processing_id=str(uuid.uuid4()),
            subject_id=subject_id,
            purpose=purpose,
            data_categories=data_categories,
            legal_basis=legal_basis,
            timestamp=datetime.now(timezone.utc),
            processor=processor,
            retention_period=retention_period,
            cross_border_transfer=cross_border_transfer,
            automated_decision_making=automated_decision_making,
            metadata=metadata or {}
        )
        
        self.processing_records.append(processing_record)
        
        # Update subject's last activity
        if subject_id in self.data_subjects:
            self.data_subjects[subject_id].last_activity = processing_record.timestamp
            self.data_subjects[subject_id].data_categories.update(data_categories)
        
        self.logger.debug(f"Recorded processing activity: {processing_record.processing_id}")
        return processing_record
    
    def handle_data_subject_request(
        self, 
        subject_id: str, 
        request_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle data subject rights requests (GDPR Article 15-22).
        
        Args:
            subject_id: ID of the data subject
            request_type: Type of request (access, portability, erasure, rectification)
            metadata: Additional request metadata
            
        Returns:
            Response data for the request
        """
        if subject_id not in self.data_subjects:
            return {"error": "Data subject not found", "status": "rejected"}
        
        subject = self.data_subjects[subject_id]
        
        if request_type == "access":
            # Article 15 - Right of access
            return self._handle_access_request(subject)
        elif request_type == "portability":
            # Article 20 - Right to data portability
            return self._handle_portability_request(subject)
        elif request_type == "erasure":
            # Article 17 - Right to erasure ('right to be forgotten')
            return self._handle_erasure_request(subject)
        elif request_type == "rectification":
            # Article 16 - Right to rectification
            return self._handle_rectification_request(subject, metadata or {})
        elif request_type == "restrict_processing":
            # Article 18 - Right to restriction of processing
            return self._handle_restriction_request(subject)
        elif request_type == "object_processing":
            # Article 21 - Right to object
            return self._handle_objection_request(subject)
        else:
            return {"error": "Unknown request type", "status": "rejected"}
    
    def anonymize_data(self, subject_id: str) -> Dict[str, Any]:
        """Anonymize all data for a subject."""
        if subject_id not in self.data_subjects:
            return {"error": "Data subject not found"}
        
        # Generate anonymous ID
        anonymous_id = hashlib.sha256(f"{subject_id}:{datetime.now()}".encode()).hexdigest()[:16]
        
        # Record the anonymization mapping (encrypted)
        mapping_data = {
            "original_id": subject_id,
            "anonymous_id": anonymous_id,
            "anonymized_at": datetime.now(timezone.utc).isoformat(),
            "method": "hash_based"
        }
        
        encrypted_mapping = self.cipher_suite.encrypt(json.dumps(mapping_data).encode())
        self.anonymization_records[anonymous_id] = {
            "encrypted_mapping": encrypted_mapping,
            "anonymized_at": datetime.now(timezone.utc)
        }
        
        # Update processing records
        for record in self.processing_records:
            if record.subject_id == subject_id:
                record.subject_id = anonymous_id
        
        # Remove subject from active subjects
        del self.data_subjects[subject_id]
        
        self.logger.info(f"Anonymized data for subject {subject_id} -> {anonymous_id}")
        
        return {
            "status": "completed",
            "anonymous_id": anonymous_id,
            "anonymized_at": datetime.now(timezone.utc).isoformat()
        }
    
    def get_compliance_report(self, regulation: ComplianceRegulation) -> Dict[str, Any]:
        """Generate compliance report for specific regulation."""
        report = {
            "regulation": regulation.value,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_subjects": {
                "total": len(self.data_subjects),
                "with_consent": 0,
                "with_expired_consent": 0,
                "by_jurisdiction": {}
            },
            "processing_activities": {
                "total": len(self.processing_records),
                "by_purpose": {},
                "by_legal_basis": {},
                "cross_border_transfers": 0,
                "automated_decisions": 0
            },
            "consent_management": {
                "granted": 0,
                "withdrawn": 0,
                "expired": 0,
                "pending": 0
            },
            "data_subject_requests": {
                "access": 0,
                "erasure": 0,
                "portability": 0,
                "rectification": 0
            },
            "compliance_score": 0.0
        }
        
        # Analyze data subjects
        for subject in self.data_subjects.values():
            # Count by jurisdiction
            jurisdiction = subject.jurisdiction or "unknown"
            report["data_subjects"]["by_jurisdiction"][jurisdiction] = \
                report["data_subjects"]["by_jurisdiction"].get(jurisdiction, 0) + 1
            
            # Count consent status
            has_valid_consent = False
            has_expired_consent = False
            
            for consent in subject.consents.values():
                if consent.status == ConsentStatus.GRANTED:
                    report["consent_management"]["granted"] += 1
                    if consent.is_valid():
                        has_valid_consent = True
                    else:
                        has_expired_consent = True
                elif consent.status == ConsentStatus.WITHDRAWN:
                    report["consent_management"]["withdrawn"] += 1
                elif consent.status == ConsentStatus.PENDING:
                    report["consent_management"]["pending"] += 1
                elif consent.status == ConsentStatus.EXPIRED:
                    report["consent_management"]["expired"] += 1
            
            if has_valid_consent:
                report["data_subjects"]["with_consent"] += 1
            if has_expired_consent:
                report["data_subjects"]["with_expired_consent"] += 1
        
        # Analyze processing activities
        for record in self.processing_records:
            # By purpose
            purpose = record.purpose.value
            report["processing_activities"]["by_purpose"][purpose] = \
                report["processing_activities"]["by_purpose"].get(purpose, 0) + 1
            
            # By legal basis
            legal_basis = record.legal_basis
            report["processing_activities"]["by_legal_basis"][legal_basis] = \
                report["processing_activities"]["by_legal_basis"].get(legal_basis, 0) + 1
            
            # Special processing
            if record.cross_border_transfer:
                report["processing_activities"]["cross_border_transfers"] += 1
            if record.automated_decision_making:
                report["processing_activities"]["automated_decisions"] += 1
        
        # Calculate compliance score (simplified)
        total_subjects = len(self.data_subjects)
        if total_subjects > 0:
            consent_score = report["data_subjects"]["with_consent"] / total_subjects
            expired_penalty = report["data_subjects"]["with_expired_consent"] / total_subjects * 0.5
            report["compliance_score"] = max(0.0, min(100.0, (consent_score - expired_penalty) * 100))
        else:
            report["compliance_score"] = 100.0
        
        self.compliance_reports.append(report)
        return report
    
    def _handle_access_request(self, subject: DataSubject) -> Dict[str, Any]:
        """Handle data access request."""
        # Collect all data about the subject
        subject_data = {
            "subject_info": {
                "subject_id": subject.subject_id,
                "email": subject.email,
                "jurisdiction": subject.jurisdiction,
                "created_at": subject.created_at.isoformat(),
                "last_activity": subject.last_activity.isoformat()
            },
            "consents": [
                {
                    "purpose": consent.purpose.value,
                    "status": consent.status.value,
                    "granted_at": consent.granted_at.isoformat() if consent.granted_at else None,
                    "expires_at": consent.expires_at.isoformat() if consent.expires_at else None,
                    "legal_basis": consent.legal_basis
                }
                for consent in subject.consents.values()
            ],
            "processing_activities": [
                {
                    "processing_id": record.processing_id,
                    "purpose": record.purpose.value,
                    "data_categories": [cat.value for cat in record.data_categories],
                    "timestamp": record.timestamp.isoformat(),
                    "processor": record.processor,
                    "legal_basis": record.legal_basis
                }
                for record in self.processing_records
                if record.subject_id == subject.subject_id
            ],
            "data_categories": [cat.value for cat in subject.data_categories]
        }
        
        return {
            "status": "completed",
            "request_type": "access",
            "data": subject_data,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
    
    def _handle_portability_request(self, subject: DataSubject) -> Dict[str, Any]:
        """Handle data portability request."""
        # Export data in machine-readable format
        portable_data = {
            "subject_id": subject.subject_id,
            "export_format": "json",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "data": self._handle_access_request(subject)["data"]
        }
        
        return {
            "status": "completed",
            "request_type": "portability",
            "data": portable_data,
            "format": "json"
        }
    
    def _handle_erasure_request(self, subject: DataSubject) -> Dict[str, Any]:
        """Handle data erasure request."""
        subject_id = subject.subject_id
        
        # Check if erasure is allowed
        active_legal_obligations = self._check_legal_obligations(subject_id)
        if active_legal_obligations:
            return {
                "status": "rejected",
                "reason": "Legal obligations prevent erasure",
                "obligations": active_legal_obligations
            }
        
        # Perform erasure
        erasure_result = self.anonymize_data(subject_id)
        
        return {
            "status": "completed",
            "request_type": "erasure",
            "anonymization_result": erasure_result,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
    
    def _handle_rectification_request(self, subject: DataSubject, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data rectification request."""
        # Apply updates to subject data
        updated_fields = []
        
        if "email" in updates:
            subject.email = updates["email"]
            updated_fields.append("email")
        
        if "jurisdiction" in updates:
            subject.jurisdiction = updates["jurisdiction"]
            updated_fields.append("jurisdiction")
        
        return {
            "status": "completed",
            "request_type": "rectification",
            "updated_fields": updated_fields,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
    
    def _handle_restriction_request(self, subject: DataSubject) -> Dict[str, Any]:
        """Handle processing restriction request."""
        # Mark all consents as restricted (implementation specific)
        restricted_purposes = []
        
        for purpose, consent in subject.consents.items():
            if consent.status == ConsentStatus.GRANTED:
                consent.metadata["restricted"] = True
                restricted_purposes.append(purpose)
        
        return {
            "status": "completed",
            "request_type": "restriction",
            "restricted_purposes": restricted_purposes,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
    
    def _handle_objection_request(self, subject: DataSubject) -> Dict[str, Any]:
        """Handle objection to processing request."""
        # Withdraw consents for non-essential purposes
        objected_purposes = []
        
        for purpose, consent in subject.consents.items():
            if consent.legal_basis in ["legitimate_interest", "marketing"]:
                consent.status = ConsentStatus.WITHDRAWN
                consent.withdrawn_at = datetime.now(timezone.utc)
                objected_purposes.append(purpose)
        
        return {
            "status": "completed",
            "request_type": "objection",
            "objected_purposes": objected_purposes,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
    
    def _check_legal_obligations(self, subject_id: str) -> List[str]:
        """Check if there are legal obligations preventing data erasure."""
        obligations = []
        
        # Check for ongoing legal proceedings, regulatory requirements, etc.
        # This is a simplified implementation
        for record in self.processing_records:
            if record.subject_id == subject_id:
                if record.legal_basis == "legal_obligation":
                    obligations.append(f"Legal obligation: {record.purpose.value}")
                if record.retention_period:
                    retention_end = record.timestamp + record.retention_period
                    if datetime.now(timezone.utc) < retention_end:
                        obligations.append(f"Retention period for {record.purpose.value} not yet expired")
        
        return obligations


class ComplianceManager:
    """
    Main compliance management system.
    
    Orchestrates data privacy management and regulatory compliance
    across multiple jurisdictions and regulations.
    """
    
    def __init__(self, supported_regulations: Optional[List[ComplianceRegulation]] = None):
        """Initialize compliance manager."""
        self.logger = logging.getLogger(__name__)
        
        self.supported_regulations = supported_regulations or [
            ComplianceRegulation.GDPR,
            ComplianceRegulation.CCPA,
            ComplianceRegulation.PDPA
        ]
        
        # Initialize privacy manager
        self.privacy_manager = DataPrivacyManager()
        
        # Compliance policies
        self.policies: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_policies()
        
        # Audit trail
        self.audit_logs: List[Dict[str, Any]] = []
        
        self.logger.info(f"ComplianceManager initialized with {len(self.supported_regulations)} regulations")
    
    def get_applicable_regulations(self, jurisdiction: Optional[str] = None) -> List[ComplianceRegulation]:
        """Get applicable regulations for jurisdiction."""
        if not jurisdiction:
            return self.supported_regulations
        
        # Map jurisdictions to regulations
        jurisdiction_mapping = {
            "EU": [ComplianceRegulation.GDPR],
            "US-CA": [ComplianceRegulation.CCPA],
            "SG": [ComplianceRegulation.PDPA],
            "BR": [ComplianceRegulation.LGPD],
            "CA": [ComplianceRegulation.PIPEDA],
            "UK": [ComplianceRegulation.DPA],
            "JP": [ComplianceRegulation.APPI]
        }
        
        return jurisdiction_mapping.get(jurisdiction.upper(), self.supported_regulations)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report for all regulations."""
        comprehensive_report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "regulations": {},
            "overall_compliance": {
                "average_score": 0.0,
                "total_subjects": len(self.privacy_manager.data_subjects),
                "total_processing_activities": len(self.privacy_manager.processing_records)
            },
            "recommendations": []
        }
        
        # Generate report for each regulation
        scores = []
        for regulation in self.supported_regulations:
            regulation_report = self.privacy_manager.get_compliance_report(regulation)
            comprehensive_report["regulations"][regulation.value] = regulation_report
            scores.append(regulation_report["compliance_score"])
        
        # Calculate overall metrics
        if scores:
            comprehensive_report["overall_compliance"]["average_score"] = sum(scores) / len(scores)
        
        # Generate recommendations
        avg_score = comprehensive_report["overall_compliance"]["average_score"]
        if avg_score < 80:
            comprehensive_report["recommendations"].append("Improve consent management processes")
        if avg_score < 60:
            comprehensive_report["recommendations"].append("Review and update data processing legal basis")
        if avg_score < 40:
            comprehensive_report["recommendations"].append("Implement comprehensive data governance framework")
        
        return comprehensive_report
    
    def _initialize_default_policies(self):
        """Initialize default compliance policies."""
        self.policies = {
            "data_retention": {
                "default_period_days": 365,
                "sensitive_data_period_days": 30,
                "anonymization_after_days": 1095
            },
            "consent_management": {
                "consent_expiry_days": 730,
                "require_explicit_consent": True,
                "allow_consent_withdrawal": True
            },
            "cross_border_transfer": {
                "require_adequacy_decision": True,
                "require_safeguards": True,
                "log_all_transfers": True
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Compliance and Data Privacy System...")
    
    # Initialize managers
    privacy_manager = DataPrivacyManager()
    compliance_manager = ComplianceManager()
    
    print("\n1. Testing data subject registration and consent:")
    
    # Register data subjects
    subjects = [
        ("user123", "user@example.com", "EU"),
        ("user456", "user2@example.com", "US-CA"),
        ("user789", "user3@example.com", "SG")
    ]
    
    for subject_id, email, jurisdiction in subjects:
        subject = privacy_manager.register_data_subject(subject_id, email, jurisdiction)
        
        # Record consent for research
        privacy_manager.record_consent(
            subject_id,
            ProcessingPurpose.RESEARCH,
            ConsentStatus.GRANTED,
            expires_at=datetime.now(timezone.utc) + timedelta(days=365),
            metadata={"allowed_categories": [DataCategory.BEHAVIORAL.value]}
        )
        
        print(f"  Registered {subject_id} ({jurisdiction}) with research consent")
    
    print("\n2. Testing processing activity recording:")
    
    # Record processing activities
    for subject_id, _, _ in subjects:
        privacy_manager.record_processing_activity(
            subject_id,
            ProcessingPurpose.RESEARCH,
            {DataCategory.BEHAVIORAL},
            processor="SNN-Fusion System",
            retention_period=timedelta(days=365)
        )
    
    print("  Recorded processing activities for all subjects")
    
    print("\n3. Testing data subject rights requests:")
    
    # Test access request
    access_response = privacy_manager.handle_data_subject_request("user123", "access")
    print(f"  Access request response status: {access_response['status']}")
    print(f"  Found {len(access_response.get('data', {}).get('processing_activities', []))} processing activities")
    
    # Test consent withdrawal
    withdrawal_success = privacy_manager.withdraw_consent("user123", ProcessingPurpose.RESEARCH)
    print(f"  Consent withdrawal: {'Success' if withdrawal_success else 'Failed'}")
    
    print("\n4. Testing compliance reporting:")
    
    # Generate compliance reports
    for regulation in [ComplianceRegulation.GDPR, ComplianceRegulation.CCPA]:
        report = privacy_manager.get_compliance_report(regulation)
        print(f"  {regulation.value} compliance score: {report['compliance_score']:.1f}%")
        print(f"    Total subjects: {report['data_subjects']['total']}")
        print(f"    With valid consent: {report['data_subjects']['with_consent']}")
    
    print("\n5. Testing comprehensive compliance report:")
    
    comprehensive_report = compliance_manager.generate_comprehensive_report()
    print(f"  Overall compliance score: {comprehensive_report['overall_compliance']['average_score']:.1f}%")
    print(f"  Supported regulations: {len(comprehensive_report['regulations'])}")
    
    if comprehensive_report["recommendations"]:
        print("  Recommendations:")
        for rec in comprehensive_report["recommendations"]:
            print(f"    - {rec}")
    
    print("\n6. Testing data anonymization:")
    
    anonymization_result = privacy_manager.anonymize_data("user789")
    print(f"  Anonymization status: {anonymization_result['status']}")
    if anonymization_result['status'] == 'completed':
        print(f"  New anonymous ID: {anonymization_result['anonymous_id']}")
    
    print("\n✓ Compliance and data privacy test completed!")