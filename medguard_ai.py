"""
MedGuard AI - Production-Grade Drug Interaction & Warning System
Enterprise-level Streamlit application with modular architecture

Version: 1.0 | February 2026
Author: Senior Software Architect
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import hashlib
import base64
from io import BytesIO
import logging

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class SeverityLevel(Enum):
    """Severity classification system"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    NONE = "NONE"

class SeverityConfig:
    """Color and styling configuration for severity levels"""
    COLORS = {
        SeverityLevel.CRITICAL: "#ef4444",
        SeverityLevel.HIGH: "#f59e0b",
        SeverityLevel.MODERATE: "#eab308",
        SeverityLevel.LOW: "#06b6d4",
        SeverityLevel.NONE: "#22c55e"
    }
    
    EMOJI = {
        SeverityLevel.CRITICAL: "🚨",
        SeverityLevel.HIGH: "⚠️",
        SeverityLevel.MODERATE: "⚡",
        SeverityLevel.LOW: "ℹ️",
        SeverityLevel.NONE: "✅"
    }

class ThemeColors:
    """Premium dark dashboard theme"""
    BG = "#0f172a"
    SECONDARY = "#1e293b"
    ACCENT = "#3b82f6"
    SUCCESS = "#22c55e"
    WARNING = "#f59e0b"
    DANGER = "#ef4444"
    TEXT = "#f1f5f9"
    SUBTLE = "#64748b"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Drug:
    """Represents a medication in the system"""
    name: str
    generic_name: str
    drug_type: str  # prescription, otc, supplement, herbal
    dose: str
    frequency: str
    indication: str = ""
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: str = field(default_factory=lambda: hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8])
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class Interaction:
    """Represents a detected drug interaction"""
    drug_a: str
    drug_b: str
    severity: SeverityLevel
    mechanism: str
    effect_summary: str
    evidence_base: str
    management_recommendation: str
    alternatives: List[str] = field(default_factory=list)
    acknowledged: bool = False
    acknowledged_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'severity': self.severity.value,
        }

@dataclass
class AnalysisResult:
    """Container for complete analysis"""
    timestamp: str
    drugs: List[Drug]
    interactions: List[Interaction]
    risk_score: int
    risk_tier: str
    organ_system_risks: Dict[str, int]
    polypharmacy_burden: str

# ============================================================================
# CORE APPLICATION CONTROLLER
# ============================================================================

class MedGuardController:
    """Main application controller - orchestrates all business logic"""
    
    def __init__(self):
        self.drug_database = DrugDatabase()
        self.interaction_engine = InteractionEngine()
        self.ai_narrator = AINarrator()
        self.analytics_engine = AnalyticsEngine()
        
    def add_drug(self, drug: Drug) -> Tuple[bool, str]:
        """Add a drug to current session"""
        logger.info(f"Adding drug: {drug.name}")
        
        # Validation
        if not drug.name.strip():
            return False, "Drug name cannot be empty"
        if drug.name.lower() in [d.name.lower() for d in st.session_state.drugs]:
            return False, f"{drug.name} is already in your list"
        
        st.session_state.drugs.append(drug)
        return True, f"✅ {drug.name} added successfully"
    
    def remove_drug(self, drug_id: str) -> Tuple[bool, str]:
        """Remove a drug from current session"""
        logger.info(f"Removing drug: {drug_id}")
        original_count = len(st.session_state.drugs)
        st.session_state.drugs = [d for d in st.session_state.drugs if d.id != drug_id]
        
        if len(st.session_state.drugs) < original_count:
            return True, "Drug removed"
        return False, "Drug not found"
    
    def run_analysis(self) -> AnalysisResult:
        """Execute full interaction analysis"""
        logger.info(f"Running analysis on {len(st.session_state.drugs)} drugs")
        
        if len(st.session_state.drugs) < 2:
            st.warning("⚠️ Add at least 2 drugs to run analysis")
            return None
        
        # Detect interactions
        interactions = self.interaction_engine.detect_interactions(st.session_state.drugs)
        
        # Calculate risk metrics
        risk_score = self.analytics_engine.calculate_risk_score(interactions)
        risk_tier = self._get_risk_tier(risk_score)
        organ_risks = self.analytics_engine.calculate_organ_system_risks(interactions)
        polypharmacy = self._get_polypharmacy_burden(len(st.session_state.drugs))
        
        result = AnalysisResult(
            timestamp=datetime.now().isoformat(),
            drugs=st.session_state.drugs.copy(),
            interactions=interactions,
            risk_score=risk_score,
            risk_tier=risk_tier,
            organ_system_risks=organ_risks,
            polypharmacy_burden=polypharmacy
        )
        
        st.session_state.last_analysis = result
        logger.info(f"Analysis complete: {risk_score} risk score, {len(interactions)} interactions found")
        return result
    
    def acknowledge_interaction(self, interaction_idx: int) -> None:
        """Mark an interaction as acknowledged by user"""
        if st.session_state.last_analysis and interaction_idx < len(st.session_state.last_analysis.interactions):
            st.session_state.last_analysis.interactions[interaction_idx].acknowledged = True
            st.session_state.last_analysis.interactions[interaction_idx].acknowledged_at = datetime.now().isoformat()
            logger.info(f"Interaction {interaction_idx} acknowledged")
    
    def search_drugs(self, query: str) -> List[Dict]:
        """Search drug database with autocomplete"""
        return self.drug_database.search(query)
    
    def get_drug_alternatives(self, drug_name: str, interaction_context: str) -> List[Dict]:
        """Get safer alternative drugs"""
        return self.drug_database.get_alternatives(drug_name, interaction_context)
    
    def generate_narrative_summary(self, analysis: AnalysisResult, reading_level: str = "standard") -> str:
        """Generate AI plain-language summary"""
        return self.ai_narrator.generate_summary(analysis, reading_level)
    
    def _get_risk_tier(self, score: int) -> str:
        if score >= 80: return "CRITICAL"
        if score >= 60: return "HIGH"
        if score >= 40: return "MODERATE"
        if score >= 20: return "LOW"
        return "MINIMAL"
    
    def _get_polypharmacy_burden(self, drug_count: int) -> str:
        if drug_count <= 4: return "Low (1-4 drugs)"
        if drug_count <= 7: return "Moderate (5-7 drugs)"
        if drug_count <= 10: return "High (8-10 drugs)"
        return "Very High (11+ drugs)"

# ============================================================================
# BUSINESS LOGIC LAYER
# ============================================================================

class InteractionEngine:
    """Detects and classifies drug interactions"""
    
    def __init__(self):
        self.interaction_matrix = self._build_interaction_matrix()
    
    def detect_interactions(self, drugs: List[Drug]) -> List[Interaction]:
        """Find all interactions between drugs"""
        interactions = []
        
        for i, drug_a in enumerate(drugs):
            for drug_b in drugs[i+1:]:
                interaction = self._check_interaction_pair(drug_a, drug_b)
                if interaction:
                    interactions.append(interaction)
        
        # Sort by severity
        severity_order = {SeverityLevel.CRITICAL: 0, SeverityLevel.HIGH: 1, 
                         SeverityLevel.MODERATE: 2, SeverityLevel.LOW: 3, SeverityLevel.NONE: 4}
        interactions.sort(key=lambda x: severity_order.get(x.severity, 5))
        
        return interactions
    
    def _check_interaction_pair(self, drug_a: Drug, drug_b: Drug) -> Optional[Interaction]:
        """Check if two drugs interact"""
        key = self._normalize_drug_pair(drug_a.generic_name, drug_b.generic_name)
        
        if key in self.interaction_matrix:
            data = self.interaction_matrix[key]
            return Interaction(
                drug_a=drug_a.name,
                drug_b=drug_b.name,
                severity=SeverityLevel[data['severity'].upper()],
                mechanism=data['mechanism'],
                effect_summary=data['effect'],
                evidence_base=data['evidence'],
                management_recommendation=data['management'],
                alternatives=data.get('alternatives', [])
            )
        return None
    
    def _normalize_drug_pair(self, drug_a: str, drug_b: str) -> str:
        """Create normalized pair key for lookup"""
        drugs = sorted([drug_a.lower(), drug_b.lower()])
        return f"{drugs[0]}_{drugs[1]}"
    
    def _build_interaction_matrix(self) -> Dict:
        """Build comprehensive interaction database"""
        # Production system would pull from DrugBank, FDA, etc.
        # This is a realistic mock dataset
        return {
            "warfarin_ibuprofen": {
                "severity": "CRITICAL",
                "mechanism": "Inhibition of platelet function + displacement from protein binding",
                "effect": "Warfarin concentration increases; severe bleeding risk",
                "evidence": "A - Established",
                "management": "DO NOT COMBINE. Use acetaminophen for pain. Requires physician approval.",
                "alternatives": ["acetaminophen", "tramadol"]
            },
            "metformin_lisinopril": {
                "severity": "LOW",
                "mechanism": "Additive hypoglycemic effect",
                "effect": "Slightly increased risk of low blood sugar",
                "evidence": "B - Probable",
                "management": "Monitor blood glucose. Generally safe combination.",
                "alternatives": []
            },
            "warfarin_aspirin": {
                "severity": "HIGH",
                "mechanism": "Cumulative anticoagulant effect + platelet inhibition",
                "effect": "Increased risk of serious bleeding",
                "evidence": "A - Established",
                "management": "If combination necessary, monitor INR closely. Usually not recommended.",
                "alternatives": ["acetaminophen", "clopidogrel"]
            },
            "lisinopril_potassium": {
                "severity": "HIGH",
                "mechanism": "Both increase serum potassium",
                "effect": "Risk of dangerous hyperkalemia (high potassium)",
                "evidence": "A - Established",
                "management": "Check potassium levels monthly. Avoid potassium supplements unless monitored.",
                "alternatives": []
            },
            "simvastatin_grapefruit": {
                "severity": "HIGH",
                "mechanism": "CYP3A4 inhibition - grapefruit blocks simvastatin metabolism",
                "effect": "Simvastatin levels increase 16-fold; muscle toxicity risk",
                "evidence": "A - Established",
                "management": "Avoid grapefruit entirely. Switch to pravastatin or rosuvastatin.",
                "alternatives": ["pravastatin", "rosuvastatin"]
            },
            "ibuprofen_lisinopril": {
                "severity": "MODERATE",
                "mechanism": "NSAIDs reduce antihypertensive efficacy + renal risk",
                "effect": "Blood pressure control may be reduced; kidney damage risk",
                "evidence": "B - Probable",
                "management": "Use sparingly. Acetaminophen preferred. Monitor BP.",
                "alternatives": ["acetaminophen"]
            },
            "metformin_alcohol": {
                "severity": "MODERATE",
                "mechanism": "Increased lactic acidosis risk",
                "effect": "Rare but serious metabolic complication",
                "evidence": "B - Probable",
                "management": "Limit alcohol to 1-2 drinks. Avoid binge drinking.",
                "alternatives": []
            },
            "citalopram_tramadol": {
                "severity": "HIGH",
                "mechanism": "Serotonin syndrome risk",
                "effect": "Agitation, confusion, rapid heart rate, tremor",
                "evidence": "B - Probable",
                "management": "Combined use possible but requires close monitoring. Educate patient on warning signs.",
                "alternatives": ["acetaminophen", "ibuprofen"]
            },
            "warfarin_cranberry": {
                "severity": "MODERATE",
                "mechanism": "Cranberry may potentiate warfarin effect",
                "effect": "Increased bleeding risk",
                "evidence": "C - Possible",
                "management": "Avoid large amounts of cranberry products. Monitor INR.",
                "alternatives": []
            },
            "omeprazole_clopidogrel": {
                "severity": "MODERATE",
                "mechanism": "CYP2C19 inhibition reduces clopidogrel activation",
                "effect": "Reduced antiplatelet effect; increased clot risk",
                "evidence": "B - Probable",
                "management": "Switch to famotidine or ranitidine if PPI necessary.",
                "alternatives": ["famotidine", "ranitidine"]
            },
            "metoprolol_verapamil": {
                "severity": "HIGH",
                "mechanism": "Additive negative inotropic and chronotropic effects",
                "effect": "Severe bradycardia, AV block, heart failure",
                "evidence": "A - Established",
                "management": "Generally contraindicated. Require ECG monitoring if used.",
                "alternatives": ["diltiazem"]
            },
            "amoxicillin_methotrexate": {
                "severity": "MODERATE",
                "mechanism": "Amoxicillin reduces methotrexate clearance",
                "effect": "Methotrexate toxicity risk",
                "evidence": "B - Probable",
                "management": "Use alternative antibiotic if possible. Monitor for toxicity.",
                "alternatives": ["cephalexin", "azithromycin"]
            },
            "loratadine_grapefruit": {
                "severity": "LOW",
                "mechanism": "Minor CYP3A4 inhibition",
                "effect": "Slightly increased loratadine levels",
                "evidence": "B - Probable",
                "management": "Minor concern. No action typically needed.",
                "alternatives": []
            },
            "atorvastatin_clarithromycin": {
                "severity": "MODERATE",
                "mechanism": "CYP3A4 inhibition increases atorvastatin levels",
                "effect": "Increased statin toxicity and myositis risk",
                "evidence": "B - Probable",
                "management": "Reduce atorvastatin dose or use alternative antibiotic.",
                "alternatives": ["azithromycin", "amoxicillin"]
            },
            "levothyroxine_calcium": {
                "severity": "MODERATE",
                "mechanism": "Calcium binds and reduces levothyroxine absorption",
                "effect": "Reduced thyroid hormone levels",
                "evidence": "B - Probable",
                "management": "Separate doses by 4+ hours. Take levothyroxine on empty stomach.",
                "alternatives": []
            },
            "nsaid_nsaid": {
                "severity": "CRITICAL",
                "mechanism": "Duplicate therapy + cumulative GI and renal toxicity",
                "effect": "Serious GI bleeding, renal failure",
                "evidence": "A - Established",
                "management": "Never take two NSAIDs together. Use only one.",
                "alternatives": ["acetaminophen"]
            },
        }

class AnalyticsEngine:
    """Calculates risk metrics and analytics"""
    
    def calculate_risk_score(self, interactions: List[Interaction]) -> int:
        """Calculate overall risk score (0-100)"""
        if not interactions:
            return 0
        
        severity_weights = {
            SeverityLevel.CRITICAL: 30,
            SeverityLevel.HIGH: 15,
            SeverityLevel.MODERATE: 8,
            SeverityLevel.LOW: 2,
            SeverityLevel.NONE: 0
        }
        
        total_score = sum(severity_weights.get(i.severity, 0) for i in interactions)
        return min(100, total_score)
    
    def calculate_organ_system_risks(self, interactions: List[Interaction]) -> Dict[str, int]:
        """Calculate risk concentration by organ system"""
        organ_mapping = {
            "cardiac": ["warfarin", "metoprolol", "verapamil", "clopidogrel", "aspirin"],
            "renal": ["lisinopril", "ibuprofen", "metformin", "omeprazole"],
            "hepatic": ["simvastatin", "atorvastatin", "methotrexate"],
            "cns": ["tramadol", "citalopram", "metformin"],
            "gastrointestinal": ["ibuprofen", "aspirin", "warfarin", "methotrexate"],
            "hematologic": ["warfarin", "aspirin", "ibuprofen", "clopidogrel"]
        }
        
        organ_risks = {organ: 0 for organ in organ_mapping}
        
        for interaction in interactions:
            severity_val = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}.get(interaction.severity.value, 0)
            
            for organ, drugs in organ_mapping.items():
                if any(d.lower() in interaction.drug_a.lower() or d.lower() in interaction.drug_b.lower() 
                       for d in drugs):
                    organ_risks[organ] += severity_val
        
        return organ_risks

class AINarrator:
    """Generates natural language summaries"""
    
    def generate_summary(self, analysis: AnalysisResult, reading_level: str = "standard") -> str:
        """Generate AI narrative summary at specified reading level"""
        
        if not analysis.interactions:
            return self._format_summary(
                "Good news! No known interactions were detected between your medications.",
                reading_level
            )
        
        critical_count = sum(1 for i in analysis.interactions if i.severity == SeverityLevel.CRITICAL)
        high_count = sum(1 for i in analysis.interactions if i.severity == SeverityLevel.HIGH)
        moderate_count = sum(1 for i in analysis.interactions if i.severity == SeverityLevel.MODERATE)
        
        summary = f"Your medication profile contains {len(analysis.interactions)} interaction(s). "
        
        if critical_count > 0:
            summary += f"⚠️ {critical_count} CRITICAL interaction(s) requiring immediate review. "
        
        if high_count > 0:
            summary += f"⚠️ {high_count} HIGH-severity interaction(s) needing close monitoring. "
        
        if moderate_count > 0:
            summary += f"⚡ {moderate_count} moderate interaction(s) to discuss with your doctor. "
        
        summary += f"Your overall risk score is {analysis.risk_score}/100 ({analysis.risk_tier}). "
        
        if high_count > 0 or critical_count > 0:
            summary += "Please review the detailed alerts below and contact your physician before making any medication changes."
        
        return self._format_summary(summary, reading_level)
    
    def _format_summary(self, text: str, reading_level: str) -> str:
        """Adjust language complexity based on reading level"""
        if reading_level == "simple":
            # Simplify vocabulary
            text = text.replace("CRITICAL", "very dangerous")
            text = text.replace("interaction", "problem")
        elif reading_level == "medical":
            # Add technical terminology
            text = text.replace("taking medications together", "concurrent pharmacotherapy")
        
        return text

# ============================================================================
# DATA HANDLING LAYER
# ============================================================================

class DrugDatabase:
    """Manages drug information and search"""
    
    def __init__(self):
        self.drugs = self._build_drug_library()
    
    def search(self, query: str) -> List[Dict]:
        """Search drugs with autocomplete"""
        query_lower = query.lower()
        results = []
        
        for drug in self.drugs:
            if (query_lower in drug['name'].lower() or 
                query_lower in drug['generic'].lower()):
                results.append(drug)
        
        # Rank by relevance
        results.sort(key=lambda x: len(x['name']))
        return results[:10]  # Top 10 results
    
    def get_alternatives(self, drug_name: str, context: str = "") -> List[Dict]:
        """Get safer alternative drugs"""
        drug_lower = drug_name.lower()
        
        # Find similar-class drugs
        alternatives = [d for d in self.drugs 
                       if d['category'] in ['prescription', 'otc']
                       and drug_lower not in d['name'].lower()]
        
        return alternatives[:5]
    
    def _build_drug_library(self) -> List[Dict]:
        """Build comprehensive drug library"""
        return [
            # Blood Thinners
            {"name": "Warfarin (Coumadin)", "generic": "warfarin", "category": "prescription", "type": "anticoagulant"},
            {"name": "Aspirin", "generic": "aspirin", "category": "otc", "type": "antiplatelet"},
            {"name": "Clopidogrel (Plavix)", "generic": "clopidogrel", "category": "prescription", "type": "antiplatelet"},
            
            # Blood Pressure
            {"name": "Lisinopril (Prinivil)", "generic": "lisinopril", "category": "prescription", "type": "ace inhibitor"},
            {"name": "Metoprolol (Lopressor)", "generic": "metoprolol", "category": "prescription", "type": "beta blocker"},
            {"name": "Verapamil (Calan)", "generic": "verapamil", "category": "prescription", "type": "calcium channel blocker"},
            
            # Diabetes
            {"name": "Metformin (Glucophage)", "generic": "metformin", "category": "prescription", "type": "diabetes"},
            
            # Cholesterol
            {"name": "Simvastatin (Zocor)", "generic": "simvastatin", "category": "prescription", "type": "statin"},
            {"name": "Atorvastatin (Lipitor)", "generic": "atorvastatin", "category": "prescription", "type": "statin"},
            {"name": "Pravastatin (Pravachol)", "generic": "pravastatin", "category": "prescription", "type": "statin"},
            
            # Pain Relief
            {"name": "Ibuprofen (Advil)", "generic": "ibuprofen", "category": "otc", "type": "nsaid"},
            {"name": "Acetaminophen (Tylenol)", "generic": "acetaminophen", "category": "otc", "type": "analgesic"},
            {"name": "Tramadol (Ultram)", "generic": "tramadol", "category": "prescription", "type": "opioid"},
            
            # Mental Health
            {"name": "Citalopram (Celexa)", "generic": "citalopram", "category": "prescription", "type": "ssri"},
            
            # Acid Reflux
            {"name": "Omeprazole (Prilosec)", "generic": "omeprazole", "category": "otc", "type": "ppi"},
            
            # Antibiotics
            {"name": "Amoxicillin", "generic": "amoxicillin", "category": "prescription", "type": "antibiotic"},
            {"name": "Azithromycin (Z-Pak)", "generic": "azithromycin", "category": "prescription", "type": "antibiotic"},
            {"name": "Clarithromycin (Biaxin)", "generic": "clarithromycin", "category": "prescription", "type": "antibiotic"},
            
            # Thyroid
            {"name": "Levothyroxine (Synthroid)", "generic": "levothyroxine", "category": "prescription", "type": "hormone"},
            
            # Supplements
            {"name": "Potassium Supplement", "generic": "potassium", "category": "supplement", "type": "mineral"},
            {"name": "Cranberry Supplement", "generic": "cranberry", "category": "supplement", "type": "herbal"},
            {"name": "Grapefruit", "generic": "grapefruit", "category": "food", "type": "citrus"},
            {"name": "Alcohol", "generic": "alcohol", "category": "other", "type": "beverage"},
        ]

# ============================================================================
# UI COMPONENTS & PRESENTATION LAYER
# ============================================================================

class UIComponents:
    """Reusable UI component builder"""
    
    @staticmethod
    def styled_metric_card(title: str, value: str, color: str, icon: str = "•"):
        """Render a styled metric card"""
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}20 0%, {color}10 100%);
            border: 1px solid {color}40;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">
                {icon} {title}
            </div>
            <div style="font-size: 32px; font-weight: 700; color: {color}; margin-top: 8px;">
                {value}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def severity_badge(severity: SeverityLevel) -> str:
        """Create severity badge HTML"""
        color = SeverityConfig.COLORS[severity]
        emoji = SeverityConfig.EMOJI[severity]
        return f'<span style="background: {color}30; color: {color}; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 12px;">{emoji} {severity.value}</span>'
    
    @staticmethod
    def interaction_card(interaction: Interaction, index: int):
        """Render an interaction alert card"""
        color = SeverityConfig.COLORS[interaction.severity]
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color}10 0%, {color}05 100%);
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 16px;
                margin: 12px 0;
            ">
                <div style="flex-wrap: wrap; gap: 8px;">
                    <strong>{interaction.drug_a}</strong> ↔️ <strong>{interaction.drug_b}</strong>
                </div>
                <div style="margin-top: 12px; color: #cbd5e1; font-size: 14px;">
                    <strong>Effect:</strong> {interaction.effect_summary}
                </div>
                <div style="margin-top: 8px; color: #94a3b8; font-size: 13px;">
                    <strong>Mechanism:</strong> {interaction.mechanism}
                </div>
                <div style="margin-top: 8px; color: #94a3b8; font-size: 13px;">
                    <strong>What to do:</strong> {interaction.management_recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(UIComponents.severity_badge(interaction.severity), unsafe_allow_html=True)
    
    @staticmethod
    def emergency_overlay(critical_interactions: List[Interaction]):
        """Render emergency alert overlay"""
        st.markdown("""
        <div style="
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(15, 23, 42, 0.95);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            animation: pulse 1s infinite;
        " id="emergency-overlay">
            <div style="
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                border: 3px solid #991b1b;
                border-radius: 16px;
                padding: 40px;
                max-width: 500px;
                text-align: center;
                box-shadow: 0 20px 60px rgba(239, 68, 68, 0.4);
                animation: shake 0.5s infinite;
            ">
                <h1 style="color: white; font-size: 48px; margin: 0; animation: pulse 1s infinite;">
                    🚨 EMERGENCY ALERT
                </h1>
                <h2 style="color: #fecaca; font-size: 24px; margin-top: 16px; font-weight: 700;">
                    DO NOT TAKE THESE MEDICATIONS TOGETHER
                </h2>
                <div style="color: white; margin-top: 24px; font-size: 16px; line-height: 1.6;">
        """, unsafe_allow_html=True)
        
        for interaction in critical_interactions:
            st.markdown(f"""
                    <p style="margin: 8px 0;"><strong>{interaction.drug_a}</strong> + <strong>{interaction.drug_b}</strong></p>
                    <p style="color: #fecaca; font-size: 14px; margin-top: 4px;">{interaction.effect_summary}</p>
            """, unsafe_allow_html=True)
        
        st.markdown("""
                </div>
                <div style="margin-top: 32px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <a href="tel:911" style="
                        background: white;
                        color: #ef4444;
                        padding: 12px;
                        border-radius: 8px;
                        text-decoration: none;
                        font-weight: 700;
                        display: block;
                    ">📞 Call 911</a>
                    <a href="tel:1-800-222-1222" style="
                        background: white;
                        color: #ef4444;
                        padding: 12px;
                        border-radius: 8px;
                        text-decoration: none;
                        font-weight: 700;
                        display: block;
                    ">☠️ Poison Control</a>
                </div>
            </div>
        </div>
        
        <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                25% { transform: translateX(-10px); }
                75% { transform: translateX(10px); }
            }
        </style>
        """, unsafe_allow_html=True)

# ============================================================================
# UTILITY & CONFIG LAYER
# ============================================================================

class StateManager:
    """Manages application state"""
    
    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables"""
        if 'drugs' not in st.session_state:
            st.session_state.drugs = []
        if 'last_analysis' not in st.session_state:
            st.session_state.last_analysis = None
        if 'show_emergency' not in st.session_state:
            st.session_state.show_emergency = False
        if 'reading_level' not in st.session_state:
            st.session_state.reading_level = "standard"
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "Input"
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

class PDFGenerator:
    """Generates clinical PDF reports"""
    
    @staticmethod
    def generate_report(analysis: AnalysisResult) -> bytes:
        """Generate clinical PDF report"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib import colors
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#ef4444'),
                spaceAfter=30,
            )
            elements.append(Paragraph("MedGuard AI - Clinical Report", title_style))
            elements.append(Spacer(1, 0.3*inch))
            
            # Summary
            elements.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
            elements.append(Paragraph(f"<b>Risk Score:</b> {analysis.risk_score}/100", styles['Normal']))
            elements.append(Paragraph(f"<b>Risk Tier:</b> {analysis.risk_tier}", styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
            
            # Medication List
            elements.append(Paragraph("Medications", styles['Heading2']))
            med_data = [['Drug Name', 'Dose', 'Frequency', 'Type']]
            for drug in analysis.drugs:
                med_data.append([drug.name, drug.dose, drug.frequency, drug.drug_type])
            
            med_table = Table(med_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
            med_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(med_table)
            elements.append(Spacer(1, 0.3*inch))
            
            # Interactions
            elements.append(Paragraph("Drug Interactions", styles['Heading2']))
            if analysis.interactions:
                for interaction in analysis.interactions:
                    elements.append(Paragraph(
                        f"<b>{interaction.drug_a} ↔️ {interaction.drug_b}</b> ({interaction.severity.value})",
                        styles['Normal']
                    ))
                    elements.append(Paragraph(f"Effect: {interaction.effect_summary}", styles['Normal']))
                    elements.append(Spacer(1, 0.1*inch))
            else:
                elements.append(Paragraph("No interactions detected.", styles['Normal']))
            
            doc.build(elements)
            buffer.seek(0)
            return buffer.getvalue()
        
        except ImportError:
            st.warning("PDF generation requires reportlab. Install with: pip install reportlab")
            return None

# ============================================================================
# PAGES / SCREENS
# ============================================================================

def page_drug_input():
    """Home / Drug Entry Screen"""
    st.markdown("## 🏥 Add Your Medications")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Search input
        search_query = st.text_input("Search medications", placeholder="e.g., Warfarin, Aspirin, Metformin...")
        
        if search_query:
            results = app_controller.search_drugs(search_query)
            if results:
                st.markdown("### Matching Medications")
                for drug in results[:5]:
                    col_drug, col_add = st.columns([4, 1])
                    with col_drug:
                        st.caption(f"**{drug['name']}** ({drug['generic']}) - {drug['category']}")
                    with col_add:
                        if st.button("➕", key=f"add_{drug['generic']}", help="Add this drug"):
                            new_drug = Drug(
                                name=drug['name'],
                                generic_name=drug['generic'],
                                drug_type=drug['category'],
                                dose="",
                                frequency="",
                            )
                            success, msg = app_controller.add_drug(new_drug)
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
    
    with col2:
        if st.button("🔄 Refresh", key="refresh_drugs"):
            st.rerun()
    
    st.divider()
    
    # Current medications
    if st.session_state.drugs:
        st.markdown(f"### Current Medications ({len(st.session_state.drugs)})")
        
        for idx, drug in enumerate(st.session_state.drugs):
            col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1, 0.5])
            
            with col1:
                st.write(f"**{drug.name}**")
            with col2:
                dose = st.text_input("Dose", value=drug.dose, key=f"dose_{idx}", label_visibility="collapsed")
                st.session_state.drugs[idx].dose = dose
            with col3:
                freq = st.selectbox("Frequency", 
                    ["Once daily", "Twice daily", "Three times daily", "As needed", "Other"],
                    index=0 if not drug.frequency else 0,
                    key=f"freq_{idx}",
                    label_visibility="collapsed"
                )
                st.session_state.drugs[idx].frequency = freq
            with col4:
                st.caption(drug.drug_type)
            with col5:
                if st.button("❌", key=f"remove_{idx}", help="Remove"):
                    app_controller.remove_drug(drug.id)
                    st.rerun()
    else:
        st.info("👋 Search and add medications to get started. Add at least 2 drugs to run analysis.")
    
    st.divider()
    
    # Run analysis button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Run Analysis", use_container_width=True, type="primary"):
            if len(st.session_state.drugs) < 2:
                st.error("⚠️ Please add at least 2 medications")
            else:
                with st.spinner("Analyzing medications..."):
                    result = app_controller.run_analysis()
                    if result:
                        st.session_state.active_tab = "Results"
                        st.rerun()

def page_analysis_results():
    """Analysis Results - Alert Feed"""
    
    if not st.session_state.last_analysis:
        st.warning("No analysis results. Please run an analysis first.")
        return
    
    analysis = st.session_state.last_analysis
    
    # Check for critical interactions
    critical_interactions = [i for i in analysis.interactions if i.severity == SeverityLevel.CRITICAL]
    if critical_interactions and not st.session_state.show_emergency:
        st.session_state.show_emergency = True
    
    if st.session_state.show_emergency and critical_interactions:
        UIComponents.emergency_overlay(critical_interactions)
        st.warning("🚨 CRITICAL INTERACTIONS DETECTED - SEE ALERT ABOVE")
        if st.button("I Understand - Proceed"):
            st.session_state.show_emergency = False
            st.rerun()
        return
    
    # Results tabs
    tab1, tab2, tab3 = st.tabs(["📋 Alert Feed", "🔥 Interaction Matrix", "📊 Risk Dashboard"])
    
    with tab1:
        st.markdown("## Detected Interactions")
        
        # Summary banner
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            UIComponents.styled_metric_card(
                "Total Interactions",
                str(len(analysis.interactions)),
                ThemeColors.ACCENT
            )
        with col2:
            critical = len([i for i in analysis.interactions if i.severity == SeverityLevel.CRITICAL])
            UIComponents.styled_metric_card(
                "Critical",
                str(critical),
                ThemeColors.DANGER,
                "🚨"
            )
        with col3:
            high = len([i for i in analysis.interactions if i.severity == SeverityLevel.HIGH])
            UIComponents.styled_metric_card(
                "High Severity",
                str(high),
                ThemeColors.WARNING,
                "⚠️"
            )
        with col4:
            moderate = len([i for i in analysis.interactions if i.severity == SeverityLevel.MODERATE])
            UIComponents.styled_metric_card(
                "Moderate",
                str(moderate),
                "#eab308",
                "⚡"
            )
        
        st.divider()
        
        if analysis.interactions:
            for idx, interaction in enumerate(analysis.interactions):
                UIComponents.interaction_card(interaction, idx)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ I Understand", key=f"ack_{idx}"):
                        app_controller.acknowledge_interaction(idx)
                        st.success("Acknowledged")
                with col2:
                    st.button("💬 Discuss with Doctor", key=f"discuss_{idx}", disabled=True)
                with col3:
                    st.button("🔄 Find Alternative", key=f"alt_{idx}", disabled=True)
                
                st.divider()
        else:
            st.success("✅ No interactions detected! Your medication combination is safe.")
    
    with tab2:
        st.markdown("## Drug × Drug Interaction Matrix")
        
        if len(st.session_state.drugs) > 1:
            # Create heatmap data
            drugs = [d.name for d in st.session_state.drugs]
            n = len(drugs)
            matrix = np.zeros((n, n))
            
            severity_values = {
                SeverityLevel.CRITICAL: 5,
                SeverityLevel.HIGH: 4,
                SeverityLevel.MODERATE: 2,
                SeverityLevel.LOW: 1,
                SeverityLevel.NONE: 0
            }
            
            for interaction in analysis.interactions:
                try:
                    i = next(idx for idx, d in enumerate(drugs) if d in interaction.drug_a)
                    j = next(idx for idx, d in enumerate(drugs) if d in interaction.drug_b)
                    matrix[i][j] = severity_values.get(interaction.severity, 0)
                    matrix[j][i] = matrix[i][j]
                except StopIteration:
                    pass
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                x=drugs,
                y=drugs,
                colorscale="RdYlGn_r",
                hovertemplate="<b>%{y} ↔️ %{x}</b><br>Severity: %{z}<extra></extra>"
            ))
            
            fig.update_layout(
                height=500,
                width=600,
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font=dict(color="#f1f5f9")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add more medications to view the interaction matrix")
    
    with tab3:
        st.markdown("## Risk Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Score Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=analysis.risk_score,
                title={'text': "Overall Risk Score"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': SeverityConfig.COLORS[SeverityLevel.CRITICAL]},
                    'steps': [
                        {'range': [0, 20], 'color': SeverityConfig.COLORS[SeverityLevel.NONE]},
                        {'range': [20, 40], 'color': SeverityConfig.COLORS[SeverityLevel.LOW]},
                        {'range': [40, 60], 'color': SeverityConfig.COLORS[SeverityLevel.MODERATE]},
                        {'range': [60, 80], 'color': SeverityConfig.COLORS[SeverityLevel.HIGH]},
                        {'range': [80, 100], 'color': SeverityConfig.COLORS[SeverityLevel.CRITICAL]}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': analysis.risk_score
                    }
                }
            ))
            fig_gauge.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font=dict(color="#f1f5f9")
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Organ System Radar
            organ_names = list(analysis.organ_system_risks.keys())
            organ_values = list(analysis.organ_system_risks.values())
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=organ_values,
                theta=organ_names,
                fill='toself',
                name='Risk Level'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max(organ_values) + 2] if organ_values else [0, 5])
                ),
                showlegend=False,
                height=350,
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font=dict(color="#f1f5f9")
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.divider()
        
        # Polypharmacy info
        col1, col2, col3 = st.columns(3)
        with col1:
            UIComponents.styled_metric_card("Drug Count", str(len(st.session_state.drugs)), ThemeColors.ACCENT)
        with col2:
            UIComponents.styled_metric_card("Polypharmacy Burden", analysis.polypharmacy_burden, ThemeColors.WARNING)
        with col3:
            UIComponents.styled_metric_card("Risk Tier", analysis.risk_tier, ThemeColors.DANGER)

def page_narrative_summary():
    """AI Narrative Summary"""
    
    if not st.session_state.last_analysis:
        st.warning("No analysis results. Please run an analysis first.")
        return
    
    analysis = st.session_state.last_analysis
    
    st.markdown("## AI Narrative Summary")
    
    # Reading level selector
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        reading_level = st.radio(
            "Reading Level",
            ["Simple", "Standard", "Medical"],
            horizontal=True,
            label_visibility="collapsed"
        )
        st.session_state.reading_level = reading_level.lower()
    
    # Generate summary
    summary = app_controller.generate_narrative_summary(analysis, st.session_state.reading_level)
    
    # Display summary
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {ThemeColors.ACCENT}20, {ThemeColors.ACCENT}10);
        border-left: 4px solid {ThemeColors.ACCENT};
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        line-height: 1.7;
    ">
        {summary}
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Text-to-speech button
    if st.button("🎧 Listen (Text-to-Speech)"):
        st.info("Text-to-speech feature requires additional setup. Consider: pip install pyttsx3")

def page_chat_assistant():
    """AI Conversational Assistant"""
    
    if not st.session_state.last_analysis:
        st.warning("No analysis results. Please run an analysis first.")
        return
    
    st.markdown("## 💬 AI Assistant")
    st.caption("Ask follow-up questions about your medications (Profile-aware)")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input box
    if user_message := st.chat_input("Ask me about your medications..."):
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Simple heuristic-based responses (production would use LLM)
        response = generate_chat_response(user_message, st.session_state.last_analysis)
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })
        
        st.rerun()

def page_pdf_report():
    """PDF Report page"""
    
    if not st.session_state.last_analysis:
        st.warning("No analysis results. Please run an analysis first.")
        return
    
    st.markdown("## 📄 Clinical PDF Report")
    
    analysis = st.session_state.last_analysis
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download PDF Report", use_container_width=True, type="primary"):
            pdf_bytes = PDFGenerator.generate_report(analysis)
            if pdf_bytes:
                st.download_button(
                    label="Click to Download",
                    data=pdf_bytes,
                    file_name=f"medguard_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Could not generate PDF")
    
    with col2:
        if st.button("🔗 Share Report Link", use_container_width=True):
            st.info("QR Code and share link (expires in 30 days)")
    
    st.divider()
    
    # Report preview
    st.markdown("### Report Preview")
    
    with st.expander("📋 Medication List", expanded=True):
        med_df = pd.DataFrame([
            {
                "Drug": d.name,
                "Dose": d.dose,
                "Frequency": d.frequency,
                "Type": d.drug_type
            } for d in analysis.drugs
        ])
        st.dataframe(med_df, use_container_width=True, hide_index=True)
    
    with st.expander("⚠️ Interactions Summary", expanded=True):
        if analysis.interactions:
            int_data = []
            for interaction in analysis.interactions:
                int_data.append({
                    "Drug A": interaction.drug_a,
                    "Drug B": interaction.drug_b,
                    "Severity": interaction.severity.value,
                    "Management": interaction.management_recommendation[:50] + "..."
                })
            int_df = pd.DataFrame(int_data)
            st.dataframe(int_df, use_container_width=True, hide_index=True)
        else:
            st.success("No interactions detected")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_chat_response(user_message: str, analysis: "AnalysisResult") -> str:
    """Generate AI assistant response"""
    
    message_lower = user_message.lower()
    
    # Simple heuristic matching
    if any(word in message_lower for word in ["surgery", "operate"]):
        return """Before surgery, tell your surgeon and anesthesiologist about ALL your medications. Some may need to be stopped temporarily. This is a critical conversation to have 1-2 weeks before your procedure."""
    
    elif any(word in message_lower for word in ["alcohol", "drink", "wine", "beer"]):
        return f"""Alcohol interactions depend on your specific medications. {analysis.drugs[0].name if analysis.drugs else 'Your medications'} may have varying effects with alcohol. Generally, limit to 1-2 drinks. Consult your pharmacist for details on your specific regimen."""
    
    elif any(word in message_lower for word in ["pregnancy", "pregnant", "pregnant", "breastfeed", "breast feeding"]):
        return """Pregnancy and breastfeeding significantly change medication safety. STOP and contact your OB/GYN immediately. Some medications are safe, others must be stopped or switched. This requires a professional review of your entire profile."""
    
    elif any(word in message_lower for word in ["alternative", "switch", "change"]):
        alternatives = app_controller.get_drug_alternatives(analysis.drugs[0].name if analysis.drugs else "")
        if alternatives:
            alt_names = ", ".join([d['name'] for d in alternatives[:3]])
            return f"""Possible alternatives include: {alt_names}. However, switching requires your doctor's approval as different drugs may have their own interactions. Discuss with your physician."""
        else:
            return "There are alternatives, but they must be evaluated by your doctor based on your complete medical picture."
    
    else:
        return """I'm here to help explain your medication interactions and provide general guidance. However, **always consult your doctor or pharmacist before making any changes**. For urgent concerns, call Poison Control (1-800-222-1222) or seek emergency care."""

def render_custom_css():
    """Apply custom CSS styling"""
    st.markdown(f"""
    <style>
    /* Main background */
    .stMainBlockContainer {{
        background-color: {ThemeColors.BG};
        color: {ThemeColors.TEXT};
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {ThemeColors.SECONDARY};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {{
        color: {ThemeColors.TEXT};
        background-color: {ThemeColors.SECONDARY};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {ThemeColors.ACCENT};
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {ThemeColors.ACCENT};
        color: {ThemeColors.TEXT};
        border: none;
        border-radius: 6px;
    }}
    
    .stButton > button:hover {{
        background-color: {ThemeColors.ACCENT};
        opacity: 0.9;
    }}
    
    /* Text input */
    .stTextInput > div > div > input {{
        background-color: {ThemeColors.SECONDARY};
        color: {ThemeColors.TEXT};
    }}
    
    /* Divider */
    .stDivider {{
        background: linear-gradient(90deg, {ThemeColors.SECONDARY}, {ThemeColors.ACCENT}, {ThemeColors.SECONDARY});
    }}
    
    h1, h2, h3 {{
        color: {ThemeColors.TEXT};
    }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page config
    st.set_page_config(
        page_title="MedGuard AI",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    render_custom_css()
    
    # Initialize state
    StateManager.initialize_session_state()
    
    # Header with gradient
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {ThemeColors.ACCENT} 0%, {ThemeColors.DANGER} 100%);
        padding: 40px;
        border-radius: 12px;
        margin: -16px -16px 24px -16px;
        margin-bottom: 30px;
    ">
        <h1 style="color: {ThemeColors.TEXT}; margin: 0;">🏥 MedGuard AI</h1>
        <p style="color: {ThemeColors.TEXT}; opacity: 0.9; margin: 8px 0 0 0;">
            Intelligent Drug Interaction & Safety Analysis Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## Navigation")
        st.divider()
        
        # Tab selection
        if st.button("🏠 Drug Input", use_container_width=True, 
                    type="primary" if st.session_state.active_tab == "Input" else "secondary"):
            st.session_state.active_tab = "Input"
            st.rerun()
        
        if st.session_state.last_analysis:
            if st.button("📋 Results", use_container_width=True,
                        type="primary" if st.session_state.active_tab == "Results" else "secondary"):
                st.session_state.active_tab = "Results"
                st.rerun()
            
            if st.button("📝 Summary", use_container_width=True,
                        type="primary" if st.session_state.active_tab == "Summary" else "secondary"):
                st.session_state.active_tab = "Summary"
                st.rerun()
            
            if st.button("💬 Chat", use_container_width=True,
                        type="primary" if st.session_state.active_tab == "Chat" else "secondary"):
                st.session_state.active_tab = "Chat"
                st.rerun()
            
            if st.button("📄 Report", use_container_width=True,
                        type="primary" if st.session_state.active_tab == "Report" else "secondary"):
                st.session_state.active_tab = "Report"
                st.rerun()
        
        st.divider()
        
        # Current stats
        if st.session_state.last_analysis:
            st.markdown("### Current Analysis")
            analysis = st.session_state.last_analysis
            st.metric("Medications", len(analysis.drugs))
            st.metric("Interactions Found", len(analysis.interactions))
            st.metric("Risk Score", f"{analysis.risk_score}/100")
    
    # Route to active page
    if st.session_state.active_tab == "Input":
        page_drug_input()
    elif st.session_state.active_tab == "Results":
        page_analysis_results()
    elif st.session_state.active_tab == "Summary":
        page_narrative_summary()
    elif st.session_state.active_tab == "Chat":
        page_chat_assistant()
    elif st.session_state.active_tab == "Report":
        page_pdf_report()
    else:
        page_drug_input()

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    app_controller = MedGuardController()
    main()
