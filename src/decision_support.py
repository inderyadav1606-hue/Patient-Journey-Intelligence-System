"""
Clinical Decision Support System (CDSS) & Counterfactual What-If Simulator
Translates machine learning risk predictions and longitudinal trajectory markers into:
- Evidence-based clinical recommendations and tailored care pathways
- Automated clinical alert triggers (renal decline, acute utilization, non-adherence)
- Counterfactual "What-If" scenario simulation for proactive risk mitigation
"""

import pandas as pd
import numpy as np


class ClinicalDecisionSupport:
    def __init__(self, risk_engine=None):
        self.risk_engine = risk_engine

    def generate_clinical_alerts(self, patient_row):
        """Generates real-time rule-based clinical warning flags."""
        alerts = []

        # 1. Blood Pressure Escalation
        if patient_row.get('Delta_Systolic_BP', 0) >= 15:
            alerts.append({
                'Severity': 'CRITICAL',
                'Category': 'Cardiovascular Drift',
                'Message': f"Systolic BP climbed by +{patient_row['Delta_Systolic_BP']} mmHg across recent encounters. Urgent antihypertensive titration indicated."
            })
        elif patient_row.get('Avg_Systolic_BP', 0) >= 150:
            alerts.append({
                'Severity': 'HIGH',
                'Category': 'Cardiovascular',
                'Message': f"Average Systolic BP ({patient_row['Avg_Systolic_BP']} mmHg) exceeds stage-2 hypertension threshold."
            })

        # 2. Glycemic Deterioration
        if patient_row.get('Delta_HbA1c', 0) >= 0.8:
            alerts.append({
                'Severity': 'HIGH',
                'Category': 'Endocrine / Glycemic',
                'Message': f"HbA1c escalated by +{patient_row['Delta_HbA1c']}% over longitudinal timeline. Assess insulin resistance and therapy intensification."
            })
        elif patient_row.get('Avg_HbA1c', 0) >= 9.0:
            alerts.append({
                'Severity': 'HIGH',
                'Category': 'Endocrine',
                'Message': f"Marked hyperglycemia (Average HbA1c {patient_row['Avg_HbA1c']}%). Elevated microvascular complication hazard."
            })

        # 3. Renal Decline
        if patient_row.get('Delta_eGFR', 0) <= -8.0 or patient_row.get('Min_eGFR', 100) < 45:
            alerts.append({
                'Severity': 'CRITICAL',
                'Category': 'Renal Decompensation',
                'Message': f"Renal function deteriorating (eGFR delta: {patient_row.get('Delta_eGFR', 0)} mL/min). Nephrology consult required."
            })

        # 4. Acute Care Utilization & Readmission Hazard
        if patient_row.get('Readmission_30d_Count', 0) > 0:
            alerts.append({
                'Severity': 'CRITICAL',
                'Category': 'Readmission Vulnerability',
                'Message': "Patient experienced a 30-day acute readmission. High priority for nurse case management enrollment."
            })
        elif patient_row.get('Acute_Utilization_Ratio', 0) >= 0.35:
            alerts.append({
                'Severity': 'HIGH',
                'Category': 'High Utilization',
                'Message': f"High acute utilization ratio ({patient_row.get('Acute_Utilization_Ratio', 0):.1%}). Evaluate care gaps."
            })

        # 5. Medication Adherence
        if patient_row.get('Mean_Med_Adherence', 1.0) < 0.70:
            alerts.append({
                'Severity': 'MEDIUM',
                'Category': 'Adherence Deficit',
                'Message': f"Low medication adherence ({patient_row.get('Mean_Med_Adherence', 0):.0%}). Pharmacist-led medication reconciliation advised."
            })

        # 6. NLP Clinical Progress Acuity
        if patient_row.get('NLP_Clinical_Acuity_Score', 0) >= 8.0:
            alerts.append({
                'Severity': 'HIGH',
                'Category': 'Clinical Progress Note Acuity',
                'Message': "Physician documentation indicates active symptom escalation, functional decline, or decompensation."
            })

        return alerts

    def recommend_interventions(self, risk_tier, patient_row, top_risk_drivers):
        """Generates structured clinical care pathway recommendations."""
        recommendations = []

        if risk_tier == 'High':
            recommendations.append({
                'Action': 'Mandatory Post-Discharge Transition Program',
                'Timeframe': 'Within 48-72 hours',
                'Details': 'Deploy community health worker / tele-nurse outreach to verify medication reconciliation and recovery trajectory.'
            })
            recommendations.append({
                'Action': 'Multi-Disciplinary Specialist Review',
                'Timeframe': 'Within 7 days',
                'Details': 'Schedule priority consultation with primary specialist (Cardiology, Endocrinology, or Nephrology).'
            })
            recommendations.append({
                'Action': 'Continuous Biometric Remote Patient Monitoring (RPM)',
                'Timeframe': 'Immediate enrollment',
                'Details': 'Equip patient with connected BP cuff, glucometer, or weight scale for daily biometric transmission.'
            })
        elif risk_tier == 'Moderate':
            recommendations.append({
                'Action': 'Targeted Care Coordination Check-in',
                'Timeframe': 'Within 14-21 days',
                'Details': 'Conduct telephone clinical check-in to review symptom stabilization and prescription refills.'
            })
            recommendations.append({
                'Action': 'Lifestyle & Diabetes/Hypertension Education',
                'Timeframe': 'Within 30 days',
                'Details': 'Enroll in structured dietary modification, sodium restriction, and disease self-management workshops.'
            })
        else:
            recommendations.append({
                'Action': 'Standard Preventative Wellness Track',
                'Timeframe': 'Annual / 6 months',
                'Details': 'Maintain current therapeutic protocol. Schedule routine laboratory screenings and wellness follow-up.'
            })

        # Custom tailoring based on top risk drivers
        driver_str = " ".join([d.lower() for d in top_risk_drivers])
        if 'bp' in driver_str or 'systolic' in driver_str:
            recommendations.append({
                'Action': 'Antihypertensive Optimization Protocol',
                'Timeframe': 'Next encounter',
                'Details': 'Assess combination therapy (ACE/ARB + CCB/diuretic) to achieve target BP < 130/80 mmHg.'
            })
        if 'glucose' in driver_str or 'hba1c' in driver_str:
            recommendations.append({
                'Action': 'Glycemic Intensification & Continuous Glucose Monitoring',
                'Timeframe': 'Within 2 weeks',
                'Details': 'Consider SGLT2 inhibitor / GLP-1 receptor agonist addition for organ protection and glycemic control.'
            })
        if 'adherence' in driver_str or 'compliance' in driver_str:
            recommendations.append({
                'Action': 'Pharmacy Adherence Packaging & Reminder App',
                'Timeframe': 'Immediate',
                'Details': 'Transition regimens to once-daily dosing where clinically feasible and implement blister-pack delivery.'
            })

        return recommendations

    def simulate_what_if_scenario(self, original_features, adjustments, risk_engine):
        """
        Simulates counterfactual clinical improvements:
        e.g., reductions in BP, glucose, or improvements in medication adherence.
        """
        modified_features = original_features.copy()

        for feature, new_val in adjustments.items():
            if feature in modified_features:
                modified_features[feature] = new_val

        # Predict baseline vs simulated
        baseline_res = risk_engine.predict_single_patient(original_features)
        simulated_res = risk_engine.predict_single_patient(modified_features)

        prob_delta = simulated_res['Risk_Probability'] - baseline_res['Risk_Probability']
        pct_reduction = (
            abs(prob_delta) / baseline_res['Risk_Probability'] * 100
            if baseline_res['Risk_Probability'] > 0 else 0
        )

        return {
            'Baseline_Risk_Prob': baseline_res['Risk_Probability'],
            'Baseline_Risk_Tier': baseline_res['Risk_Tier'],
            'Simulated_Risk_Prob': simulated_res['Risk_Probability'],
            'Simulated_Risk_Tier': simulated_res['Risk_Tier'],
            'Absolute_Risk_Change': prob_delta,
            'Relative_Risk_Reduction_Pct': pct_reduction,
            'Modified_Features': modified_features
        }
