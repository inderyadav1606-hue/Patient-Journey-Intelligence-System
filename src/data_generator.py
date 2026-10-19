"""
Synthetic Longitudinal Healthcare EHR Data Generator
Generates realistic multi-table longitudinal EHR datasets including:
- Demographics & Patient Profiles
- Multi-encounter Longitudinal Timelines
- Longitudinal Vital Signs & Lab Biomarkers
- Unstructured Clinical Notes with Medical Narratives
- Prescribed Medications & Adherence
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

CONDITIONS = [
    {
        'name': 'Type 2 Diabetes',
        'icd10': 'E11.9',
        'risk_prob': 0.72,
        'meds': [('Metformin', 'Oral Hypoglycemic', '1000mg BID'), ('Glipizide', 'Sulfonylurea', '10mg Daily'), ('Insulin Glargine', 'Basal Insulin', '25 units QHS')],
        'symptoms': ['polydipsia and polyuria', 'blurred vision', 'diabetic peripheral neuropathy', 'poor glycemic control', 'fasting hyperglycemia']
    },
    {
        'name': 'Essential Hypertension',
        'icd10': 'I10',
        'risk_prob': 0.58,
        'meds': [('Lisinopril', 'ACE Inhibitor', '20mg Daily'), ('Amlodipine', 'Calcium Channel Blocker', '10mg Daily'), ('Hydrochlorothiazide', 'Thiazide Diuretic', '25mg Daily')],
        'symptoms': ['occipital headache', 'elevated blood pressure readings', 'episodic dizziness', 'palpitations', 'sub-optimal pressure control']
    },
    {
        'name': 'Congestive Heart Failure',
        'icd10': 'I50.9',
        'risk_prob': 0.88,
        'meds': [('Carvedilol', 'Beta Blocker', '25mg BID'), ('Furosemide', 'Loop Diuretic', '40mg Daily'), ('Spironolactone', 'Aldosterone Antagonist', '25mg Daily')],
        'symptoms': ['dyspnea on minimal exertion', 'orthopnea requiring 3 pillows', 'bilateral 2+ lower extremity edema', 'jugular venous distension', 'nocturnal paroxysmal dyspnea']
    },
    {
        'name': 'Chronic Obstructive Pulmonary Disease',
        'icd10': 'J44.9',
        'risk_prob': 0.75,
        'meds': [('Albuterol Inhaler', 'Short-Acting Beta Agonist', '2 puffs PRN'), ('Tiotropium', 'Anticholinergic', '18mcg Daily'), ('Fluticasone/Salmeterol', 'Inhaled Corticosteroid/LABA', '250/50mcg BID')],
        'symptoms': ['chronic productive cough', 'wheezing on expiration', 'exertional hypoxemia', 'frequent acute exacerbations', 'decreased exercise tolerance']
    },
    {
        'name': 'Chronic Kidney Disease',
        'icd10': 'N18.3',
        'risk_prob': 0.80,
        'meds': [('Losartan', 'ARB', '50mg Daily'), ('Sodium Bicarbonate', 'Alkalinizing Agent', '650mg BID'), ('Erythropoietin', 'Colony Stimulating Factor', '4000 units weekly')],
        'symptoms': ['declining eGFR trajectory', 'microalbuminuria', 'fatigue and metabolic acidosis', 'electrolyte imbalance', 'fluid retention']
    },
    {
        'name': 'Healthy / Routine Care',
        'icd10': 'Z00.0',
        'risk_prob': 0.10,
        'meds': [('Multivitamin', 'Nutritional Supplement', '1 tablet Daily'), ('Omega-3 Fish Oil', 'Supplement', '1000mg Daily')],
        'symptoms': ['no acute somatic complaints', 'well-controlled vital parameters', 'routine wellness evaluation', 'normal exercise tolerance', 'healthy dietary habits']
    }
]

ENCOUNTER_TYPES = ['Outpatient Visit', 'Emergency Department', 'Inpatient Admission', 'Telehealth Follow-up', 'Specialist Consult']


def generate_longitudinal_dataset(num_patients=1200, output_dir='data'):
    """Generate realistic longitudinal EHR data tables."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating longitudinal EHR dataset for {num_patients} patients...")

    patient_rows = []
    encounter_rows = []
    lab_rows = []
    note_rows = []
    medication_rows = []

    base_start_date = datetime(2024, 1, 1)

    for i in range(1, num_patients + 1):
        pid = f"PJ-{i:05d}"
        gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
        first_name = fake.first_name_male() if gender == 'Male' else fake.first_name_female()
        last_name = fake.last_name()
        age = int(np.random.normal(56, 17))
        age = max(19, min(89, age))

        # Primary condition assignment
        if age < 35:
            cond_weights = [0.10, 0.12, 0.03, 0.05, 0.03, 0.67]
        elif age < 60:
            cond_weights = [0.30, 0.32, 0.12, 0.10, 0.08, 0.08]
        else:
            cond_weights = [0.26, 0.28, 0.22, 0.12, 0.10, 0.02]

        cond_idx = np.random.choice(len(CONDITIONS), p=cond_weights)
        primary_cond = CONDITIONS[cond_idx]

        # BMI
        if primary_cond['name'] in ['Type 2 Diabetes', 'Essential Hypertension', 'Congestive Heart Failure']:
            bmi = np.round(np.random.normal(32.4, 4.8), 1)
        elif primary_cond['name'] == 'Healthy / Routine Care':
            bmi = np.round(np.random.normal(23.8, 2.9), 1)
        else:
            bmi = np.round(np.random.normal(27.5, 4.0), 1)
        bmi = max(16.5, min(52.0, bmi))

        smoking = np.random.choice(
            ['Never', 'Former', 'Active'],
            p=[0.35, 0.40, 0.25] if primary_cond['name'] in ['COPD', 'Congestive Heart Failure'] else [0.60, 0.28, 0.12]
        )
        insurance = np.random.choice(
            ['Medicare', 'Commercial', 'Medicaid', 'Uninsured'],
            p=[0.55, 0.30, 0.12, 0.03] if age >= 65 else [0.05, 0.68, 0.20, 0.07]
        )

        patient_rows.append({
            'Patient_ID': pid,
            'Name': f"{first_name} {last_name}",
            'Age': age,
            'Gender': gender,
            'BMI': bmi,
            'Smoking_Status': smoking,
            'Insurance_Type': insurance,
            'Primary_Condition': primary_cond['name'],
            'Primary_ICD10': primary_cond['icd10']
        })

        # Number of encounters across a 2-year timeline
        if primary_cond['name'] == 'Healthy / Routine Care':
            num_encounters = np.random.randint(1, 4)
        elif primary_cond['name'] in ['Congestive Heart Failure', 'Chronic Kidney Disease']:
            num_encounters = np.random.randint(4, 9)
        else:
            num_encounters = np.random.randint(3, 7)

        # Generating timeline
        days_offset = np.random.randint(0, 120)
        curr_date = base_start_date + timedelta(days=days_offset)

        # Baseline vitals and progression trajectory
        is_high_risk_trajectory = (np.random.rand() < primary_cond['risk_prob'])
        sys_bp_trend = 1.8 if is_high_risk_trajectory else -0.5
        glucose_trend = 2.4 if is_high_risk_trajectory else -0.8
        egfr_trend = -2.0 if is_high_risk_trajectory else -0.2

        curr_sys_bp = np.random.normal(145 if primary_cond['name'] in ['Essential Hypertension', 'Congestive Heart Failure'] else 122, 10)
        curr_glucose = np.random.normal(160 if primary_cond['name'] == 'Type 2 Diabetes' else 95, 20)
        curr_hba1c = np.random.normal(8.4 if primary_cond['name'] == 'Type 2 Diabetes' else 5.4, 0.8)
        curr_egfr = np.random.normal(48 if primary_cond['name'] == 'Chronic Kidney Disease' else 92, 10)

        for enc_idx in range(num_encounters):
            enc_id = f"ENC-{pid}-{enc_idx+1:02d}"
            interval_days = int(np.random.exponential(scale=65) + 15)
            curr_date += timedelta(days=interval_days)

            # Encounter type
            if is_high_risk_trajectory and enc_idx >= (num_encounters - 2):
                enc_type = np.random.choice(['Inpatient Admission', 'Emergency Department', 'Outpatient Visit'], p=[0.45, 0.35, 0.20])
            else:
                enc_type = np.random.choice(ENCOUNTER_TYPES, p=[0.60, 0.15, 0.08, 0.12, 0.05])

            los = np.random.randint(2, 9) if enc_type == 'Inpatient Admission' else (1 if enc_type == 'Emergency Department' else 0)
            cost = float(np.round(np.random.normal(12500, 3200) if enc_type == 'Inpatient Admission' else (np.random.normal(1850, 450) if enc_type == 'Emergency Department' else np.random.normal(260, 60)), 2))
            is_readmit_30d = 1 if (is_high_risk_trajectory and enc_idx > 0 and interval_days <= 30 and enc_type in ['Inpatient Admission', 'Emergency Department']) else 0

            encounter_rows.append({
                'Encounter_ID': enc_id,
                'Patient_ID': pid,
                'Encounter_Date': curr_date.strftime('%Y-%m-%d'),
                'Encounter_Type': enc_type,
                'Length_of_Stay_Days': los,
                'Primary_Diagnosis': primary_cond['name'],
                'Primary_ICD10': primary_cond['icd10'],
                'Encounter_Cost_USD': cost,
                'Is_Readmission_30d': is_readmit_30d
            })

            # Update labs with temporal drift
            curr_sys_bp += sys_bp_trend + np.random.normal(0, 4)
            curr_dia_bp = curr_sys_bp * 0.62 + np.random.normal(0, 3)
            curr_glucose += glucose_trend + np.random.normal(0, 6)
            curr_hba1c += (glucose_trend * 0.04) + np.random.normal(0, 0.15)
            curr_egfr += egfr_trend + np.random.normal(0, 1.5)

            # Clamping physiological limits
            sys_bp = float(np.clip(curr_sys_bp, 85, 215))
            dia_bp = float(np.clip(curr_dia_bp, 50, 130))
            glucose = float(np.clip(curr_glucose, 65, 380))
            hba1c = float(np.clip(curr_hba1c, 4.5, 14.2))
            egfr = float(np.clip(curr_egfr, 12, 125))

            cholesterol = float(np.random.normal(215 if primary_cond['name'] != 'Healthy / Routine Care' else 175, 25))
            ldl = float(np.random.normal(135 if is_high_risk_trajectory else 95, 20))
            hdl = float(np.random.normal(42 if is_high_risk_trajectory else 56, 8))
            creatinine = float(np.clip(np.random.normal(2.1 if primary_cond['name'] == 'Chronic Kidney Disease' else 0.95, 0.3), 0.5, 6.0))
            crp = float(np.clip(np.random.exponential(scale=6.5 if is_high_risk_trajectory else 1.2), 0.2, 45.0))
            troponin = float(np.clip(np.random.exponential(scale=0.08 if primary_cond['name'] == 'Congestive Heart Failure' else 0.01), 0.005, 1.8))

            lab_rows.append({
                'Lab_ID': f"LAB-{enc_id}",
                'Encounter_ID': enc_id,
                'Patient_ID': pid,
                'Test_Date': curr_date.strftime('%Y-%m-%d'),
                'Systolic_BP': round(sys_bp, 1),
                'Diastolic_BP': round(dia_bp, 1),
                'Glucose_mg_dL': round(glucose, 1),
                'HbA1c_pct': round(hba1c, 2),
                'Total_Cholesterol': round(cholesterol, 1),
                'LDL_Cholesterol': round(ldl, 1),
                'HDL_Cholesterol': round(hdl, 1),
                'Serum_Creatinine': round(creatinine, 2),
                'eGFR': round(egfr, 1),
                'CRP_mg_L': round(crp, 2),
                'Troponin_ng_mL': round(troponin, 3)
            })

            # Clinical Note Generation with domain medical language
            physician = fake.last_name()
            symptom_sample = random.sample(primary_cond['symptoms'], min(2, len(primary_cond['symptoms'])))
            symptom_str = " and ".join(symptom_sample)

            if is_high_risk_trajectory and (enc_type in ['Inpatient Admission', 'Emergency Department'] or enc_idx >= num_encounters - 2):
                narrative = (
                    f"CHIEF COMPLAINT: Acute exacerbation of {primary_cond['name']} with progressive deterioration. "
                    f"HISTORY: Patient presents reporting severe {symptom_str}. "
                    f"Symptoms have intensified over the preceding 2 weeks with noticeable functional decline and poor treatment adherence. "
                    f"PHYSICAL EXAM: Vitals markedly labile. Notable peripheral edema, respiratory distress, and signs of end-organ stress. "
                    f"ASSESSMENT & PLAN: High risk for 30-day hospital readmission and decompensation. Ordered urgent cardiology/endocrinology consult. "
                    f"Titrated pharmacotherapy aggressively and scheduled mandatory 7-day post-discharge care management follow-up."
                )
                note_risk = 1
            elif primary_cond['name'] == 'Healthy / Routine Care':
                narrative = (
                    f"CHIEF COMPLAINT: Routine scheduled preventive wellness evaluation. "
                    f"HISTORY: Patient reports feeling well with {symptom_str}. No complaints of chest pain, shortness of breath, or palpitations. "
                    f"PHYSICAL EXAM: Alert, oriented x4, no acute distress. Heart regular rate and rhythm, lungs clear to auscultation bilaterally. "
                    f"ASSESSMENT & PLAN: Stable baseline profile. Excellent therapeutic compliance. Continue current dietary regimen and annual routine follow-up."
                )
                note_risk = 0
            else:
                narrative = (
                    f"CHIEF COMPLAINT: Scheduled follow-up for chronic management of {primary_cond['name']}. "
                    f"HISTORY: Patient reports mild {symptom_str}. Compliance with prescribed medications is fair to moderate. "
                    f"PHYSICAL EXAM: Blood pressure moderately elevated at {round(sys_bp)}/{round(dia_bp)}. No acute decompensation noted today. "
                    f"ASSESSMENT & PLAN: Sub-optimal control of chronic condition. Adjusted daily dosing schedule and reinforced low-sodium diet."
                )
                note_risk = 1 if (sys_bp > 150 or glucose > 180 or hba1c > 8.5) else 0

            note_rows.append({
                'Note_ID': f"NOTE-{enc_id}",
                'Encounter_ID': enc_id,
                'Patient_ID': pid,
                'Note_Date': curr_date.strftime('%Y-%m-%d'),
                'Note_Type': 'Discharge Summary' if enc_type == 'Inpatient Admission' else ('Emergency Consult' if enc_type == 'Emergency Department' else 'Progress Note'),
                'Physician': f"Dr. {physician}, MD",
                'Clinical_Note': narrative,
                'High_Risk_Note_Flag': note_risk
            })

            # Medications
            for med_name, drug_class, dosage in primary_cond['meds']:
                medication_rows.append({
                    'Patient_ID': pid,
                    'Encounter_ID': enc_id,
                    'Medication_Name': med_name,
                    'Drug_Class': drug_class,
                    'Dosage': dosage,
                    'Adherence_Score': round(np.random.uniform(0.40, 0.75) if is_high_risk_trajectory else np.random.uniform(0.85, 1.00), 2)
                })

    df_patients = pd.DataFrame(patient_rows)
    df_encounters = pd.DataFrame(encounter_rows)
    df_labs = pd.DataFrame(lab_rows)
    df_notes = pd.DataFrame(note_rows)
    df_meds = pd.DataFrame(medication_rows)

    df_patients.to_csv(f"{output_dir}/patients.csv", index=False)
    df_encounters.to_csv(f"{output_dir}/encounters.csv", index=False)
    df_labs.to_csv(f"{output_dir}/lab_results.csv", index=False)
    df_notes.to_csv(f"{output_dir}/clinical_notes.csv", index=False)
    df_meds.to_csv(f"{output_dir}/medications.csv", index=False)

    print("Generated datasets successfully:")
    print(f"  Patients:    {len(df_patients)} records -> {output_dir}/patients.csv")
    print(f"  Encounters:  {len(df_encounters)} records -> {output_dir}/encounters.csv")
    print(f"  Lab Results: {len(df_labs)} records -> {output_dir}/lab_results.csv")
    print(f"  Notes:       {len(df_notes)} records -> {output_dir}/clinical_notes.csv")
    print(f"  Medications: {len(df_meds)} records -> {output_dir}/medications.csv")

    return df_patients, df_encounters, df_labs, df_notes, df_meds


if __name__ == '__main__':
    generate_longitudinal_dataset()
