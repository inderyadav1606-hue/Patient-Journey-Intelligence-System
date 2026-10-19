"""
EHR Data Cleaning, Integration & Longitudinal Feature Pipeline
Extracts multi-encounter patient trajectories, longitudinal biomarker trends,
and clinical utilization metrics.
"""

import os
import pandas as pd
import numpy as np


class HealthcareDataPipeline:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.patients = None
        self.encounters = None
        self.labs = None
        self.notes = None
        self.medications = None
        self.df_master = None

    def load_data(self):
        """Loads all raw EHR tables."""
        self.patients = pd.read_csv(os.path.join(self.data_dir, 'patients.csv'))
        self.encounters = pd.read_csv(os.path.join(self.data_dir, 'encounters.csv'))
        self.labs = pd.read_csv(os.path.join(self.data_dir, 'lab_results.csv'))
        self.notes = pd.read_csv(os.path.join(self.data_dir, 'clinical_notes.csv'))
        self.medications = pd.read_csv(os.path.join(self.data_dir, 'medications.csv'))

        # Date parsing
        self.encounters['Encounter_Date'] = pd.to_datetime(self.encounters['Encounter_Date'])
        self.labs['Test_Date'] = pd.to_datetime(self.labs['Test_Date'])
        self.notes['Note_Date'] = pd.to_datetime(self.notes['Note_Date'])

        return self.patients, self.encounters, self.labs, self.notes, self.medications

    def clean_data(self):
        """Cleans and validates records across EHR tables."""
        # Clean patients
        self.patients['Age'] = self.patients['Age'].clip(18, 100)
        self.patients['BMI'] = self.patients['BMI'].clip(15.0, 60.0)
        self.patients['BMI'] = self.patients['BMI'].fillna(self.patients['BMI'].median())

        # Clean encounters
        self.encounters['Length_of_Stay_Days'] = self.encounters['Length_of_Stay_Days'].fillna(0).clip(lower=0)
        self.encounters['Encounter_Cost_USD'] = self.encounters['Encounter_Cost_USD'].fillna(self.encounters['Encounter_Cost_USD'].median())
        self.encounters['Is_Readmission_30d'] = self.encounters['Is_Readmission_30d'].fillna(0).astype(int)

        # Clean labs
        num_lab_cols = ['Systolic_BP', 'Diastolic_BP', 'Glucose_mg_dL', 'HbA1c_pct', 
                        'Total_Cholesterol', 'LDL_Cholesterol', 'HDL_Cholesterol', 
                        'Serum_Creatinine', 'eGFR', 'CRP_mg_L', 'Troponin_ng_mL']
        for col in num_lab_cols:
            if col in self.labs.columns:
                self.labs[col] = self.labs[col].fillna(self.labs[col].median())

        # Clean notes
        self.notes['Clinical_Note'] = self.notes['Clinical_Note'].fillna("Routine encounter. No acute complaints.")

        # Clean medications
        self.medications['Adherence_Score'] = self.medications['Adherence_Score'].fillna(0.75).clip(0.0, 1.0)

    def extract_longitudinal_features(self):
        """
        Engineers temporal features tracking patient trajectory over time:
        - Encounter volume & acute utilization (ED, Inpatient visits)
        - 30-day readmissions
        - Trajectory slopes / deltas for vital biomarkers (Glucose, BP, eGFR, HbA1c)
        - Biomarker variability (standard deviation across visits)
        - Medication adherence and polypharmacy count
        """
        # 1. Encounter Features
        enc_agg = self.encounters.groupby('Patient_ID').agg(
            Total_Encounters=('Encounter_ID', 'count'),
            Inpatient_Visits=('Encounter_Type', lambda x: (x == 'Inpatient Admission').sum()),
            Emergency_Visits=('Encounter_Type', lambda x: (x == 'Emergency Department').sum()),
            Outpatient_Visits=('Encounter_Type', lambda x: (x == 'Outpatient Visit').sum()),
            Total_LOS_Days=('Length_of_Stay_Days', 'sum'),
            Total_Healthcare_Cost=('Encounter_Cost_USD', 'sum'),
            Readmission_30d_Count=('Is_Readmission_30d', 'sum')
        ).reset_index()

        enc_agg['Acute_Utilization_Ratio'] = (
            (enc_agg['Inpatient_Visits'] + enc_agg['Emergency_Visits']) / enc_agg['Total_Encounters'].replace(0, 1)
        ).round(3)

        # 2. Longitudinal Lab Features (Averages, Variability, and Temporal Slopes/Deltas)
        sorted_labs = self.labs.sort_values(by=['Patient_ID', 'Test_Date'])

        # Group stats
        lab_summary = sorted_labs.groupby('Patient_ID').agg(
            Avg_Systolic_BP=('Systolic_BP', 'mean'),
            Max_Systolic_BP=('Systolic_BP', 'max'),
            Std_Systolic_BP=('Systolic_BP', lambda x: x.std() if len(x) > 1 else 0.0),
            Avg_Diastolic_BP=('Diastolic_BP', 'mean'),
            Avg_Glucose=('Glucose_mg_dL', 'mean'),
            Max_Glucose=('Glucose_mg_dL', 'max'),
            Avg_HbA1c=('HbA1c_pct', 'mean'),
            Max_HbA1c=('HbA1c_pct', 'max'),
            Avg_eGFR=('eGFR', 'mean'),
            Min_eGFR=('eGFR', 'min'),
            Avg_Creatinine=('Serum_Creatinine', 'mean'),
            Avg_CRP=('CRP_mg_L', 'mean'),
            Max_CRP=('CRP_mg_L', 'max'),
            Avg_Troponin=('Troponin_ng_mL', 'mean')
        ).reset_index()

        # Compute longitudinal Delta (Last value - First value)
        first_labs = sorted_labs.groupby('Patient_ID').first().reset_index()
        last_labs = sorted_labs.groupby('Patient_ID').last().reset_index()

        delta_df = pd.DataFrame({'Patient_ID': first_labs['Patient_ID']})
        delta_df['Delta_Systolic_BP'] = (last_labs['Systolic_BP'] - first_labs['Systolic_BP']).round(1)
        delta_df['Delta_Glucose'] = (last_labs['Glucose_mg_dL'] - first_labs['Glucose_mg_dL']).round(1)
        delta_df['Delta_HbA1c'] = (last_labs['HbA1c_pct'] - first_labs['HbA1c_pct']).round(2)
        delta_df['Delta_eGFR'] = (last_labs['eGFR'] - first_labs['eGFR']).round(1)

        # 3. Medication & Adherence Features
        med_agg = self.medications.groupby('Patient_ID').agg(
            Mean_Med_Adherence=('Adherence_Score', 'mean'),
            Unique_Medications_Count=('Medication_Name', 'nunique')
        ).reset_index()

        # 4. Integrate all longitudinal features with Patient Demographics
        df_merged = self.patients.merge(enc_agg, on='Patient_ID', how='left')
        df_merged = df_merged.merge(lab_summary, on='Patient_ID', how='left')
        df_merged = df_merged.merge(delta_df, on='Patient_ID', how='left')
        df_merged = df_merged.merge(med_agg, on='Patient_ID', how='left')

        # Fill any remaining NaNs in numerical columns
        num_cols = df_merged.select_dtypes(include=[np.number]).columns
        df_merged[num_cols] = df_merged[num_cols].fillna(0)

        # 5. Define Ground Truth Target Risk (Deterioration & Readmission Risk)
        # Clinical criteria: Readmission occurred OR high acute utilization with worsening trajectory
        risk_condition = (
            (df_merged['Readmission_30d_Count'] > 0) |
            (df_merged['Inpatient_Visits'] >= 2) |
            ((df_merged['Emergency_Visits'] >= 2) & (df_merged['Avg_Systolic_BP'] > 140)) |
            ((df_merged['Avg_HbA1c'] >= 8.5) & (df_merged['Delta_HbA1c'] > 0.5)) |
            ((df_merged['Min_eGFR'] < 45) & (df_merged['Delta_eGFR'] < -5.0)) |
            (df_merged['Max_CRP'] > 12.0)
        )
        df_merged['Target_Risk'] = risk_condition.astype(int)

        self.df_master = df_merged
        return self.df_master

    def run_full_pipeline(self):
        """Executes full loading, cleaning and feature extraction."""
        self.load_data()
        self.clean_data()
        master = self.extract_longitudinal_features()
        print(f"Data pipeline complete. Master dataset shape: {master.shape}")
        print(f"Target Risk Distribution: {master['Target_Risk'].value_counts().to_dict()}")
        return master


if __name__ == '__main__':
    pipeline = HealthcareDataPipeline()
    pipeline.run_full_pipeline()
