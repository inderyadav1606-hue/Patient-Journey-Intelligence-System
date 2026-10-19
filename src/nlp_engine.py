"""
Clinical NLP Engine
Processes unstructured clinical notes from longitudinal encounters:
- TF-IDF clinical feature extraction
- Clinical entity & acuity keyword extraction
- Clinical Note Risk Severity Index
- Merges patient-level NLP representations with longitudinal tabular data
"""

import re
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

CLINICAL_KEYWORDS = {
    'Cardiovascular': ['chest pain', 'dyspnea', 'palpitations', 'edema', 'orthopnea', 'cardiology', 'heart failure', 'troponin'],
    'Metabolic': ['glucose', 'glycemic', 'polyuria', 'polydipsia', 'diabetic', 'hba1c', 'hypoglycemia', 'hyperglycemia'],
    'Respiratory': ['cough', 'wheezing', 'hypoxemia', 'exacerbation', 'copd', 'inhaler', 'dyspnea on exertion'],
    'Renal': ['egfr', 'creatinine', 'acidosis', 'kidney', 'fluid retention', 'proteinuria'],
    'Acuity_Decompensation': ['acute exacerbation', 'decompensation', 'progressive deterioration', 'severe', 'readmission', 'labile', 'distress'],
    'Compliance_Deficit': ['non-adherent', 'poor adherence', 'sub-optimal', 'non-compliant', 'missed doses']
}


class ClinicalNLPEngine:
    def __init__(self, max_features=40, model_dir='models'):
        self.max_features = max_features
        self.model_dir = model_dir
        self.vectorizer = None
        self.feature_names = []

    def clean_text(self, text):
        """Standardizes clinical note text."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def aggregate_patient_notes(self, df_notes):
        """
        Concatenates all longitudinal clinical notes for each patient,
        preserving note chronology.
        """
        sorted_notes = df_notes.sort_values(by=['Patient_ID', 'Note_Date'])
        grouped = sorted_notes.groupby('Patient_ID').agg(
            Aggregated_Notes=('Clinical_Note', lambda texts: " ".join(texts)),
            Total_Notes_Count=('Note_ID', 'count'),
            High_Risk_Note_Count=('High_Risk_Note_Flag', 'sum')
        ).reset_index()

        grouped['Cleaned_Notes'] = grouped['Aggregated_Notes'].apply(self.clean_text)
        return grouped

    def compute_rule_based_nlp_features(self, df_patient_notes):
        """Extracts clinically grounded domain features from notes."""
        records = []
        for _, row in df_patient_notes.iterrows():
            text = row['Cleaned_Notes']
            feats = {'Patient_ID': row['Patient_ID']}

            # Domain condition flags
            for category, terms in CLINICAL_KEYWORDS.items():
                count = sum(text.count(term.lower()) for term in terms)
                feats[f'NLP_Count_{category}'] = count
                feats[f'NLP_Flag_{category}'] = 1 if count > 0 else 0

            # Composite Clinical Acuity Score
            acuity = (
                feats['NLP_Count_Acuity_Decompensation'] * 2.5 +
                feats['NLP_Count_Compliance_Deficit'] * 2.0 +
                feats['NLP_Count_Cardiovascular'] * 1.0 +
                feats['NLP_Count_Metabolic'] * 1.0 +
                feats['NLP_Count_Respiratory'] * 1.0 +
                feats['NLP_Count_Renal'] * 1.2
            )
            feats['NLP_Clinical_Acuity_Score'] = round(float(acuity), 2)
            records.append(feats)

        return pd.DataFrame(records)

    def fit_transform_tfidf(self, df_patient_notes):
        """Computes TF-IDF representations over aggregated clinical notes."""
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = self.vectorizer.fit_transform(df_patient_notes['Cleaned_Notes']).toarray()
        self.feature_names = [f"tfidf_{w.replace(' ', '_')}" for w in self.vectorizer.get_feature_names_out()]

        tfidf_df = pd.DataFrame(tfidf_matrix, columns=self.feature_names)
        tfidf_df['Patient_ID'] = df_patient_notes['Patient_ID'].values

        # Save fitted vectorizer
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(self.model_dir, 'clinical_tfidf_vectorizer.pkl'))
        return tfidf_df

    def transform_tfidf(self, df_patient_notes):
        """Transforms text using pre-fitted vectorizer."""
        if self.vectorizer is None:
            vec_path = os.path.join(self.model_dir, 'clinical_tfidf_vectorizer.pkl')
            if os.path.exists(vec_path):
                self.vectorizer = joblib.load(vec_path)
                self.feature_names = [f"tfidf_{w.replace(' ', '_')}" for w in self.vectorizer.get_feature_names_out()]
            else:
                raise ValueError("Vectorizer has not been fitted and no saved model found.")

        tfidf_matrix = self.vectorizer.transform(df_patient_notes['Cleaned_Notes']).toarray()
        tfidf_df = pd.DataFrame(tfidf_matrix, columns=self.feature_names)
        tfidf_df['Patient_ID'] = df_patient_notes['Patient_ID'].values
        return tfidf_df

    def process_and_merge(self, df_master, df_notes, is_training=True):
        """
        End-to-end NLP processing and merging with structured master DataFrame.
        """
        # Aggregate notes
        patient_notes = self.aggregate_patient_notes(df_notes)

        # Rule-based acuity & condition flags
        rule_features = self.compute_rule_based_nlp_features(patient_notes)

        # TF-IDF features
        if is_training:
            tfidf_features = self.fit_transform_tfidf(patient_notes)
        else:
            tfidf_features = self.transform_tfidf(patient_notes)

        # Combine all NLP representations
        nlp_combined = patient_notes[['Patient_ID', 'Total_Notes_Count', 'High_Risk_Note_Count']].merge(
            rule_features, on='Patient_ID', how='left'
        )
        nlp_combined = nlp_combined.merge(tfidf_features, on='Patient_ID', how='left')

        # Merge with master dataset
        df_full = df_master.merge(nlp_combined, on='Patient_ID', how='left')

        # Fill NaNs in text features with 0
        fill_cols = [c for c in df_full.columns if c.startswith('NLP_') or c.startswith('tfidf_') or c.endswith('_Count')]
        df_full[fill_cols] = df_full[fill_cols].fillna(0)

        return df_full


if __name__ == '__main__':
    from data_pipeline import HealthcareDataPipeline
    pipeline = HealthcareDataPipeline()
    master = pipeline.run_full_pipeline()
    nlp = ClinicalNLPEngine()
    enriched = nlp.process_and_merge(master, pipeline.notes, is_training=True)
    print(f"NLP enriched dataset shape: {enriched.shape}")
