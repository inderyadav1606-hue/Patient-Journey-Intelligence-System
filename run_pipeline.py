"""
Master Pipeline Orchestrator for Patient Journey Intelligence System
Executes:
1. Longitudinal EHR Data Generation / Ingestion
2. Data Cleaning & Trajectory Feature Engineering
3. Clinical NLP Engine (TF-IDF & Acuity Scoring)
4. Unsupervised Patient Clustering (K-Means + PCA)
5. Predictive Risk Modeling (XGBoost + Baselines)
6. Model Evaluation & Benchmarking
7. Explainable AI (SHAP TreeExplainer)
8. Artifact Serialization for Streamlit Dashboard
"""

import os
import sys
import time
import joblib
import pandas as pd

# Add current folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import generate_longitudinal_dataset
from data_pipeline import HealthcareDataPipeline
from nlp_engine import ClinicalNLPEngine
from clustering import PatientSegmentationEngine
from predictive_models import RiskPredictionEngine
from explainability import ExplainableAIEngine


def run_complete_pipeline(num_patients=1200, force_regenerate=False):
    start_time = time.time()
    print("=" * 70)
    print("PATIENT JOURNEY INTELLIGENCE SYSTEM (PJIS) - END-TO-END PIPELINE")
    print("=" * 70)

    data_dir = 'data'
    model_dir = 'models'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. Data Generation / Ingestion
    patients_file = os.path.join(data_dir, 'patients.csv')
    if force_regenerate or not os.path.exists(patients_file):
        print("\n[Step 1/7] Generating synthetic longitudinal EHR dataset...")
        generate_longitudinal_dataset(num_patients=num_patients, output_dir=data_dir)
    else:
        print("\n[Step 1/7] Found existing EHR dataset in 'data/' directory.")

    # 2. Data Cleaning & Longitudinal Feature Extraction
    print("\n[Step 2/7] Ingesting and extracting longitudinal trajectory features...")
    pipeline = HealthcareDataPipeline(data_dir=data_dir)
    master_df = pipeline.run_full_pipeline()

    # 3. Clinical NLP Engine
    print("\n[Step 3/7] Processing clinical notes via Clinical NLP Engine...")
    nlp = ClinicalNLPEngine(max_features=35, model_dir=model_dir)
    enriched_df = nlp.process_and_merge(master_df, pipeline.notes, is_training=True)
    print(f"Enriched dataset shape after NLP: {enriched_df.shape}")

    # 4. Unsupervised Patient Segmentation
    print("\n[Step 4/7] Applying K-Means clustering and PCA patient segmentation...")
    segmenter = PatientSegmentationEngine(n_clusters=4, model_dir=model_dir)
    df_segmented, sil_score, cluster_means = segmenter.fit_segmentation(enriched_df)
    cohort_profiles = segmenter.get_cluster_profiles(df_segmented)
    cohort_profiles.to_csv(os.path.join(data_dir, 'cohort_profiles.csv'), index=False)

    # Save complete enriched master dataset
    df_segmented.to_csv(os.path.join(data_dir, 'processed_patient_master.csv'), index=False)
    print(f"Saved processed master dataset ({len(df_segmented)} patients) to data/processed_patient_master.csv")

    # 5. Predictive Risk Modeling (XGBoost + Benchmarks)
    print("\n[Step 5/7] Training XGBoost and benchmark models for 30-day readmission risk...")
    risk_engine = RiskPredictionEngine(model_dir=model_dir)
    X_train_scaled, X_test_scaled, y_train, y_test = risk_engine.prepare_data(df_segmented)
    metrics = risk_engine.train_models()

    # 6. Explainable AI (SHAP)
    print("\n[Step 6/7] Initializing SHAP TreeExplainer for XGBoost model...")
    xai = ExplainableAIEngine(model_dir=model_dir)
    # Use 150 background samples for SHAP stability
    sample_size = min(150, len(X_test_scaled))
    xai.initialize_explainer(risk_engine.models['XGBoost'], X_test_scaled.iloc[:sample_size])
    global_importance = xai.get_global_feature_importance(top_n=20)
    global_importance.to_csv(os.path.join(data_dir, 'shap_global_importance.csv'), index=False)
    print("Top Global SHAP Risk Drivers:\n", global_importance.head(5))

    # 7. Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("=" * 70)
    print(f"Total Patients:       {len(df_segmented)}")
    print(f"Clustering Silhouette:{sil_score:.3f}")
    print(f"XGBoost Test ROC-AUC: {metrics['XGBoost']['ROC_AUC']:.4f}")
    print(f"XGBoost Test F1:      {metrics['XGBoost']['F1_Score']:.4f}")
    print(f"XGBoost Test Recall:  {metrics['XGBoost']['Recall']:.4f}")
    print("=" * 70)

    return df_segmented, metrics, global_importance


if __name__ == '__main__':
    run_complete_pipeline()
