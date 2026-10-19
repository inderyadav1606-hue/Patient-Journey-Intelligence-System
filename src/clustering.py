"""
Patient Segmentation & Unsupervised Clustering Module
Applies K-Means and Principal Component Analysis (PCA) to discover distinct
longitudinal patient phenotyping cohorts for targeted healthcare interventions.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

CLUSTER_FEATURES = [
    'Age',
    'BMI',
    'Total_Encounters',
    'Inpatient_Visits',
    'Emergency_Visits',
    'Acute_Utilization_Ratio',
    'Avg_Systolic_BP',
    'Avg_Glucose',
    'Avg_HbA1c',
    'Avg_eGFR',
    'Max_CRP',
    'Mean_Med_Adherence',
    'NLP_Clinical_Acuity_Score'
]

COHORT_LABELS = {
    0: "Cohort A: Low-Risk / Well-Controlled",
    1: "Cohort B: Cardiometabolic Moderate-Risk",
    2: "Cohort C: High-Risk Multi-Morbid / Frequent Inpatient",
    3: "Cohort D: Acute Episodic / Transitional"
}


class PatientSegmentationEngine:
    def __init__(self, n_clusters=4, model_dir='models'):
        self.n_clusters = n_clusters
        self.model_dir = model_dir
        self.scaler = None
        self.kmeans = None
        self.pca = None
        self.cluster_names = {}

    def fit_segmentation(self, df):
        """
        Fits StandardScaler, KMeans, and PCA on patient feature matrix.
        """
        # Ensure required features exist
        available_features = [f for f in CLUSTER_FEATURES if f in df.columns]
        X = df[available_features].copy().fillna(0)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)

        # Evaluate clustering
        sil_score = silhouette_score(X_scaled, clusters)

        # Fit PCA for 2D visualization
        self.pca = PCA(n_components=2, random_state=42)
        pca_coords = self.pca.fit_transform(X_scaled)

        # Add results to dataframe
        df_segmented = df.copy()
        df_segmented['Cluster'] = clusters
        df_segmented['PCA_1'] = pca_coords[:, 0]
        df_segmented['PCA_2'] = pca_coords[:, 1]

        # Name cohorts dynamically based on risk profile
        cluster_means = df_segmented.groupby('Cluster')[available_features].mean()
        self._assign_cohort_names(cluster_means)
        df_segmented['Cohort_Name'] = df_segmented['Cluster'].map(self.cluster_names)

        # Save artifacts
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'clustering_scaler.pkl'))
        joblib.dump(self.kmeans, os.path.join(self.model_dir, 'kmeans_model.pkl'))
        joblib.dump(self.pca, os.path.join(self.model_dir, 'pca_model.pkl'))
        joblib.dump(self.cluster_names, os.path.join(self.model_dir, 'cluster_names.pkl'))

        print(f"Clustering complete. Silhouette Score: {sil_score:.3f}")
        return df_segmented, sil_score, cluster_means

    def _assign_cohort_names(self, cluster_means):
        """Maps clusters to meaningful clinical cohort descriptions."""
        # Rank by risk indicators
        if 'NLP_Clinical_Acuity_Score' in cluster_means.columns and 'Acute_Utilization_Ratio' in cluster_means.columns:
            risk_rank = (cluster_means['NLP_Clinical_Acuity_Score'] + cluster_means['Acute_Utilization_Ratio'] * 5).rank().astype(int)
            for c_id in cluster_means.index:
                rank = risk_rank.loc[c_id]
                if rank == self.n_clusters:
                    self.cluster_names[c_id] = "High-Risk Multi-Morbid Complex"
                elif rank == 1:
                    self.cluster_names[c_id] = "Stable / Routine Management"
                elif rank == 2:
                    self.cluster_names[c_id] = "Cardiometabolic At-Risk"
                else:
                    self.cluster_names[c_id] = "Acute Episodic / Fluctuating"
        else:
            self.cluster_names = COHORT_LABELS

    def get_cluster_profiles(self, df_segmented):
        """Generates cohort summary tables for comparative dashboards."""
        available_features = [f for f in CLUSTER_FEATURES if f in df_segmented.columns]
        summary = df_segmented.groupby(['Cluster', 'Cohort_Name'])[available_features + ['Target_Risk']].mean().reset_index()
        return summary


if __name__ == '__main__':
    from data_pipeline import HealthcareDataPipeline
    from nlp_engine import ClinicalNLPEngine

    dp = HealthcareDataPipeline()
    master = dp.run_full_pipeline()
    nlp = ClinicalNLPEngine()
    enriched = nlp.process_and_merge(master, dp.notes)

    segmenter = PatientSegmentationEngine(n_clusters=4)
    df_clustered, sil, means = segmenter.fit_segmentation(enriched)
    print("Cohort Distribution:")
    print(df_clustered['Cohort_Name'].value_counts())
