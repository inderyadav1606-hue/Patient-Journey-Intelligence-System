"""
Explainable AI (SHAP) Module
Provides clinical interpretability for XGBoost predictions:
- Global SHAP feature importances & beeswarm summary
- Local patient-level SHAP attributions (Waterfall explanations)
- Clinician-interpretable risk driver narratives
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


class ExplainableAIEngine:
    def __init__(self, model=None, feature_names=None, model_dir='models'):
        self.model_dir = model_dir
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.sample_data = None

    def initialize_explainer(self, model, X_sample):
        """Initializes SHAP TreeExplainer with background sample data."""
        self.model = model
        self.feature_names = list(X_sample.columns)
        self.sample_data = X_sample

        print("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer(self.sample_data)

        # Cache explainer
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.explainer, os.path.join(self.model_dir, 'shap_tree_explainer.pkl'))
        return self.explainer, self.shap_values

    def get_global_feature_importance(self, top_n=15):
        """Calculates global mean absolute SHAP value for each feature."""
        if self.shap_values is None:
            raise ValueError("SHAP values have not been computed yet.")

        vals = self.shap_values.values
        if len(vals.shape) == 3:
            vals = vals[:, :, 1]

        mean_abs_shap = np.abs(vals).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_Absolute_SHAP': mean_abs_shap
        }).sort_values(by='Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)

        return importance_df.head(top_n)

    def explain_patient(self, patient_features_scaled_series, top_k=6):
        """
        Computes local SHAP explanation for an individual patient.
        Returns top positive (risk-increasing) and negative (risk-reducing) features.
        """
        if self.explainer is None:
            exp_path = os.path.join(self.model_dir, 'shap_tree_explainer.pkl')
            if os.path.exists(exp_path):
                self.explainer = joblib.load(exp_path)
            else:
                raise ValueError("SHAP explainer not initialized.")

        # Ensure 2D DataFrame
        if isinstance(patient_features_scaled_series, pd.Series):
            df_patient = pd.DataFrame([patient_features_scaled_series])
        else:
            df_patient = patient_features_scaled_series

        patient_shap = self.explainer(df_patient)
        values = patient_shap.values[0]
        if len(values.shape) > 1 and values.shape[-1] == 2:
            values = values[:, 1]

        base_val = patient_shap.base_values[0]
        if isinstance(base_val, np.ndarray):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]

        impact_df = pd.DataFrame({
            'Feature': df_patient.columns,
            'SHAP_Value': values,
            'Feature_Value': df_patient.iloc[0].values
        }).sort_values(by='SHAP_Value', key=abs, ascending=False).reset_index(drop=True)

        risk_drivers = impact_df[impact_df['SHAP_Value'] > 0].head(top_k)
        protective_factors = impact_df[impact_df['SHAP_Value'] < 0].head(top_k)

        # Clinician explanation narrative
        top_driver_names = [f.replace('_', ' ') for f in risk_drivers['Feature'].head(3)]
        top_prot_names = [f.replace('_', ' ') for f in protective_factors['Feature'].head(2)]

        summary_narrative = (
            f"Primary clinical factors driving elevated risk: {', '.join(top_driver_names)}. "
            f"Mitigating factors stabilizing patient: {', '.join(top_prot_names) if top_prot_names else 'None observed'}."
        )

        return {
            'Base_Value': float(base_val),
            'Impact_DataFrame': impact_df,
            'Risk_Drivers': risk_drivers,
            'Protective_Factors': protective_factors,
            'Clinical_Narrative': summary_narrative,
            'Patient_SHAP_Object': patient_shap
        }


if __name__ == '__main__':
    from data_pipeline import HealthcareDataPipeline
    from nlp_engine import ClinicalNLPEngine
    from clustering import PatientSegmentationEngine
    from predictive_models import RiskPredictionEngine

    dp = HealthcareDataPipeline()
    master = dp.run_full_pipeline()
    nlp = ClinicalNLPEngine()
    enriched = nlp.process_and_merge(master, dp.notes)
    seg = PatientSegmentationEngine()
    df_clustered, _, _ = seg.fit_segmentation(enriched)

    rpe = RiskPredictionEngine()
    X_train_scaled, X_test_scaled, y_train, y_test = rpe.prepare_data(df_clustered)
    rpe.train_models()

    xai = ExplainableAIEngine()
    xai.initialize_explainer(rpe.models['XGBoost'], X_test_scaled.iloc[:100])
    top_feats = xai.get_global_feature_importance()
    print("Top Global SHAP Drivers:\n", top_feats.head(7))
