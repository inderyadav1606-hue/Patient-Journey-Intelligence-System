"""
Predictive Risk Modeling Engine
Trains, compares, and evaluates Machine Learning classifiers with a primary focus on
XGBoost for longitudinal patient deterioration and 30-day readmission risk prediction.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

EXCLUDED_COLS = [
    'Patient_ID', 'Name', 'Primary_ICD10', 'Primary_Condition',
    'Cohort_Name', 'Target_Risk', 'PCA_1', 'PCA_2'
]


class RiskPredictionEngine:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.models = {}
        self.metrics = {}
        self.best_model_name = 'XGBoost'
        self.feature_names = []
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, df):
        """Prepares feature matrix X and target y with categorical encoding."""
        df_ml = df.copy()

        # Categorical encoding
        if 'Gender' in df_ml.columns:
            df_ml['Gender_Male'] = (df_ml['Gender'] == 'Male').astype(int)
            df_ml = df_ml.drop(columns=['Gender'])

        if 'Smoking_Status' in df_ml.columns:
            smoking_dummies = pd.get_dummies(df_ml['Smoking_Status'], prefix='Smoking', drop_first=True, dtype=int)
            df_ml = pd.concat([df_ml.drop(columns=['Smoking_Status']), smoking_dummies], axis=1)

        if 'Insurance_Type' in df_ml.columns:
            ins_dummies = pd.get_dummies(df_ml['Insurance_Type'], prefix='Ins', drop_first=True, dtype=int)
            df_ml = pd.concat([df_ml.drop(columns=['Insurance_Type']), ins_dummies], axis=1)

        # Select predictive features
        feature_cols = [c for c in df_ml.columns if c not in EXCLUDED_COLS]
        self.feature_names = feature_cols

        X = df_ml[feature_cols].copy().fillna(0)
        y = df_ml['Target_Risk'].astype(int)

        # Train-Test Split (Stratified 80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        self.X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=self.X_train.index)
        self.X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=self.X_test.index)

        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def train_models(self):
        """Trains Logistic Regression, Random Forest, and XGBoost classifiers."""
        print("Training predictive models...")

        # 1. XGBoost (Flagship Model)
        xgb = XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            random_state=42,
            eval_metric='logloss'
        )
        xgb.fit(self.X_train_scaled, self.y_train)
        self.models['XGBoost'] = xgb

        # 2. Random Forest
        rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_split=4,
            random_state=42
        )
        rf.fit(self.X_train_scaled, self.y_train)
        self.models['Random Forest'] = rf

        # 3. Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(self.X_train_scaled, self.y_train)
        self.models['Logistic Regression'] = lr

        # Evaluate all models
        self.evaluate_all_models()

        # Save artifacts
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.models['XGBoost'], os.path.join(self.model_dir, 'xgboost_risk_model.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'risk_scaler.pkl'))
        joblib.dump(self.feature_names, os.path.join(self.model_dir, 'feature_names.pkl'))
        joblib.dump(self.metrics, os.path.join(self.model_dir, 'evaluation_metrics.pkl'))

        return self.metrics

    def evaluate_all_models(self):
        """Computes comprehensive evaluation metrics for each model."""
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test_scaled)
            y_prob = model.predict_proba(self.X_test_scaled)[:, 1]

            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            auc = roc_auc_score(self.y_test, y_prob)
            pr_auc = average_precision_score(self.y_test, y_prob)
            cm = confusion_matrix(self.y_test, y_pred)
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            prec_curve, rec_curve, _ = precision_recall_curve(self.y_test, y_prob)

            # 5-fold cross validation on train set
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='roc_auc')

            self.metrics[name] = {
                'Accuracy': float(acc),
                'Precision': float(prec),
                'Recall': float(rec),
                'F1_Score': float(f1),
                'ROC_AUC': float(auc),
                'PR_AUC': float(pr_auc),
                'CV_ROC_AUC_Mean': float(cv_scores.mean()),
                'CV_ROC_AUC_Std': float(cv_scores.std()),
                'Confusion_Matrix': cm.tolist(),
                'FPR': fpr.tolist(),
                'TPR': tpr.tolist(),
                'Precision_Curve': prec_curve.tolist(),
                'Recall_Curve': rec_curve.tolist(),
                'Classification_Report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            print(f"[{name}] AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {rec:.4f} | Acc: {acc:.4f}")

    def predict_single_patient(self, patient_features):
        """Generates real-time risk score for an individual patient."""
        xgb = self.models.get('XGBoost')
        if xgb is None:
            xgb = joblib.load(os.path.join(self.model_dir, 'xgboost_risk_model.pkl'))
        scaler = self.scaler or joblib.load(os.path.join(self.model_dir, 'risk_scaler.pkl'))

        # Align columns
        aligned = pd.DataFrame([patient_features])[self.feature_names].fillna(0)
        scaled = scaler.transform(aligned)
        prob = xgb.predict_proba(scaled)[0, 1]
        pred = int(prob >= 0.5)

        return {
            'Risk_Probability': float(prob),
            'Risk_Prediction': pred,
            'Risk_Tier': 'High' if prob >= 0.65 else ('Moderate' if prob >= 0.35 else 'Low')
        }


if __name__ == '__main__':
    from data_pipeline import HealthcareDataPipeline
    from nlp_engine import ClinicalNLPEngine
    from clustering import PatientSegmentationEngine

    dp = HealthcareDataPipeline()
    master = dp.run_full_pipeline()
    nlp = ClinicalNLPEngine()
    enriched = nlp.process_and_merge(master, dp.notes)
    seg = PatientSegmentationEngine()
    df_clustered, _, _ = seg.fit_segmentation(enriched)

    rpe = RiskPredictionEngine()
    rpe.prepare_data(df_clustered)
    metrics = rpe.train_models()
