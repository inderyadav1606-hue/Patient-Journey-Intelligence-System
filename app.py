"""
Patient Journey Intelligence System (PJIS)
Interactive Streamlit Analytics Dashboard
Provides longitudinal patient analysis, clinical NLP insights, unsupervised clustering,
XGBoost predictive risk modeling, SHAP explainability, and clinical decision support.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from predictive_models import RiskPredictionEngine
from explainability import ExplainableAIEngine
from decision_support import ClinicalDecisionSupport

# Page configuration
st.set_page_config(
    page_title="Patient Journey Intelligence System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern healthcare styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .alert-critical {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        color: #991B1B;
    }
    .alert-high {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        color: #92400E;
    }
    .alert-info {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        color: #1E40AF;
    }
    .risk-badge-high {
        background-color: #EF4444;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
    .risk-badge-moderate {
        background-color: #F59E0B;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
    .risk-badge-low {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_datasets():
    """Loads cached tables from data directory."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    patients = pd.read_csv(os.path.join(data_dir, 'patients.csv'))
    encounters = pd.read_csv(os.path.join(data_dir, 'encounters.csv'))
    labs = pd.read_csv(os.path.join(data_dir, 'lab_results.csv'))
    notes = pd.read_csv(os.path.join(data_dir, 'clinical_notes.csv'))
    meds = pd.read_csv(os.path.join(data_dir, 'medications.csv'))
    master = pd.read_csv(os.path.join(data_dir, 'processed_patient_master.csv'))
    cohort_profiles = pd.read_csv(os.path.join(data_dir, 'cohort_profiles.csv'))
    shap_importance = pd.read_csv(os.path.join(data_dir, 'shap_global_importance.csv'))

    encounters['Encounter_Date'] = pd.to_datetime(encounters['Encounter_Date'])
    labs['Test_Date'] = pd.to_datetime(labs['Test_Date'])
    notes['Note_Date'] = pd.to_datetime(notes['Note_Date'])

    return patients, encounters, labs, notes, meds, master, cohort_profiles, shap_importance


@st.cache_resource
def load_models_and_engines():
    """Loads serialized models and initializes engines."""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    xgb_model = joblib.load(os.path.join(model_dir, 'xgboost_risk_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'risk_scaler.pkl'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    metrics = joblib.load(os.path.join(model_dir, 'evaluation_metrics.pkl'))

    risk_engine = RiskPredictionEngine(model_dir=model_dir)
    risk_engine.models['XGBoost'] = xgb_model
    risk_engine.scaler = scaler
    risk_engine.feature_names = feature_names

    xai_engine = ExplainableAIEngine(model_dir=model_dir)
    xai_engine.model = xgb_model
    xai_engine.feature_names = feature_names

    cdss = ClinicalDecisionSupport(risk_engine=risk_engine)

    return risk_engine, xai_engine, cdss, metrics


# Load data and models
patients_df, encounters_df, labs_df, notes_df, meds_df, master_df, cohort_profiles_df, shap_importance_df = load_all_datasets()
risk_engine, xai_engine, cdss, metrics = load_models_and_engines()

# Header
st.markdown('<div class="main-header">🏥 Patient Journey Intelligence System (PJIS)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-end longitudinal healthcare analytics platform integrating EHR encounters, biomarker trajectories, and clinical NLP notes for precision risk prediction and decision support.</div>', unsafe_allow_html=True)

# Executive KPI Ribbon
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Patient Cohort", f"{len(master_df):,}")
with col2:
    st.metric("Longitudinal Encounters", f"{len(encounters_df):,}")
with col3:
    high_risk_pct = (master_df['Target_Risk'].mean() * 100)
    st.metric("High-Risk Decompensation Rate", f"{high_risk_pct:.1f}%")
with col4:
    readmits = master_df['Readmission_30d_Count'].sum()
    st.metric("30-Day Readmissions", f"{int(readmits):,}")
with col5:
    xgb_auc = metrics['XGBoost']['ROC_AUC']
    st.metric("XGBoost Model ROC-AUC", f"{xgb_auc:.4f}")

st.markdown("---")

# Navigation Tabs
tabs = st.tabs([
    "📊 Population Health & EDA",
    "🗺️ Patient Journey Explorer",
    "👥 Cohort Segmentation",
    "🤖 Predictive Modeling (XGBoost)",
    "🔍 Explainable AI (SHAP)",
    "💡 Decision Support & What-If"
])

# ==========================================
# TAB 1: Population Health & EDA
# ==========================================
with tabs[0]:
    st.subheader("Population Health Overview & Exploratory Data Analysis")
    st.markdown("Comprehensive demographic, diagnostic, and clinical encounter distributions across the active cohort.")

    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        st.write("##### Primary Clinical Conditions Distribution")
        fig_cond, ax_cond = plt.subplots(figsize=(8, 4))
        cond_counts = master_df['Primary_Condition'].value_counts()
        sns.barplot(x=cond_counts.values, y=cond_counts.index, palette="Blues_r", ax=ax_cond)
        ax_cond.set_xlabel("Number of Patients")
        ax_cond.set_title("Condition Prevalence")
        st.pyplot(fig_cond)

    with r1_col2:
        st.write("##### Age vs. BMI Stratified by Risk Category")
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            data=master_df,
            x='Age', y='BMI',
            hue='Target_Risk',
            palette={0: '#10B981', 1: '#EF4444'},
            alpha=0.7,
            ax=ax_scatter
        )
        ax_scatter.set_title("Age vs BMI Distribution (0: Low Risk, 1: High Risk)")
        st.pyplot(fig_scatter)

    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        st.write("##### Longitudinal Encounter Types Breakdown")
        fig_enc, ax_enc = plt.subplots(figsize=(8, 4))
        enc_counts = encounters_df['Encounter_Type'].value_counts()
        ax_enc.pie(enc_counts, labels=enc_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
        ax_enc.set_title("Encounter Care Setting Volume")
        st.pyplot(fig_enc)

    with r2_col2:
        st.write("##### Total Healthcare Expenditure by Primary Condition")
        fig_cost, ax_cost = plt.subplots(figsize=(8, 4))
        cost_df = master_df.groupby('Primary_Condition')['Total_Healthcare_Cost'].mean().sort_values()
        sns.barplot(x=cost_df.values, y=cost_df.index, palette="viridis", ax=ax_cost)
        ax_cost.set_xlabel("Mean Total Expenditure (USD)")
        ax_cost.set_title("Average Cost per Patient by Diagnosis")
        st.pyplot(fig_cost)


# ==========================================
# TAB 2: Longitudinal Patient Journey Explorer
# ==========================================
with tabs[1]:
    st.subheader("Longitudinal Patient Trajectory & Journey Explorer")
    st.markdown("Track an individual patient's medical timeline across chronological encounters, lab fluctuations, and progress notes.")

    # Patient Selector
    patient_ids = master_df['Patient_ID'].tolist()
    selected_pid = st.selectbox("Select Patient to Inspect Journey:", patient_ids, index=0)

    p_data = master_df[master_df['Patient_ID'] == selected_pid].iloc[0]
    p_encounters = encounters_df[encounters_df['Patient_ID'] == selected_pid].sort_values('Encounter_Date')
    p_labs = labs_df[labs_df['Patient_ID'] == selected_pid].sort_values('Test_Date')
    p_notes = notes_df[notes_df['Patient_ID'] == selected_pid].sort_values('Note_Date')
    p_meds = meds_df[meds_df['Patient_ID'] == selected_pid]

    # Patient Demographics Card
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        st.write(f"**Name:** {p_data.get('Name', 'N/A')}")
        st.write(f"**Patient ID:** `{selected_pid}`")
    with p_col2:
        st.write(f"**Age / Gender:** {int(p_data['Age'])} yrs / {p_data['Gender']}")
        st.write(f"**BMI:** {p_data['BMI']} kg/m²")
    with p_col3:
        st.write(f"**Primary Diagnosis:** {p_data['Primary_Condition']}")
        st.write(f"**ICD-10 Code:** `{p_data['Primary_ICD10']}`")
    with p_col4:
        st.write(f"**Insurance:** {p_data['Insurance_Type']}")
        st.write(f"**Smoking Status:** {p_data['Smoking_Status']}")
    with p_col5:
        tier = 'HIGH RISK' if p_data['Target_Risk'] == 1 else 'LOW / STABLE'
        badge_class = 'risk-badge-high' if p_data['Target_Risk'] == 1 else 'risk-badge-low'
        st.markdown(f"**Status:** <span class='{badge_class}'>{tier}</span>", unsafe_allow_html=True)
        st.write(f"**Adherence Score:** {p_data['Mean_Med_Adherence']:.0%}")

    st.write("---")

    # Longitudinal Trajectory Biomarker Trends
    st.write("#### 📈 Longitudinal Biomarker Trajectories (Time-Series Trends)")
    traj_col1, traj_col2, traj_col3 = st.columns(3)

    with traj_col1:
        fig_bp, ax_bp = plt.subplots(figsize=(6, 3))
        ax_bp.plot(p_labs['Test_Date'].dt.strftime('%b %d'), p_labs['Systolic_BP'], marker='o', color='#EF4444', label='Systolic BP')
        ax_bp.plot(p_labs['Test_Date'].dt.strftime('%b %d'), p_labs['Diastolic_BP'], marker='s', color='#3B82F6', label='Diastolic BP')
        ax_bp.axhline(140, color='gray', linestyle='--', alpha=0.6, label='Stage 2 Limit')
        ax_bp.set_title(f"BP Trajectory (Delta: +{p_data['Delta_Systolic_BP']} mmHg)")
        ax_bp.set_ylabel("mmHg")
        ax_bp.tick_params(axis='x', rotation=30)
        ax_bp.legend(fontsize=8)
        st.pyplot(fig_bp)

    with traj_col2:
        fig_glu, ax_glu = plt.subplots(figsize=(6, 3))
        ax_glu.plot(p_labs['Test_Date'].dt.strftime('%b %d'), p_labs['Glucose_mg_dL'], marker='^', color='#F59E0B', label='Glucose')
        ax_glu.plot(p_labs['Test_Date'].dt.strftime('%b %d'), p_labs['HbA1c_pct'] * 20, marker='d', color='#8B5CF6', label='HbA1c (x20)')
        ax_glu.set_title(f"Glycemic Drift (HbA1c: {p_data['Avg_HbA1c']}%)")
        ax_glu.set_ylabel("mg/dL")
        ax_glu.tick_params(axis='x', rotation=30)
        ax_glu.legend(fontsize=8)
        st.pyplot(fig_glu)

    with traj_col3:
        fig_egfr, ax_egfr = plt.subplots(figsize=(6, 3))
        ax_egfr.plot(p_labs['Test_Date'].dt.strftime('%b %d'), p_labs['eGFR'], marker='o', color='#10B981', label='eGFR')
        ax_egfr.axhline(60, color='red', linestyle='--', alpha=0.6, label='CKD Threshold')
        ax_egfr.set_title(f"Renal Function (eGFR Delta: {p_data['Delta_eGFR']})")
        ax_egfr.set_ylabel("mL/min/1.73m²")
        ax_egfr.tick_params(axis='x', rotation=30)
        ax_egfr.legend(fontsize=8)
        st.pyplot(fig_egfr)

    # Encounter Journey Timeline Table
    st.write("#### 🏥 Chronological Encounter Journey History")
    enc_view = p_encounters[['Encounter_Date', 'Encounter_Type', 'Length_of_Stay_Days', 'Primary_Diagnosis', 'Encounter_Cost_USD', 'Is_Readmission_30d']].copy()
    enc_view['Encounter_Date'] = enc_view['Encounter_Date'].dt.strftime('%Y-%m-%d')
    enc_view.rename(columns={
        'Length_of_Stay_Days': 'LOS (Days)',
        'Encounter_Cost_USD': 'Cost ($)',
        'Is_Readmission_30d': '30d Readmission'
    }, inplace=True)
    st.dataframe(enc_view, use_container_width=True)

    # Clinical Progress Notes Expander
    st.write("#### 📝 Longitudinal Clinical Progress Notes")
    for _, note_row in p_notes.iterrows():
        date_str = note_row['Note_Date'].strftime('%Y-%m-%d')
        acuity_tag = "🚨 High Acuity Note" if note_row.get('High_Risk_Note_Flag', 0) == 1 else "✅ Routine Note"
        with st.expander(f"📅 {date_str} — {note_row['Note_Type']} ({note_row['Physician']}) — {acuity_tag}"):
            st.markdown(f"**Clinical Narrative:**")
            st.write(note_row['Clinical_Note'])


# ==========================================
# TAB 3: Cohort Segmentation
# ==========================================
with tabs[2]:
    st.subheader("Unsupervised Patient Segmentation & Cohort Phenotyping")
    st.markdown("K-Means unsupervised clustering combined with PCA dimensionality reduction to isolate high-risk phenotypes for population management.")

    seg_col1, seg_col2 = st.columns([3, 2])

    with seg_col1:
        st.write("##### 2D PCA Projection of Patient Phenotypes")
        fig_pca, ax_pca = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=master_df,
            x='PCA_1', y='PCA_2',
            hue='Cohort_Name',
            palette='tab10',
            alpha=0.8,
            s=50,
            ax=ax_pca
        )
        ax_pca.set_title("Patient Clusters in Latent Space (PCA)")
        ax_pca.set_xlabel("Principal Component 1")
        ax_pca.set_ylabel("Principal Component 2")
        ax_pca.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        st.pyplot(fig_pca)

    with seg_col2:
        st.write("##### Cohort Size Distribution")
        cohort_counts = master_df['Cohort_Name'].value_counts()
        st.dataframe(pd.DataFrame({'Patients': cohort_counts, 'Share': (cohort_counts / len(master_df) * 100).round(1).astype(str) + '%'}), use_container_width=True)

        st.info("💡 **Clinical Phenotyping Utility:** Clusters isolate specific clinical profiles ranging from complex multi-morbid patients requiring intensive nurse coordination to stable preventive cohorts.")

    st.write("##### Cohort Profiling Comparison Matrix")
    st.dataframe(cohort_profiles_df, use_container_width=True)


# ==========================================
# TAB 4: Predictive Modeling (XGBoost)
# ==========================================
with tabs[3]:
    st.subheader("Predictive Risk Modeling: XGBoost vs. Baseline Benchmarks")
    st.markdown("Supervised machine learning pipeline predicting 30-day readmission and severe clinical deterioration risk.")

    # Benchmark Comparison Table
    benchmarks = []
    for model_name, m in metrics.items():
        benchmarks.append({
            'Model': model_name,
            'ROC-AUC': f"{m['ROC_AUC']:.4f}",
            'PR-AUC': f"{m['PR_AUC']:.4f}",
            'Accuracy': f"{m['Accuracy']:.2%}",
            'Precision': f"{m['Precision']:.2%}",
            'Recall': f"{m['Recall']:.2%}",
            'F1-Score': f"{m['F1_Score']:.4f}",
            '5-Fold CV AUC': f"{m['CV_ROC_AUC_Mean']:.4f} ± {m['CV_ROC_AUC_Std']:.4f}"
        })
    st.dataframe(pd.DataFrame(benchmarks), use_container_width=True)

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.write("##### ROC Curve Comparison")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        for m_name, m in metrics.items():
            ax_roc.plot(m['FPR'], m['TPR'], label=f"{m_name} (AUC = {m['ROC_AUC']:.3f})")
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Receiver Operating Characteristic (ROC)")
        ax_roc.legend(fontsize=8)
        st.pyplot(fig_roc)

    with m_col2:
        st.write("##### XGBoost Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        cm = np.array(metrics['XGBoost']['Confusion_Matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                    xticklabels=['Predicted Low', 'Predicted High'],
                    yticklabels=['Actual Low', 'Actual High'])
        ax_cm.set_title("XGBoost Confusion Matrix")
        st.pyplot(fig_cm)


# ==========================================
# TAB 5: Explainable AI (SHAP) Studio
# ==========================================
with tabs[4]:
    st.subheader("Explainable AI (SHAP) Studio")
    st.markdown("Uncovering model interpretability at both population (Global SHAP) and individual patient (Local SHAP) levels using SHAP TreeExplainer.")

    shap_tab_global, shap_tab_local = st.tabs(["🌐 Global Feature Importance", "👤 Patient-Level Waterfall Attribution"])

    with shap_tab_global:
        st.write("##### Top 15 Global Clinical Risk Drivers")
        st.markdown("Features with greatest average marginal contribution (mean absolute SHAP value) to risk scoring:")
        fig_shap_bar, ax_shap_bar = plt.subplots(figsize=(9, 5))
        top_shap = shap_importance_df.head(15).sort_values('Mean_Absolute_SHAP')
        ax_shap_bar.barh(top_shap['Feature'].str.replace('_', ' '), top_shap['Mean_Absolute_SHAP'], color='#1E40AF')
        ax_shap_bar.set_xlabel("Mean |SHAP Value| (Impact on Model Log-Odds)")
        ax_shap_bar.set_title("Global XGBoost Feature Attribution")
        st.pyplot(fig_shap_bar)

    with shap_tab_local:
        st.write(f"##### Local SHAP Feature Attribution for Selected Patient: `{selected_pid}`")
        try:
            # Prepare patient scaled vector
            features_list = risk_engine.feature_names
            p_features_dict = {}
            for col in features_list:
                if col in p_data:
                    p_features_dict[col] = p_data[col]
                elif col == 'Gender_Male':
                    p_features_dict[col] = 1 if p_data.get('Gender') == 'Male' else 0
                elif col.startswith('Smoking_'):
                    p_features_dict[col] = 1 if p_data.get('Smoking_Status') in col else 0
                elif col.startswith('Ins_'):
                    p_features_dict[col] = 1 if p_data.get('Insurance_Type') in col else 0
                else:
                    p_features_dict[col] = 0

            p_series = pd.Series(p_features_dict)
            p_scaled_arr = risk_engine.scaler.transform(pd.DataFrame([p_series])[features_list])
            p_scaled = pd.Series(p_scaled_arr[0], index=features_list)

            explanation = xai_engine.explain_patient(p_scaled, top_k=6)

            st.info(f"**Clinician Summary:** {explanation['Clinical_Narrative']}")

            s_col1, s_col2 = st.columns(2)
            with s_col1:
                st.write("🔴 **Top Risk Escalation Factors (+SHAP):**")
                st.dataframe(explanation['Risk_Drivers'][['Feature', 'SHAP_Value']], use_container_width=True)
            with s_col2:
                st.write("🟢 **Top Stabilizing Protective Factors (-SHAP):**")
                st.dataframe(explanation['Protective_Factors'][['Feature', 'SHAP_Value']], use_container_width=True)

            # Local Waterfall Plot
            fig_local, ax_local = plt.subplots(figsize=(8, 4))
            local_top = explanation['Impact_DataFrame'].head(10).sort_values('SHAP_Value')
            colors = ['#EF4444' if v > 0 else '#10B981' for v in local_top['SHAP_Value']]
            ax_local.barh(local_top['Feature'].str.replace('_', ' '), local_top['SHAP_Value'], color=colors)
            ax_local.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax_local.set_xlabel("SHAP Impact on Patient Prediction (+ Risk / - Protective)")
            ax_local.set_title(f"Patient {selected_pid} Feature Attributions")
            st.pyplot(fig_local)
        except Exception as e:
            st.warning(f"Could not compute local SHAP explanation: {e}")


# ==========================================
# TAB 6: Decision Support & What-If Simulator
# ==========================================
with tabs[5]:
    st.subheader("Clinical Decision Support System (CDSS) & Counterfactual Simulator")
    st.markdown("Actionable clinical guidance, automated alerts, and interactive simulation of intervention outcomes.")

    # 1. Automated Clinical Alerts
    st.write("#### 🚨 Automated Real-Time Clinical Warning Alerts")
    alerts = cdss.generate_clinical_alerts(p_data.to_dict())
    if alerts:
        for alert in alerts:
            alert_class = 'alert-critical' if alert['Severity'] == 'CRITICAL' else 'alert-high'
            st.markdown(f"""
            <div class="{alert_class}">
                <strong>[{alert['Severity']} ALERT] {alert['Category']}:</strong> {alert['Message']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-info">
            <strong>✅ NO CRITICAL WARNINGS:</strong> Patient vitals and longitudinal trajectories are within stable thresholds.
        </div>
        """, unsafe_allow_html=True)

    # 2. Recommended Care Pathways
    st.write("#### 📋 Recommended Evidence-Based Care Pathways")
    patient_tier = 'High' if p_data['Target_Risk'] == 1 else 'Low'
    recs = cdss.recommend_interventions(patient_tier, p_data.to_dict(), top_risk_drivers=['Systolic_BP', 'Glucose', 'Adherence'])
    for r in recs:
        with st.container():
            st.markdown(f"**• {r['Action']}** `({r['Timeframe']})`")
            st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;{r['Details']}")

    st.write("---")

    # 3. Counterfactual What-If Risk Simulator
    st.write("#### 🧪 Interactive Counterfactual \"What-If\" Risk Simulator")
    st.markdown("Simulate the quantitative impact of clinical interventions (e.g. lowering BP, glycemic control, improved medication adherence) on this patient's predicted risk score.")

    sim_col1, sim_col2 = st.columns([1, 1])

    with sim_col1:
        st.write("##### Adjust Clinical Interventions:")
        new_sbp = st.slider("Target Systolic BP (mmHg)", 90, 200, int(p_data.get('Avg_Systolic_BP', 130)))
        new_glucose = st.slider("Target Glucose (mg/dL)", 70, 300, int(p_data.get('Avg_Glucose', 120)))
        new_hba1c = st.slider("Target HbA1c (%)", 4.5, 13.0, float(p_data.get('Avg_HbA1c', 7.0)), step=0.1)
        new_adherence = st.slider("Target Medication Adherence (%)", 20, 100, int(p_data.get('Mean_Med_Adherence', 0.8) * 100)) / 100.0

    with sim_col2:
        st.write("##### Risk Recalculation Results:")
        adjustments = {
            'Avg_Systolic_BP': new_sbp,
            'Max_Systolic_BP': new_sbp + 5,
            'Delta_Systolic_BP': min(0.0, float(new_sbp - p_data.get('Avg_Systolic_BP', 130))),
            'Avg_Glucose': new_glucose,
            'Max_Glucose': new_glucose + 10,
            'Delta_Glucose': min(0.0, float(new_glucose - p_data.get('Avg_Glucose', 120))),
            'Avg_HbA1c': new_hba1c,
            'Max_HbA1c': new_hba1c + 0.3,
            'Delta_HbA1c': min(0.0, float(new_hba1c - p_data.get('Avg_HbA1c', 7.0))),
            'Mean_Med_Adherence': new_adherence
        }

        sim_result = cdss.simulate_what_if_scenario(p_features_dict, adjustments, risk_engine)

        base_p = sim_result['Baseline_Risk_Prob']
        sim_p = sim_result['Simulated_Risk_Prob']
        delta_p = sim_result['Absolute_Risk_Change']

        res_c1, res_c2 = st.columns(2)
        with res_c1:
            st.metric("Baseline Risk Score", f"{base_p:.1%}", delta=None)
            st.write(f"Baseline Tier: **{sim_result['Baseline_Risk_Tier']}**")
        with res_c2:
            st.metric("Simulated Risk Score", f"{sim_p:.1%}", delta=f"{delta_p:+.1%}", delta_color="inverse")
            st.write(f"Simulated Tier: **{sim_result['Simulated_Risk_Tier']}**")

        if delta_p < 0:
            st.success(f"🎯 **Projected Outcome:** This care plan achieves a **{sim_result['Relative_Risk_Reduction_Pct']:.1f}% relative risk reduction** for Patient `{selected_pid}`.")
        else:
            st.info("Adjust the sliders on the left to simulate proactive risk reduction.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.85rem;">
    Patient Journey Intelligence System (PJIS) &copy; 2026 | Built by Inderjeet | Tech Stack: Python, Pandas, NumPy, XGBoost, Scikit-Learn, NLP, SHAP, Streamlit
</div>
""", unsafe_allow_html=True)
