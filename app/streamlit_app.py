
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from app.components.charts import render_global_importance
import streamlit as st
from app.components.charts import render_global_importance

# --------------------------------------------------
# PROJECT PATHS AND IMPORTS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.utils.prediction import predict_applicant

MODELS_PATH = ROOT / "models"
RESULTS_PATH = ROOT / "reports" / "model_results"
DATA_PATH = (
    ROOT / "data" / "interim" / "application_train_engineered.csv"
)

# --------------------------------------------------
# STREAMLIT CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
)

sns.set_theme(style="whitegrid")

st.title("💳 Credit Risk Intelligence")
st.caption(
    "Machine learning • Probability of default • "
    "Risk analysis • Explainability"
)

st.warning(
    "Portfolio prototype only. Decisions and thresholds are illustrative, "
    "not suitable for real lending decisions."
)

# --------------------------------------------------
# LOAD MODELS AND DATA
# --------------------------------------------------


@st.cache_resource
def load_model():
    """Load the calibrated model used for risk prediction."""
    return joblib.load(MODELS_PATH / "calibrated_model.joblib")


@st.cache_resource
def load_explanation_model():
    """Load the model used for SHAP explanations."""
    return joblib.load(MODELS_PATH / "model.joblib")


@st.cache_data
def load_engineered_data():
    """Load the engineered application dataset."""
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_prediction_summary():
    """Load the saved development prediction summary."""
    return pd.read_csv(
        RESULTS_PATH / "development_prediction_summary.csv"
    )


try:
    model = load_model()
    explanation_model = load_explanation_model()
    engineered_df = load_engineered_data()
    summary_df = load_prediction_summary()

except Exception as exc:
    st.error(
        "Could not load a required file. Check that the model artifacts, "
        "prediction summary, and interim dataset exist."
    )
    st.exception(exc)
    st.stop()

# --------------------------------------------------
# SIDEBAR: DECISION POLICY
# --------------------------------------------------

st.sidebar.header("Decision Policy")

review_threshold = st.sidebar.slider(
    "Manual review threshold",
    min_value=0.01,
    max_value=0.30,
    value=0.10,
    step=0.01,
)

reject_threshold = st.sidebar.slider(
    "Reject threshold",
    min_value=0.05,
    max_value=0.60,
    value=0.25,
    step=0.01,
)

if review_threshold >= reject_threshold:
    st.sidebar.error(
        "The review threshold must be lower than the reject threshold."
    )
    st.stop()

st.sidebar.caption(
    "Changing these thresholds changes the illustrative policy, "
    "not the model."
)

# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

overview_tab, performance_tab, applicant_tab, data_tab = st.tabs(
    [
        "📊 Portfolio Overview",
        "🎯 Model Performance",
        "🧑 Applicant Assessment",
        "🔎 Data Explorer",
    ]
)

# --------------------------------------------------
# PORTFOLIO OVERVIEW
# --------------------------------------------------

with overview_tab:
    st.subheader("Portfolio Risk Overview")

    total_applicants = len(summary_df)

    observed_default_rate = (
        summary_df["Actual_Default"].mean() * 100
    )

    mean_pd = summary_df["PD_Calibrated"].mean() * 100

    decisions = summary_df["PD_Calibrated"].apply(
        lambda probability: (
            "Approve"
            if probability < review_threshold
            else "Manual Review"
            if probability < reject_threshold
            else "Reject"
        )
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Test applicants",
        f"{total_applicants:,}",
    )
    col2.metric(
        "Observed default rate",
        f"{observed_default_rate:.2f}%",
    )
    col3.metric(
        "Mean predicted PD",
        f"{mean_pd:.2f}%",
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Predicted risk distribution**")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            summary_df["PD_Calibrated"] * 100,
            bins=30,
            kde=True,
            ax=ax,
        )

        ax.set_xlabel("Predicted probability of default (%)")
        ax.set_ylabel("Applicants")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Illustrative decision distribution**")

        decision_counts = decisions.value_counts().reindex(
            ["Approve", "Manual Review", "Reject"],
            fill_value=0,
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        decision_counts.plot(kind="bar", ax=ax)

        ax.set_ylabel("Applicants")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "The observed default rate is historical. The mean predicted "
        "PD is a model estimate; these quantities are not interchangeable."
    )

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

with performance_tab:
    st.subheader("Model Performance")

    comparison_file = RESULTS_PATH / "all_model_comparison.csv"

    if comparison_file.exists():
        comparison = pd.read_csv(
            comparison_file,
            index_col=0,
        )
      

        st.dataframe(
            comparison.round(4),
            use_container_width=True,
        )

        metric_columns = [
            column
            for column in [
                "ROC-AUC",
                "PR-AUC",
                "Precision",
                "Recall",
                "F1",
            ]
            if column in comparison.columns
        ]

        if metric_columns:
            st.bar_chart(comparison[metric_columns])

    else:
        st.info(
            "Model comparison file not found. Run the model comparison "
            "notebook cell first."
        )

    tuning_file = RESULTS_PATH / "xgboost_tuning_results.csv"

    if tuning_file.exists():
        with st.expander("View tuning results"):
            tuning_df = pd.read_csv(tuning_file)

            tuning_columns = [
                column
                for column in [
                    "rank_test_score",
                    "mean_test_score",
                    "std_test_score",
                    "mean_train_score",
                ]
                if column in tuning_df.columns
            ]

            if tuning_columns:
                st.dataframe(
                    tuning_df[tuning_columns],
                    use_container_width=True,
                )
            else:
                st.dataframe(
                    tuning_df,
                    use_container_width=True,
                )

    st.caption(
        "Reported scores are development results. They are not a "
        "substitute for independent final validation."
    )

# --------------------------------------------------
# APPLICANT ASSESSMENT
# --------------------------------------------------

with applicant_tab:
    st.subheader("Applicant Risk Assessment")

    st.write(
        "Select a historical application as a starting point, then "
        "optionally adjust the financial inputs. The resulting output "
        "is a model estimate under an illustrative policy."
    )

    # Keep the applicant ID for display, but never pass it to the model.
    if "SK_ID_CURR" not in engineered_df.columns:
        st.error("The dataset is missing the SK_ID_CURR column.")
        st.stop()

    available_df = (
        engineered_df
        .dropna(subset=["SK_ID_CURR"])
        .reset_index(drop=True)
    )

    if available_df.empty:
        st.error("No applications with valid applicant IDs were found.")
        st.stop()

    sample_size = min(500, len(available_df))

    selected_position = st.selectbox(
        "Select application row",
        options=list(range(sample_size)),
        format_func=lambda index: (
            f"Sample {index + 1} — ID "
            f"{int(available_df.iloc[index]['SK_ID_CURR'])}"
        ),
    )

    applicant_row = available_df.iloc[[selected_position]].copy()

    applicant_id = int(
        applicant_row["SK_ID_CURR"].iloc[0]
    )

    st.caption(
        f"Selected historical application ID: {applicant_id}"
    )

    with st.form("applicant_form"):
        st.markdown("**Adjust selected applicant's financial inputs**")

        col1, col2 = st.columns(2)

        # Use safe defaults if a source value is missing or invalid.
        def numeric_default(column, fallback=0.0):
            value = pd.to_numeric(
                applicant_row[column].iloc[0],
                errors="coerce",
            )

            if pd.isna(value) or not np.isfinite(value):
                return float(fallback)

            return float(value)

        income_default = max(
            0.0,
            numeric_default("AMT_INCOME_TOTAL"),
        )
        credit_default = max(
            0.0,
            numeric_default("AMT_CREDIT"),
        )
        annuity_default = max(
            0.0,
            numeric_default("AMT_ANNUITY"),
        )
        family_default = max(
            1.0,
            numeric_default("CNT_FAM_MEMBERS", 1.0),
        )

        with col1:
            income = st.number_input(
                "Annual income",
                min_value=0.0,
                value=income_default,
                step=1000.0,
            )

            credit = st.number_input(
                "Credit amount",
                min_value=0.0,
                value=credit_default,
                step=1000.0,
            )

        with col2:
            annuity = st.number_input(
                "Loan annuity",
                min_value=0.0,
                value=annuity_default,
                step=100.0,
            )

            family_size = st.number_input(
                "Family members",
                min_value=1.0,
                value=family_default,
                step=1.0,
            )

        submitted = st.form_submit_button(
            "Calculate risk",
            type="primary",
        )

    # All prediction, metrics, download, and explanation code belongs
    # inside this submitted block.
    if submitted:
        try:
            modified = applicant_row.copy()

            modified.loc[:, "AMT_INCOME_TOTAL"] = income
            modified.loc[:, "AMT_CREDIT"] = credit
            modified.loc[:, "AMT_ANNUITY"] = annuity
            modified.loc[:, "CNT_FAM_MEMBERS"] = family_size

            # Recalculate engineered features affected by the inputs.
            modified.loc[:, "CREDIT_INCOME_RATIO"] = (
                credit / income if income > 0 else np.nan
            )

            modified.loc[:, "ANNUITY_INCOME_RATIO"] = (
                annuity / income if income > 0 else np.nan
            )

            if "AMT_GOODS_PRICE" in modified.columns:
                goods_price = pd.to_numeric(
                    modified["AMT_GOODS_PRICE"],
                    errors="coerce",
                )

                modified.loc[:, "GOODS_CREDIT_RATIO"] = (
                    goods_price / credit if credit > 0 else np.nan
                )

            modified.loc[:, "INCOME_PER_FAMILY_MEMBER"] = (
                income / family_size if family_size > 0 else np.nan
            )

            features = modified.drop(
                columns=["TARGET", "SK_ID_CURR"],
                errors="ignore",
            )

            # ------------------------------------------
            # PREDICTION
            # ------------------------------------------

            result = predict_applicant(
                model,
                features,
                review_threshold,
                reject_threshold,
            )

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Probability of default",
                f"{result['probability_of_default']:.2%}",
            )

            p2.metric(
                "Project risk score",
                result["risk_score"],
            )

            p3.metric(
                "Risk category",
                result["risk_category"],
            )

            st.subheader("Illustrative triage result")

            st.write(f"**Decision:** {result['decision']}")

            st.caption(
                "This demonstration uses a historical row with selected "
                "fields changed. It is not a validated real-world "
                "lending decision."
            )

            # Correct dictionary unpacking: **result belongs inside
            # the dictionary literal.
            result_df = pd.DataFrame(
                [
                    {
                        "Application_ID": applicant_id,
                        **result,
                        "Review_Threshold": review_threshold,
                        "Reject_Threshold": reject_threshold,
                    }
                ]
            )

            st.download_button(
                "Download applicant result",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="applicant_risk_assessment.csv",
                mime="text/csv",
            )

            # ------------------------------------------
            # SHAP EXPLANATION
            # ------------------------------------------

            st.subheader("Why did the model make this prediction?")

            try:
                xgb_pipeline = explanation_model

                if not hasattr(xgb_pipeline, "named_steps"):
                    raise TypeError(
                        "model.joblib must contain a pipeline with "
                        "'preprocessor' and 'model' steps."
                    )

                xgb_preprocessor = (
                    xgb_pipeline.named_steps["preprocessor"]
                )

                xgb_estimator = xgb_pipeline.named_steps["model"]

                transformed_applicant = (
                    xgb_preprocessor.transform(features)
                )

                feature_names = (
                    xgb_preprocessor.get_feature_names_out()
                )

                # TreeExplainer explains the uncalibrated tree model,
                # not necessarily the calibrated probability.
                explainer = shap.TreeExplainer(xgb_estimator)

                applicant_shap_values = explainer.shap_values(
                    transformed_applicant
                )

                # SHAP versions can return a list for each class,
                # a 2D array, or a 3D array.
                if isinstance(applicant_shap_values, list):
                    if len(applicant_shap_values) > 1:
                        shap_row = np.asarray(
                            applicant_shap_values[1]
                        )[0]
                    else:
                        shap_row = np.asarray(
                            applicant_shap_values[0]
                        )[0]

                else:
                    shap_array = np.asarray(
                        applicant_shap_values
                    )

                    if shap_array.ndim == 3:
                        # Binary classification: select class 1.
                        shap_row = shap_array[0, :, 1]

                    elif shap_array.ndim == 2:
                        shap_row = shap_array[0]

                    else:
                        raise ValueError(
                            "Unexpected SHAP output shape: "
                            f"{shap_array.shape}"
                        )

                feature_names = np.asarray(feature_names)

                if len(feature_names) != len(shap_row):
                    raise ValueError(
                        "Feature-name count does not match SHAP output: "
                        f"{len(feature_names)} names vs "
                        f"{len(shap_row)} SHAP values."
                    )

                shap_importance = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "SHAP_Impact": shap_row,
                    }
                )

                shap_importance["Absolute_Impact"] = (
                    shap_importance["SHAP_Impact"].abs()
                )

                shap_importance = (
                    shap_importance
                    .sort_values(
                        "Absolute_Impact",
                        ascending=False,
                    )
                    .head(10)
                )

                fig, ax = plt.subplots(figsize=(9, 5))

                plot_data = shap_importance.sort_values(
                    "SHAP_Impact"
                )

                ax.barh(
                    plot_data["Feature"],
                    plot_data["SHAP_Impact"],
                )

                ax.axvline(
                    0,
                    linestyle="--",
                    linewidth=1,
                )

                ax.set_xlabel(
                    "SHAP value (model raw output)"
                )

                ax.set_title("Top Feature Contributions")

                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "Positive SHAP values increase the tree model's "
                    "raw output for the explained class; negative "
                    "values decrease it. For a binary XGBoost model, "
                    "these values commonly represent contributions "
                    "to the log-odds, not percentage-point changes "
                    "in calibrated probability of default."
                )

            except Exception as exc:
                st.warning(
                    "The explanation could not be generated for "
                    "this applicant. The prediction was still calculated."
                )
                st.exception(exc)

        except Exception as exc:
            st.error(
                "Prediction failed. Check that the input features match "
                "the model and that the prediction utility is configured "
                "correctly."
            )
            st.exception(exc)

# --------------------------------------------------
# DATA EXPLORER
# --------------------------------------------------

with data_tab:
    st.subheader("Dataset Explorer")

    st.write(
        "Preview of the engineered historical application data."
    )

    st.dataframe(
        engineered_df.head(100),
        use_container_width=True,
    )

    st.caption(
        "Missing values shown below describe the full dataset, "
        "not just the first 100 rows."
    )

    missing_pct = (
        engineered_df.isnull()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .head(15)
    )

    st.bar_chart(missing_pct)
