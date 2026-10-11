import pandas as pd
import streamlit as st


def render_global_importance(importance_path, top_n=15):
    path = pd.io.common.stringify_path(importance_path)

    try:
        importance = pd.read_csv(path)
    except FileNotFoundError:
        st.info("Run the SHAP analysis notebook to generate feature importance.")
        return

    required = {"Feature", "Mean_Absolute_SHAP"}
    if not required.issubset(importance.columns):
        st.warning("The feature importance file has an unexpected format.")
        return

    importance = importance.nlargest(top_n, "Mean_Absolute_SHAP")
    importance = importance.sort_values("Mean_Absolute_SHAP")

    st.subheader("Global Feature Importance")
    st.bar_chart(
        importance.set_index("Feature")["Mean_Absolute_SHAP"]
    )