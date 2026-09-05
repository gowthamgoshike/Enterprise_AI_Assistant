import streamlit as st

from apps.ui.services.api_client import (
    get_api_health,
    get_model_health,
)


def render_sidebar():

    st.sidebar.title(
        "System Status"
    )

    try:

        api_health = (
            get_api_health()
        )

        if (
            api_health.get("status")
            == "ok"
        ):

            st.sidebar.success(
                "API Online"
            )

        else:

            st.sidebar.error(
                "API Unhealthy"
            )

    except Exception:

        st.sidebar.error(
            "API Offline"
        )

    try:

        model_health = (
            get_model_health()
        )

        if (
            model_health.get("status")
            == "ok"
        ):

            st.sidebar.success(
                "LLM Online"
            )

        else:

            st.sidebar.warning(
                "LLM Unavailable"
            )

    except Exception:

        st.sidebar.error(
            "Model Service Offline"
        )

