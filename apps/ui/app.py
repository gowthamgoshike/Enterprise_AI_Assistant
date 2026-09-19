import sys
import uuid
from pathlib import Path

import requests
import streamlit as st


project_root = Path(__file__).resolve().parents[2]

if str(project_root) not in sys.path:

    sys.path.insert(
        0,
        str(project_root),
    )

from apps.ui.components.sidebar import (
    render_sidebar,
)
from apps.ui.services.api_client import (
    send_chat,
)


def render_activity(
    tools_used: list[str],
    route: str | None = None,
    confidence: float | None = None,
) -> None:
    if tools_used or route or confidence is not None:
        with st.expander("Assistant activity"):
            if route:
                st.write("Route:", route)
            if confidence is not None:
                st.write("Routing confidence:", round(confidence, 2))
            for tool_name in tools_used:
                st.write(f"Tool: {tool_name}")


st.set_page_config(
    page_title=(
        "Enterprise AI Assistant"
    ),
    page_icon="🤖",
    layout="wide",
)


if "messages" not in st.session_state:

    st.session_state.messages = []


if "session_id" not in st.session_state:

    st.session_state.session_id = str(
        uuid.uuid4()
    )


render_sidebar()


st.sidebar.caption(
    "Provider: Ollama"
)

if "model_name" in st.session_state:

    st.sidebar.caption(
        f"Model: "
        f"{st.session_state.model_name}"
    )

st.sidebar.caption(
    f"Session: "
    f"{st.session_state.session_id[:8]}"
)

if st.sidebar.button(
    "Clear conversation"
):

    st.session_state.messages = []
    st.session_state.session_id = str(
        uuid.uuid4()
    )

    st.rerun()


message_count = len(
    st.session_state.messages
)

st.sidebar.metric(
    "Messages",
    message_count,
)


st.title(
    "Enterprise Transaction "
    "Intelligence Assistant"
)


st.caption(
    "AI assistant powered by "
    "an open-weight local model"
)

st.info(
    "Ask about specific transactions, top products by revenue, "
    "aggregate revenue, or transaction counts by status."
)


if not st.session_state.messages:

    st.markdown(
        """
        ### What can I help with?

        Try asking:

        - What are our top 5 products?
        - How is the business doing?
        - How many transactions have failed?
        - What is the status of TX-123?
        """
    )


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )
        render_activity(
            message.get("tools_used", []),
            message.get("route"),
            message.get("route_confidence"),
        )


prompt = st.chat_input(
    "Ask about enterprise transactions..."
)


if prompt:

    user_message = {
        "role": "user",
        "content": prompt,
    }

    st.session_state.messages.append(
        user_message
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            prompt
        )

    with st.chat_message(
        "assistant"
    ):

        tools_used = []
        route = None
        confidence = None

        try:

            with st.spinner(
                "Thinking..."
            ):

                MAX_CONTEXT_MESSAGES = 50

                context_messages = (
                    st.session_state.messages[
                        -MAX_CONTEXT_MESSAGES:
                    ]
                )

                result = send_chat(
                    messages=[
                        {"role": message["role"], "content": message["content"]}
                        for message in context_messages
                    ],
                    session_id=st.session_state.session_id,
                )

            assistant_response = (
                result["response"]
            )

            st.markdown(
                assistant_response
            )

            tools_used = result.get("tools_used", [])
            route = result.get("route")
            confidence = result.get("route_confidence")
            render_activity(tools_used, route, confidence)

            model_name = result.get(
                "model"
            )

            if model_name is not None:

                st.session_state.model_name = (
                    model_name
                )

            latency = result.get(
                "latency_seconds"
            )

            if latency is not None:

                st.caption(
                    f"Latency: "
                    f"{latency:.2f}s"
                )

        except requests.Timeout:

            assistant_response = (
                "The AI service timed out."
            )

            st.error(
                assistant_response
            )

        except requests.ConnectionError:

            assistant_response = (
                "Unable to connect "
                "to the API."
            )

            st.error(
                assistant_response
            )

        except requests.HTTPError as exc:

            assistant_response = (
                "The AI service returned "
                f"an error: {exc}"
            )

            st.error(
                assistant_response
            )

        except Exception:

            assistant_response = (
                "An unexpected "
                "error occurred."
            )

            st.error(
                assistant_response
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response,
            "tools_used": tools_used,
            "route": route,
            "route_confidence": confidence,
        }
    )
