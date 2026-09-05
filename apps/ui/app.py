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
    "The assistant currently has "
    "general model knowledge only. "
    "Enterprise database tools will be "
    "added in later stages."
)


if not st.session_state.messages:

    st.markdown(
        """
        ### What can I help with?

        Try asking:

        - Explain our streaming architecture.
        - Why do we use Kafka?
        - What is a Gold data layer?
        - What is tool calling?
        """
    )


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
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

        try:

            with st.spinner(
                "Thinking..."
            ):

                MAX_CONTEXT_MESSAGES = 20

                context_messages = (
                    st.session_state.messages[
                        -MAX_CONTEXT_MESSAGES:
                    ]
                )

                result = send_chat(
                    context_messages
                )

            assistant_response = (
                result["response"]
            )

            st.markdown(
                assistant_response
            )

            tool_used = result.get(
                "tool_used"
            )

            if tool_used:

                with st.expander(
                    "Assistant activity"
                ):

                    st.write(
                        "Tool used:",
                        tool_used,
                    )

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
        }
    )