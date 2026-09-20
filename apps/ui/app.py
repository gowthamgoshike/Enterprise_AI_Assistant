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
    SessionExpiredError,
    login,
    send_chat,
    submit_approval,
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


if "access_token" not in st.session_state:

    st.session_state.access_token = None


if "authenticated" not in st.session_state:

    st.session_state.authenticated = False


if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None


def clear_authenticated_session() -> None:
    st.session_state.access_token = None
    st.session_state.authenticated = False
    st.session_state.messages = []
    st.session_state.session_id = str(
        uuid.uuid4()
    )
    st.session_state.pending_approval = None
    st.session_state.pop(
        "model_name",
        None,
    )


if (
    not st.session_state.authenticated
    or not st.session_state.access_token
):
    st.title(
        "Enterprise AI Assistant"
    )

    if "authentication_notice" in st.session_state:
        st.warning(
            st.session_state.pop(
                "authentication_notice"
            )
        )

    email = st.text_input(
        "Email"
    )

    password = st.text_input(
        "Password",
        type="password",
    )

    if st.button(
        "Login"
    ):
        try:
            result = login(
                email,
                password,
            )
            access_token = result[
                "access_token"
            ]
            if not isinstance(
                access_token,
                str,
            ) or not access_token:
                raise TypeError(
                    "Missing access token"
                )

        except (
            requests.RequestException,
            KeyError,
            TypeError,
        ):
            st.error(
                "Invalid email or password."
            )

        else:
            st.session_state.access_token = access_token
            st.session_state.authenticated = True
            st.session_state.messages = []
            st.session_state.session_id = str(
                uuid.uuid4()
            )
            st.rerun()

    st.stop()


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
    "Logout"
):
    clear_authenticated_session()
    st.rerun()

if st.sidebar.button(
    "Clear conversation"
):

    st.session_state.messages = []
    st.session_state.session_id = str(
        uuid.uuid4()
    )
    st.session_state.pending_approval = None

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


pending_approval = st.session_state.pending_approval
if pending_approval:
    action = pending_approval["action"]
    st.warning("Human approval required")
    st.write("Transaction:", action["transaction_id"])
    st.write(
        "Amount:",
        f"{action['currency']} {action['amount_cents'] / 100:,.2f}",
    )
    st.write("Reason:", action["reason"])
    approval_comment = st.text_input(
        "Approval comment (optional)",
        key="approval_comment",
    )
    approve_column, reject_column = st.columns(2)
    approve_clicked = approve_column.button(
        "Approve",
        type="primary",
    )
    reject_clicked = reject_column.button("Reject")

    if approve_clicked or reject_clicked:
        try:
            with st.spinner("Recording decision..."):
                approval_result = submit_approval(
                    session_id=st.session_state.session_id,
                    approved=approve_clicked,
                    comment=approval_comment or None,
                    access_token=st.session_state.access_token,
                )
            if approval_result.get("status") == "approval_required":
                st.session_state.pending_approval = approval_result[
                    "approval_request"
                ]
                final_response = (
                    "Another action proposal requires human approval."
                )
            else:
                st.session_state.pending_approval = None
                final_response = approval_result.get("response") or (
                    "The refund proposal was approved for future execution."
                    if approve_clicked
                    else "The refund proposal was rejected."
                )
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response,
                "tools_used": approval_result.get("tools_used", []),
                "route": approval_result.get("route"),
                "route_confidence": approval_result.get("route_confidence"),
            })
            st.rerun()
        except SessionExpiredError:
            clear_authenticated_session()
            st.session_state.authentication_notice = (
                "Your session expired. Please log in again."
            )
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Unable to submit approval: {exc}")


prompt = st.chat_input(
    "Ask about enterprise transactions...",
    disabled=pending_approval is not None,
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
        approval_pending = False

        try:

            with st.spinner(
                "Thinking..."
            ):

                MAX_CONTEXT_MESSAGES = 1

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
                    access_token=st.session_state.access_token,
                )

            if result.get("status") == "approval_required":
                st.session_state.pending_approval = result["approval_request"]
                assistant_response = (
                    "A refund proposal is ready and requires human approval."
                )
                approval_pending = True
            else:
                assistant_response = result["response"]

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

        except SessionExpiredError:

            clear_authenticated_session()
            st.session_state.authentication_notice = (
                "Your session expired. Please log in again."
            )
            st.rerun()

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
    if approval_pending:
        st.rerun()
