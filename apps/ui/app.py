import sys
import uuid
from pathlib import Path

import requests
import httpx
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
    cancel_background_job,
    create_failed_transaction_report,
    get_background_job,
    login,
    submit_approval,
)
from apps.ui.services.chat_client import stream_chat


def render_activity(
    tools_used: list[str],
    route: str | None = None,
    confidence: float | None = None,
    latency_seconds: float | None = None,
) -> None:
    if tools_used or route or confidence is not None or latency_seconds is not None:
        with st.expander("Assistant activity", expanded=True):
            if route:
                st.write("Route:", route)
            if confidence is not None:
                st.write("Routing confidence:", round(confidence, 2))
            for tool_name in tools_used:
                st.write(f"Tool: {tool_name}")
            if latency_seconds is not None:
                st.write(f"Latency: {latency_seconds:.2f}s")


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


if "report_job" not in st.session_state:
    st.session_state.report_job = None


def clear_authenticated_session() -> None:
    st.session_state.access_token = None
    st.session_state.authenticated = False
    st.session_state.messages = []
    st.session_state.session_id = str(
        uuid.uuid4()
    )
    st.session_state.pending_approval = None
    st.session_state.report_job = None
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

        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                st.error("Invalid email or password.")
            else:
                st.error("Login request failed. Check the API status and try again.")
        except requests.RequestException:
            st.error("Cannot reach the API. Check that the server is running.")
        except (KeyError, TypeError):
            st.error("The API returned an invalid login response.")

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


def render_report_job() -> None:
    report_job = st.session_state.report_job
    if not report_job:
        return

    if report_job["status"] in {"queued", "running"}:
        try:
            report_job = get_background_job(
                job_id=report_job["job_id"],
                access_token=st.session_state.access_token,
            )
            st.session_state.report_job = report_job
            if report_job["status"] not in {"queued", "running"}:
                # Rebuild the fragment without its polling timer.
                st.rerun()
        except SessionExpiredError:
            clear_authenticated_session()
            st.session_state.authentication_notice = (
                "Your session expired. Please log in again."
            )
            st.rerun()
        except requests.RequestException:
            st.warning("Unable to refresh the report status; retrying.")

    st.success("Report submitted.")
    st.code(report_job["job_id"])
    st.write("Status:", report_job["status"].title())
    if report_job.get("stage"):
        st.write("Stage:", report_job["stage"].replace("_", " ").title())

    if report_job["status"] in {"queued", "running"}:
        refresh_column, cancel_column = st.columns(2)
        if refresh_column.button("Refresh report status"):
            st.rerun(scope="fragment")
        if report_job["status"] == "queued" and cancel_column.button(
            "Cancel report"
        ):
            try:
                st.session_state.report_job = cancel_background_job(
                    job_id=report_job["job_id"],
                    access_token=st.session_state.access_token,
                )
                st.rerun()
            except SessionExpiredError:
                clear_authenticated_session()
                st.session_state.authentication_notice = (
                    "Your session expired. Please log in again."
                )
                st.rerun()
            except requests.RequestException:
                st.error("Unable to cancel the report.")

    if report_job["status"] == "completed" and report_job.get("result"):
        result = report_job["result"]
        st.metric("Failed transactions", result["failed_count"])
        st.metric(
            "Total failed amount",
            f"${result['total_failed_amount_cents'] / 100:,.2f}",
        )
        st.dataframe(result.get("top_failure_products", []))
    elif report_job["status"] == "failed":
        st.error(f"Report failed: {report_job.get('error_code', 'unknown')}")
    elif report_job["status"] == "cancelled":
        st.info("Report cancelled.")


report_poll_seconds = (
    3
    if st.session_state.report_job
    and st.session_state.report_job["status"] in {"queued", "running"}
    else None
)
render_report_job_fragment = st.fragment(
    render_report_job,
    run_every=report_poll_seconds,
)


with st.expander("Failed transaction report", expanded=False):
    if st.button("Generate 30-Day Failure Report"):
        try:
            st.session_state.report_job = create_failed_transaction_report(
                days=30,
                access_token=st.session_state.access_token,
            )
            st.rerun()
        except SessionExpiredError:
            clear_authenticated_session()
            st.session_state.authentication_notice = (
                "Your session expired. Please log in again."
            )
            st.rerun()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                st.error("Your role cannot generate analytics reports.")
            else:
                st.error("Unable to submit the report job.")
        except requests.RequestException:
            st.error("Cannot reach the API.")

    render_report_job_fragment()


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
            message.get("latency_seconds"),
        )


pending_approval = st.session_state.pending_approval
if pending_approval:
    action = pending_approval.get("action", pending_approval)
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
                    "The approved refund was executed successfully."
                    if approve_clicked
                    else "The refund proposal was rejected."
                )
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response,
                "tools_used": approval_result.get("tools_used", []),
                "route": approval_result.get("route"),
                "route_confidence": approval_result.get("route_confidence"),
                "latency_seconds": approval_result.get("latency_seconds"),
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
        latency = None
        approval_pending = False
        assistant_response = ""

        try:
            activity_box = st.empty()
            answer_box = st.empty()
            stream_state = {"done": {}, "approval": None, "error": None}

            def token_stream():
                saw_token = False
                for event in stream_chat(
                    message=prompt,
                    session_id=st.session_state.session_id,
                    access_token=st.session_state.access_token,
                ):
                    event_type = event["type"]
                    data = event["data"]
                    if event_type in {"status", "tool_start"}:
                        activity_box.info(data.get("message", "Working..."))
                    elif event_type == "tool_end":
                        activity_box.empty()
                    elif event_type == "token":
                        saw_token = True
                        yield data["text"]
                    elif event_type == "approval_required":
                        stream_state["approval"] = data
                    elif event_type == "error":
                        stream_state["error"] = data
                        activity_box.error(data["message"])
                    elif event_type == "done":
                        stream_state["done"] = data
                if not saw_token:
                    yield ""

            full_response = answer_box.write_stream(token_stream())
            result = stream_state["done"]
            if not result:
                raise RuntimeError("The AI stream ended before completion")

            if stream_state["approval"]:
                st.session_state.pending_approval = stream_state["approval"]
                assistant_response = (
                    "A refund proposal is ready and requires human approval."
                )
                answer_box.markdown(assistant_response)
                activity_box.empty()
                approval_pending = True
            else:
                assistant_response = (
                    result.get("response")
                    or full_response
                    or (stream_state["error"] or {}).get("message", "")
                )
                if assistant_response != full_response:
                    answer_box.markdown(assistant_response)
                if stream_state["error"] is None:
                    activity_box.empty()

            tools_used = result.get("tools_used", [])
            route = result.get("route")
            confidence = result.get("route_confidence")
            latency = result.get("latency_seconds")
            render_activity(tools_used, route, confidence, latency)

            model_name = result.get(
                "model"
            )

            if model_name is not None:

                st.session_state.model_name = (
                    model_name
                )

        except SessionExpiredError:

            clear_authenticated_session()
            st.session_state.authentication_notice = (
                "Your session expired. Please log in again."
            )
            st.rerun()

        except (requests.Timeout, httpx.TimeoutException):

            assistant_response = (
                "The AI service timed out."
            )

            st.error(
                assistant_response
            )

        except (requests.ConnectionError, httpx.ConnectError):

            assistant_response = (
                "Unable to connect "
                "to the API."
            )

            st.error(
                assistant_response
            )

        except (requests.HTTPError, httpx.HTTPStatusError):

            assistant_response = (
                "The request could not be completed."
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
            "latency_seconds": latency,
        }
    )
    if approval_pending:
        st.rerun()
