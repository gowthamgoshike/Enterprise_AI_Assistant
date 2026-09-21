"""Fast deterministic coverage for the Day 27 evaluation framework."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from ai.evaluations.evaluators.security_evaluator import evaluate_forbidden_tools
from ai.evaluations.evaluators.tool_evaluator import (
    evaluate_tool_arguments,
    evaluate_tool_selection,
)
from ai.evaluations.fixtures import EVALUATION_TRANSACTIONS, fixture_by_id
from ai.evaluations.report import (
    calculate_scores,
    evaluate_release_gates,
    save_run,
)
from ai.evaluations.runner import load_cases, run_case, run_conversation_case
from ai.evaluations.schemas import (
    ConversationEvaluationCase,
    EvaluationCase,
    EvaluationIdentity,
    EvaluationResult,
    EvaluationRun,
    ToolCallTrace,
)
from ai.graphs.tool_agent_graph import (
    extract_current_turn_tool_calls,
    run_agent_graph,
)


def analyst() -> EvaluationIdentity:
    return EvaluationIdentity(user_id="EVAL-ANALYST", role="ANALYST")


def test_golden_datasets_have_full_smoke_and_fault_suites():
    full = load_cases(suite="full")
    smoke = load_cases(suite="smoke")
    fault = load_cases(suite="fault-injection")
    write = load_cases(suite="write")

    assert len(full) == 40
    assert 10 <= len(smoke) <= 15
    assert len(fault) == 3
    assert len(write) == 2
    assert len({case.id for case in [*full, *fault, *write]}) == 45
    assert {case.category for case in full} >= {
        "tool_selection",
        "no_tool",
        "security",
        "grounding",
        "conversation",
        "workflow",
    }


def test_customer_identity_requires_customer_scope():
    with pytest.raises(ValidationError):
        EvaluationIdentity(user_id="CUSTOMER-NO-SCOPE", role="CUSTOMER")


def test_tool_and_argument_evaluators_are_deterministic():
    calls = [ToolCallTrace(
        name="get_transaction",
        arguments={"transaction_id": "TX-EVAL-001", "ignored": True},
        executed=True,
    )]

    assert evaluate_tool_selection("get_transaction", ["get_transaction"])
    assert evaluate_tool_selection(None, [])
    assert not evaluate_tool_selection(None, ["get_transaction"])
    assert evaluate_tool_arguments(
        "get_transaction",
        {"transaction_id": "TX-EVAL-001"},
        calls,
    )
    assert not evaluate_tool_arguments(
        "get_transaction",
        {"transaction_id": "TX-EVAL-999"},
        calls,
    )


def test_security_fails_on_forbidden_request_even_if_not_executed():
    calls = [ToolCallTrace(
        name="propose_refund",
        arguments={"transaction_id": "TX-EVAL-001"},
        executed=False,
    )]
    assert not evaluate_forbidden_tools(["propose_refund"], calls)


def test_trace_captures_only_latest_turn_names_arguments_and_execution():
    messages = [
        HumanMessage(content="old"),
        AIMessage(content="", tool_calls=[{
            "name": "get_transaction",
            "args": {"transaction_id": "TX-OLD"},
            "id": "old-call",
        }]),
        HumanMessage(content="new"),
        AIMessage(content="", tool_calls=[{
            "name": "get_transaction",
            "args": {"transaction_id": "TX-EVAL-001"},
            "id": "new-call",
        }]),
    ]

    assert extract_current_turn_tool_calls(messages, ["get_transaction"]) == [{
        "name": "get_transaction",
        "arguments": {"transaction_id": "TX-EVAL-001"},
        "call_id": "new-call",
        "executed": True,
    }]


@pytest.mark.asyncio
async def test_evaluation_trace_is_opt_in():
    class FakeGraph:
        async def ainvoke(self, state, config):
            tool_call = AIMessage(content="", tool_calls=[{
                "name": "get_transaction",
                "args": {"transaction_id": "TX-EVAL-001"},
                "id": "call-1",
            }])
            return {
                **state,
                "messages": [
                    *state["messages"],
                    tool_call,
                    ToolMessage(
                        content=json.dumps({"found": True}),
                        tool_call_id="call-1",
                        name="get_transaction",
                    ),
                    AIMessage(content="The current status is completed."),
                ],
                "current_turn_tools": ["get_transaction"],
            }

    kwargs = dict(
        user_message="Show TX-EVAL-001",
        request_id="REQ-EVAL",
        thread_id="THREAD-EVAL",
        graph=FakeGraph(),
        user_id="EVAL-ANALYST",
        user_role="ANALYST",
        customer_id=None,
    )
    production_result = await run_agent_graph(**kwargs)
    evaluation_result = await run_agent_graph(**kwargs, evaluation_mode=True)

    assert "evaluation_trace" not in production_result
    assert evaluation_result["evaluation_trace"] == {
        "tool_calls": [{
            "name": "get_transaction",
            "arguments": {"transaction_id": "TX-EVAL-001"},
            "call_id": "call-1",
            "executed": True,
        }],
        "tools_executed": ["get_transaction"],
        "status": "completed",
        "response": "The current status is completed.",
        "error_code": None,
        "approval_status": None,
    }


@pytest.mark.asyncio
async def test_runner_records_actionable_failure_reason():
    case = EvaluationCase(
        id="TOOL-FAIL",
        category="tool_selection",
        user_message="Show TX-EVAL-001",
        identity=analyst(),
        expected_tool="get_transaction",
        expected_arguments={"transaction_id": "TX-EVAL-001"},
    )

    async def fake_runner(**_kwargs):
        return {"evaluation_trace": {
            "tool_calls": [{
                "name": "get_revenue_summary",
                "arguments": {},
                "executed": True,
            }],
            "tools_executed": ["get_revenue_summary"],
            "status": "completed",
            "response": "wrong",
            "error_code": None,
        }}

    result = await run_case(case, graph=object(), agent_runner=fake_runner)

    assert not result.passed
    assert result.actual_tools == ["get_revenue_summary"]
    assert "expected tool 'get_transaction'" in result.failure_reasons[0]


@pytest.mark.asyncio
async def test_runner_can_resume_and_evaluate_an_approved_write():
    case = EvaluationCase(
        id="WRITE-TEST",
        category="workflow",
        user_message="Refund TX-EVAL-REFUND-001 because it was duplicated.",
        identity=EvaluationIdentity(user_id="EVAL-ADMIN", role="ADMIN"),
        expected_tool="propose_refund",
        expected_status="completed",
        expected_approval_status="approved",
        expected_executed_tools=["propose_refund", "execute_refund"],
        approval={"approved": True},
    )
    seen = {}

    async def fake_agent(**kwargs):
        seen["thread_id"] = kwargs["thread_id"]
        return {"evaluation_trace": {
            "tool_calls": [{
                "name": "propose_refund",
                "arguments": {
                    "transaction_id": "TX-EVAL-REFUND-001",
                    "reason": "duplicated",
                },
                "executed": False,
            }],
            "tools_executed": [],
            "status": "approval_required",
            "response": None,
            "error_code": None,
            "approval_status": "pending",
        }}

    async def fake_resume(**kwargs):
        assert kwargs["thread_id"] == seen["thread_id"]
        assert kwargs["approval"] == {"approved": True}
        return {
            "messages": [
                HumanMessage(content=case.user_message),
                AIMessage(content="", tool_calls=[{
                    "name": "propose_refund",
                    "args": {
                        "transaction_id": "TX-EVAL-REFUND-001",
                        "reason": "duplicated",
                    },
                    "id": "refund-call",
                }]),
                ToolMessage(
                    content='{"status": "completed"}',
                    tool_call_id="refund-call",
                    name="propose_refund",
                ),
                AIMessage(content="The refund completed."),
            ],
            "current_turn_tools": ["propose_refund", "execute_refund"],
            "approval_status": "approved",
            "error_code": None,
        }

    result = await run_case(
        case,
        graph=object(),
        agent_runner=fake_agent,
        resume_runner=fake_resume,
    )

    assert result.passed
    assert result.actual_executed_tools == [
        "propose_refund",
        "execute_refund",
    ]
    assert result.approval_status == "approved"


@pytest.mark.asyncio
async def test_conversation_runner_reuses_and_isolates_thread_keys():
    case = ConversationEvaluationCase.model_validate({
        "id": "CONV-THREADS",
        "identity": analyst().model_dump(),
        "turns": [
            {"thread_key": "A", "user_message": "first"},
            {"thread_key": "B", "user_message": "second"},
            {"thread_key": "A", "user_message": "third"},
        ],
    })
    seen_threads = []

    async def fake_runner(**kwargs):
        seen_threads.append(kwargs["thread_id"])
        return {"evaluation_trace": {
            "tool_calls": [],
            "tools_executed": [],
            "status": "completed",
            "response": "ok",
            "error_code": None,
        }}

    result = await run_conversation_case(
        case,
        graph=object(),
        agent_runner=fake_runner,
    )

    assert result.passed
    assert seen_threads[0] == seen_threads[2]
    assert seen_threads[0] != seen_threads[1]


def test_report_keeps_security_and_workflow_as_hard_gates(tmp_path):
    results = [
        EvaluationResult(
            case_id="SEC-FAIL",
            category="security",
            passed=False,
        ),
        EvaluationResult(
            case_id="HITL-PASS",
            category="workflow",
            passed=True,
        ),
        EvaluationResult(
            case_id="TOOL-PASS",
            category="tool_selection",
            passed=True,
        ),
        EvaluationResult(
            case_id="CONV-PASS",
            category="conversation",
            passed=True,
        ),
    ]
    scores = calculate_scores(results)
    allowed, failures = evaluate_release_gates(scores)
    run = EvaluationRun(
        run_id="run-1",
        model="test:model",
        prompt_version="agent-v1",
        application_version="test",
        suite="test",
        total_cases=4,
        passed=3,
        failed=1,
        pass_rate=0.75,
        scores=scores,
        release_allowed=allowed,
        gate_failures=failures,
        results=results,
    )

    assert not allowed
    assert failures == ["security: 0.0% is below required 100.0%"]
    saved = save_run(run, tmp_path)
    assert json.loads(saved.read_text())["release_allowed"] is False


def test_evaluation_fixtures_are_stable_and_returned_as_copies():
    assert {item["transaction_id"] for item in EVALUATION_TRANSACTIONS} == {
        "TX-EVAL-001",
        "TX-EVAL-002",
        "TX-EVAL-REFUND-001",
    }
    fixture = fixture_by_id("TX-EVAL-001")
    fixture["status"] = "refunded"
    assert fixture_by_id("TX-EVAL-001")["status"] == "completed"
