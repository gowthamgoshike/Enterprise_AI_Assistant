"""Run golden behavioral cases against the real secured LangGraph agent.

Examples:
    python -m ai.evaluations.runner --suite smoke
    python -m ai.evaluations.runner --suite full --load-fixtures
"""

import argparse
import asyncio
import inspect
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from ai.evaluations.evaluators.conversation_evaluator import (
    evaluate_conversation,
)
from ai.evaluations.evaluators.response_evaluator import evaluate_response
from ai.evaluations.evaluators.security_evaluator import evaluate_forbidden_tools
from ai.evaluations.evaluators.tool_evaluator import (
    evaluate_tool_arguments,
    evaluate_tool_selection,
)
from ai.evaluations.evaluators.workflow_evaluator import evaluate_workflow
from ai.evaluations.report import (
    RELEASE_GATES,
    calculate_scores,
    evaluate_release_gates,
    render_terminal_report,
    save_run,
)
from ai.evaluations.schemas import (
    ConversationEvaluationCase,
    EvaluationCase,
    EvaluationResult,
    EvaluationRun,
    ToolCallTrace,
)
from ai.graphs.tool_agent_graph import (
    AGENT_PROMPT_VERSION,
    build_evaluation_trace,
    build_enterprise_tool_graph,
    resume_agent_graph,
    run_agent_graph,
)
from ai.models.langchain_ollama import OLLAMA_MODEL


DATASET_DIR = Path(__file__).with_name("datasets")
DEFAULT_RESULT_DIR = Path("evaluation_results")
APPLICATION_VERSION = "0.1.0"
AgentRunner = Callable[..., Awaitable[dict]]
ResumeRunner = Callable[..., Awaitable[dict]]
FixtureHook = Callable[[str], Any]


def load_cases(
    dataset_dir: str | Path = DATASET_DIR,
    *,
    suite: str = "full",
) -> list[EvaluationCase | ConversationEvaluationCase]:
    cases: list[EvaluationCase | ConversationEvaluationCase] = []
    for path in sorted(Path(dataset_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw_case in payload:
            case = (
                ConversationEvaluationCase.model_validate(raw_case)
                if "turns" in raw_case
                else EvaluationCase.model_validate(raw_case)
            )
            if suite in case.tags:
                cases.append(case)
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation case IDs must be unique")
    return cases


def _trace_from_result(result: dict) -> dict:
    return result.get("evaluation_trace") or {
        "tool_calls": [],
        "tools_executed": result.get("current_turn_tools", []),
        "status": "failed" if result.get("error_code") else "completed",
        "response": None,
        "error_code": result.get("error_code"),
    }


def _evaluate_expectation(
    *,
    case_id: str,
    category: str,
    expectation,
    trace: dict,
    latency_ms: float,
) -> EvaluationResult:
    calls = [ToolCallTrace.model_validate(call) for call in trace["tool_calls"]]
    actual_tools = [call.name for call in calls]
    if not calls:
        actual_tools = list(trace.get("tools_executed", []))
    failures = []
    if not evaluate_tool_selection(expectation.expected_tool, actual_tools):
        failures.append(
            f"expected tool {expectation.expected_tool!r}, got {actual_tools!r}"
        )
    if not evaluate_tool_arguments(
        expectation.expected_tool,
        expectation.expected_arguments,
        calls,
    ):
        failures.append(
            f"arguments for {expectation.expected_tool!r} did not match "
            f"{expectation.expected_arguments!r}"
        )
    if not evaluate_forbidden_tools(expectation.forbidden_tools, calls):
        requested = sorted(
            {call.name for call in calls} & set(expectation.forbidden_tools)
        )
        failures.append(f"forbidden tools requested: {requested!r}")
    missing_executions = sorted(
        set(expectation.expected_executed_tools)
        - set(trace.get("tools_executed", []))
    )
    if missing_executions:
        failures.append(f"expected tools were not executed: {missing_executions!r}")
    _, workflow_failures = evaluate_workflow(
        expectation.expected_status,
        trace.get("status"),
        expectation.expected_error_code,
        trace.get("error_code"),
        expectation.expected_approval_status,
        trace.get("approval_status"),
    )
    failures.extend(workflow_failures)
    _, response_failures = evaluate_response(
        trace.get("response"),
        expected_contains=expectation.expected_response_contains,
        forbidden_contains=expectation.forbidden_response_contains,
        expected_facts=expectation.expected_facts,
    )
    failures.extend(response_failures)
    return EvaluationResult(
        case_id=case_id,
        category=category,
        passed=not failures,
        expected_tool=expectation.expected_tool,
        actual_tools=actual_tools,
        actual_executed_tools=list(trace.get("tools_executed", [])),
        actual_tool_calls=calls,
        expected_status=expectation.expected_status,
        actual_status=trace.get("status"),
        latency_ms=latency_ms,
        response=trace.get("response"),
        error_code=trace.get("error_code"),
        approval_status=trace.get("approval_status"),
        failure_reasons=failures,
    )


async def _apply_fixture_hook(hook: FixtureHook | None, action: str | None) -> None:
    if hook is None or action is None:
        return
    result = hook(action)
    if inspect.isawaitable(result):
        await result


async def run_case(
    case: EvaluationCase,
    *,
    graph,
    agent_runner: AgentRunner = run_agent_graph,
    resume_runner: ResumeRunner = resume_agent_graph,
    fixture_hook: FixtureHook | None = None,
) -> EvaluationResult:
    await _apply_fixture_hook(fixture_hook, case.fixture)
    started_at = time.perf_counter()
    thread_id = f"eval-{case.id}-{uuid.uuid4()}"
    try:
        raw_result = await agent_runner(
            user_message=case.user_message,
            request_id=f"EVAL-{case.id}-{uuid.uuid4()}",
            thread_id=thread_id,
            graph=graph,
            user_id=case.identity.user_id,
            user_role=case.identity.role,
            customer_id=case.identity.customer_id,
            evaluation_mode=True,
        )
        trace = _trace_from_result(raw_result)
        if case.approval is not None and trace["status"] == "approval_required":
            raw_result = await resume_runner(
                thread_id=thread_id,
                approval=case.approval,
                graph=graph,
            )
            trace = build_evaluation_trace(raw_result)
    except Exception as exc:  # A benchmark records failures instead of aborting.
        trace = {
            "tool_calls": [],
            "tools_executed": [],
            "status": "failed",
            "response": None,
            "error_code": type(exc).__name__,
        }
    latency_ms = (time.perf_counter() - started_at) * 1000
    return _evaluate_expectation(
        case_id=case.id,
        category=case.category,
        expectation=case,
        trace=trace,
        latency_ms=latency_ms,
    )


async def run_conversation_case(
    case: ConversationEvaluationCase,
    *,
    graph,
    agent_runner: AgentRunner = run_agent_graph,
    resume_runner: ResumeRunner = resume_agent_graph,
    fixture_hook: FixtureHook | None = None,
) -> EvaluationResult:
    await _apply_fixture_hook(fixture_hook, case.fixture)
    run_suffix = str(uuid.uuid4())
    thread_ids: dict[str, str] = {}
    turns = []
    started_at = time.perf_counter()
    for index, turn in enumerate(case.turns, start=1):
        await _apply_fixture_hook(fixture_hook, turn.fixture_action)
        thread_id = thread_ids.setdefault(
            turn.thread_key,
            f"eval-{case.id}-{turn.thread_key}-{run_suffix}",
        )
        turn_started = time.perf_counter()
        try:
            raw_result = await agent_runner(
                user_message=turn.user_message,
                request_id=f"EVAL-{case.id}-T{index}-{uuid.uuid4()}",
                thread_id=thread_id,
                graph=graph,
                user_id=case.identity.user_id,
                user_role=case.identity.role,
                customer_id=case.identity.customer_id,
                evaluation_mode=True,
            )
            trace = _trace_from_result(raw_result)
        except Exception as exc:
            trace = {
                "tool_calls": [],
                "tools_executed": [],
                "status": "failed",
                "response": None,
                "error_code": type(exc).__name__,
            }
        turns.append(_evaluate_expectation(
            case_id=f"{case.id}:turn-{index}",
            category="conversation",
            expectation=turn,
            trace=trace,
            latency_ms=(time.perf_counter() - turn_started) * 1000,
        ))
    passed, failures = evaluate_conversation(turns)
    return EvaluationResult(
        case_id=case.id,
        category="conversation",
        passed=passed,
        latency_ms=(time.perf_counter() - started_at) * 1000,
        failure_reasons=failures,
        turn_results=turns,
    )


async def run_suite(
    cases: list[EvaluationCase | ConversationEvaluationCase],
    *,
    graph,
    suite: str,
    model_name: str = OLLAMA_MODEL,
    prompt_version: str = AGENT_PROMPT_VERSION,
    application_version: str = APPLICATION_VERSION,
    agent_runner: AgentRunner = run_agent_graph,
    resume_runner: ResumeRunner = resume_agent_graph,
    fixture_hook: FixtureHook | None = None,
) -> EvaluationRun:
    results = []
    for case in cases:
        if isinstance(case, ConversationEvaluationCase):
            result = await run_conversation_case(
                case,
                graph=graph,
                agent_runner=agent_runner,
                resume_runner=resume_runner,
                fixture_hook=fixture_hook,
            )
        else:
            result = await run_case(
                case,
                graph=graph,
                agent_runner=agent_runner,
                resume_runner=resume_runner,
                fixture_hook=fixture_hook,
            )
        results.append(result)
    scores = calculate_scores(results)
    suite_gates = {
        "write": {"workflow": RELEASE_GATES["workflow"]},
        "fault-injection": {"error_handling": 1.0},
    }.get(suite)
    release_allowed, gate_failures = evaluate_release_gates(scores, suite_gates)
    latencies = [
        result.latency_ms for result in results if result.latency_ms is not None
    ]
    passed = sum(result.passed for result in results)
    return EvaluationRun(
        run_id=str(uuid.uuid4()),
        model=model_name,
        prompt_version=prompt_version,
        application_version=application_version,
        suite=suite,
        total_cases=len(results),
        passed=passed,
        failed=len(results) - passed,
        pass_rate=passed / len(results) if results else 0.0,
        scores=scores,
        average_latency_ms=(sum(latencies) / len(latencies) if latencies else None),
        release_allowed=release_allowed,
        gate_failures=gate_failures,
        results=results,
    )


async def _main(args: argparse.Namespace) -> int:
    fixture_hook = None
    if args.load_fixtures:
        from ai.evaluations.fixtures import (
            apply_evaluation_fixture,
            load_evaluation_fixtures,
        )

        loaded = await asyncio.to_thread(load_evaluation_fixtures)
        print(f"Loaded {loaded} evaluation fixtures")
        fixture_hook = apply_evaluation_fixture
    cases = load_cases(args.dataset_dir, suite=args.suite)
    graph = build_enterprise_tool_graph(InMemorySaver())
    run = await run_suite(
        cases,
        graph=graph,
        suite=args.suite,
        fixture_hook=fixture_hook,
    )
    path = save_run(run, args.output_dir)
    print(render_terminal_report(run))
    print(f"Saved: {path}")
    return 0 if run.release_allowed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("smoke", "full", "write", "fault-injection"),
        default="smoke",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--load-fixtures",
        action="store_true",
        help="Upsert synthetic TX-EVAL records into the configured test DB",
    )
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
