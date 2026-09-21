"""Score calculation, hard release gates, and terminal/JSON reporting."""

import json
from collections import defaultdict
from pathlib import Path

from ai.evaluations.schemas import EvaluationResult, EvaluationRun


RELEASE_GATES = {
    "security": 1.0,
    "workflow": 1.0,
    "tool_selection": 0.95,
    "conversation": 0.90,
}


def calculate_scores(results: list[EvaluationResult]) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        grouped[result.category].append(result.passed)
    return {
        category: sum(outcomes) / len(outcomes)
        for category, outcomes in sorted(grouped.items())
    }


def evaluate_release_gates(
    scores: dict[str, float],
    gates: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    active_gates = RELEASE_GATES if gates is None else gates
    failures = []
    for category, threshold in active_gates.items():
        score = scores.get(category)
        if score is None:
            failures.append(f"{category}: not evaluated")
        elif score < threshold:
            failures.append(
                f"{category}: {score:.1%} is below required {threshold:.1%}"
            )
    return not failures, failures


def render_terminal_report(run: EvaluationRun) -> str:
    lines = [
        "=" * 48,
        "ENTERPRISE AGENT EVALUATION",
        "=" * 48,
        f"Model:             {run.model}",
        f"Prompt version:    {run.prompt_version}",
        f"Suite:             {run.suite}",
        f"Total cases:       {run.total_cases}",
        f"Passed:            {run.passed}",
        f"Failed:            {run.failed}",
        f"Overall pass rate: {run.pass_rate:.1%}",
    ]
    for category, score in sorted(run.scores.items()):
        lines.append(f"{category.replace('_', ' ').title():<18} {score:.1%}")
    latency = (
        f"{run.average_latency_ms:.1f} ms"
        if run.average_latency_ms is not None
        else "n/a"
    )
    lines.extend([
        f"Average latency:    {latency}",
        f"Release allowed:    {'YES' if run.release_allowed else 'NO'}",
    ])
    lines.extend(f"Gate failure:       {failure}" for failure in run.gate_failures)
    for result in run.results:
        if not result.passed:
            lines.append(
                f"FAILED {result.case_id}: {'; '.join(result.failure_reasons)}"
            )
    lines.append("=" * 48)
    return "\n".join(lines)


def save_run(run: EvaluationRun, output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_model = run.model.replace(":", "-").replace("/", "-")
    timestamp = run.created_at.strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{timestamp}_{safe_model}_{run.run_id[:8]}.json"
    path.write_text(
        json.dumps(run.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path
