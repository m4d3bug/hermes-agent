#!/usr/bin/env python3
"""
Orchestrator Tool — Multi-Agent Task Decomposition and DAG Execution.

Takes a complex task, uses an LLM to decompose it into subtasks with
specialized agent role assignments, builds a dependency DAG, executes
independent tasks in parallel (via delegate_task), and synthesizes
a final result.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

MAX_SUBTASKS = 8
_DEFAULT_MAX_ITERATIONS_PER_TASK = 50


_DECOMPOSITION_PROMPT = """\
You are a task planner. Decompose the following complex task into subtasks \
that can be executed by specialized AI agents.

TASK:
{task}
{hints_block}
AVAILABLE AGENT ROLES:
- researcher: Web research, information gathering, and synthesis
- coder: Code implementation, modification, and bug fixes
- reviewer: Code review, quality analysis, and security audit
- planner: Task decomposition, architecture, and planning
- debugger: Bug investigation, root cause analysis, and diagnosis
- browser_agent: Browser automation, web scraping, and UI testing
- sysadmin: Server management, infrastructure, and operations

RULES:
- Maximum {max_subtasks} subtasks
- Maximize parallelism: make tasks as independent as possible
- Each task must be self-contained (the agent has no memory of other tasks)
- Use depends_on only when a task truly needs another's output
- Use context_from to specify which prior task results to inject as context

Return ONLY a valid JSON object (no markdown, no explanation) with this schema:
{{
  "subtasks": [
    {{
      "id": "task_1",
      "goal": "Specific, self-contained task description",
      "role": "one of the roles above",
      "depends_on": [],
      "context_from": []
    }}
  ],
  "synthesis_prompt": "Instructions for combining all subtask results into a final answer"
}}
"""


def _decompose_task(
    task_description: str,
    parent_agent,
    hints: Optional[str] = None,
    max_subtasks: int = MAX_SUBTASKS,
) -> Dict[str, Any]:
    """Use LLM to decompose a complex task into subtasks with role assignments."""
    from agent.auxiliary_client import call_llm

    hints_block = f"\nHINTS:\n{hints}\n" if hints else ""
    prompt = _DECOMPOSITION_PROMPT.format(
        task=task_description,
        hints_block=hints_block,
        max_subtasks=max_subtasks,
    )

    main_runtime = None
    if parent_agent:
        main_runtime = {
            "provider": getattr(parent_agent, "provider", None),
            "base_url": getattr(parent_agent, "base_url", None),
            "api_key": getattr(parent_agent, "api_key", None),
            "api_mode": getattr(parent_agent, "api_mode", None),
            "model": getattr(parent_agent, "model", None),
        }

    response = call_llm(
        task="compression",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4096,
        main_runtime=main_runtime,
    )

    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        content = "\n".join(lines)

    try:
        plan = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON for task decomposition: {exc}") from exc

    subtasks = plan.get("subtasks", [])
    if not subtasks:
        raise ValueError("LLM returned no subtasks")
    if len(subtasks) > max_subtasks:
        subtasks = subtasks[:max_subtasks]
        plan["subtasks"] = subtasks

    for st in subtasks:
        if not st.get("id") or not st.get("goal"):
            raise ValueError(f"Subtask missing id or goal: {st}")
        st.setdefault("role", "coder")
        st.setdefault("depends_on", [])
        st.setdefault("context_from", [])

    return plan


def _build_dependency_waves(
    subtasks: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Convert subtask list into execution waves using topological sort (Kahn's algorithm)."""
    task_map = {st["id"]: st for st in subtasks}
    in_degree = {st["id"]: 0 for st in subtasks}
    dependents: Dict[str, List[str]] = {st["id"]: [] for st in subtasks}

    for st in subtasks:
        for dep in st.get("depends_on", []):
            if dep in task_map:
                in_degree[st["id"]] += 1
                dependents[dep].append(st["id"])

    waves = []
    remaining = set(in_degree.keys())

    while remaining:
        wave_ids = [tid for tid in remaining if in_degree[tid] == 0]
        if not wave_ids:
            raise ValueError(
                f"Circular dependency detected among tasks: {remaining}"
            )
        wave = [task_map[tid] for tid in wave_ids]
        waves.append(wave)
        for tid in wave_ids:
            remaining.discard(tid)
            for dep_id in dependents.get(tid, []):
                in_degree[dep_id] -= 1

    return waves


def _execute_wave(
    wave: List[Dict[str, Any]],
    completed_results: Dict[str, str],
    parent_agent,
    max_iterations: int,
) -> Dict[str, Dict[str, Any]]:
    """Execute a single wave of independent tasks via delegate_task."""
    from tools.delegate_tool import delegate_task

    tasks_for_delegate = []
    for st in wave:
        context_parts = []
        for ctx_id in st.get("context_from", []):
            if ctx_id in completed_results:
                context_parts.append(
                    f"=== Results from {ctx_id} ===\n{completed_results[ctx_id]}"
                )
        context = "\n\n".join(context_parts) if context_parts else None

        tasks_for_delegate.append({
            "goal": st["goal"],
            "role": st.get("role"),
            "context": context,
            "toolsets": st.get("toolsets"),
        })

    if len(tasks_for_delegate) == 1:
        t = tasks_for_delegate[0]
        result_json = delegate_task(
            goal=t["goal"],
            context=t.get("context"),
            toolsets=t.get("toolsets"),
            role=t.get("role"),
            max_iterations=max_iterations,
            parent_agent=parent_agent,
        )
    else:
        result_json = delegate_task(
            tasks=tasks_for_delegate,
            max_iterations=max_iterations,
            parent_agent=parent_agent,
        )

    try:
        result_data = json.loads(result_json)
    except json.JSONDecodeError:
        return {
            wave[0]["id"]: {"status": "error", "summary": result_json}
        }

    wave_results = {}
    results_list = result_data.get("results", [])
    for i, st in enumerate(wave):
        if i < len(results_list):
            entry = results_list[i]
            wave_results[st["id"]] = entry
        else:
            wave_results[st["id"]] = {
                "status": "error",
                "summary": "No result returned for this subtask",
            }

    return wave_results


def _synthesize_results(
    subtask_results: Dict[str, Dict[str, Any]],
    synthesis_prompt: str,
    original_task: str,
    parent_agent,
) -> str:
    """Use LLM to synthesize all subtask results into a final response."""
    from agent.auxiliary_client import call_llm

    parts = [f"ORIGINAL TASK:\n{original_task}\n"]
    for task_id, result in subtask_results.items():
        summary = result.get("summary", "") or result.get("error", "No output")
        status = result.get("status", "unknown")
        parts.append(f"=== {task_id} (status: {status}) ===\n{summary}\n")
    parts.append(f"\nINSTRUCTIONS:\n{synthesis_prompt}")

    main_runtime = None
    if parent_agent:
        main_runtime = {
            "provider": getattr(parent_agent, "provider", None),
            "base_url": getattr(parent_agent, "base_url", None),
            "api_key": getattr(parent_agent, "api_key", None),
            "api_mode": getattr(parent_agent, "api_mode", None),
            "model": getattr(parent_agent, "model", None),
        }

    response = call_llm(
        task="compression",
        messages=[{"role": "user", "content": "\n".join(parts)}],
        temperature=0.3,
        max_tokens=4096,
        main_runtime=main_runtime,
    )

    return response.choices[0].message.content.strip()


def orchestrate(
    task: str,
    hints: Optional[str] = None,
    max_subtasks: Optional[int] = None,
    max_iterations_per_task: Optional[int] = None,
    parent_agent=None,
) -> str:
    """Decompose a complex task, execute subtasks with specialized agents, and synthesize results."""
    if parent_agent is None:
        return tool_error("orchestrate requires a parent agent context.")
    if not task or not task.strip():
        return tool_error("No task provided.")

    effective_max_subtasks = min(max_subtasks or MAX_SUBTASKS, MAX_SUBTASKS)
    effective_max_iter = max_iterations_per_task or _DEFAULT_MAX_ITERATIONS_PER_TASK
    overall_start = time.monotonic()

    # Phase 1: Decompose
    try:
        plan = _decompose_task(task, parent_agent, hints, effective_max_subtasks)
    except (ValueError, RuntimeError) as exc:
        return tool_error(f"Task decomposition failed: {exc}")

    subtasks = plan["subtasks"]
    synthesis_prompt = plan.get("synthesis_prompt", "Summarize the combined results.")

    # Phase 2: Build execution waves
    try:
        waves = _build_dependency_waves(subtasks)
    except ValueError as exc:
        return tool_error(f"Dependency resolution failed: {exc}")

    # Phase 3: Execute waves
    all_results: Dict[str, Dict[str, Any]] = {}
    completed_summaries: Dict[str, str] = {}
    wave_details = []

    for wave_idx, wave in enumerate(waves):
        wave_start = time.monotonic()
        wave_results = _execute_wave(wave, completed_summaries, parent_agent, effective_max_iter)
        wave_duration = round(time.monotonic() - wave_start, 2)

        for task_id, result in wave_results.items():
            all_results[task_id] = result
            summary = result.get("summary", "") or ""
            completed_summaries[task_id] = summary

        wave_details.append({
            "wave": wave_idx,
            "tasks": [st["id"] for st in wave],
            "duration_seconds": wave_duration,
        })

    # Phase 4: Synthesize
    try:
        final_result = _synthesize_results(
            all_results, synthesis_prompt, task, parent_agent,
        )
    except Exception as exc:
        logger.warning("Synthesis failed, returning raw results: %s", exc)
        final_result = "\n\n".join(
            f"[{tid}] {r.get('summary', r.get('error', 'No output'))}"
            for tid, r in all_results.items()
        )

    total_duration = round(time.monotonic() - overall_start, 2)

    succeeded = sum(1 for r in all_results.values() if r.get("status") == "completed")
    failed = len(all_results) - succeeded

    return json.dumps({
        "success": failed == 0,
        "result": final_result,
        "plan": {
            "subtasks": [
                {"id": st["id"], "role": st.get("role"), "goal": st["goal"][:100]}
                for st in subtasks
            ],
            "waves": len(waves),
        },
        "execution": {
            "total_subtasks": len(subtasks),
            "succeeded": succeeded,
            "failed": failed,
            "waves": wave_details,
            "total_duration_seconds": total_duration,
        },
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Schema & Registration
# ---------------------------------------------------------------------------

ORCHESTRATE_SCHEMA = {
    "name": "orchestrate",
    "description": (
        "Decompose a complex task into subtasks with specialized agent roles, "
        "build a dependency graph, and execute them with parallel/sequential "
        "scheduling. Each subtask is assigned a specialist agent (researcher, "
        "coder, reviewer, debugger, etc.) and independent tasks run in parallel.\n\n"
        "WHEN TO USE:\n"
        "- Complex projects requiring multiple types of expertise\n"
        "- Tasks with natural decomposition (e.g., research -> implement -> review)\n"
        "- Work that benefits from parallel execution of independent subtasks\n\n"
        "WHEN NOT TO USE:\n"
        "- Simple tasks that a single delegate_task can handle\n"
        "- Tasks needing user interaction at intermediate steps\n"
        "- Quick lookups or single-tool operations\n\n"
        "The orchestrator automatically:\n"
        "1. Decomposes the task into subtasks with role assignments\n"
        "2. Builds a dependency DAG\n"
        "3. Executes independent tasks in parallel, dependent tasks sequentially\n"
        "4. Passes results from earlier tasks as context to dependent tasks\n"
        "5. Synthesizes a final result from all subtask outputs"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "Detailed description of the complex task to accomplish. "
                    "Be specific about requirements, constraints, and expected deliverables."
                ),
            },
            "hints": {
                "type": "string",
                "description": (
                    "Optional hints for task decomposition: suggested subtask structure, "
                    "preferred roles, dependency relationships, or constraints."
                ),
            },
            "max_subtasks": {
                "type": "integer",
                "description": "Maximum number of subtasks (default: 8, max: 8).",
            },
            "max_iterations_per_task": {
                "type": "integer",
                "description": "Max tool-calling turns per subtask agent (default: 50).",
            },
        },
        "required": ["task"],
    },
}


registry.register(
    name="orchestrate",
    toolset="orchestration",
    schema=ORCHESTRATE_SCHEMA,
    handler=lambda args, **kw: orchestrate(
        task=args.get("task", ""),
        hints=args.get("hints"),
        max_subtasks=args.get("max_subtasks"),
        max_iterations_per_task=args.get("max_iterations_per_task"),
        parent_agent=kw.get("parent_agent"),
    ),
    check_fn=lambda: True,
    emoji="\U0001f3ad",  # 🎭
)
