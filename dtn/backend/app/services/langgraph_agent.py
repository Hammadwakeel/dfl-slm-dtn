"""LangGraph control loop for model-selected, schema-bound coordinator tools."""

import json
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    tool_calls: int
    retryable_tool_calls: int
    trace: list[dict[str, Any]]
    paused: bool
    terminated: bool


ModelCall = Callable[[list[dict[str, Any]]], dict[str, Any]]
ToolCall = Callable[[str, dict[str, Any]], dict[str, Any]]


def build_agent_graph(
    *,
    model_call: ModelCall,
    tool_call: ToolCall,
    max_tool_calls: int,
    max_retryable_tool_calls: int = 2,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    stop_requested: Callable[[], bool] | None = None,
):
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be at least one")
    if max_retryable_tool_calls < 1:
        raise ValueError("max_retryable_tool_calls must be at least one")

    def completed_tool_calls(state: AgentState) -> int:
        return state["tool_calls"] - state["retryable_tool_calls"]

    def call_model(state: AgentState) -> dict[str, Any]:
        if stop_requested and stop_requested():
            return {
                "messages": [
                    *state["messages"],
                    {
                        "role": "assistant",
                        "content": "The Agent paused before choosing its next action because a stop was requested.",
                    },
                ],
                "paused": True,
            }
        message = model_call(state["messages"])
        if message.get("role") != "assistant":
            raise RuntimeError("The agent model returned a non-assistant message")
        return {"messages": [*state["messages"], message], "paused": False}

    def route_after_model(state: AgentState) -> str:
        if state["paused"]:
            return END
        latest = state["messages"][-1]
        if not latest.get("tool_calls"):
            return END
        if completed_tool_calls(state) >= max_tool_calls:
            return "limit"
        return "tools"

    def execute_tool(state: AgentState) -> dict[str, Any]:
        assistant_message = state["messages"][-1]
        calls = assistant_message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            raise RuntimeError("The agent attempted to execute an empty tool-call batch")

        tool_messages: list[dict[str, Any]] = []
        trace = list(state["trace"])
        call = calls[0]
        if not isinstance(call, dict):
            raise RuntimeError("The agent returned a malformed tool call")
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise RuntimeError("The agent returned a malformed function call")
        name = function["name"]
        decision = assistant_message.get("content")
        if not isinstance(decision, str):
            decision = ""
        decision = decision.strip()[:2000]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if stop_requested and stop_requested():
            result: dict[str, Any] = {
                "status": "paused",
                "error": "The selected action was not executed because a stop was requested.",
            }
        elif not isinstance(arguments, dict):
            result: dict[str, Any] = {"error": "Tool arguments must be a JSON object"}
        else:
            result = tool_call(name, arguments)

        tool_messages.append(
            {
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            }
        )
        trace.append({
            "tool": name,
            "arguments": arguments,
            "result": result,
            "decision": decision,
        })
        paused = result.get("status") == "paused"
        terminated = result.get("status") == "workflow_complete"
        retryable_tool_call = result.get("retryable") is True or (
            "error" in result and result.get("status") is None
        )
        if paused:
            tool_messages.append(
                {
                    "role": "assistant",
                    "content": "The selected action was not run. The Agent is paused and can be resumed.",
                }
            )
        elif terminated:
            summary = result.get("summary")
            outcome = result.get("outcome")
            content = (
                f"The SLM ended this workflow ({outcome}): {summary}"
                if isinstance(summary, str) and isinstance(outcome, str)
                else "The SLM ended this workflow."
            )
            tool_messages.append({"role": "assistant", "content": content})

        for ignored in calls[1:]:
            ignored_function = ignored.get("function", {}) if isinstance(ignored, dict) else {}
            ignored_name = (
                ignored_function.get("name", "unknown")
                if isinstance(ignored_function, dict)
                else "unknown"
            )
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_name": str(ignored_name),
                    "content": json.dumps(
                        {"error": "Call one tool at a time; choose the next action after reviewing this result."}
                    ),
                }
            )
            trace.append(
                {
                    "tool": ignored_name,
                    "arguments": {},
                    "result": {"error": "Only the first call in this batch was executed"},
                }
            )
        return {
            "messages": [*state["messages"], *tool_messages],
            "tool_calls": state["tool_calls"] + 1,
            "retryable_tool_calls": (
                state["retryable_tool_calls"] + 1
                if retryable_tool_call
                else state["retryable_tool_calls"]
            ),
            "trace": trace,
            "paused": paused,
            "terminated": terminated,
        }

    def stop_at_limit(state: AgentState) -> dict[str, Any]:
        if state["retryable_tool_calls"] >= max_retryable_tool_calls:
            message = (
                "I stopped after repeated tool-validation failures. The validation errors "
                "are shown above; choose a schema-valid action before continuing."
            )
        else:
            message = (
                "I stopped after reaching the configured agent tool-call limit. "
                "The completed tool results are shown above; review the current system state "
                "before requesting another action."
            )
        return {
            "messages": [*state["messages"], {"role": "assistant", "content": message}]
        }

    def route_after_tool(state: AgentState) -> str:
        if state["paused"] or state["terminated"]:
            return END
        if (
            state["retryable_tool_calls"] >= max_retryable_tool_calls
            or completed_tool_calls(state) >= max_tool_calls
        ):
            return "limit"
        return "model"

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", execute_tool)
    graph.add_node("limit", stop_at_limit)
    graph.set_entry_point("model")
    graph.add_conditional_edges(
        "model",
        route_after_model,
        {"tools": "tools", "limit": "limit", END: END},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_tool,
        {"model": "model", "limit": "limit", END: END},
    )
    graph.add_edge("limit", END)
    return graph.compile(checkpointer=checkpointer)


def run_agent(
    *,
    messages: list[dict[str, Any]],
    model_call: ModelCall,
    tool_call: ToolCall,
    max_tool_calls: int,
    max_retryable_tool_calls: int = 2,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    thread_id: str | None = None,
    resume: bool = False,
    stop_requested: Callable[[], bool] | None = None,
    on_update: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    graph = build_agent_graph(
        model_call=model_call,
        tool_call=tool_call,
        max_tool_calls=max_tool_calls,
        max_retryable_tool_calls=max_retryable_tool_calls,
        checkpointer=checkpointer,
        stop_requested=stop_requested,
    )
    config: dict[str, Any] = {
        "recursion_limit": (max_tool_calls + max_retryable_tool_calls) * 2 + 3
    }
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    if checkpointer and not thread_id:
        raise ValueError("A thread_id is required when checkpointing an agent run")
    initial_state: dict[str, Any] | None
    if resume:
        if not thread_id:
            raise ValueError("A thread_id is required to resume an agent run")
        checkpoint = graph.get_state(config)
        if not checkpoint.values:
            raise RuntimeError("No LangGraph checkpoint is available for this run")
        initial_state = dict(checkpoint.values)
        initial_state["paused"] = False
        initial_state.setdefault("retryable_tool_calls", 0)
        if messages and initial_state["messages"]:
            initial_state["messages"][0] = messages[0]
        last_message = initial_state["messages"][-1] if initial_state["messages"] else {}
        pending_calls = last_message.get("tool_calls") if isinstance(last_message, dict) else None
        if isinstance(pending_calls, list) and pending_calls:
            for pending_call in pending_calls:
                function = (
                    pending_call.get("function", {})
                    if isinstance(pending_call, dict)
                    else {}
                )
                tool_name = (
                    function.get("name", "unknown")
                    if isinstance(function, dict)
                    else "unknown"
                )
                initial_state["messages"].append(
                    {
                        "role": "tool",
                        "tool_name": str(tool_name),
                        "content": (
                            "The coordinator stopped before recording this action's result. "
                            "Its outcome is unknown. Do not retry it until a read-only check "
                            "confirms the current system state."
                        ),
                    }
                )
            initial_state["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "Recovery note: the last selected action has an unknown outcome. "
                        "I will verify current state before considering another state change."
                    ),
                }
            )
            recovery_details = {"pending_tools": len(pending_calls)}
        else:
            recovery_details = None
        graph.update_state(config, initial_state, as_node="tools")
        if on_update and recovery_details:
            on_update({"recovery": recovery_details}, initial_state)
        initial_state = None
    else:
        initial_state = {
            "messages": messages,
            "tool_calls": 0,
            "retryable_tool_calls": 0,
            "trace": [],
            "paused": False,
            "terminated": False,
        }
    if on_update:
        for update in graph.stream(initial_state, config=config, stream_mode="updates"):
            snapshot = graph.get_state(config)
            on_update(update, dict(snapshot.values))
        state = dict(graph.get_state(config).values)
    else:
        state = graph.invoke(initial_state, config=config)
    final_message = state["messages"][-1]
    reply = final_message.get("content")
    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("The agent completed without a user-facing response")
    return {
        "reply": reply.strip(),
        "trace": state["trace"],
        "tool_calls": state["tool_calls"],
        "paused": state["paused"],
        "terminated": state.get("terminated", False),
    }
