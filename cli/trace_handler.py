import ast
import json
import threading
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="milliseconds")


def _short_id(run_id: Any) -> str:
    return str(run_id)[:8]


def _parse_input(input_str: str) -> Any:
    try:
        return json.loads(input_str)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(input_str)
    except Exception:
        return input_str


def _extract_output(output: Any) -> Any:
    if hasattr(output, "content"):
        return output.content
    return str(output)


def _serialize_message(msg: Any) -> dict:
    if isinstance(msg, BaseMessage):
        content = msg.content
        # content can be str or list of content blocks
        return {"role": msg.type, "content": content}
    return {"role": "unknown", "content": str(msg)}


class TraceCallbackHandler(BaseCallbackHandler):
    """Streams per-agent trace events to a JSONL file.

    Each line is one JSON event with: ts, event, node (agent name), run_id,
    and event-specific payload (messages / content / tool name / output).
    """

    def __init__(self, trace_path: Path) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._file = open(trace_path, "w", encoding="utf-8")
        # run_id (str) → node name
        self._chain_names: Dict[str, str] = {}

    def _write(self, event: dict) -> None:
        line = json.dumps(event, default=str, ensure_ascii=False) + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()

    def _node_for(self, parent_run_id: Optional[Any]) -> str:
        if parent_run_id is None:
            return "unknown"
        return self._chain_names.get(str(parent_run_id), "unknown")

    # ------------------------------------------------------------------
    # Chain hooks — used to map run_id → LangGraph node name
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Any,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        node_name = name or (serialized or {}).get("name") or "unknown"
        with self._lock:
            self._chain_names[str(run_id)] = node_name

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._chain_names.pop(str(run_id), None)

    # ------------------------------------------------------------------
    # LLM hooks
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        node = self._node_for(parent_run_id)
        serialized_msgs = [
            [_serialize_message(m) for m in batch if getattr(m, "type", None) != "system"]
            for batch in messages
        ]
        self._write({
            "ts": _now(),
            "event": "llm_start",
            "node": node,
            "run_id": _short_id(run_id),
            "messages": serialized_msgs,
        })

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        node = self._node_for(parent_run_id)
        content = ""
        tokens_in = 0
        tokens_out = 0
        try:
            gen = response.generations[0][0]
            if hasattr(gen, "message"):
                content = gen.message.content
                usage = getattr(gen.message, "usage_metadata", None) or {}
                tokens_in = usage.get("input_tokens", 0)
                tokens_out = usage.get("output_tokens", 0)
            else:
                content = getattr(gen, "text", "")
        except (IndexError, TypeError):
            pass
        event: Dict[str, Any] = {
            "ts": _now(),
            "event": "llm_end",
            "node": node,
            "run_id": _short_id(run_id),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
        if content:
            event["content"] = content
        self._write(event)

    # ------------------------------------------------------------------
    # Tool hooks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        node = self._node_for(parent_run_id)
        tool_name = (serialized or {}).get("name", "unknown")
        self._write({
            "ts": _now(),
            "event": "tool_start",
            "node": node,
            "run_id": _short_id(run_id),
            "tool": tool_name,
            "input": _parse_input(input_str),
        })

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        node = self._node_for(parent_run_id)
        raw = _extract_output(output)
        if isinstance(raw, str) and len(raw) > 500:
            raw = raw[:500] + f"…[truncated {len(raw) - 500} chars]"
        self._write({
            "ts": _now(),
            "event": "tool_end",
            "node": node,
            "run_id": _short_id(run_id),
            "output": raw,
        })

    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._file.close()
