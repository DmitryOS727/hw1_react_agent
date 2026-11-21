"""Траєкторія одного запиту. Час події в UTC, тривалість за монотонним годинником."""

import asyncio
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from langchain_core.messages import AIMessage, ToolMessage


def message_record(message):
    data = {"role": message.type, "content": message.content}
    if message.name:
        data["name"] = message.name
    if isinstance(message, AIMessage) and message.tool_calls:
        data["tool_calls"] = message.tool_calls
    if isinstance(message, AIMessage) and message.invalid_tool_calls:
        data["invalid_tool_calls"] = message.invalid_tool_calls
    if isinstance(message, ToolMessage):
        data.update(tool_call_id=message.tool_call_id, name=message.name, status=message.status)
    return data


def write_json(path: Path, data):
    """Замінюємо готовий файл одним кроком, щоб не залишити обірваний JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False)
    try:
        with f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(f.name, path)
    finally:
        Path(f.name).unlink(missing_ok=True)


class TrajectoryLogger:
    def __init__(self):
        self.started = monotonic()
        self.events = []
        self.observations = []
        self.agent_steps = 0

    @contextmanager
    def node(self, name, input_data, tool_calls=()):
        started = monotonic()
        event = {
            "step_number": len(self.events) + 1,
            "node_name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "input": input_data,
            "output": None,
            "tool_calls": list(tool_calls),
        }
        self.events.append(event)
        try:
            yield event
        except asyncio.CancelledError:
            event["output"] = {"interrupted": True}
            raise
        except Exception as exc:
            event["output"] = {"error": str(exc), "type": type(exc).__name__}
            raise
        finally:
            event["duration_ms"] = round((monotonic() - started) * 1000, 2)

    def observe(self, message: ToolMessage, call: dict):
        try:
            value = json.loads(message.content)
        except (ValueError, TypeError):
            value = message.content
        self.observations.append({
            "id": message.tool_call_id,
            "name": call["name"],
            "args": call["args"],
            "status": message.status,
            "result": value,
        })

    def sources(self):
        return list(dict.fromkeys(o["name"] for o in self.observations if o["status"] == "success"))
