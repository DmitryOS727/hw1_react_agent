"""ReAct-агент для матеріалів на заняття. Запуск прикладу описано в README."""

import argparse
import asyncio
import json
import os
import re
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from logger import TrajectoryLogger, message_record, write_json
from safety import Limits, LoopDetector
from tools import TOOLS, get_pack


DEFAULT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
SYSTEM = """Help an educator order classroom supplies. Answer the user in Ukrainian.
A group calculation requires BOTH the group size and the units per child from the user.
If either is missing, stop and ask for that number. A typical group size is never evidence.
For a catalogue question, call get_pack for the requested material. Never ask permission
to read the catalogue. Read the result before planning. To find packs or leftovers, call
plan_packs with the user's numbers and the observed pack_size. Only on_hand defaults to zero.
To find the total cost, call order_cost with the number of packs and the observed price_cents.
Continue calling tools until you have every requested fact. A single pack price is not an order
total. Reply briefly, with digits and units. For example, when a group size is missing, ask
“Скільки дітей братимуть участь у занятті?”."""


class ConfidenceEstimate(BaseModel):
    confidence: float = Field(ge=0, le=1, description="Самооцінка моделі від 0 до 1.")


class FinalAnswer(BaseModel):
    answer: str = Field(min_length=1, description="Текст відповіді або уточнення від вузла agent.")
    confidence: float = Field(ge=0, le=1, description="Самооцінка моделі від 0 до 1.")
    sources: list[str] = Field(description="Імена успішно виконаних інструментів у порядку виклику.")


class AgentState(MessagesState):
    step_count: int
    final: FinalAnswer | None
    stop_reason: str | None
    quote_issue: str | None


def make_model(model_name: str):
    llm = ChatOllama(
        model=model_name, temperature=0, seed=17, num_ctx=4096, num_predict=320,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    catalog_only = llm.bind_tools([get_pack])
    with_calculators = llm.bind_tools(TOOLS)

    async def decide(messages, catalog_ready):
        model = with_calculators if catalog_ready else catalog_only
        return await model.ainvoke(messages)

    return decide, llm.with_structured_output(ConfidenceEstimate, method="json_schema")


def halt(log: TrajectoryLogger, reason: str, explanation: str, calls=()):
    successes = [o for o in log.observations if o["status"] == "success"]
    text = explanation
    if successes:
        evidence = [f"{o['name']} → {json.dumps(o['result'], ensure_ascii=False)}" for o in successes]
        text += " Отримані дані: " + " / ".join(dict.fromkeys(evidence)) + "."
    final = FinalAnswer(answer=text, confidence=0.0, sources=log.sources())
    with log.node("guard", {"reason": reason, "agent_steps": log.agent_steps}, calls) as event:
        event["output"] = final.model_dump()
    return {"stop_reason": reason, "final": final}


def quote_feedback(answer: str, observations: list[dict]) -> str | None:
    """Перевірити грошовий підсумок після планування закупівлі одного матеріалу."""
    currency = r"(?:\bCAD\b|(?:CA|C)?\$|(?:канадськ\w*\s+)?долар\w*|(?:Canadian\s+)?dollars?)"
    if not re.search(rf"{currency}|вартіст|кошту", answer, re.IGNORECASE):
        return None
    successful = {o["name"]: o for o in observations if o["status"] == "success"}
    if "plan_packs" not in successful or "get_pack" not in successful:
        return None
    count = successful["plan_packs"]["result"]["packs_to_buy"]
    price = successful["get_pack"]["result"]["price_cents"]
    cost = successful.get("order_cost")
    if not cost or cost["args"] != {"packs": count, "price_cents": price}:
        return (f"Підтвердь загальну вартість. Виклич order_cost з packs={count} і "
                f"price_cents={price}, дочекайся результату й дай відповідь користувачу.")
    # Число має стояти поряд із позначенням валюти. Кількість упаковок або залишок
    # у тексті не можуть випадково підтвердити грошову суму.
    number = r"\d+(?:[.,]\d{1,2})?"
    matches = re.findall(rf"(?<![\d.,])({number})\s*{currency}|{currency}\s*({number})(?!\d|[.,]\d)",
                         answer, re.IGNORECASE)
    amounts = {Decimal((left or right).replace(",", ".")) for left, right in matches}
    total = Decimal(cost["result"]["total_cad"])
    unit = Decimal(successful["get_pack"]["result"]["price_cad"])
    if total not in amounts or amounts - {total, unit}:
        return f"Звір грошові суми з інструментами. Вкажи підтверджений підсумок {total:.2f} CAD."
    return None


def build_graph(decide, assess_confidence, limits: Limits, log: TrajectoryLogger):
    detector = LoopDetector(limits.max_repeats)

    async def check_catalog(request, execute):
        # Pydantic перевіряє тип, а тут звіряємо число з даними, які агент уже отримав.
        call = request.tool_call
        field = {"plan_packs": "pack_size", "order_cost": "price_cents"}.get(call["name"])
        if field:
            verified = [o["result"][field] for o in log.observations
                        if o["name"] == "get_pack" and o["status"] == "success"]
            if call["args"].get(field) not in verified:
                return ToolMessage(
                    content=f"Поле {field} має походити з get_pack. Звір його з каталогом: {verified}.",
                    name=call["name"], tool_call_id=call["id"], status="error",
                )
        return await execute(request)

    executor = ToolNode(TOOLS, handle_tool_errors=True, awrap_tool_call=check_catalog)

    async def agent(state: AgentState):
        if state["step_count"] >= limits.max_steps:
            return halt(log, "max_steps", f"Досягнуто ліміт рішень моделі ({limits.max_steps}).")
        messages = state["messages"]
        with log.node("agent", [message_record(m) for m in messages]) as event:
            log.agent_steps += 1  # Враховуємо також розпочатий і перерваний запит.
            catalog_ready = "get_pack" in log.sources()
            event["available_tools"] = [t.name for t in TOOLS] if catalog_ready else ["get_pack"]
            reply = await decide(messages, catalog_ready)
            if not isinstance(reply, AIMessage):
                raise TypeError("Очікувалося повідомлення AIMessage.")
            event["output"] = message_record(reply)
            event["tool_calls"] = reply.tool_calls
        update = {"messages": [reply], "step_count": log.agent_steps, "quote_issue": None}
        if reply.invalid_tool_calls:
            update.update(halt(log, "invalid_tool_call", "Модель повернула некоректний виклик інструмента."))
        elif any(detector.repeated(call) for call in reply.tool_calls):
            note = f"Зупинено після {limits.max_repeats} однакових запитів поспіль."
            update.update(halt(log, "loop_detected", note, reply.tool_calls))
        elif not reply.tool_calls:
            update["quote_issue"] = quote_feedback(reply.text, log.observations)
        return update

    async def tools_node(state: AgentState):
        calls = state["messages"][-1].tool_calls
        by_id = {call["id"]: call for call in calls}
        with log.node("tools", {"requested": calls}, calls) as event:
            update = await executor.ainvoke({"messages": state["messages"]})
            for message in update["messages"]:
                log.observe(message, by_id[message.tool_call_id])
            event["output"] = [message_record(m) for m in update["messages"]]
        return update

    async def formatter(state: AgentState):
        # Зберігаємо дослівний текст агента, зокрема його запитання про відсутні дані.
        answer = state["messages"][-1].text.strip()
        query = next(m.content for m in state["messages"] if isinstance(m, HumanMessage))
        payload = {"query": query, "answer": answer, "observations": log.observations}
        messages = [
            SystemMessage(content=(
                "Оціни від 0 до 1, наскільки відповідь або уточнення обґрунтовані запитом "
                "та спостереженнями. Поверни тільки поле confidence."
            )),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
        with log.node("formatter", [message_record(m) for m in messages]) as event:
            score = ConfidenceEstimate.model_validate(await assess_confidence.ainvoke(messages))
            final = FinalAnswer(answer=answer, confidence=score.confidence, sources=log.sources())
            event["output"] = final.model_dump()
        return {"final": final, "stop_reason": "completed"}

    def verify_quote(state: AgentState):
        note = HumanMessage(content=state["quote_issue"], name="quote_check")
        with log.node("verify_quote", message_record(state["messages"][-1])) as event:
            event["output"] = message_record(note)
        return {"messages": [note], "quote_issue": None}

    def route(state: AgentState) -> Literal["tools", "verify_quote", "formatter", "__end__"]:
        if state["stop_reason"]:
            return END
        if state["quote_issue"]:
            return "verify_quote"
        return "tools" if state["messages"][-1].tool_calls else "formatter"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools_node)
    graph.add_node("verify_quote", verify_quote)
    graph.add_node("formatter", formatter)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("tools", "agent")
    graph.add_edge("verify_quote", "agent")
    graph.add_edge("formatter", END)
    return graph.compile()


async def run_agent(query: str, decide, assess_confidence, *, run_id: str, backend: str, limits=Limits()):
    log = TrajectoryLogger()
    # Єдиний дедлайн на весь граф охоплює модель, інструменти й форматування.
    budget = asyncio.timeout(limits.timeout_seconds)
    try:
        async with budget:
            app = build_graph(decide, assess_confidence, limits, log)
            initial = {
                "messages": [SystemMessage(content=SYSTEM), HumanMessage(content=query)],
                "step_count": 0, "final": None, "stop_reason": None, "quote_issue": None,
            }
            state = await app.ainvoke(initial, config={"recursion_limit": 2 * limits.max_steps + 6})
    except TimeoutError as exc:
        if budget.expired():
            state = halt(log, "timeout", f"Вичерпано загальний час ({limits.timeout_seconds:g} с).")
        else:
            state = halt(log, "error", f"Інструмент або провайдер перевищив свій час очікування: {exc}")
    except Exception as exc:
        state = halt(log, "error", f"Помилка виконання ({type(exc).__name__}): {exc}")

    reason = state["stop_reason"]
    summary = {
        "run_id": run_id, "backend": backend, "query": query,
        "actual": state["final"].model_dump(), "stop_reason": reason,
        "steps": log.agent_steps, "tool_calls": log.observations,
        "elapsed_ms": round((monotonic() - log.started) * 1000, 2),
        "max_steps_reached": reason == "max_steps", "timed_out": reason == "timeout",
        "loop_detected": reason == "loop_detected",
    }
    trajectory = {
        "run_id": run_id, "backend": backend, "query": query, "limits": asdict(limits),
        "stop_reason": reason, "elapsed_ms": summary["elapsed_ms"],
        "agent_steps": log.agent_steps, "trajectory": log.events, "final": summary["actual"],
    }
    return summary, trajectory


async def main():
    parser = argparse.ArgumentParser(description="Планувальник матеріалів для заняття")
    parser.add_argument("query", help="Запит українською")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120.0, help="Загальний час у секундах")
    parser.add_argument("--log", type=Path, default=Path("trajectory_single.json"))
    args = parser.parse_args()
    result, trace = await run_agent(
        args.query, *make_model(args.model), run_id="single", backend=f"ollama/{args.model}",
        limits=Limits(max_steps=args.max_steps, timeout_seconds=args.timeout),
    )
    write_json(args.log, trace)
    print(json.dumps(result["actual"], ensure_ascii=False, indent=2))
    return 0 if result["stop_reason"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
