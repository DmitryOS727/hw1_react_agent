"""Запускає предметні кейси з Ollama та окремі контрольовані перевірки захисту."""

import argparse
import asyncio
import os
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from agent import DEFAULT_MODEL, make_model, run_agent
from logger import write_json
from safety import Limits
from tools import TOOLS


@dataclass
class Case:
    id: str
    complexity: str
    query: str
    expected: str
    calls: list
    numbers: tuple = ()
    needs_clarification: bool = False


CASES = [
    Case("TC-001", "simple",
         "Скільки аркушів в упаковці паперу з каталогу і яка ціна однієї упаковки в CAD?",
         "25 аркушів, 4.50 CAD за упаковку.",
         [("get_pack", {"item": "paper"})], ("25", "4.50")),
    Case("TC-002", "medium",
         "На заняття прийдуть 18 дітей. Кожній дитині потрібно по 2 клей-олівці. "
         "Запасу немає. Скільки цілих упаковок клею з каталогу купити?",
         "Потрібно 36 клей-олівців, тобто 6 упаковок по 6. Запитана величина — 6 упаковок.",
         [("get_pack", {"item": "glue"}),
          ("plan_packs", {"children": 18, "per_child": 2, "pack_size": 6, "on_hand": 0})], ("6",)),
    Case("TC-003", "complex",
         "Для колажу 24 дітям потрібно по 3 аркуші паперу. У шафі вже є 10 аркушів. "
         "За каталогом визнач, скільки упаковок купити, скільки аркушів залишиться після заняття "
         "і якою буде загальна вартість покупки в CAD.",
         "Купити 3 упаковки, після заняття залишиться 13 аркушів, вартість 13.50 CAD.",
         [("get_pack", {"item": "paper"}),
          ("plan_packs", {"children": 24, "per_child": 3, "pack_size": 25, "on_hand": 10}),
          ("order_cost", {"packs": 3, "price_cents": 450})], ("3", "13", "13.50")),
    Case("TC-004", "clarification",
         "Планую малювання. Кожній дитині потрібно по 2 фломастери. "
         "Скільки упаковок купити для моєї групи?",
         "Уточнити кількість дітей. До відповіді користувача розрахунок неможливий.",
         [], needs_clarification=True),
    Case("TC-005", "complex",
         "Для 12 дітей потрібно по 2 фломастери. Є 3 фломастери. Скільки упаковок купити і скільки це коштуватиме?",
         "Потрібно купити 3 упаковки, перевірена загальна вартість — 19.20 CAD.",
         [("get_pack", {"item": "markers"}),
          ("plan_packs", {"children": 12, "per_child": 2, "pack_size": 8, "on_hand": 3}),
          ("order_cost", {"packs": 3, "price_cents": 640})], ("3", "19.20")),
]


def amount_with_unit(text: str, number: int, noun: str, label: str) -> bool:
    # Обидва поширені записи: «3 упаковки» і «Упаковок до купівлі — 3».
    value = rf"(?<![\d.,]){number}(?!\d|[.,]\d)"
    forward = rf"{value}\s*(?:ціл\w*\s+)?(?:{noun})"
    backward = rf"(?:{label})\s*[:=–—-]?\s*{value}"
    return bool(re.search(forward, text) or re.search(backward, text))


def check_case(case: Case, result: dict):
    calls = result["tool_calls"]
    text = result["actual"]["answer"].lower().replace("*", "")
    checks = {"completed": result["stop_reason"] == "completed"}
    if case.needs_clarification:
        checks["no_premature_calculation"] = all(
            c["name"] == "get_pack" and c["status"] == "success" for c in calls
        )
        checks["asks_group_size"] = (
            bool(re.search(r"діт|дит|вихован|учн", text))
            and bool(re.search(r"скільки|кількіст|чисельніст|розмір груп", text))
            and ("?" in text or bool(re.search(r"уточн|вкаж|скаж|повідом|вказ|потрібно знати", text)))
        )
        return checks

    # Перевіряємо також аргументи. Пропущене значення on_hand має той самий ефект, що й 0.
    known = {t.name: t.args_schema for t in TOOLS}
    sequence = []
    for c in calls:
        try:
            arguments = known[c["name"]].model_validate(c["args"]).model_dump()
        except (KeyError, ValidationError):
            arguments = c["args"]
        sequence.append((c["name"], arguments))
    checks["tool_sequence_and_arguments"] = sequence == case.calls and all(c["status"] == "success" for c in calls)
    # Десяткову кому та крапку трактуємо однаково. Число 13 у 13.50 не рахуємо окремо.
    written = {Decimal(n.replace(",", ".")) for n in re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)", text)}
    checks["requested_numbers"] = all(Decimal(n) in written for n in case.numbers)
    if case.id in ("TC-002", "TC-003", "TC-005"):
        count = 6 if case.id == "TC-002" else 3
        checks["packs_in_answer"] = amount_with_unit(
            text, count, r"упак\w*|пач\w*|пакован\w*", r"упаков\w*(?: до (?:купівлі|закупівлі))?"
        )
    if case.id == "TC-003":
        checks["leftover_in_answer"] = amount_with_unit(
            text, 13, r"аркуш\w*|одиниц\w*|шт", r"залиш\w*(?: аркуш\w*| після заняття)?"
        )
    if case.id in ("TC-001", "TC-003", "TC-005"):
        checks["currency"] = bool(re.search(r"cad|канадськ|долар|\$", text))
    return checks


async def safety_demos():
    # Тестовий драйвер створює граничні умови. Граф, ToolNode, годинник і журнал — ті самі.
    async def repeat(messages, catalog_ready):
        index = 1 + sum(isinstance(m, AIMessage) for m in messages)
        call = {"name": "get_pack", "args": {"item": "paper"}, "id": f"demo-{index}"}
        return AIMessage(content="", tool_calls=[call])

    async def slow(messages, catalog_ready):
        await asyncio.sleep(2)
        return AIMessage(content="Відповідь після затримки.")

    def unexpected_formatter(_):
        raise AssertionError("Захисний сценарій мав завершитися до форматування.")

    demos = [
        ("G-01", "Лише одне рішення моделі. Перший виклик каталогу встигає виконатися.",
         repeat, Limits(max_steps=1), "max_steps", 1, 1),
        ("G-02", "Три запити того самого матеріалу поспіль. Третій виклик блокується.",
         repeat, Limits(), "loop_detected", 3, 2),
        ("G-03", "Модель чекає 2 с, загальний бюджет запуску 0.15 с.",
         slow, Limits(timeout_seconds=0.15), "timeout", 1, 0),
    ]
    results, traces = [], []
    for key, explanation, policy, limits, reason, steps, tool_count in demos:
        result, trace = await run_agent(
            explanation, policy, RunnableLambda(unexpected_formatter),
            limits=limits, run_id=key, backend="controlled_test_driver",
        )
        checks = {"reason": result["stop_reason"] == reason, "steps": result["steps"] == steps,
                  "completed_tool_calls": len(result["tool_calls"]) == tool_count}
        if reason == "timeout":
            checks["interrupted_in_flight"] = any(
                e["node_name"] == "agent" and e["output"] == {"interrupted": True} for e in trace["trajectory"]
            )
            checks["finished_before_2s"] = result["elapsed_ms"] < 2000
        results.append(dict(result, expected=reason, checks=checks, passed=all(checks.values())))
        traces.append(trace)
    return results, traces


async def main():
    parser = argparse.ArgumentParser(description="П’ять запитів до Ollama та три перевірки захисту.")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    backend = f"ollama/{args.model}"
    decide, assess_confidence = make_model(args.model)
    results, traces = [], []
    for case in CASES:
        print(f"{case.id}  {case.complexity} …", flush=True)
        result, trace = await run_agent(
            case.query, decide, assess_confidence, run_id=case.id, backend=backend,
        )
        checks = check_case(case, result)
        results.append(dict(
            result, expected=case.expected, complexity=case.complexity,
            checks=checks, passed=all(checks.values()),
        ))
        traces.append(trace)
        print(f"  {result['stop_reason']} | рішень моделі — {result['steps']} | {result['elapsed_ms']} мс | {checks}", flush=True)
        print(f"  {result['actual']['answer']}", flush=True)

    guards, guard_traces = await safety_demos()
    write_json(args.out_dir / "trajectory.json", {"domain_runs": traces, "safety_runs": guard_traces})
    write_json(args.out_dir / "test_results.json", {"domain_cases": results, "safety_demos": guards})
    passed = sum(r["passed"] for r in results)
    protected = sum(r["passed"] for r in guards)
    print(f"Предметні кейси: {passed}/{len(results)}. Перевірки захисту: {protected}/{len(guards)}.")
    return 0 if passed == len(results) and protected == len(guards) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
