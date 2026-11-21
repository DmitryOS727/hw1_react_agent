"""Регресійні тести фасування, ціни та збереження тексту уточнення."""

import unittest

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agent import run_agent


class AgentRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_calculators_require_observed_values(self):
        correct = {"children": 24, "per_child": 3, "pack_size": 25, "on_hand": 10}
        requests = [
            ("get_pack", {"item": "paper"}),
            ("plan_packs", correct | {"pack_size": 50}),
            ("plan_packs", correct),
            ("order_cost", {"packs": 3, "price_cents": 900}),
            ("order_cost", {"packs": 3, "price_cents": 450}),
        ]

        async def decide(messages, catalog_ready):
            i = sum(isinstance(m, AIMessage) for m in messages)
            self.assertEqual(catalog_ready, i > 0)
            if i == len(requests):
                return AIMessage(content="Три упаковки коштують 13,50 CAD.")
            name, args = requests[i]
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"check-{i}"}])

        scorer = RunnableLambda(lambda _: {"confidence": 1.0})
        result, _ = await run_agent(
            "24 дітям потрібно по 3 аркуші, 10 уже є.", decide, scorer,
            run_id="regression", backend="controlled_test_driver",
        )
        self.assertEqual(result["stop_reason"], "completed")
        self.assertEqual([c["status"] for c in result["tool_calls"]],
                         ["success", "error", "success", "error", "success"])
        self.assertEqual(result["tool_calls"][2]["result"]["leftover_units"], 13)
        self.assertEqual(result["tool_calls"][4]["result"]["total_cents"], 1350)

    async def test_formatting_preserves_clarifying_question(self):
        question = "Скільки дітей братимуть участь у занятті?"

        async def decide(messages, catalog_ready):
            return AIMessage(content=question)

        result, _ = await run_agent(
            "Скільки фломастерів потрібно моїй групі?", decide,
            RunnableLambda(lambda _: {"confidence": 0.9}), run_id="format", backend="controlled_test_driver",
        )
        self.assertEqual(result["actual"], {"answer": question, "confidence": 0.9, "sources": []})

    async def test_total_needs_a_cost_observation_and_a_matching_amount(self):
        def call(name, args, index):
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"quote-{index}"}])

        replies = [
            call("get_pack", {"item": "markers"}, 1),
            call("plan_packs", {"children": 12, "per_child": 2, "pack_size": 8, "on_hand": 3}, 2),
            AIMessage(content="3 упаковки коштують 19,20 CAD."),
            call("order_cost", {"packs": 3, "price_cents": 640}, 3),
            AIMessage(content="3 упаковки коштують 29,20 CAD."),
            AIMessage(content="3 упаковки коштують 19,20 CAD."),
        ]

        async def decide(messages, catalog_ready):
            return replies[sum(isinstance(m, AIMessage) for m in messages)]

        result, trace = await run_agent(
            "12 дітям по 2 фломастери, є 3. Яка загальна вартість?", decide,
            RunnableLambda(lambda _: {"confidence": 1.0}), run_id="price", backend="controlled_test_driver",
        )
        self.assertEqual(result["stop_reason"], "completed")
        self.assertEqual(result["actual"]["answer"], replies[-1].content)
        self.assertEqual(sum(e["node_name"] == "verify_quote" for e in trace["trajectory"]), 2)


if __name__ == "__main__":
    unittest.main()
