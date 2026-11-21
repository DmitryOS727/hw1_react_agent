"""Окремі перевірки інструментів: python -m unittest -v test_tools."""

import unittest

from pydantic import ValidationError

from tools import get_pack, order_cost, plan_packs


class ToolTests(unittest.TestCase):
    def test_catalog(self):
        self.assertEqual(get_pack.invoke({"item": " PAPER "}), {
            "item": "paper", "pack_size": 25, "price_cents": 450, "price_cad": "4.50",
        })

    def test_invalid_catalog(self):
        for args in ({"item": "watercolor"}, {"item": ""}, {"item": "paper", "store": "A"}):
            with self.subTest(args=args), self.assertRaises(ValidationError):
                get_pack.invoke(args)

    def test_pack_rounding(self):
        result = plan_packs.invoke({"children": 24, "per_child": 3, "pack_size": 25, "on_hand": 10})
        self.assertEqual(result, {"required_units": 72, "packs_to_buy": 3, "leftover_units": 13})

    def test_stock_and_exact_multiple(self):
        for stock, packs, left in [(0, 2, 0), (10, 0, 0), (13, 0, 3)]:
            with self.subTest(stock=stock):
                args = {"children": 5, "per_child": 2, "pack_size": 5, "on_hand": stock}
                result = plan_packs.invoke(args)
                self.assertEqual((result["packs_to_buy"], result["leftover_units"]), (packs, left))

    def test_invalid_quantity(self):
        valid = {"children": 5, "per_child": 2, "pack_size": 5}
        for change in ({"children": 0}, {"pack_size": 0}, {"on_hand": -1}, {"per_child": 2.5},
                       {"children": True}, {"children": "5"}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                plan_packs.invoke(valid | change)

    def test_cost(self):
        self.assertEqual(order_cost.invoke({"packs": 3, "price_cents": 450}),
                         {"total_cents": 1350, "total_cad": "13.50"})
        self.assertEqual(order_cost.invoke({"packs": 0, "price_cents": 450})["total_cad"], "0.00")

    def test_invalid_cost(self):
        for args in ({"packs": -1, "price_cents": 450}, {"packs": 1, "price_cents": 4.5},
                     {"packs": 1, "price_cents": -1}):
            with self.subTest(args=args), self.assertRaises(ValidationError):
                order_cost.invoke(args)


if __name__ == "__main__":
    unittest.main()
