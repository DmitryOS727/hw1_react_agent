"""Обмеження одного запуску та детектор послідовних однакових викликів."""

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    max_steps: int = 10
    timeout_seconds: float = 120.0
    max_repeats: int = 3

    def __post_init__(self):
        if self.max_steps < 1 or self.max_repeats < 1:
            raise ValueError("Ліміти кроків і повторів мають бути додатними.")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Тайм-аут має бути скінченним додатним числом.")


class LoopDetector:
    def __init__(self, max_repeats: int):
        self.max_repeats = max_repeats
        self.previous = None
        self.count = 0

    def repeated(self, call: dict) -> bool:
        # Різні ID та порядок ключів JSON не змінюють змісту запиту.
        signature = (call["name"], json.dumps(call["args"], sort_keys=True, ensure_ascii=False))
        self.count = self.count + 1 if signature == self.previous else 1
        self.previous = signature
        return self.count >= self.max_repeats
