"""Три інструменти для планування закупівлі матеріалів на заняття."""

from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Демонстраційний каталог: кількість одиниць в упаковці та ціна в центах CAD.
CATALOG = {"paper": (25, 450), "glue": (6, 720), "markers": (8, 640)}


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CatalogInput(ToolInput):
    item: Literal["paper", "glue", "markers"] = Field(
        description="Код матеріалу: paper — папір, glue — клей-олівці, markers — фломастери."
    )

    @field_validator("item", mode="before")
    @classmethod
    def known_item(cls, value):
        if isinstance(value, str):
            value = value.strip().lower()
            if value in CATALOG:
                return value
        raise ValueError("Оберіть один із кодів: paper, glue, markers.")


class PacksInput(ToolInput):
    children: int = Field(ge=1, le=1000, description="Кількість дітей у групі.")
    per_child: int = Field(ge=1, le=1000, description="Скільки одиниць потрібно одній дитині.")
    pack_size: int = Field(ge=1, le=10000, description="Кількість одиниць в одній упаковці.")
    on_hand: int = Field(default=0, ge=0, le=1000000, description="Наявний запас в одиницях.")

    @field_validator("children", "per_child", "pack_size", "on_hand", mode="before")
    @classmethod
    def whole_units(cls, value):
        if type(value) is not int:
            raise ValueError("Кількість має бути цілим числом.")
        return value


class CostInput(ToolInput):
    packs: int = Field(ge=0, le=1000000, description="Кількість упаковок до купівлі.")
    price_cents: int = Field(ge=0, le=1000000, description="Ціна однієї упаковки в центах CAD.")

    @field_validator("packs", "price_cents", mode="before")
    @classmethod
    def nonnegative_integer(cls, value):
        if type(value) is not int or value < 0:
            raise ValueError("Кількість упаковок і ціна в центах мають бути невід’ємними цілими.")
        return value


def cad(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


@tool(args_schema=CatalogInput)
def get_pack(item: str) -> dict:
    """Знайти фасування та ціну однієї упаковки у навчальному каталозі.

    Викликай для запитів про папір (paper), клей-олівці (glue) та фломастери
    (markers), коли фасування або ціну користувач не надав. Повертає pack_size,
    price_cents і price_cad. Ціна стосується цілої упаковки.
    """
    size, price = CATALOG[item]
    return {"item": item, "pack_size": size, "price_cents": price, "price_cad": cad(price)}


@tool(args_schema=PacksInput)
def plan_packs(children: int, per_child: int, pack_size: int, on_hand: int = 0) -> dict:
    """Порахувати, скільки цілих упаковок купити і скільки одиниць залишиться.

    Передай кількість дітей, норму на дитину, відоме фасування та наявний запас.
    Якщо кількість дітей або норма невідомі, запитай користувача. Результат
    packs_to_buy можна передати в order_cost, щоб отримати загальну вартість.
    """
    required = children * per_child
    missing = max(0, required - on_hand)
    packs = (missing + pack_size - 1) // pack_size
    return {
        "required_units": required,
        "packs_to_buy": packs,
        "leftover_units": on_hand + packs * pack_size - required,
    }


@tool(args_schema=CostInput)
def order_cost(packs: int, price_cents: int) -> dict:
    """Розрахувати підсумок за упаковки одного матеріалу в CAD.

    Передай кількість упаковок і ціну однієї упаковки саме в центах.
    Повертає total_cents та total_cad. Нуль упаковок коштує 0.00 CAD.
    """
    total = packs * price_cents
    return {"total_cents": total, "total_cad": cad(total)}


TOOLS = [get_pack, plan_packs, order_cost]
