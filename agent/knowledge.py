"""知识层加载。

设计原则：知识与代码分离。更换设备型号时只替换 knowledge/ 下的 5 个 JSON 文件，
不需要改动任何代码——这是「方案可复制到同行业其他客户」这一商业卖点的技术兑现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


@dataclass
class KnowledgeBase:
    root: Path = KNOWLEDGE_DIR

    def _load(self, name: str) -> dict[str, Any]:
        with (self.root / f"{name}.json").open(encoding="utf-8") as fh:
            return json.load(fh)

    @cached_property
    def device(self) -> dict[str, Any]:
        return self._load("device_params")

    @cached_property
    def alarms(self) -> dict[str, dict[str, Any]]:
        return {a["code"]: a for a in self._load("alarm_codes")["alarm_codes"]}

    @cached_property
    def safety_rules(self) -> list[dict[str, Any]]:
        return self._load("safety_rules")["safety_rules"]

    @cached_property
    def cases(self) -> list[dict[str, Any]]:
        """说明书 §4 的历史故障案例。厂商给的依据，只读，Agent 不得改写。"""
        return self._load("history_cases")["cases"]

    @cached_property
    def cases_learned(self) -> list[dict[str, Any]]:
        """工单闭环后沉淀的现场案例，单独成表以便审计与回滚。

        与 history_cases 分表存放是刻意的：说明书原文属于设备厂商，AI 改不了也不该改；
        现场经验属于客户，要能被单独查看、单独撤销。文件不存在时视为空表。
        """
        path = self.root / "cases_learned.json"
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            return json.load(fh).get("cases", [])

    @cached_property
    def all_cases(self) -> list[dict[str, Any]]:
        """说明书案例 + 现场沉淀案例。新案例一沉淀就参与后续诊断，这是数据飞轮。"""
        return [*self.cases, *self.cases_learned]

    @cached_property
    def maintenance(self) -> dict[str, Any]:
        return self._load("maintenance_records")

    @cached_property
    def work_orders(self) -> dict[str, dict[str, Any]]:
        return {r["work_order"]: r for r in self.maintenance["records"]}

    @cached_property
    def params_by_key(self) -> dict[str, dict[str, Any]]:
        return {p["key"]: p for p in self.device["normal_params"]}

    def cases_for(self, alarm_code: str | None) -> list[dict[str, Any]]:
        if not alarm_code:
            return []
        return [c for c in self.all_cases if c.get("alarm_code") == alarm_code]

    def work_orders_for(self, alarm_code: str | None) -> list[dict[str, Any]]:
        if not alarm_code:
            return []
        return [r for r in self.maintenance["records"] if r.get("alarm_code") == alarm_code]

    def is_covered(self, alarm_code: str | None) -> bool:
        """SAFE-05 的覆盖度判定：报警码是否存在于知识库。"""
        return bool(alarm_code and alarm_code in self.alarms)

    def source_tag(self, kind: str, ref: str = "") -> str:
        return f"{kind}:{ref}" if ref else kind
