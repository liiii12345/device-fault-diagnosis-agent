"""运行指标：把工单与用量流水汇成客户能看懂的一张表。

指标刻意分成两半，因为它们回答的是两个不同人的问题：

  客户设备主管关心 —— 报警数、平均处置时长、专家升级率、Top 故障模式
  飞书商业化关心   —— AI 调用量、知识库覆盖率、沉淀案例数（可复制、可扩容的证据）

覆盖率与沉淀案例数合起来就是"这个方案会自己长大"的量化证明：
每闭环一单，知识库多一条经验，下一次同类故障的 AI 判断就更准、人工介入更少。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .client import Client
from .workflow import CAND_PENDING, Paths

_TS = "%Y-%m-%d %H:%M:%S"


@dataclass
class Metrics:
    work_orders: int = 0
    ai_calls: int = 0
    escalated: int = 0
    closed: int = 0
    selfcheck_failed: int = 0
    coverage: Counter = field(default_factory=Counter)
    alarms: Counter = field(default_factory=Counter)
    learned_cases: int = 0
    pending_candidates: int = 0
    avg_resolve_minutes: float | None = None
    top_causes: Counter = field(default_factory=Counter)

    @property
    def escalate_rate(self) -> float:
        return self.escalated / self.work_orders if self.work_orders else 0.0

    @property
    def covered_rate(self) -> float:
        hit = self.coverage.get("full", 0) + self.coverage.get("partial", 0)
        return hit / self.work_orders if self.work_orders else 0.0

    def render(self) -> str:
        pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
        lines = [
            "## APX-240 诊断 Agent 运行看板",
            "",
            "**处置效率**",
            f"- 工单总数：{self.work_orders}",
            f"- 已闭环：{self.closed}"
            + (f"（平均处置时长 {'<1 分钟' if self.avg_resolve_minutes < 1 else f'{self.avg_resolve_minutes:.0f} 分钟'}）"
               if self.avg_resolve_minutes is not None else ""),
            f"- 专家升级率：{pct(self.escalate_rate)}（{self.escalated}/{self.work_orders}）",
            f"- 输出自检未通过而失效关闭：{self.selfcheck_failed} 单",
            "",
            "**AI 用量与知识覆盖**",
            f"- AI 调用量：{self.ai_calls} 次",
            f"- 知识库覆盖率：{pct(self.covered_rate)}"
            f"（full {self.coverage.get('full', 0)} / partial {self.coverage.get('partial', 0)}"
            f" / none {self.coverage.get('none', 0)}）",
            f"- 已沉淀入库案例：{self.learned_cases} 条　待现场确认：{self.pending_candidates} 条",
        ]
        if self.alarms:
            lines += ["", "**Top 报警码**"]
            lines += [f"- {code or '（无报警码）'}：{n} 单" for code, n in self.alarms.most_common(5)]
        if self.top_causes:
            lines += ["", "**Top 已确认根因**"]
            lines += [f"- {c}：{n} 单" for c, n in self.top_causes.most_common(5)]
        return "\n".join(lines)


def _minutes(a: str, b: str) -> float | None:
    try:
        return (time.mktime(time.strptime(b, _TS)) - time.mktime(time.strptime(a, _TS))) / 60
    except (ValueError, TypeError):
        return None


def collect(client: Client | None = None, paths: Paths | None = None) -> Metrics:
    client = client or Client()
    paths = paths or Paths()
    m = Metrics()

    for wo in client.work_orders().values():
        m.work_orders += 1
        m.ai_calls += int(wo.get("AI调用次数") or 0)
        if wo.get("是否升级专家") == "是":
            m.escalated += 1
        if wo.get("输出自检") == "未通过":
            m.selfcheck_failed += 1
        m.coverage[wo.get("知识库覆盖度") or "none"] += 1
        m.alarms[wo.get("报警码") or ""] += 1
        if wo.get("闭环时间"):
            m.closed += 1
            if wo.get("实际根因"):
                m.top_causes[wo["实际根因"]] += 1

    spans = [d for d in (_minutes(wo.get("报修时间", ""), wo.get("闭环时间", ""))
                         for wo in client.work_orders().values() if wo.get("闭环时间"))
             if d is not None and d >= 0]
    if spans:
        m.avg_resolve_minutes = sum(spans) / len(spans)

    if paths.usage.exists():
        # 用量流水里也含未建单的即席诊断（run.py diagnose），一并计入 AI 调用量
        extra = sum(int(r.get("calls") or 0)
                    for r in _jsonl(paths.usage) if not r.get("wo_id", "").startswith("RX-"))
        m.ai_calls += extra

    if paths.learned.exists():
        m.learned_cases = len(json.loads(paths.learned.read_text(encoding="utf-8")).get("cases", []))
    if paths.candidates.exists():
        m.pending_candidates = sum(
            1 for c in json.loads(paths.candidates.read_text(encoding="utf-8")).get("candidates", [])
            if c.get("status") == CAND_PENDING)
    return m


def _jsonl(path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out
