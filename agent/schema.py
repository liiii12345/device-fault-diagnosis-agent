"""APX-240 故障诊断 Agent 的输出 schema：8 个字段。

现场决定能不能开工所依赖的 6 项一项不少——故障现象、可能原因、证据来源、排查顺序、需要补充的信息、
是否升级专家。另外两项是本方案自己加的：「已知事实」让每条信息都能回指现场原话，
「安全前置条件」把责任边界写死，两者都是防幻觉与免责的支点。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Confidence = Literal["强", "中", "弱"]

FIELD_LABELS: dict[str, str] = {
    "phenomenon": "故障现象",
    "known_facts": "已知事实",
    "possible_causes": "可能原因",
    "evidence": "证据来源",
    "safety_preconditions": "安全前置条件",
    "inspection_steps": "排查顺序",
    "missing_info": "需要补充的信息",
    "stop_and_escalate": "停止条件/升级",
}


@dataclass
class Fact:
    """已知事实。source 是强制字段——没有来源的信息不允许写进已知事实。"""

    content: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class Cause:
    """可能原因。永远是「可能」，不存在「已确认」这一档。"""

    cause: str
    confidence: Confidence
    basis: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class Step:
    """排查顺序中的一步。order 不可重排，authorization 标注授权边界。"""

    order: int
    action: str
    authorization: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Escalation:
    """停止条件/升级。"""

    must_stop: bool
    escalate_to_expert: bool
    expert_type: str
    stop_conditions: list[str] = field(default_factory=list)
    triggered_rules: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Diagnosis:
    """8 字段诊断结果 + 运行元数据。"""

    # --- 8 个输出字段 ---
    phenomenon: str
    known_facts: list[Fact] = field(default_factory=list)
    possible_causes: list[Cause] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    safety_preconditions: list[str] = field(default_factory=list)
    inspection_steps: list[Step] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    stop_and_escalate: Escalation = field(default_factory=lambda: Escalation(
        must_stop=False, escalate_to_expert=False, expert_type=""
    ))

    # --- 运行元数据（不属于 8 字段，用于仪表盘与审计） ---
    alarm_code: str | None = None
    coverage: Literal["full", "partial", "none"] = "none"
    safety_gates: list[str] = field(default_factory=list)
    refused_requests: list[str] = field(default_factory=list)
    degraded: bool = False
    model_used: str = "deterministic-core"
    warnings: list[str] = field(default_factory=list)
    # 由模型兜底层补齐的信号（只记字段名，不记数值——本字段也会被数字溯源自检扫到）
    parse_assisted: list[str] = field(default_factory=list)
    # 第二轮回填时，现场答了但解析层读不出可判定信号的待补项。⑦ 因此变空时必须靠它
    # 说清"不是都齐了，是这几项答了也没进证据链"，否则卡片会谎称观测项都已给出。
    unresolved_answers: list[str] = field(default_factory=list)
    # 由 agent.validate 回填；类型用 Any 以免 schema 反向依赖 validate 形成循环导入
    validation: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "故障现象": self.phenomenon,
            "已知事实": [f.to_dict() for f in self.known_facts],
            "可能原因": [c.to_dict() for c in self.possible_causes],
            "证据来源": self.evidence,
            "安全前置条件": self.safety_preconditions,
            "排查顺序": [s.to_dict() for s in self.inspection_steps],
            "需要补充的信息": self.missing_info,
            "停止条件/升级": self.stop_and_escalate.to_dict(),
            "_meta": {
                "报警码": self.alarm_code,
                "知识库覆盖度": self.coverage,
                "触发的安全红线": self.safety_gates,
                "已拒绝的违规请求": self.refused_requests,
                "降级运行": self.degraded,
                "推理引擎": self.model_used,
                "模型辅助解析": self.parse_assisted,
                "答复未进证据链的项": self.unresolved_answers,
                "警告": self.warnings,
                "输出自检": None if self.validation is None else {
                    "是否通过": self.validation.ok,
                    "检查项": [
                        {"项目": c.name, "通过": c.ok, "未通过原因": c.detail}
                        for c in self.validation.checks
                    ],
                },
            },
        }

    def render(self, verbose_check: bool = False) -> str:
        """渲染成现场工程师能直接读懂的文本，用于飞书群消息与演示。"""
        lines: list[str] = []
        lines.append(f"## 故障现象\n{self.phenomenon}")

        lines.append("\n## 已知事实")
        if self.known_facts:
            lines += [f"- {f.content} 〔来源：{f.source}〕" for f in self.known_facts]
        else:
            lines.append("- 无。现场描述中未包含可引用的确定信息。")

        lines.append("\n## 可能原因（按证据强弱排序，均为可能而非已确认）")
        if self.possible_causes:
            lines += [
                f"{i}. 可能为「{c.cause}」｜证据强度：{c.confidence}｜依据：{c.basis} 〔{c.source}〕"
                for i, c in enumerate(self.possible_causes, 1)
            ]
        else:
            lines.append("- 知识库未覆盖，无法给出有证据的可能原因。")

        lines.append("\n## 证据来源")
        lines += [f"- {e}" for e in self.evidence] if self.evidence else ["- 无"]

        lines.append("\n## 安全前置条件")
        lines += [f"- {s}" for s in self.safety_preconditions] if self.safety_preconditions else ["- 无"]

        lines.append("\n## 排查顺序")
        if self.inspection_steps:
            lines += [
                f"{s.order}. {s.action}"
                + (f"　【仅限：{s.authorization}】" if s.authorization and s.authorization != "现场人员可执行" else "")
                + f" 〔{s.source}〕"
                for s in self.inspection_steps
            ]
        else:
            lines.append("- 不执行排查。已触发停止条件。")

        lines.append("\n## 需要补充的信息")
        lines += [f"- {m}" for m in self.missing_info] if self.missing_info else ["- 无"]

        esc = self.stop_and_escalate
        lines.append("\n## 停止条件 / 升级")
        lines.append(f"- 是否必须停止排查：{'是' if esc.must_stop else '否'}")
        lines.append(f"- 是否升级专家：{'是' if esc.escalate_to_expert else '否'}")
        if esc.escalate_to_expert and esc.expert_type:
            lines.append(f"- 升级对象：{esc.expert_type}")
        if esc.triggered_rules:
            lines.append(f"- 触发红线：{'、'.join(esc.triggered_rules)}")
        if esc.reason:
            lines.append(f"- 判定理由：{esc.reason}")
        lines += [f"- 停止条件：{c}" for c in esc.stop_conditions]

        if self.refused_requests:
            lines.append("\n## 已拒绝的请求")
            lines += [f"- {r}" for r in self.refused_requests]

        if self.warnings:
            lines.append("\n## 提示")
            lines += [f"- {w}" for w in self.warnings]

        if self.validation is not None:
            if self.validation.ok and not verbose_check:
                lines.append(f"\n---\n本次输出已通过 {len(self.validation.checks)} 项自检"
                             f"（来源可追溯、无断言措辞、无违规建议、数字全部可溯源、排查顺序与说明书一致）。")
            else:
                lines.append("\n---\n" + self.validation.render())

        return "\n".join(lines)
