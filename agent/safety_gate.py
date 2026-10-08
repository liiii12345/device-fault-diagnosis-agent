"""安全门控层。

这一层的输出是硬约束，不允许被语言模型覆盖。设计上它先于任何自由生成执行：
一旦命中 SAFE-01 / A205 一类的立即停机条件，诊断链路会短路到「停机 + 升级」，
不再让模型去"权衡"。这是回答客户「AI 乱指挥出了安全事故谁担责」这一签约异议的
技术兑现——责任边界写死在代码里，不在 prompt 里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .knowledge import KnowledgeBase
from .parse import Signals


@dataclass
class GateResult:
    triggered: list[str] = field(default_factory=list)
    mandatory_actions: list[str] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    escalate_to_expert: bool = False
    expert_types: list[str] = field(default_factory=list)
    forbid_continue_run: bool = False
    must_stop: bool = False
    severity: str = "中"
    notes: list[str] = field(default_factory=list)

    def add_action(self, action: str) -> None:
        if action not in self.mandatory_actions:
            self.mandatory_actions.append(action)

    def add_precondition(self, pre: str) -> None:
        if pre not in self.preconditions:
            self.preconditions.append(pre)

    def add_rule(self, rule_id: str) -> None:
        if rule_id not in self.triggered:
            self.triggered.append(rule_id)

    def escalate(self, expert_type: str) -> None:
        self.escalate_to_expert = True
        if expert_type and expert_type not in self.expert_types:
            self.expert_types.append(expert_type)


def _rule(kb: KnowledgeBase, rule_id: str) -> dict[str, Any]:
    for r in kb.safety_rules:
        if r["id"] == rule_id:
            return r
    return {}


def evaluate(sig: Signals, kb: KnowledgeBase) -> GateResult:
    gate = GateResult()
    alarm = kb.alarms.get(sig.alarm_code) if sig.alarm_code else None
    derived = (alarm or {}).get("derived", {})

    # --- SAFE-01：立即停机信号 ---
    r = _rule(kb, "SAFE-01")
    hit = [s for s in r.get("signals", []) if getattr(sig, s, False) is True]
    if hit:
        gate.add_rule("SAFE-01")
        for a in r.get("mandatory_actions", []):
            gate.add_action(a)
        gate.escalate("专家")
        gate.forbid_continue_run = True
        gate.must_stop = True
        gate.severity = "高"
        labels = "、".join(f"「{sig.spans.get(h, h)}」" for h in hit)
        gate.notes.append(f"现场描述出现 SAFE-01 列举的立即停机征兆：{labels}。")

    # --- SAFE-03：违规请求，硬拒绝 ---
    if sig.user_request:
        r = _rule(kb, "SAFE-03")
        gate.add_rule("SAFE-03")
        gate.refusals.append(
            f"拒绝执行「{sig.user_request.strip('？?。， ')}」。"
            f"该操作被 SAFE-03 明确禁止：{r.get('requirement', '')}"
        )
        for a in r.get("mandatory_actions", []):
            gate.add_action(a)
        gate.escalate("授权电气/机械维修人员")
        gate.must_stop = True
        gate.severity = "高"

    # --- 报警码自带的强制属性 ---
    if derived:
        if derived.get("forbid_continue_run"):
            gate.forbid_continue_run = True
            gate.must_stop = True
        if derived.get("escalate_to_expert"):
            gate.escalate(derived.get("authorization", "专家"))
        if derived.get("severity") == "高":
            gate.severity = "高"
        if derived.get("requires_power_isolation"):
            gate.add_precondition("隔离电源")
        if derived.get("requires_cooldown"):
            gate.add_precondition("等待冷却至安全温度后再接近")
        if derived.get("requires_depressurization"):
            gate.add_precondition("停机泄压，确认管路无残余压力")
        if derived.get("forbid_bypass_interlock"):
            gate.add_precondition("禁止短接或屏蔽安全门联锁（SAFE-03）")
        if derived.get("requires_lockout"):
            gate.add_rule("SAFE-02")
            gate.add_precondition("断电、锁定、挂牌，并确认残余能量释放后方可开罩或接触部件")
        if derived.get("authorization"):
            gate.add_rule("SAFE-04")
            gate.add_precondition(f"涉及线路/加热回路/伺服/气路拆装的环节仅限授权人员执行：{derived['authorization']}")

    # --- 继续运行请求 与 禁止试运行 对撞 ---
    if sig.wants_continue_run and gate.forbid_continue_run:
        gate.refusals.append(
            f"拒绝「{sig.spans.get('wants_continue_run', '继续运行')}」这一请求。"
            + (
                f"{sig.alarm_code} 的说明书安全排查顺序明确规定「禁止继续试运行」。"
                if alarm and "禁止继续试运行" in " ".join(alarm.get("steps", []))
                else "当前条件已触发立即停机要求，继续运行会扩大风险。"
            )
        )
        gate.must_stop = True
        gate.severity = "高"

    # --- A401 专项：安全门联锁不得绕过 ---
    if sig.alarm_code == "A401" and sig.door_closed is True:
        gate.add_precondition("门体已关严仍报警时不得反复强行复位或绕过联锁，应停机由授权人员检查")

    return gate


def apply_safe05(gate: GateResult, reason: str) -> None:
    """SAFE-05 由覆盖度与证据一致性计算触发，不在关键词层判定。"""
    gate.add_rule("SAFE-05")
    for a in _SAFE05_ACTIONS:
        gate.add_action(a)
    gate.escalate("专家")
    gate.must_stop = True
    gate.forbid_continue_run = True
    gate.severity = "高"
    gate.notes.append(reason)


_SAFE05_ACTIONS = ["停止进一步排查", "升级专家", "列出需要补充的信息"]
