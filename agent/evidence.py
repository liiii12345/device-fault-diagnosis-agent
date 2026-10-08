"""证据比对：历史案例与维修记录的三态/四态匹配。

说明书 §4 原文约束：「历史案例不是当前故障的结论；现场读数或安全风险不一致时，
必须以当前证据和报警码顺序为准。」§5 原文约束：「维修记录只可作为候选证据，
不能因为『以前修过』而直接断言当前根因相同。」

因此案例匹配不能是相似度打分，必须是逐信号的可反驳比对：
  refuted     —— 决定性信号被现场读数反驳，禁止引用该案例根因，必须解释差异
  differs     —— 非决定性读数与案例不同，根因降为弱，必须先解释差异
  unverified  —— 决定性信号缺失，案例只能作为候选假设，不得写成结论
  strong_match—— 全部信号一致，可作为证据强度最高的「可能」原因（仍非已确认）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .parse import Signals

MatchState = Literal["strong_match", "differs", "unverified", "refuted"]

# 各信号的容差。温度读数以 °C 计，压力以 MPa 计。
TOLERANCE = {"setpoint_temp": 0.5, "actual_temp": 0.5, "upstream_pressure": 0.005, "device_pressure": 0.005}

SIGNAL_LABELS = {
    "material_present": "物料是否到位",
    "p1_light": "P1 指示灯状态",
    "setpoint_temp": "设定温度",
    "actual_temp": "实际温度",
    "heating_current_state": "加热电流状态",
    "upstream_pressure": "上游压力",
    "device_pressure": "设备端压力",
    "leak_sound": "漏气声",
    "door_closed": "门体是否关严",
    "alarm_intermittent": "报警是否间歇出现",
    "friction_sound": "摩擦声",
    "friction_periodic": "摩擦声是否周期性",
    "preheated": "是否已预热",
}


UNITS = {
    "setpoint_temp": "°C",
    "actual_temp": "°C",
    "heating_current": " A",
    "upstream_pressure": " MPa",
    "device_pressure": " MPa",
}


def _fmt(value: Any, key: str = "") -> str:
    """读数格式化。带上 key 时补单位——现场工程师看「165」不如看「165°C」。"""
    if value is None:
        return "未提供"
    if isinstance(value, bool):
        return "是" if value else "否"
    if value == "zero":
        return "0 A（无电流）"
    if value == "normal":
        return "有正常读数"
    if value == "on":
        return "亮"
    if value == "off":
        return "不亮"
    if isinstance(value, (int, float)):
        return f"{value:g}{UNITS.get(key, '')}"
    return str(value)


def _values_equal(key: str, expected: Any, actual: Any) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(expected, bool):
        return abs(float(expected) - float(actual)) <= TOLERANCE.get(key, 0.001)
    return expected == actual


@dataclass
class CaseMatch:
    case_id: str
    alarm_code: str
    state: MatchState
    root_cause: str
    disposition: str
    authorization: str
    critical_signal: str | None
    confirmed: list[tuple[str, Any, Any]] = field(default_factory=list)
    differs: list[tuple[str, Any, Any]] = field(default_factory=list)
    unknown: list[tuple[str, Any]] = field(default_factory=list)
    explanation: str = ""
    # 出处：说明书 §4 为 MANUAL:§4:Hxx，现场沉淀案例为 KB:LEARNED:CAND-xxx
    source_ref: str = ""
    learned: bool = False
    # 报修人的现场署名。双人确认闸门上线前沉淀的案例这一项是空的，
    # 依据栏要如实说清它只过了专家一个人，不能含混成"两人都签了"。
    field_confirmed_by: str = ""

    @property
    def usable_as_cause(self) -> bool:
        """refuted 状态下禁止把该案例根因写进可能原因。"""
        return self.state != "refuted"

    @property
    def confidence(self) -> str:
        return {"strong_match": "强", "differs": "弱", "unverified": "弱"}.get(self.state, "弱")


def match_case(case: dict[str, Any], sig: Signals) -> CaseMatch:
    critical = case.get("critical_signal")
    confirmed: list[tuple[str, Any, Any]] = []
    differs: list[tuple[str, Any, Any]] = []
    unknown: list[tuple[str, Any]] = []
    critical_state = "unknown"

    for key, expected in case.get("match_signals", {}).items():
        actual = getattr(sig, key, None)
        if actual is None:
            unknown.append((key, expected))
            if key == critical:
                critical_state = "unknown"
        elif _values_equal(key, expected, actual):
            confirmed.append((key, expected, actual))
            if key == critical:
                critical_state = "confirmed"
        else:
            differs.append((key, expected, actual))
            if key == critical:
                critical_state = "refuted"

    critical_differs = any(k == critical for k, _, _ in differs)
    noncritical_differs = any(k != critical for k, _, _ in differs)

    if critical_state == "refuted":
        state: MatchState = "refuted"
    elif noncritical_differs or critical_differs:
        state = "differs"
    elif unknown:
        state = "unverified"
    else:
        state = "strong_match"

    m = CaseMatch(
        case_id=case["id"],
        alarm_code=case.get("alarm_code", ""),
        state=state,
        root_cause=case["root_cause"],
        disposition=case["disposition"],
        authorization=case.get("authorization", ""),
        critical_signal=critical,
        confirmed=confirmed,
        differs=differs,
        unknown=unknown,
        source_ref=case.get("source_ref") or f"MANUAL:§4:{case['id']}",
        learned=bool(case.get("learned")),
        field_confirmed_by=str(case.get("field_confirmed_by") or "").strip(),
    )
    m.explanation = _explain(m)
    return m


def _explain(m: CaseMatch) -> str:
    """生成「当前证据与历史案例的相同点和差异」——说明书 §5 明确要求。"""
    parts: list[str] = []
    if m.learned:
        parts.append(f"{m.case_id} 为本厂工单闭环后沉淀的现场案例，非说明书 §4 原文，证据分量按厂内经验对待")
    if m.confirmed:
        same = "、".join(f"{SIGNAL_LABELS.get(k, k)}＝{_fmt(a, k)}" for k, _, a in m.confirmed)
        parts.append(f"与 {m.case_id} 相同点：{same}")

    if m.differs:
        diff = "；".join(
            f"{SIGNAL_LABELS.get(k, k)}：案例为 {_fmt(e, k)}，当前为 {_fmt(a, k)}" for k, e, a in m.differs
        )
        parts.append(f"差异：{diff}")

    if m.unknown:
        miss = "、".join(SIGNAL_LABELS.get(k, k) for k, _ in m.unknown)
        parts.append(f"当前未提供：{miss}")

    if m.state == "refuted":
        key = SIGNAL_LABELS.get(m.critical_signal or "", m.critical_signal)
        parts.append(
            f"{m.case_id} 的决定性证据是{key}，当前现场读数与之矛盾，"
            f"因此不得引用 {m.case_id} 的根因「{m.root_cause}」，须按报警码顺序重新排查。"
        )
    elif m.state == "differs":
        parts.append(
            f"现场读数与 {m.case_id} 不一致，按说明书要求应优先解释差异并按报警码顺序排查，"
            f"「{m.root_cause}」只能作为弱证据的候选原因。"
        )
    elif m.state == "unverified":
        parts.append(
            f"{m.case_id} 的决定性证据当前缺失，该案例只能作为候选假设，"
            f"不得写成结论；需补齐后才能判断是否同因。"
        )
    else:
        parts.append(
            f"{m.case_id} 的全部特征与当前现场一致，可作为证据强度最高的候选原因，"
            f"但历史案例不是当前故障的结论，仍需按报警码顺序验证。"
        )
    return " ".join(parts)


@dataclass
class WorkOrderNote:
    work_order: str
    date: str
    confirmed_cause: str
    work_done: str
    verification: str
    text: str


def note_work_orders(sig: Signals, kb) -> list[WorkOrderNote]:
    """处理现场描述中援引的维修记录，强制附带「不得因以前修过就断言同因」的约束。"""
    notes: list[WorkOrderNote] = []
    for wo_id in sig.mentioned_work_orders:
        rec = kb.work_orders.get(wo_id)
        if not rec:
            continue
        missing = [
            SIGNAL_LABELS.get(k, k)
            for k in _observations_needed_for(kb, sig.alarm_code)
            if getattr(sig, _obs_to_signal(k), None) is None
        ]
        if missing:
            verdict = f"当前「{'、'.join(missing)}」未提供，无法判断是否与当时同因。"
        else:
            verdict = "需将当前读数与当时确认状态逐项比对后才能判断是否同因。"
        text = (
            f"{wo_id}（{rec['date']}）当时已确认原因为「{rec['confirmed_cause']}」，"
            f"执行了「{rec['work_done']}」，复机验证「{rec['verification']}」。"
            f"但维修记录只代表设备当时的已确认状态，不能替代当前诊断，"
            f"不得因为「以前修过」而直接断言当前根因相同。{verdict}"
        )
        notes.append(WorkOrderNote(
            work_order=wo_id, date=rec["date"], confirmed_cause=rec["confirmed_cause"],
            work_done=rec["work_done"], verification=rec["verification"], text=text,
        ))

    # 未被现场提及、但与当前报警码同类的历史工单，只作为候选证据列出
    for rec in kb.work_orders_for(sig.alarm_code):
        if rec["work_order"] in sig.mentioned_work_orders:
            continue
        notes.append(WorkOrderNote(
            work_order=rec["work_order"], date=rec["date"], confirmed_cause=rec["confirmed_cause"],
            work_done=rec["work_done"], verification=rec["verification"],
            text=(
                f"{rec['work_order']}（{rec['date']}）曾就同类报警确认原因为「{rec['confirmed_cause']}」，"
                f"执行「{rec['work_done']}」。仅作为候选证据，不能据此断言当前根因相同。"
            ),
        ))
    return notes


# 说明书观测项 → 解析层信号名。没登记在这里的观测项永远算「缺失」，
# 现场明明答了也会被反复追问——新增观测项时必须同步补这张表。
_OBS_TO_SIGNAL = {
    "设定温度": "setpoint_temp",
    "实际温度": "actual_temp",
    "加热电流": "heating_current_state",
    "是否已完成预热": "preheated",
    "上游压力": "upstream_pressure",
    "设备端压力": "device_pressure",
    "是否听到持续漏气声": "leak_sound",
    "物料是否到位": "material_present",
    "P1 指示灯状态": "p1_light",
    "门体是否已关严": "door_closed",
    "门内是否有异物": "door_obstructed",
    "报警是否间歇出现": "alarm_intermittent",
    "是否有可见卡阻": "visible_jam",
    "输送带是否打滑": "belt_slipping",
    "输送段是否有摩擦声及其周期性": "friction_periodic",
}


def _obs_to_signal(obs: str) -> str:
    return _OBS_TO_SIGNAL.get(obs, "")


def _observations_needed_for(kb, alarm_code: str | None) -> list[str]:
    if not alarm_code or alarm_code not in kb.alarms:
        return []
    return kb.alarms[alarm_code]["derived"].get("required_observations", [])
