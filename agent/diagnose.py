"""诊断主链路：把现场描述组装成 8 字段诊断输出。

链路顺序（全部确定性，语言模型不参与事实判定）：
  信号解析 → 安全门控 → 覆盖度判定 → 参数越界比对 → 证据比对
  → 可能原因排序 → 缺失信息推导 → 8 字段组装 → 输出后校验

安全门控先于一切生成：命中立即停机条件时短路，不给模型"权衡"的机会。
输出后校验晚于一切生成：自检不通过即失效关闭——撤回可能原因与排查顺序、
强制升级专家，而不是把可疑结论发给现场。两道闸门都在代码里，不在 prompt 里。
"""

from __future__ import annotations

from typing import Any

from .evidence import _OBS_TO_SIGNAL, SIGNAL_LABELS, _fmt, match_case, note_work_orders
from .knowledge import KnowledgeBase
from .llm_parse import Caller, augment
from .parse import Signals, param_deviation, parse
from .safety_gate import GateResult, apply_safe05, evaluate
from .schema import Cause, Diagnosis, Escalation, Fact, Step
from .validate import enforce, validate

STRENGTH_ORDER = {"强": 0, "中": 1, "弱": 2}


def _eval_condition(cond: dict[str, Any], sig: Signals, kb: KnowledgeBase) -> bool | None:
    """三态条件求值：True 成立 / False 不成立 / None 信号缺失无法判定。"""
    op = cond["op"]
    actual = getattr(sig, cond.get("signal", ""), None)

    if op == "unknown":
        return actual is None
    if op in ("below_param", "above_param"):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return None
        param = kb.params_by_key.get(cond.get("param_key", ""), {})
        lo, hi = param.get("min"), param.get("max")
        if op == "below_param":
            return None if lo is None else actual < lo
        return None if hi is None else actual > hi
    if actual is None:
        return None
    if op == "eq":
        return actual == cond.get("value")
    if op == "ne":
        return actual != cond.get("value")
    if op == "true":
        return actual is True
    if op == "false":
        return actual is False
    return None


def _obs_missing(obs: str, sig: Signals) -> bool:
    key = _OBS_TO_SIGNAL.get(obs, "")
    return True if not key else getattr(sig, key, None) is None


# 信号 → 现场描述里的中文说法，用于「故障现象」复述与「已知事实」
_SIGNAL_PHRASE = {
    "alarm_code": "报警码",
    "occurrence_time": "发生时间",
    "setpoint_temp": "热封设定温度",
    "actual_temp": "热封实际温度",
    "heating_current": "加热电流",
    "heating_current_state": "加热电流状态",
    "preheated": "是否已预热",
    "upstream_pressure": "上游气压",
    "device_pressure": "设备端气压",
    "leak_sound": "漏气声",
    "material_present": "物料是否到位",
    "p1_light": "P1 指示灯",
    "belt_slipping": "输送带打滑",
    "door_closed": "安全门是否关严",
    "door_obstructed": "门内异物",
    "alarm_intermittent": "报警是否间歇",
    "friction_sound": "摩擦声",
    "friction_periodic": "摩擦声周期性",
    "visible_jam": "可见卡阻",
    "smoke": "烟雾",
    "burning_smell": "焦味",
    "abnormal_high_temp": "异常高温",
    "violent_vibration": "剧烈振动",
    "metal_friction_sound": "金属摩擦声",
    "part_loose": "部件松脱",
    "discoloration": "物料变色",
}

_PRESENCE_ONLY = {
    "smoke", "burning_smell", "abnormal_high_temp", "violent_vibration",
    "metal_friction_sound", "part_loose", "discoloration", "belt_slipping",
    "door_obstructed", "visible_jam", "leak_sound", "friction_sound", "friction_periodic",
}


def _build_phenomenon(sig: Signals, kb: KnowledgeBase, alarm: dict[str, Any] | None) -> str:
    """复述报警、现场读数、时间点和可见异常——只复述现场描述里确实存在的内容。"""
    parts: list[str] = []
    if sig.alarm_code and alarm:
        parts.append(f"设备报出 {sig.alarm_code}（{alarm['meaning']}）")
    elif sig.alarm_code:
        parts.append(f"设备报出 {sig.alarm_code}，但该报警码不在本设备知识库覆盖范围内")
    else:
        parts.append("现场未报告任何报警码")

    if sig.occurrence_time:
        parts.append(f"发生时间 {sig.occurrence_time}")

    readings = []
    for key in ("setpoint_temp", "actual_temp", "heating_current", "upstream_pressure", "device_pressure"):
        val = getattr(sig, key, None)
        if val is not None:
            readings.append(f"{_SIGNAL_PHRASE[key]} {_fmt(val, key)}")
    if readings:
        parts.append("现场读数：" + "、".join(readings))

    observed = []
    for key in sorted(_PRESENCE_ONLY):
        if getattr(sig, key, False) is True:
            observed.append(f"{_SIGNAL_PHRASE[key]}（原文「{sig.spans.get(key, '')}」）")
    for key in ("preheated", "material_present", "p1_light", "door_closed", "alarm_intermittent",
                "heating_current_state"):
        val = getattr(sig, key, None)
        if val is not None:
            observed.append(f"{_SIGNAL_PHRASE[key]}：{_fmt(val, key)}")
    if observed:
        parts.append("可见/可听异常与状态：" + "；".join(observed))

    if sig.mentioned_work_orders:
        parts.append("现场援引了历史维修记录：" + "、".join(sig.mentioned_work_orders))
    if sig.user_request:
        parts.append(f"现场提出的请求：「{sig.user_request.strip('？?。， ')}」")
    if sig.wants_continue_run:
        parts.append(f"现场希望继续运行：「{sig.spans.get('wants_continue_run', '')}」")

    return "；".join(parts) + "。"


def _build_known_facts(sig: Signals, kb: KnowledgeBase, alarm: dict[str, Any] | None,
                       gate: GateResult, deviations: list[tuple[str, str, str]],
                       matches: list, wo_notes: list) -> list[Fact]:
    facts: list[Fact] = []

    if alarm:
        facts.append(Fact(
            content=f"{alarm['code']} 的含义为「{alarm['meaning']}」",
            source=f"MANUAL:§3:{alarm['code']}",
        ))

    for item, verdict, basis in deviations:
        facts.append(Fact(content=f"{item}{verdict}", source=f"MANUAL:§1 正常参数（{basis}）"))

    for key in ("setpoint_temp", "actual_temp", "heating_current", "heating_current_state",
                "upstream_pressure", "device_pressure", "material_present", "p1_light",
                "door_closed", "alarm_intermittent", "preheated", "friction_periodic",
                "occurrence_time"):
        val = getattr(sig, key, None)
        if val is None:
            continue
        facts.append(Fact(
            content=f"现场报告{_SIGNAL_PHRASE.get(key, key)}：{_fmt(val, key)}",
            source=f"INPUT:「{sig.spans.get(key, '')}」",
        ))

    for key in sorted(_PRESENCE_ONLY):
        if getattr(sig, key, False) is True:
            facts.append(Fact(
                content=f"现场出现{_SIGNAL_PHRASE.get(key, key)}",
                source=f"INPUT:「{sig.spans.get(key, '')}」",
            ))

    for rule_id in gate.triggered:
        rule = next((r for r in kb.safety_rules if r["id"] == rule_id), None)
        if rule:
            facts.append(Fact(content=f"安全红线 {rule_id}：{rule['requirement']}",
                              source=f"MANUAL:§2:{rule_id}"))

    for m in matches:
        kind = "现场沉淀案例" if m.learned else "历史案例"
        facts.append(Fact(content=f"{kind} {m.case_id}（{m.alarm_code}）：{m.explanation}",
                          source=m.source_ref))

    for n in wo_notes:
        facts.append(Fact(content=n.text, source=f"MANUAL:§5:{n.work_order}"))

    return facts


def _build_causes(sig: Signals, kb: KnowledgeBase, alarm: dict[str, Any] | None,
                  matches: list) -> tuple[list[Cause], list[str]]:
    """按证据强弱排序的可能原因。全部标为「可能」，不存在「已确认」这一档。"""
    if not alarm:
        return [], []

    case_by_id = {m.case_id: m for m in matches}
    causes: list[Cause] = []
    warnings: list[str] = []

    for entry in alarm.get("cause_evidence", []):
        cause = entry["cause"]
        strength = entry["strength"]
        results = [_eval_condition(c, sig, kb) for c in entry.get("when", [])]
        missing_needs = [n for n in entry.get("needs", []) if _obs_missing(n, sig)]

        linked = entry.get("linked_case")
        linked_m = case_by_id.get(linked) if linked else None
        if linked_m and linked_m.state == "refuted":
            warnings.append(
                f"「{cause}」对应的历史案例 {linked} 已被当前现场读数反驳，故不列为可能原因；"
                f"逐项比对见「已知事实」中 {linked} 一条。"
            )
            continue

        if any(r is False for r in results):
            continue  # 条件明确不成立，不是候选原因

        basis_bits: list[str] = []
        if not results:
            basis_bits.append(entry.get("note", "无直接现场证据支持，需靠排除法确认"))
            strength = "弱"
        elif any(r is None for r in results):
            basis_bits.append("判定该原因所需的关键观测当前缺失")
            strength = "弱"
        else:
            basis_bits.append("现场信号满足该原因的成立条件")
            if linked_m and linked_m.state == "strong_match":
                basis_bits.append(f"且与历史案例 {linked} 特征完全一致")

        if missing_needs:
            basis_bits.append(f"仍需补齐：{'、'.join(missing_needs)}")
            if strength == "强":
                strength = "中"

        wo = entry.get("linked_work_order")
        if wo and wo in kb.work_orders:
            rec = kb.work_orders[wo]
            basis_bits.append(f"历史工单 {wo}（{rec['date']}）曾确认过类似原因，仅作候选证据")

        source = f"MANUAL:§3:{alarm['code']} 常见原因"
        if linked:
            source += f"；{linked_m.source_ref}" if linked_m else f"；MANUAL:§4:{linked}"

        causes.append(Cause(
            cause=cause,
            confidence=strength,
            basis="；".join(basis_bits),
            source=source,
        ))

    # 现场沉淀案例的根因也可作为候选原因，但分量必须低于说明书依据：
    # 特征完全一致只给「中」，读数有差异给「弱」，未验证或已被反驳则不采纳。
    seen = {c.cause for c in causes}
    for m in matches:
        if not m.learned or m.root_cause in seen:
            continue
        if m.state == "strong_match":
            strength, wording = "中", "特征一致"
        elif m.state == "differs":
            strength, wording = "弱", "特征部分一致，存在读数差异"
        else:
            continue
        gate = ("由已闭环工单经专家提交、报修人现场确认，双人签字入库"
                if m.field_confirmed_by else
                "由已闭环工单经专家确认入库，早于双人确认规则、没有现场署名")
        causes.append(Cause(
            cause=m.root_cause,
            confidence=strength,
            basis=f"本厂沉淀案例 {m.case_id}（{gate}）与当前现场{wording}；"
                  f"属厂内经验而非说明书原文，须经授权人员现场验证后方可采信",
            source=m.source_ref,
        ))
        seen.add(m.root_cause)

    causes.sort(key=lambda c: STRENGTH_ORDER.get(c.confidence, 3))
    return causes, warnings


def _build_steps(alarm: dict[str, Any] | None, gate: GateResult, coverage: str) -> list[Step]:
    if coverage == "none" or not alarm:
        return []
    steps = []
    for i, action in enumerate(alarm["steps"], 1):
        auth = "现场人员可执行"
        lowered = action
        if any(k in lowered for k in ("授权人员", "授权电气", "专家", "查线路")):
            auth = alarm["derived"].get("authorization", "授权人员")
        elif any(k in lowered for k in ("停机", "锁定", "隔离电源", "泄压", "清洁", "校准")):
            auth = "现场人员可执行（须先满足安全前置条件）"
        steps.append(Step(order=i, action=action, authorization=auth,
                          source=f"MANUAL:§3:{alarm['code']} 安全排查顺序"))
    return steps


def _build_missing(sig: Signals, kb: KnowledgeBase, alarm: dict[str, Any] | None,
                   coverage: str) -> list[str]:
    missing: list[str] = []
    if not sig.alarm_code:
        missing.append("操作面板上的报警码（当前描述未提供任何报警码，无法定位到说明书条目）")
    elif coverage == "none":
        missing.append(f"报警码 {sig.alarm_code} 不在本设备知识库中，需补充该设备的说明书条目")

    if alarm:
        for obs in alarm["derived"].get("required_observations", []):
            if _obs_missing(obs, sig):
                missing.append(obs)

    if not sig.occurrence_time:
        missing.append("故障发生时间点")
    return missing


_PRECONDITION_RULES = ("SAFE-02", "SAFE-04")
_STOP_RULES = ("SAFE-01", "SAFE-03", "SAFE-05")


def _escalation_reason(gate: GateResult, alarm: dict[str, Any] | None) -> str:
    """区分「停机类红线」与「作业前置类红线」，避免同一屏出现自相矛盾的措辞。

    must_stop 为真却没有任何理由是不可接受的——现场必须看得到为什么停。
    此时从报警码自带的说明书属性回填理由。
    """
    if gate.notes:
        return " ".join(gate.notes)

    if not gate.must_stop:
        hit = [r for r in gate.triggered if r in _PRECONDITION_RULES]
        text = f"未触发停机类红线（{'、'.join(_STOP_RULES)}）"
        if hit:
            return (f"{text}；{'、'.join(hit)} 属能量隔离与作业授权前置条件，"
                    f"须满足后方可按排查顺序处理。")
        return f"{text}，可在满足安全前置条件后按排查顺序处理。"

    derived = (alarm or {}).get("derived", {})
    bits: list[str] = []
    if alarm:
        bits.append(f"{alarm['code']}（{alarm['meaning']}）属高风险报警")
    elif "SAFE-01" in gate.triggered:
        bits.append("现场已出现 SAFE-01 列举的立即停机征兆")
    if derived.get("forbid_continue_run"):
        bits.append("说明书安全排查顺序明确「禁止继续试运行」")
    if derived.get("escalate_to_expert"):
        role = str(derived.get("authorization", "专家")).split("（")[0]
        bits.append(f"须交由{role}处置，现场不得自行处理")
    if gate.refusals:
        bits.append("现场提出的请求已被拒绝")
    if not bits:
        return "已触发停机要求。因此必须停机并升级。"
    return "；".join(bits) + "。因此必须停机并升级。"


def diagnose(text: str, kb: KnowledgeBase | None = None,
             llm: Caller | None = None) -> Diagnosis:
    kb = kb or KnowledgeBase()
    sig = parse(text)
    # 兜底解析必须排在安全门控之前：模型补齐的 SAFE-01 征照样要触发强制停机，
    # 放在 evaluate 之后等于让口语化描述里最危险的那一类信号漏过去。
    fill = augment(sig, llm) if llm else None
    gate = evaluate(sig, kb)
    alarm = kb.alarms.get(sig.alarm_code) if sig.alarm_code else None

    # --- 覆盖度判定（SAFE-05 的第一个触发条件）---
    if sig.alarm_code is None and not gate.triggered:
        coverage = "none"
    elif sig.alarm_code is None:
        # 无报警码，但命中了 SAFE-01 立即停机征兆：风险明确，不需要覆盖度
        coverage = "partial"
    elif alarm is None:
        coverage = "none"
    else:
        coverage = "full"

    if coverage == "none":
        reason = (
            "现场未提供报警码且未出现可识别的安全征兆，知识库无法覆盖该情况。"
            if sig.alarm_code is None
            else f"报警码 {sig.alarm_code} 不在本设备知识库覆盖范围内。"
        )
        apply_safe05(gate, reason + "按 SAFE-05 停止进一步操作并升级专家。")

    if len(sig.alarm_codes) > 1:
        # 说明书的排查顺序是一码一表，多码并发本身就是异常工况。挑一个诊断了事
        # 会漏掉其余码的安全属性，所以按 SAFE-05「风险无法判断」停机升级，并要求分别建单。
        apply_safe05(
            gate,
            f"现场同时报出 {'、'.join(sig.alarm_codes)}，说明书的排查顺序按单一报警码编写，"
            f"无法覆盖并发工况。本单按 {sig.alarm_code} 出具，其余各码须分别建单排查。",
        )

    deviations = param_deviation(sig, kb.params_by_key)
    matches = [match_case(c, sig) for c in kb.cases_for(sig.alarm_code)]
    wo_notes = note_work_orders(sig, kb)

    causes, cause_warnings = _build_causes(sig, kb, alarm, matches)

    # --- SAFE-05 的第二个触发条件：真正无法判断，而非"证据不够强" ---
    # 只在两种情况下触发：高风险报警却排不出任何候选原因；或案例被反驳后无原因存活。
    # 过度升级会让客户不再相信升级信号，因此"证据强度为中/弱但仍有方向"时不触发。
    refuted_any = any(m.state == "refuted" for m in matches)
    if coverage != "none" and not causes:
        if gate.severity == "高" or refuted_any:
            reason = (
                "该报警属高风险等级，但当前证据排不出任何候选原因。"
                if gate.severity == "高"
                else "历史案例已被当前读数反驳，且无其他原因存活，构成证据矛盾。"
            )
            apply_safe05(gate, reason + " 按 SAFE-05 升级专家。")

    missing = _build_missing(sig, kb, alarm, coverage)
    steps = _build_steps(alarm, gate, coverage)

    preconditions = list(gate.preconditions)
    if not preconditions:
        # 空着会被现场读成「不需要任何防护」，比不写更糟；未覆盖时连风险都判断不了，
        # 兜底只能用失效关闭那一句最保守的措辞。
        if coverage == "none":
            why = ("现场未提供报警码且无可识别的安全征兆" if sig.alarm_code is None
                   else f"报警码 {sig.alarm_code} 不在知识库覆盖范围内")
            preconditions.append(f"风险无法判断（{why}）：一切操作待专家到场确认后执行")
        else:
            preconditions.append("无额外安全前置条件，但仍须遵守 §2 安全红线")

    esc = Escalation(
        must_stop=gate.must_stop,
        escalate_to_expert=gate.escalate_to_expert,
        expert_type="、".join(gate.expert_types),
        stop_conditions=_stop_conditions(gate, alarm, coverage),
        triggered_rules=gate.triggered,
        reason=_escalation_reason(gate, alarm),
    )

    warnings = list(cause_warnings)
    assisted = [item.split(" ← ")[0] for item in fill.filled] if fill else []
    if assisted:
        warnings.append(
            f"本次由模型辅助补齐信号：{'、'.join(assisted)}。"
            "每项均已回核现场原文片段，读数无法从原文得出的已丢弃；模型只做听写，不参与事实判定。"
        )
    if fill and fill.rejected:
        # 这层失败不能让输出看起来像一次"正常的纯规则诊断"：端点超时和被复核挡掉是两回事，
        # 前者意味着"这句话可能还有一个停机征兆没被读到"，必须显式说出来。
        warnings.append(
            f"模型兜底解析本次未生效（{len(fill.rejected)} 项被拒绝）："
            + "；".join(fill.rejected)
            + "。以下结论仅基于规则层读到的信号，"
            "若现场口述中还有未被识别的异常，请以「需要补充的信息」为准复核。"
        )
    if not sig.known_keys() - {"raw"}:
        warnings.append("现场描述过于简略，规则解析未提取到有效信号，建议补充现场读数后重新报修。")

    d = Diagnosis(
        phenomenon=_build_phenomenon(sig, kb, alarm),
        known_facts=_build_known_facts(sig, kb, alarm, gate, deviations, matches, wo_notes),
        possible_causes=causes,
        evidence=_build_evidence(sig, alarm, matches, wo_notes, deviations),
        safety_preconditions=preconditions,
        inspection_steps=steps,
        missing_info=missing,
        stop_and_escalate=esc,
        alarm_code=sig.alarm_code,
        coverage=coverage,
        safety_gates=gate.triggered,
        refused_requests=gate.refusals,
        model_used="deterministic-core+llm-parse" if assisted else "deterministic-core",
        warnings=warnings,
        parse_assisted=assisted,
    )
    d.validation = validate(d, text, kb)
    return enforce(d, d.validation, text)


def _stop_conditions(gate: GateResult, alarm: dict[str, Any] | None, coverage: str) -> list[str]:
    conds: list[str] = []
    if gate.forbid_continue_run:
        conds.append("禁止继续试运行或恢复生产，直至专家确认风险已消除")
    if "SAFE-01" in gate.triggered:
        conds.append("出现烟雾、焦味、异常高温、剧烈振动、金属摩擦声或部件松脱时立即停机")
    if "SAFE-03" in gate.triggered:
        conds.append("任何要求短接安全门、绕过报警、带电插拔或徒手触碰加热部件的请求，一律停止并拒绝")
    if "SAFE-05" in gate.triggered:
        conds.append("知识库无覆盖、证据矛盾或风险无法判断时，停止进一步操作并升级专家")
    if alarm and alarm["derived"].get("requires_cooldown"):
        conds.append("未冷却至安全温度前不得接近热封部件")
    if coverage == "none":
        conds.append("在补齐报警码或说明书条目之前不执行任何排查动作")
    return conds


def _build_evidence(sig: Signals, alarm: dict[str, Any] | None, matches: list,
                    wo_notes: list, deviations: list[tuple[str, str, str]]) -> list[str]:
    out: list[str] = []
    if alarm:
        out.append(f"MANUAL:§3:{alarm['code']}「{alarm['meaning']}」及其安全排查顺序")
    if deviations:
        out.append("MANUAL:§1 设备正常参数（用于判定现场读数是否越界）")
    for m in matches:
        out.append(f"{m.source_ref}（状态：{m.state}）")
    for n in wo_notes:
        out.append(f"MANUAL:§5:{n.work_order}（{n.date}）")
    cited = [s for s in sig.spans.values() if s]
    if cited:
        out.append("INPUT:现场描述原文「" + "」「".join(dict.fromkeys(cited)) + "」")
    return out
