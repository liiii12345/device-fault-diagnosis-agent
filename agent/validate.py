"""输出后校验层：诊断结果交给现场之前的最后一道闸门。

测试集只能证明"我写过的用例不出错"，证明不了"上线后每一次输出都不出错"。
本模块把全局不变量搬到运行时，每次输出都过一遍，失败即**失效关闭**（fail closed）：
撤回可能原因与排查顺序、强制升级专家，而不是把可疑结论发给现场。

这是回答「你怎么保证 AI 不胡说」的最硬一句：不是靠 prompt 里写"请不要编造"，
而是有一层代码在输出之后逐条核对数字与来源，对不上就不许发出去。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .knowledge import KnowledgeBase
from .parse import numbers_in
from .schema import FIELD_LABELS, Diagnosis

# 数字若无法在「现场输入」或「说明书知识库」中找到出处，即视为幻觉。
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# 现场描述里出现的这些说法，无论上下文如何都不允许出现在 Agent 输出中（SAFE-03）。
BANNED_PHRASES = (
    "可以短接", "建议短接", "可以绕过", "建议绕过", "可以带电", "建议带电",
    "屏蔽联锁后", "先继续生产", "可以继续试运行", "不用停机", "无需停机",
)

# 可能原因只能是"可能"：没有任何实测证据能支撑断言，写出「已确认」本身就是幻觉。
ASSERTIVE_PHRASES = ("已确认", "确定是", "一定是", "肯定是", "必然是", "就是", "根因为")

_SOURCE_RE = re.compile(r"^(MANUAL:§\d|INPUT:|KB:)")

_QUOTE_RE = re.compile(r"「([^」]+)」")


def _own_words(blob: str, text: str) -> str:
    """只留 Agent 自己写的话：逐字引自现场原文的「…」片段不算 Agent 的主张。

    复述现场的违规请求是必需的——现场要看见自己那句话为什么被拒。可要是连引用一起扫，
    一份正确停机升级的结论会被判成"Agent 给了违规建议"，接着被 enforce() 整体撤回。
    伪装成引用的建议照样拦得住：现场原文里没有这句话，就不会被剥掉。
    """
    return _QUOTE_RE.sub(lambda m: "" if m.group(1) in text else m.group(0), blob)


def _banned_in(item: str, text: str) -> list[str]:
    own = _own_words(item, text)
    return [p for p in BANNED_PHRASES if p in own]


# 触发后必须停机并升级专家的红线。
_HARD_RULES = ("SAFE-01", "SAFE-03", "SAFE-05")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    # 出问题的具体条目标签，如「已知事实#3」；失效关闭时据此精准剔除被污染的内容
    bad: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [f"[{c.name}] {c.detail}" for c in self.checks if not c.ok]

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = ["## 输出自检"]
        for c in self.checks:
            lines.append(f"- {'通过' if c.ok else '未通过'}　{c.name}" + (f"：{c.detail}" if c.detail and not c.ok else ""))
        lines.append(f"\n结论：{'全部通过，可发出现场' if self.ok else '未通过，已失效关闭（撤回结论并强制升级）'}")
        return "\n".join(lines)


def _norm(tok: str) -> str:
    """归一化数字写法，使 188 / 188.0 / 042 能与出处对齐比较。"""
    try:
        return f"{float(tok):g}"
    except ValueError:
        return tok


def _collect_numbers(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.add(f"{obj:g}")
    elif isinstance(obj, str):
        out.update(_norm(t) for t in _NUM_RE.findall(obj))


def _traceable_numbers(text: str, kb: KnowledgeBase) -> set[str]:
    # 现场原文里的阿拉伯数字与口述中文数字，都走解析层那一个归一化函数
    allowed: set[str] = numbers_in(text)
    for obj in (kb.device, kb.alarms, kb.safety_rules, kb.all_cases, kb.maintenance):
        _collect_numbers(obj, allowed)
    return allowed


def _prose(d: Diagnosis) -> list[tuple[str, str]]:
    """待核查的正文片段（标签, 文本）。来源标签不在此列——它们由 _SOURCE_RE 单独核查。"""
    out: list[tuple[str, str]] = [("故障现象", d.phenomenon)]
    out += [(f"已知事实#{i}", f.content) for i, f in enumerate(d.known_facts, 1)]
    out += [(f"可能原因#{i}", f"{c.cause}｜{c.basis}") for i, c in enumerate(d.possible_causes, 1)]
    out += [("证据来源", e) for e in d.evidence]
    out += [("安全前置条件", p) for p in d.safety_preconditions]
    out += [(f"排查顺序#{s.order}", s.action) for s in d.inspection_steps]
    out += [("需要补充的信息", m) for m in d.missing_info]
    esc = d.stop_and_escalate
    out += [("停止条件/升级", c) for c in esc.stop_conditions]
    out += [("停止条件/升级", esc.reason)]
    out += [("提示", w) for w in d.warnings]
    return out


def validate(d: Diagnosis, text: str, kb: KnowledgeBase) -> ValidationReport:
    rep = ValidationReport()

    # 1. 8 个字段一个都不能少
    keys = set(d.to_dict())
    missing = [label for label in FIELD_LABELS.values() if label not in keys]
    rep.checks.append(Check("八字段齐全", not missing, "、".join(missing)))

    # 2. 已知事实必须带可追溯来源
    bad = [f"已知事实#{i}" for i, f in enumerate(d.known_facts, 1)
           if not _SOURCE_RE.match(f.source or "")]
    rep.checks.append(Check("已知事实来源可追溯", not bad, f"{len(bad)} 条无合法来源标签", bad=bad))

    # 3. 可能原因只能是"可能"
    assertive = [
        f"可能原因#{i}"
        for i, c in enumerate(d.possible_causes, 1)
        if any(p in c.cause or p in c.confidence for p in ASSERTIVE_PHRASES)
    ]
    illegal_conf = [f"可能原因#{i}" for i, c in enumerate(d.possible_causes, 1)
                    if c.confidence not in ("强", "中", "弱")]
    bad3 = sorted(set(assertive + illegal_conf))
    rep.checks.append(Check(
        "可能原因不含断言措辞",
        not bad3,
        f"{len(assertive)} 条含断言措辞，{len(illegal_conf)} 条证据强度非法",
        bad=bad3,
    ))

    # 4. 绝不给出违规操作建议（现场原话的引用不算 Agent 的建议）
    blob = _own_words("\n".join(t for _, t in _prose(d)), text)
    hit = [p for p in BANNED_PHRASES if p in blob]
    rep.checks.append(Check("无违规操作建议", not hit, "、".join(hit)))

    # 5. 排查顺序：一旦给出就必须与说明书逐字一致；为空则必须是停机或未覆盖所致
    alarm = kb.alarms.get(d.alarm_code) if d.alarm_code else None
    esc = d.stop_and_escalate
    orders = [s.order for s in d.inspection_steps]
    actual = [s.action for s in d.inspection_steps]
    expected = (alarm or {}).get("steps", [])
    if actual:
        step_ok = orders == list(range(1, len(orders) + 1)) and actual == expected
        step_detail = (f"编号错乱 {orders}" if orders != list(range(1, len(orders) + 1))
                       else f"应为 {expected}，实为 {actual}")
    else:
        step_ok = esc.must_stop or d.coverage == "none"
        step_detail = "既未触发停机、知识库也已覆盖，却未给出任何排查顺序"
    rep.checks.append(Check("排查顺序与说明书一致", step_ok, step_detail))

    # 6. 知识库未覆盖时不得编造流程，也不得给出强证据原因
    if d.coverage == "none":
        violations = []
        if d.inspection_steps:
            violations.append("coverage=none 却输出了排查步骤")
        if [c for c in d.possible_causes if c.confidence == "强"]:
            violations.append("coverage=none 却给出强证据原因")
        rep.checks.append(Check("未覆盖时不编造", not violations, "；".join(violations)))
    else:
        rep.checks.append(Check("未覆盖时不编造", True))

    # 7. 数字幻觉检查：正文中每个数字都要能在输入或说明书里找到出处
    allowed = _traceable_numbers(text, kb)
    stray: list[str] = []
    bad_labels: list[str] = []
    for label, body in _prose(d):
        for tok in _NUM_RE.findall(body):
            # 个位数字多为序号或位号（P1、T1、§3），不参与溯源
            if len(tok) < 2:
                continue
            n = _norm(tok)
            if n not in allowed:
                stray.append(f"{label} 中的 {tok}")
                bad_labels.append(label)
    rep.checks.append(Check(
        "数字全部可溯源", not stray,
        "；".join(dict.fromkeys(stray))[:200],
        bad=list(dict.fromkeys(bad_labels)),
    ))

    # 8. 停机/升级判定与红线、以及报警码自带的硬属性一致
    hard = [r for r in _HARD_RULES if r in d.safety_gates]
    derived = (alarm or {}).get("derived", {})
    inconsistent = []
    if hard and not esc.must_stop:
        inconsistent.append(f"触发 {'、'.join(hard)} 却未要求停机")
    if hard and not esc.escalate_to_expert:
        inconsistent.append(f"触发 {'、'.join(hard)} 却未升级专家")
    if "SAFE-03" in d.safety_gates and not d.refused_requests:
        inconsistent.append("命中 SAFE-03 却没有记录被拒绝的请求")
    # A205 一类报警即使没触发任何红线，说明书属性本身就要求升级并禁止试运行
    if derived.get("escalate_to_expert") and not esc.escalate_to_expert:
        inconsistent.append(f"{d.alarm_code} 的说明书属性要求升级专家，输出却未升级")
    if derived.get("forbid_continue_run"):
        if not esc.must_stop:
            inconsistent.append(f"{d.alarm_code} 的说明书属性禁止继续运行，输出却未要求停机")
        if not any("禁止继续试运行" in c for c in esc.stop_conditions):
            inconsistent.append(f"{d.alarm_code} 的停止条件中缺少「禁止继续试运行」")
    rep.checks.append(Check("停机与升级判定一致", not inconsistent, "；".join(inconsistent)))

    return rep


_BAD_ITEM_RE = re.compile(r"^(已知事实)#(\d+)$")


def enforce(d: Diagnosis, rep: ValidationReport, text: str = "") -> Diagnosis:
    """失效关闭：自检未通过时剔除被污染的内容、撤回结论，改为停止并升级。

    写回现场的文案不得回显出问题的数字或原文——否则降级后的输出会再次触发
    数字溯源检查，陷入自我矛盾。完整原因保留在 validation 报告里供审计日志使用。
    """
    if rep.ok:
        return d

    failed = [c.name for c in rep.checks if not c.ok]
    drop = {int(m.group(2)) for c in rep.checks if not c.ok
            for label in c.bad if (m := _BAD_ITEM_RE.match(label))}
    if drop:
        d.known_facts = [f for i, f in enumerate(d.known_facts, 1) if i not in drop]

    # 违规操作建议要单独清：它不依附于任何一条证据，撤回可能原因和排查顺序带不走它。
    # 留着就等于一边停机升级、一边把「可以短接安全门」照样发给现场。
    # 删空时补一条保守兜底——前置条件全空会让现场误以为不需要任何防护，比留着更糟。
    for owner, attr, fallback in (
            (d, "safety_preconditions", "一切操作待专家到场确认后执行"),
            (d.stop_and_escalate, "stop_conditions", "停止一切操作，等待专家复核")):
        items = getattr(owner, attr)
        kept = [x for x in items if not _banned_in(x, text)]
        if len(kept) != len(items):
            setattr(owner, attr, kept or [fallback])

    # 可能原因整体撤回：其排序依赖上面被剔除的证据，留着会给出错误的优先级
    d.possible_causes = []
    d.inspection_steps = []
    d.degraded = True
    if "SAFE-05" not in d.safety_gates:
        d.safety_gates.append("SAFE-05")

    summary = f"输出自检未通过（{'、'.join(failed)}），已撤回本次诊断结论并升级专家。"
    d.stop_and_escalate.must_stop = True
    d.stop_and_escalate.escalate_to_expert = True
    d.stop_and_escalate.expert_type = d.stop_and_escalate.expert_type or "专家"
    d.stop_and_escalate.reason = summary + " 完整原因见审计日志。"
    if "停止进一步排查，等待专家复核" not in d.stop_and_escalate.stop_conditions:
        d.stop_and_escalate.stop_conditions.insert(0, "停止进一步排查，等待专家复核")

    d.warnings.append(summary)
    d.missing_info.append("需人工复核：本次输出未通过自检，结论已撤回")
    return d
