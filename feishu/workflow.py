"""飞书报修闭环编排。

    报修（群消息 / 多维表格）
        ↓
    确定性诊断 + 输出自检
        ↓
    风险路由 ──高危──→ 专家群红色卡片 + 工单状态「已升级专家」
        │
        └──常规──→ 报修群卡片 + 工单状态「待现场排查」
        ↓
    工单写回多维表格（含证据链、自检结果、AI 调用量）
        ↓
    ⑦ 需要补充的信息 ──卡片按钮──→ 网页回填 ──→ 「原描述＋补充」重跑同一条诊断链
        ↓                                          ↓
    现场处置                              第二轮工单（关联原单）＋ 新卡片
        ↓
    专家填写处置结论（实际根因 / 处置动作 / 复机验证）──卡片按钮＋一次性确认码──→ 网页
        ↓
    沉淀为候选案例（var/candidate_cases.json，状态：待现场确认）※ 此时仍不参与诊断
        ↓
    报修人现场确认（只问现场看得见的事：照做了吗 / 恢复了吗 / 读数多少）
        ├── 通过 ──→ 入库（knowledge/cases_learned.json）→ 参与后续诊断
        └── 不通过 ─→ 一个字都不写，分歧写回工单，重发确认码退回专家群

四处刻意的设计：

1. **候选案例在两人签字前不参与诊断。** 让 AI 自己把猜测写进知识库、再拿它当证据，
   等于给幻觉开了复利。沉淀与入库之间必须隔一道人工确认，而且是两道：技术结论由
   专家署名，现场可行性由报修人作证。一个人既当运动员又当裁判，这条经验就没人能
   对它的真实性负责。
2. **现场确认只问现场事实，不问技术判断。** 报修人能作证「照这个做了、机器恢复了、
   读数回到 165°C」，作证不了「根因是继电器触点粘连」。让他给技术结论投票，等于
   造了一个假权威——这正是这套 Agent 从头到尾在防的事。
3. **沉淀表与说明书原文分表。** history_cases.json 是设备厂商给的依据，只读；
   cases_learned.json 是本厂经验，可单独查看、单独回滚。
4. **补充信息另建第二轮工单，不覆盖原单。** 第一轮的卡片已经进群、原单可能已经
   进了专家视野；改存档等于事后改口，谁在什么时候补了什么就再也查不到了。
"""

from __future__ import annotations

import json
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.diagnose import diagnose
from agent.knowledge import KnowledgeBase
from agent.llm_parse import Caller
from agent.parse import parse
from agent.schema import Diagnosis
from agent.validate import enforce, validate

from .client import Client

ROOT = Path(__file__).resolve().parent.parent

STATUS_NEW = "待现场排查"
STATUS_ESCALATED = "已升级专家"
STATUS_CLOSED = "已闭环待沉淀"        # 命令行直接 close：不走网页，没有现场确认环节
STATUS_AWAIT_FIELD = "待现场确认"      # 专家已提交技术结论，等报修人作证现场可行
STATUS_PROMOTED = "已沉淀入库"
STATUS_REJECTED = "现场确认未通过"

CAND_PENDING = "待现场确认"
CAND_REJECTED = "现场确认未通过，退回专家"


@dataclass
class Paths:
    """闭环落盘位置。默认指向项目根；演示时换成临时目录即可跑完整飞轮而不污染知识库。"""

    root: Path = ROOT

    @property
    def var(self) -> Path:
        return self.root / "var"

    @property
    def knowledge(self) -> Path:
        return self.root / "knowledge"

    @property
    def candidates(self) -> Path:
        return self.var / "candidate_cases.json"

    @property
    def learned(self) -> Path:
        return self.knowledge / "cases_learned.json"

    @property
    def usage(self) -> Path:
        return self.var / "usage.jsonl"


@dataclass
class Ticket:
    wo_id: str
    diagnosis: Diagnosis
    high_risk: bool
    record_id: str
    message_id: str
    round: int = 1
    prev_wo: str = ""


# ------------------------------------------------------------------ 卡片渲染
#
# 卡片与工单全文是同一份内容的两种排版，不是「摘要 vs 完整」：八个字段（现场必须拿到的 6 项 +
# 本方案额外 2 项）一条明细都不省，一条一行，字段名加粗，出处用最短标签跟在后面。
# 密度靠"一条一行 + 剥掉模板腔"解决，不靠删条目解决——概览删掉一半条目就不叫概览。
#
# lark_md 不认 ## 与 ---，也不支持调字号：层级靠一个字段一个 div、div 内换行分条。


def _flat(text: Any) -> str:
    """折成一行。div 内的换行被飞书当成分条，所以条目正文里不许再有换行。"""
    return " ".join(str(text).split())


def _div(content: str) -> dict[str, Any]:
    return {"tag": "div", "text": {"tag": "lark_md", "content": content}}


def _verdict(d: Diagnosis) -> str:
    if d.degraded:
        return "结论已撤回"
    if d.stop_and_escalate.must_stop:
        return "立即停机"
    if d.stop_and_escalate.escalate_to_expert:
        return "升级专家"
    return "可继续排查"


def _experts(esc) -> str:
    """两条红线可能点名同一拨专家（「专家」与「专家（禁止现场自行处理）」）。

    卡片上留最具体那条，不然显示成「专家、专家（禁止现场自行处理）」很突兀。
    只做显示层去重——工单与命令行仍保留说明书原文的完整列表。
    """
    names = [n.strip() for n in esc.expert_type.split("、") if n.strip()]
    return "、".join(n for n in names if not any(m != n and m.startswith(n) for m in names))


def _mix(sources: list[str]) -> str:
    """三类出处的条数：说明书 / 经验 / 原话。"""
    buckets = {"说明书": 0, "经验": 0, "原话": 0}
    for s in sources:
        if s.startswith("MANUAL"):
            buckets["说明书"] += 1
        elif s.startswith("KB:LEARNED"):
            buckets["经验"] += 1
        else:
            buckets["原话"] += 1
    return "·".join(f"{k}{v}" for k, v in buckets.items() if v)


def _tag(source: str) -> str:
    """出处的最短可读形式。MANUAL:§3:A205… → 说明书§3；KB:LEARNED:L01 → 经验L01。

    INPUT 的引号里那截就是出处本身（现场原话），保留不缩写。
    """
    if m := re.match(r"MANUAL:(§\d+)", source):
        return "说明书" + m.group(1)
    if m := re.match(r"KB:LEARNED:([\w-]+)", source):
        return "经验" + m.group(1)
    if source.startswith("INPUT:"):
        return "原话" + source[len("INPUT:"):]
    return "原话"


def _phenomenon(d: Diagnosis) -> str:
    text = d.phenomenon
    for noisy, tight in (("设备报出 ", ""), ("现场未报告任何报警码", "无报警码"),
                         ("，但该报警码不在本设备知识库覆盖范围内", "（知识库未收录）"),
                         ("；现场读数：", "｜"), ("；可见/可听异常与状态：", "｜"),
                         ("现场读数：", "｜"), ("发生时间 ", "｜"),
                         ("现场援引了历史维修记录：", "援引记录：")):
        text = text.replace(noisy, tight)
    # 复述项后面那个（原文「…」）和信号名重复（「焦味（原文「焦味」）」），出处已由④承担
    text = re.sub(r"（原文「[^」]*」）", "", text)
    return f"**① 故障现象**｜{_flat(text)}"


def _facts(d: Diagnosis) -> str:
    if not d.known_facts:
        return "**② 已知事实**｜无——现场描述里没有可引用的确定信息"
    rows = [f"{i}. {_flat(f.content)}〔{_tag(f.source)}〕"
            for i, f in enumerate(d.known_facts, 1)]
    return "\n".join([f"**② 已知事实**｜{len(rows)} 条"
                      f"（{_mix([f.source for f in d.known_facts])}）", *rows])


def _causes(d: Diagnosis) -> str:
    if not d.possible_causes:
        return "**③ 可能原因**｜无——证据不足，不给推测性原因"
    rows = [f"{i}. {_flat(c.cause)}｜证据强度 {c.confidence}｜依据：{_flat(c.basis)}"
            f"〔{_tag(c.source)}〕"
            for i, c in enumerate(d.possible_causes, 1)]
    return "\n".join([f"**③ 可能原因**｜{len(rows)} 条，按证据强弱排序"
                      f"（均为可能而非已确认）", *rows])


def _sources(d: Diagnosis) -> str:
    if not d.evidence:
        return "**④ 证据来源**｜无"
    rows = [f"- `{_flat(e)}`" for e in d.evidence]
    return "\n".join([f"**④ 证据来源**｜{len(rows)} 份（{_mix(list(d.evidence))}）", *rows])


def _safety(d: Diagnosis) -> str:
    if not d.safety_preconditions:
        return "**⑤ 安全前置条件**｜无"
    rows = [f"- {_flat(p)}" for p in d.safety_preconditions]
    return "\n".join([f"**⑤ 安全前置条件**｜{len(rows)} 条，未满足前不得动手", *rows])


def _steps(d: Diagnosis) -> str:
    if not d.inspection_steps:
        return "**⑥ 排查顺序**｜无——已触发停机，不安排现场排查"
    rows = [f"{s.order}. {_flat(s.action)}"
            + (f"【仅限：{_flat(s.authorization)}】" if s.authorization else "")
            + f"〔{_tag(s.source)}〕"
            for s in d.inspection_steps]
    # 停机时照样把顺序列全：这几步不是给现场做的，是专家到场后的清单，删了就没人知道查什么。
    head = (f"**⑥ 排查顺序**｜{len(rows)} 步（已停机：**现场一律不动手**，"
            f"先满足⑤，以下步骤交专家到场后按序执行）"
            if d.stop_and_escalate.must_stop else
            f"**⑥ 排查顺序**｜{len(rows)} 步，逐条抄自说明书、不得重排")
    return "\n".join([head, *rows])


def _gaps(d: Diagnosis) -> str:
    if not d.missing_info:
        if d.unresolved_answers:
            # 空 ⑦ 有两种：观测项真齐了，和"答了但读不出、已从清单摘掉"。后者不能说成前者，
            # 那等于告诉现场不用再测了。
            return ("**⑦ 需要补充的信息**｜无——本轮答复的 "
                    + "、".join(_flat(x) for x in d.unresolved_answers)
                    + " 读不出可判定信号，未据此调整证据强度，详见「提示」")
        return "**⑦ 需要补充的信息**｜无——说明书要求的观测项现场都已给出"
    rows = [f"- {_flat(m)}" for m in d.missing_info]
    return "\n".join([f"**⑦ 需要补充的信息**｜{len(rows)} 项", *rows])


def _escalation(d: Diagnosis) -> str:
    esc = d.stop_and_escalate
    if d.degraded:
        head = "🛑 **自检未通过，结论已撤回**"
    elif esc.must_stop:
        head = "🛑 **立即停机**"
    elif esc.escalate_to_expert:
        head = "⏸ **升级专家**"
    else:
        head = "✅ **无需升级，按⑥排查**"
    who = f" → {_experts(esc) or '专家'}" if esc.escalate_to_expert else ""
    parts = [f"**⑧ 是否升级专家**｜{head}{who}"]
    if esc.triggered_rules:
        # 「红线」只用于真要停机/升级的情形，否则会和上面的判定自相矛盾。
        hard = esc.must_stop or esc.escalate_to_expert
        parts.append(("触发红线 " if hard else "命中安全规则 ")
                     + "·".join(esc.triggered_rules))
    if esc.reason:
        parts.append(f"判定理由：{_flat(esc.reason)}")
    parts += [f"停止条件：{_flat(c)}" for c in esc.stop_conditions]
    if d.refused_requests:
        parts.append(f"🚫 **已当场拒绝违规请求 {len(d.refused_requests)} 项**")
        parts += [f"　{_flat(r)}" for r in d.refused_requests]
    return "\n".join(parts)


def _warnings(d: Diagnosis) -> str:
    """兜底解析未生效、案例被反驳这类降级说明。不写出来，现场会把降级版当成正常结论。"""
    if not d.warnings:
        return ""
    return "\n".join([f"⚠️ **提示**｜{len(d.warnings)} 条",
                      *[f"- {_flat(w)}" for w in d.warnings]])


def _card(title: str, color: str, elements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": title}, "template": color},
        "elements": elements,
    }


def build_card(wo_id: str, d: Diagnosis, reporter: str, device: str,
               wo_url: str = "", followup_url: str = "",
               round_no: int = 1, prev_wo: str = "",
               close_url: str = "", expert_code: str = "") -> dict[str, Any]:
    high = is_high_risk(d)
    # 诊断卡只有红蓝两色：覆盖度 none 在 diagnose 里必定触发 SAFE-05 强制升级，
    # 所以"知识库未覆盖"也是红卡，不存在介于两者之间的第三种风险状态。
    # （下面的 _field_card / _promoted_card / _rejected_card 是流程卡，颜色表示
    #   流程走到哪一步，不是风险等级，别和这里的两色混为一谈。）
    color, prefix = ("red", "🔴 高危") if high else ("blue", "🔵 常规报修")

    # 判定直接写进卡头标题：扫一眼群列表就知道这单要不要管，不用点开卡片。
    title = (f"{prefix}｜{d.alarm_code or '无报警码'}｜{_verdict(d)}｜{wo_id}"
             + (f"｜第{round_no}轮" if round_no > 1 else ""))

    sections = [
        _phenomenon(d),
        _facts(d),
        _causes(d),
        _sources(d),
        _safety(d),
        _steps(d),
        _gaps(d),
        _escalation(d),
        _warnings(d),
    ]
    if round_no > 1:
        sections.insert(0, f"🔁 **第 {round_no} 轮诊断**｜承接 {prev_wo}："
                           f"现场已补上上一轮的待补项，本卡结论取代原单")
    elements = [_div(s) for s in sections if s]
    elements.append({"tag": "hr"})

    checks = len(d.validation.checks) if d.validation else 0
    elements.append(_div(
        f"📄 工单 {wo_id}（诊断全文 {len(d.render())} 字）另有：出处标签原文、"
        f"提取信号、{checks} 项自检明细、闭环与沉淀状态"))
    # ⑦ 为空就不出这个按钮：点进去是一张没有题目的空表单，和死链一样是死胡同。
    can_follow = bool(d.missing_info and followup_url)
    if d.missing_info:
        entry = "点下方按钮" if can_follow else f"打开网页入口 /?wo={wo_id}"
        elements.append(_div(
            f"✍️ ⑦ 待补 {len(d.missing_info)} 项，都能回填：{entry}。补齐后把原描述与补充合成一段"
            f"重跑同一条诊断链（安全闸门与自检照常生效），另建关联工单并推送第 {round_no + 1} 轮卡片"))

    # 处置结论的入口只在升级专家的单上出现：常规单是现场自己排掉的，
    # 同一个人既写结论又确认"现场可行"，双人确认就成了自问自答。
    can_close = bool(expert_code and close_url)
    if expert_code:
        entry = "点下方按钮" if can_close else f"打开网页入口 /?wo={wo_id}&k={expert_code}"
        elements.append(_div(
            f"🧑‍🔧 处置完成后由专家填写实际根因与处置动作：{entry}（确认码 {expert_code}，一次性）。"
            f"提交后生成候选案例（此时不参与诊断），再由报修人确认现场可行——两人签字才入库"))

    # 没有真链接就不出按钮——死链比没有链更糟，这一条对补充信息入口同样成立。
    buttons = []
    if can_follow:
        buttons.append({"tag": "button", "type": "primary", "url": followup_url,
                        "text": {"tag": "plain_text", "content": "✍️ 补充信息，重出诊断"}})
    if can_close:
        buttons.append({"tag": "button", "type": "default", "url": close_url,
                        "text": {"tag": "plain_text", "content": "🧑‍🔧 填写处置结论"}})
    if wo_url:
        buttons.append({"tag": "button", "type": "default", "url": wo_url,
                        "text": {"tag": "plain_text", "content": "📄 打开完整工单"}})
    if buttons:
        elements.append({"tag": "action", "actions": buttons})
    ok = bool(d.validation and d.validation.ok)
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content":
        f"报修人 {reporter}｜{device}｜{time.strftime('%m-%d %H:%M')}｜"
        f"推理引擎 {d.model_used}｜知识库覆盖度 {d.coverage}｜"
        f"输出自检 {'通过' if ok else '未通过'}（{checks} 项）"}]})
    return _card(title, color, elements)


def is_high_risk(d: Diagnosis) -> bool:
    """停机、升级专家、或自检未通过而失效关闭——三者任一都按高危路由。"""
    return bool(d.stop_and_escalate.must_stop or d.stop_and_escalate.escalate_to_expert or d.degraded)


# ------------------------------------------------------------------ 一次性确认码
#
# 双人确认发生在公网链接上，页面本身没有登录态。确认码是这条链上唯一的凭证：
# 专家码只出现在专家群那张卡片的链接里，现场码只出现在报修群的链接里，
# 用过一次即作废。这不是身份认证——转发链接就等于转交权限，生产形态应当走
# 飞书事件回调拿 open_id。但在演示形态下，它把"谁能改知识库"从任何人收窄到
# 拿得到那张卡片的人，且每一步都在工单上留痕。

CODE_LEN = 6
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # 去掉 0O1I，口述与抄写都不会认错


class CodeRejected(RuntimeError):
    """确认码不对或已被使用。与"内容填错"分开：这是权限问题，网页要回 403。"""


def _new_code() -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(CODE_LEN))


def _code_match(want: Any, given: str) -> bool:
    """定长比较，不给"逐位试出来"留时间差；空码永不匹配，否则空链接就等于万能钥匙。"""
    want = str(want or "").strip()
    return bool(want) and secrets.compare_digest(str(given or "").strip().upper().encode("utf-8"),
                                                 want.encode("utf-8"))


def _stage_of(wo: dict[str, Any], code: str) -> str:
    """这张码属于哪一阶段。阶段由码决定，页面说自己是谁不算数。"""
    if _code_match(wo.get("专家码"), code):
        return "expert"
    if _code_match(wo.get("现场码"), code):
        return "field"
    return ""


def _reject_unknown(wo: dict[str, Any], wo_id: str) -> None:
    """两个码都对不上。分清"这一单没有待办了"和"码抄错了"，别让人对着 403 猜。"""
    if str(wo.get("专家码") or "").strip() or str(wo.get("现场码") or "").strip():
        raise CodeRejected("确认码不对：请从群里的卡片重新点开，不要手抄地址")
    raise CodeRejected(f"工单 {wo_id} 当前没有待处理的确认码"
                       "（可能已经处理过了，请从群里的卡片重新点开）")


def _check_code(wo: dict[str, Any], field: str, code: str, who: str) -> None:
    if not str(wo.get(field) or "").strip():
        raise CodeRejected(f"工单 {wo.get('工单号', '?')} 当前没有待{who}处理的确认码"
                           f"（可能已经处理过了，请从群里的卡片重新点开）")
    if not _code_match(wo.get(field), code):
        raise CodeRejected("确认码不对：请从群里的卡片重新点开，不要手抄地址")


# ------------------------------------------------------------------ 工单

def _extracted_signals(text: str) -> dict[str, Any]:
    """把解析出的信号存进工单，闭环时用它构造可被机器复用的案例特征。"""
    sig = parse(text)
    return {k: getattr(sig, k) for k in sorted(sig.known_keys()) if k != "raw"}


def _wo_fields(wo_id: str, text: str, reporter: str, device: str, d: Diagnosis,
               round_no: int = 1, prev_wo: str = "") -> dict[str, Any]:
    esc = d.stop_and_escalate
    top = d.possible_causes[0] if d.possible_causes else None
    return {
        "工单号": wo_id,
        "报修时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "报修人": reporter,
        "设备": device,
        "现场描述": text,
        "报警码": d.alarm_code or "",
        "知识库覆盖度": d.coverage,
        "风险等级": "高" if is_high_risk(d) else "常规",
        "是否升级专家": "是" if esc.escalate_to_expert else "否",
        "触发的安全红线": "、".join(d.safety_gates),
        "已拒绝的违规请求": "、".join(d.refused_requests),
        "首要可能原因": top.cause if top else "",
        "证据强度": top.confidence if top else "",
        "需补充信息": "、".join(d.missing_info),
        # 回填页面按这份清单出题。顿号拼接串是给人看的，拆回来不保证还原（项里可能自带顿号）。
        "需补充信息项": json.dumps(d.missing_info, ensure_ascii=False),
        "诊断轮次": str(round_no),
        # 两个方向分开存：第三轮出现时，若共用一列就会把第二轮指回第一轮的指针覆盖掉，
        # 链条从此只能单向往后走。
        "关联工单": prev_wo,
        "后续工单": "",
        "诊断全文": d.render(),
        "输出自检": "通过" if (d.validation and d.validation.ok) else "未通过",
        "推理引擎": d.model_used,
        "模型辅助解析": "、".join(d.parse_assisted),
        "AI调用次数": 1,
        "状态": STATUS_ESCALATED if is_high_risk(d) else STATUS_NEW,
        "提取信号": json.dumps(_extracted_signals(text), ensure_ascii=False),
    }


def submit(text: str, reporter: str = "现场操作员", device: str = "APX-240",
           client: Client | None = None, kb: KnowledgeBase | None = None,
           paths: Paths | None = None, llm: Caller | None = None) -> Ticket:
    """报修入口：诊断 → 路由 → 发卡 → 写工单 → 记 AI 用量。"""
    client = client or Client()
    kb = kb or KnowledgeBase()
    paths = paths or Paths()
    if not text.strip():
        raise ValueError("现场描述为空，无法建单")

    d = diagnose(text, kb, llm=llm)
    wo_id = f"RX-{time.strftime('%Y%m%d')}-{client.next_seq():04d}"

    # 先落工单再发卡：卡片上要带「打开完整工单」的直达链接，得先拿到 record_id。
    # 顺序反过来更稳——工单是存档、卡片是通知，通知失败不该留下一条指向不存在记录的链接。
    fields = _wo_fields(wo_id, text, reporter, device, d)
    # 只有升级到专家的单才发确认码：双人确认得有第二个人，常规单从头到尾只有现场一个人，
    # 让他既写结论又确认"现场可行"，那就成了自问自答。
    code = _new_code() if d.stop_and_escalate.escalate_to_expert else ""
    if code:
        fields["专家码"] = code
    record_id = client.upsert_work_order(wo_id, fields)
    card = build_card(wo_id, d, reporter, device, client.wo_url(record_id),
                      client.followup_url(wo_id),
                      close_url=client.close_url(wo_id, code), expert_code=code)
    msg_id = client.send_card(card, to_experts=is_high_risk(d))
    _log_usage(wo_id, d, paths)
    return Ticket(wo_id=wo_id, diagnosis=d, high_risk=is_high_risk(d),
                  record_id=record_id, message_id=msg_id)


def missing_items(wo: dict[str, Any]) -> list[str]:
    """这张工单还缺哪几项。回填页面按它出题，顺序与卡片上 ⑦ 一致。"""
    try:
        items = json.loads(wo.get("需补充信息项") or "[]")
    except json.JSONDecodeError:
        items = []
    if isinstance(items, list) and items:
        return [str(x) for x in items]
    # 老工单没有这一列，只能拆顿号串。当前知识库的观测项都不含顿号，拆得开。
    return [x for x in (wo.get("需补充信息") or "").split("、") if x]


_NO_INFO = re.compile(
    r"(?:没|未|无法|没法|不能)(?:测|量|看|查|确认|记录|校|拆)(?:过|了)?"
    r"|不知道|不清楚|不了解|说不清|没数|没留意")


def _no_info(answer: str) -> bool:
    """答的是"这一项没取得"，不是这一项的内容。

    整句匹配：「没测过但端子发黑」带着内容，得按内容处理，不能当成没取得。
    """
    return bool(_NO_INFO.fullmatch(answer.strip("。．.；;，,、 　")))


def followup(wo_id: str, answers: dict[str, str], client: Client | None = None,
             kb: KnowledgeBase | None = None, paths: Paths | None = None,
             llm: Caller | None = None) -> Ticket:
    """第二轮：把现场补上来的信息并进原描述，重跑同一条诊断链，另建一张关联工单。

    补充信息只是让证据变多，不是绕过闸门的理由——安全闸门、四状态证据、输出自检、
    失效关闭一项不少，第二轮照样可能停机升级。

    有三类答案要区别对待：解析层读得出信号的（「加热电流 12.4A」）自然进入证据链；
    答了内容但没有对应信号的（「继电器触点发黑」）不许悄悄丢掉，也不许假装已经判定——
    从 ⑦ 里摘掉、在「提示」里写清本轮没有据此调整证据强度；答的是"这一项没取得"
    （「没测」「不知道」）则仍然缺，留在 ⑦ 里下一轮接着问。
    """
    client = client or Client()
    kb = kb or KnowledgeBase()
    paths = paths or Paths()
    wo = client.work_orders().get(wo_id)
    if not wo:
        raise KeyError(f"工单 {wo_id} 不存在，无法补充信息")

    items = missing_items(wo)
    filled = {k: str(v).strip() for k, v in answers.items() if k in items and str(v).strip()}
    if not filled:
        raise ValueError(
            f"没有一条补充对得上工单 {wo_id} 的待补项"
            + (f"：{'、'.join(items)}" if items else "（该单已无待补项）"))

    base = str(wo.get("现场描述", "")).rstrip("。；; ")
    merged = base + "。补充信息：" + "；".join(f"{k}：{v}" for k, v in filled.items())
    reporter = str(wo.get("报修人") or "现场操作员")
    device = str(wo.get("设备") or "APX-240")
    round_no = int(wo.get("诊断轮次") or 1) + 1

    d = diagnose(merged, kb, llm=llm)
    still = [k for k in filled if k in d.missing_info]
    declined = [k for k in still if _no_info(filled[k])]
    unread = [k for k in still if k not in declined]
    if unread:
        d.missing_info = [m for m in d.missing_info if m not in unread]
        d.unresolved_answers = unread
        d.warnings.append(
            "现场已回答，但解析层未能从中读出可判定的信号，本轮未据此调整证据强度："
            + "；".join(f"{k}「{filled[k]}」" for k in unread))
    if declined:
        d.warnings.append(
            "现场答复未取得该项，仍列在待补清单，补上才会进入证据链："
            + "；".join(f"{k}「{filled[k]}」" for k in declined))
    if still:
        # 改过就要重新过闸：发出去的一切都必须是自己刚通过自检的那一份。
        d.validation = validate(d, merged, kb)
        d = enforce(d, d.validation, merged)

    new_id = f"RX-{time.strftime('%Y%m%d')}-{client.next_seq():04d}"
    fields = _wo_fields(new_id, merged, reporter, device, d, round_no, wo_id)
    code = _new_code() if d.stop_and_escalate.escalate_to_expert else ""
    if code:
        fields["专家码"] = code
    record_id = client.upsert_work_order(new_id, fields)
    client.upsert_work_order(wo_id, {"后续工单": new_id})
    card = build_card(new_id, d, reporter, device, client.wo_url(record_id),
                      client.followup_url(new_id), round_no, wo_id,
                      close_url=client.close_url(new_id, code), expert_code=code)
    msg_id = ""
    try:
        msg_id = client.send_card(card, to_experts=is_high_risk(d))
    except Exception as exc:  # noqa: BLE001
        # 第二轮的请求人就是刚填完表的那个人，结论必须回得去；卡片只是通知群里。
        # 第一轮不一样：那条链的请求人本来就在群里，所以网页壳回落到纯诊断就够了。
        # 通知失败要如实记在工单上，不能因为群发不出去就把已经成立的诊断一起吞掉。
        client.upsert_work_order(
            new_id, {"推送状态": f"卡片未送达：{type(exc).__name__}: {exc}"})
    _log_usage(new_id, d, paths)
    return Ticket(wo_id=new_id, diagnosis=d, high_risk=is_high_risk(d),
                  record_id=record_id, message_id=msg_id, round=round_no, prev_wo=wo_id)


def _log_usage(wo_id: str, d: Diagnosis, paths: Paths) -> None:
    """AI 用量流水。仪表盘上的「AI 调用量」「知识库覆盖率」都从这里来。"""
    paths.var.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wo_id": wo_id,
        "alarm_code": d.alarm_code or "",
        "coverage": d.coverage,
        "engine": d.model_used,
        "calls": 1,
        "escalated": d.stop_and_escalate.escalate_to_expert,
        "degraded": d.degraded,
        "selfcheck_ok": bool(d.validation and d.validation.ok),
    }
    with paths.usage.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ 闭环与沉淀
#
# 两条路径，同一套准入规则：
#   命令行 close + promote —— 值班工程师代录，演示与批量回填时用；
#   网页 expert_close + field_confirm —— 专家与报修人各自从群里的卡片点进来，
#       用一次性确认码代替登录态，走完双人签字。
# 无论走哪条，候选案例在两人都签字之前都不进 cases_learned.json，也就不参与诊断。

# 现场确认的题目在服务端定义，页面不得自己编：这四问全是报修人肉眼能作证的事，
# 没有一问是"专家说得对不对"——那要他来答就成了假权威。
# 每项是 (键, 题干, 控件, 必填)；必填与否也是服务端说了算，页面只负责照着画。
EXPERT_QUESTIONS = [
    ("expert", "署名：谁下的这个技术结论（进案例备查）", "text", True),
    ("root_cause", "实际根因：写清物理原因，不要只写「已修复」", "text", True),
    ("action_taken", "处置动作：现场照着做完就能复机的那几步", "text", True),
    ("verification", "复机验证：读数与判据（例：封口温度 165°C 稳定 30 分钟）", "text", False),
]

FIELD_QUESTIONS = [
    ("reporter", "你是谁：现场确认人（署名进案例）", "text", True),
    ("executed", "是否照专家写的处置动作执行了？", "yesno", True),
    ("recovered", "设备是否已恢复正常生产？", "yesno", True),
    ("reading", "复机后的关键读数是多少？（写清单位，例：封口温度 165°C 稳定 30 分钟）", "text", True),
    ("discrepancy", "与专家结论不符时，现场实际是什么情况？（不符必填）", "text", False),
]


def _ask(questions: list[tuple[str, str, str, bool]],
         values: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """题目转成网页要的格式。values 只用于预填工单上已有的事实（如报修人姓名）。"""
    values = values or {}
    return [{"key": k, "label": label, "kind": kind, "required": req, "value": values.get(k, "")}
            for k, label, kind, req in questions]


_YES = {"是", "对", "已执行", "已恢复", "y", "yes", "true", "1"}


def _yes(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in _YES


def close(wo_id: str, root_cause: str, action_taken: str, verification: str,
          expert: str, client: Client | None = None,
          paths: Paths | None = None, status: str = STATUS_CLOSED) -> dict[str, Any]:
    """登记闭环并生成候选案例（待现场确认，暂不参与诊断）。

    status 由调用方决定：命令行代录停在「已闭环待沉淀」，网页专家提交后要接着
    走现场确认，停在「待现场确认」。候选案例的准入规则两条路径完全一致。
    """
    client = client or Client()
    paths = paths or Paths()
    wo = client.work_orders().get(wo_id)
    if not wo:
        raise KeyError(f"工单 {wo_id} 不存在")
    if not (root_cause.strip() and action_taken.strip() and expert.strip()):
        raise ValueError("实际根因、处置动作、确认专家三项均为必填，缺一不得沉淀")

    candidate = {
        "id": f"CAND-{wo_id}",
        "status": CAND_PENDING,
        "alarm_code": wo.get("报警码", ""),
        "phenomenon": wo.get("现场描述", ""),
        "match_signals": json.loads(wo.get("提取信号") or "{}"),
        "critical_signal": None,
        "root_cause": root_cause,
        "disposition": action_taken,
        "verification": verification,
        "authorization": expert,
        "source_work_order": wo_id,
        "confirmed_by": expert,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not candidate["alarm_code"]:
        candidate["status"] = "无法沉淀：缺报警码"

    _append_candidate(candidate, paths)
    client.upsert_work_order(wo_id, {
        "状态": status,
        "实际根因": root_cause,
        "处置动作": action_taken,
        "复机验证": verification,
        "确认专家": expert,
        "闭环时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "候选案例": candidate["id"],
        "沉淀状态": candidate["status"],
    })
    return candidate


def promote(wo_id: str, confirmer: str, client: Client | None = None,
            paths: Paths | None = None,
            field_facts: dict[str, str] | None = None) -> dict[str, Any]:
    """现场确认通过后把候选案例提升进 cases_learned.json，从此参与后续诊断。

    两个署名分开存：技术结论由专家负责（confirmed_by），现场可行性由报修人作证
    （field_confirmed_by）。只存一个名字，事后就查不出是谁说"这法子在现场管用"。
    """
    client = client or Client()
    paths = paths or Paths()
    items = _candidates(paths)
    cand = next((c for c in items if c.get("source_work_order") == wo_id), None)
    if cand is None:
        raise KeyError(f"工单 {wo_id} 没有待提升的候选案例")
    if cand.get("status") != CAND_PENDING:
        raise ValueError(f"候选案例状态为「{cand.get('status')}」，不可提升")
    if not cand.get("match_signals"):
        raise ValueError("候选案例没有可机器比对的信号特征，提升后无法参与匹配，须先补齐")
    if not str(confirmer).strip():
        raise ValueError("现场确认人不得为空：入库的案例必须追得到是谁作证现场可行")

    learned = _learned_cases(paths)
    case_id = f"L{len(learned) + 1:02d}"
    expert = str(cand.get("confirmed_by") or confirmer).strip()
    entry = {
        "id": case_id,
        "learned": True,
        "source_ref": f"KB:LEARNED:{case_id}",
        "alarm_code": cand["alarm_code"],
        "phenomenon": cand["phenomenon"],
        "match_signals": cand["match_signals"],
        "critical_signal": cand.get("critical_signal"),
        "root_cause": cand["root_cause"],
        "disposition": cand["disposition"],
        "authorization": cand.get("authorization") or expert,
        "source_work_order": wo_id,
        "confirmed_by": expert,
        "field_confirmed_by": str(confirmer).strip(),
        "field_verification": field_facts or {},
        "promoted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    learned.append(entry)
    paths.learned.parent.mkdir(parents=True, exist_ok=True)
    paths.learned.write_text(json.dumps({
        "_meta": {
            "说明": "本厂工单闭环后沉淀的现场案例，经专家提交＋现场确认双人签字后入库。"
                    "与说明书 §4 的 history_cases.json 分表存放。",
            "只读来源": "history_cases.json（设备厂商说明书，Agent 不得改写）",
        },
        "cases": learned,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 在同一个列表对象上改状态再写回；重新读一份会把这次更新丢掉
    cand["status"] = f"已提升为 {case_id}"
    _write_candidates(items, paths)
    client.upsert_work_order(wo_id, {"状态": STATUS_PROMOTED, "案例编号": case_id})
    return entry


def _notify(client: Client, card: dict[str, Any], wo_id: str, to_experts: bool) -> str:
    """发卡失败不该把已经成立的闭环动作一起吞掉，但要如实记在工单上。"""
    try:
        return client.send_card(card, to_experts=to_experts)
    except Exception as exc:  # noqa: BLE001
        client.upsert_work_order(wo_id, {"推送状态": f"卡片未送达：{type(exc).__name__}: {exc}"})
        return ""


def expert_close(wo_id: str, code: str, expert: str, root_cause: str, action_taken: str,
                 verification: str, client: Client | None = None,
                 paths: Paths | None = None) -> dict[str, Any]:
    """专家从群里的红卡点进来填写技术结论。

    校验一次性专家码 → 生成候选案例（仍不参与诊断）→ 发一张橙卡到报修群，
    把「现场是否可行」这一问交给报修人，链接里带现场码。
    """
    client = client or Client()
    paths = paths or Paths()
    wo = client.work_orders().get(wo_id)
    if not wo:
        raise KeyError(f"工单 {wo_id} 不存在")
    _check_code(wo, "专家码", code, "专家")

    cand = close(wo_id, root_cause, action_taken, verification, expert,
                 client=client, paths=paths, status=STATUS_AWAIT_FIELD)
    if cand["status"] != CAND_PENDING:
        # 缺报警码这类根本沉淀不了的单：闭环照记，但不该去要现场确认一个永远入不了库的案例。
        # 状态也不能停在「待现场确认」——不会再有这一步，挂着就是假待办。
        client.upsert_work_order(wo_id, {"状态": STATUS_CLOSED, "专家码": "", "现场码": ""})
        return {"stage": "expert", "passed": None, "promotable": False, "expert": expert,
                "candidate": cand, "case_id": "", "field_code": "", "message_id": ""}

    field_code = _new_code()
    client.upsert_work_order(wo_id, {"专家码": "", "现场码": field_code})
    msg_id = _notify(client, _field_card(wo_id, cand, expert, field_code,
                                         client.close_url(wo_id, field_code)),
                     wo_id, to_experts=False)
    return {"stage": "expert", "passed": None, "promotable": True, "expert": expert,
            "candidate": cand, "case_id": "", "field_code": field_code, "message_id": msg_id}


def field_confirm(wo_id: str, code: str, reporter: str, executed: Any, recovered: Any,
                  reading: str, discrepancy: str = "", client: Client | None = None,
                  paths: Paths | None = None) -> dict[str, Any]:
    """报修人确认专家的结论在现场是否可行。只认现场看得见的事。

    通过 → promote 入库，两个人的署名一起存；
    不通过 → 一个字都不写进知识库，分歧写回工单、退回专家群，并重发一个新的专家码。
    退回必须有下一步，否则这条链就断在"专家不知道该改什么"。
    """
    client = client or Client()
    paths = paths or Paths()
    wo = client.work_orders().get(wo_id)
    if not wo:
        raise KeyError(f"工单 {wo_id} 不存在")
    _check_code(wo, "现场码", code, "现场")
    items, cand = _pending_candidate(wo_id, paths)

    reporter = str(reporter).strip() or str(wo.get("报修人") or "").strip()
    if not reporter:
        raise ValueError("现场确认人不得为空：入库与退回都要追得到是谁作证的")

    done, back = _yes(executed), _yes(recovered)
    reading = str(reading).strip()
    discrepancy = str(discrepancy).strip()
    facts = {"照专家处置执行": "是" if done else "否",
             "设备恢复正常生产": "是" if back else "否",
             "复机关键读数": reading or "（未提供）"}
    stamp = "；".join(f"{k}＝{v}" for k, v in facts.items())
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    if done and back and reading:
        entry = promote(wo_id, reporter, client=client, paths=paths, field_facts=facts)
        # promote 改的是它自己重新读出的那一份候选；不回写本地这份，
        # 回执就会一边说「已入库」一边把状态还显示成「待现场确认」。
        cand["status"] = f"已提升为 {entry['id']}"
        client.upsert_work_order(wo_id, {"现场码": "", "现场确认人": reporter,
                                         "现场确认时间": now, "现场确认": stamp})
        msg_id = _notify(client, _promoted_card(wo_id, entry, reporter, facts),
                         wo_id, to_experts=True)
        return {"stage": "field", "passed": True, "promotable": True, "reporter": reporter,
                "facts": facts, "candidate": cand, "expert": cand.get("confirmed_by", ""),
                "case_id": entry["id"], "message_id": msg_id}

    if not discrepancy:
        raise ValueError("现场与专家结论不符时必须写清实际情况，否则专家无从修正")
    cand["status"] = CAND_REJECTED
    cand["field_rejected"] = {"by": reporter, "at": now,
                              "facts": facts, "discrepancy": discrepancy}
    _write_candidates(items, paths)

    new_code = _new_code()
    client.upsert_work_order(wo_id, {
        "状态": STATUS_REJECTED, "现场码": "", "专家码": new_code,
        "现场确认人": reporter, "现场确认时间": now, "现场确认": stamp,
        "现场分歧": discrepancy, "沉淀状态": CAND_REJECTED,
    })
    msg_id = _notify(client, _rejected_card(wo_id, cand, reporter, facts, discrepancy,
                                            now, new_code, client.close_url(wo_id, new_code)),
                     wo_id, to_experts=True)
    return {"stage": "field", "passed": False, "promotable": True, "reporter": reporter,
            "facts": facts, "candidate": cand, "expert": cand.get("confirmed_by", ""),
            "case_id": "", "expert_code": new_code, "message_id": msg_id}


def _pending_candidate(wo_id: str, paths: Paths) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """返回 (候选列表, 该单的候选案例)。同时返回列表是为了改完状态能原样写回。"""
    items = _candidates(paths)
    cand = next((c for c in items if c.get("source_work_order") == wo_id), None)
    if cand is None:
        raise KeyError(f"工单 {wo_id} 没有待现场确认的候选案例：专家还没提交处置结论")
    if cand.get("status") != CAND_PENDING:
        raise ValueError(f"候选案例状态为「{cand.get('status')}」，不再等现场确认")
    return items, cand


def close_form(wo_id: str, code: str, client: Client | None = None,
               paths: Paths | None = None) -> dict[str, Any]:
    """双人确认页的题目。专家页与现场页共用一个网址，靠一次性码分辨该问谁。

    题目、控件类型、必填与否全部由服务端给出：现场确认那一页只许问现场看得见的事，
    这个边界不能交给页面自己决定——页面一旦能改题目，"不问技术判断"就成了空话。
    """
    client = client or Client()
    paths = paths or Paths()
    wo = client.work_orders().get(wo_id)
    if not wo:
        raise KeyError(f"工单 {wo_id} 不存在")
    stage = _stage_of(wo, code)
    if not stage:
        _reject_unknown(wo, wo_id)

    form: dict[str, Any] = {
        "wo_id": wo_id,
        "stage": stage,
        "alarm_code": wo.get("报警码", ""),
        "device": wo.get("设备", ""),
        "text": wo.get("现场描述", ""),
        "reporter": wo.get("报修人", ""),
        "status": wo.get("状态", ""),
        "round": int(wo.get("诊断轮次") or 1),
        "selfcheck": wo.get("输出自检", ""),
        "risk": wo.get("风险等级", ""),
        "escalated": wo.get("是否升级专家") == "是",
    }
    if stage == "expert":
        form["questions"] = _ask(EXPERT_QUESTIONS)
        # 上一轮被现场退回时，把分歧摆在专家眼前：不看他就是在原地再交一份同样的结论。
        form["rejected"] = {"by": wo.get("现场确认人", ""), "at": wo.get("现场确认时间", ""),
                            "facts": wo.get("现场确认", ""), "discrepancy": wo.get("现场分歧", "")}
        if not form["rejected"]["discrepancy"]:
            form["rejected"] = {}
    else:
        _, cand = _pending_candidate(wo_id, paths)
        form.update({
            "expert": cand.get("confirmed_by", ""),
            "root_cause": cand.get("root_cause", ""),
            "disposition": cand.get("disposition", ""),
            "verification": cand.get("verification", ""),
            "questions": _ask(FIELD_QUESTIONS, {"reporter": wo.get("报修人", "")}),
        })
    return form


def close_submit(wo_id: str, code: str, answers: dict[str, Any], client: Client | None = None,
                 paths: Paths | None = None) -> dict[str, Any]:
    """双人确认页的提交。走哪一阶段由码决定，不信页面报上来的 stage。"""
    client = client or Client()
    paths = paths or Paths()
    wo = client.work_orders().get(wo_id)
    if not wo:
        raise KeyError(f"工单 {wo_id} 不存在")
    stage = _stage_of(wo, code)
    if not stage:
        _reject_unknown(wo, wo_id)
    a = {k: str(v or "").strip() for k, v in answers.items()}

    if stage == "expert":
        return expert_close(wo_id, code, a.get("expert", ""), a.get("root_cause", ""),
                            a.get("action_taken", ""), a.get("verification", ""),
                            client=client, paths=paths)
    return field_confirm(wo_id, code, a.get("reporter", ""), a.get("executed", ""),
                         a.get("recovered", ""), a.get("reading", ""), a.get("discrepancy", ""),
                         client=client, paths=paths)


# ------------------------------------------------------------------ 流程卡
#
# 三张流程卡的颜色表示流程走到哪一步，与诊断卡的红蓝（风险等级）不是一套语义。
# 内容照旧一条不省：专家写的根因、处置、验证逐条列全，不折叠成"见工单"。


def _field_card(wo_id: str, cand: dict[str, Any], expert: str, code: str,
                url: str) -> dict[str, Any]:
    entry = "点下方按钮" if url else f"打开网页入口 /?wo={wo_id}&k={code}"
    elements = [
        _div(f"🧑‍🔧 **专家结论已提交，等现场作证**｜专家 {_flat(expert)}"
             f"｜{_flat(cand.get('created_at', ''))}"),
        _div(f"**实际根因**｜{_flat(cand['root_cause'])}"),
        _div(f"**处置动作**｜{_flat(cand['disposition'])}"),
        _div(f"**专家记录的复机验证**｜{_flat(cand.get('verification') or '（未填写）')}"),
        _div("你只需回答现场看得见的事：照做了吗、恢复了吗、复机读数多少。"
             "**根因对不对由专家署名负责，不用你判断。**"),
        _div(f"确认通过 → 案例入库、从此参与同类故障诊断；不通过 → 一个字都不写进知识库，"
             f"你写的分歧会退回专家群。入口：{entry}（确认码 {code}，一次性）"),
        {"tag": "hr"},
    ]
    if url:
        elements.append({"tag": "action", "actions": [
            {"tag": "button", "type": "primary", "url": url,
             "text": {"tag": "plain_text", "content": "✅ 现场确认是否可行"}}]})
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content":
        f"工单 {wo_id}｜候选案例 {cand['id']}（{cand['status']}）｜入库前不参与诊断"}]})
    return _card(f"🟠 待现场确认｜{cand.get('alarm_code') or '无报警码'}｜{wo_id}",
                 "orange", elements)


def _promoted_card(wo_id: str, entry: dict[str, Any], reporter: str,
                   facts: dict[str, str]) -> dict[str, Any]:
    elements = [
        _div(f"📚 **现场已确认，案例入库**｜{entry['id']}｜报警码 {entry['alarm_code']}"),
        _div(f"**根因（专家 {_flat(entry['confirmed_by'])} 署名）**｜{_flat(entry['root_cause'])}"),
        _div(f"**处置**｜{_flat(entry['disposition'])}"),
        _div(f"**现场作证（{_flat(reporter)}）**｜"
             + "；".join(f"{k}＝{_flat(v)}" for k, v in facts.items())),
        _div(f"已写入 cases_learned.json，从此参与同类故障诊断：证据分量封顶「中」、"
             f"出处标 经验{entry['id']}、与说明书原文分表存放，可单独回滚。"),
        {"tag": "note", "elements": [{"tag": "plain_text", "content":
            f"工单 {wo_id}｜双人签字：技术结论 {entry['confirmed_by']}＋现场 {reporter}｜"
            f"{entry['promoted_at']}"}]},
    ]
    return _card(f"📚 已沉淀入库｜{entry['id']}｜{wo_id}", "green", elements)


def _rejected_card(wo_id: str, cand: dict[str, Any], reporter: str, facts: dict[str, str],
                   discrepancy: str, stamp: str, code: str, url: str) -> dict[str, Any]:
    entry = "点下方按钮" if url else f"打开网页入口 /?wo={wo_id}&k={code}"
    elements = [
        _div(f"↩️ **现场确认未通过，退回专家**｜现场 {_flat(reporter)}｜{_flat(stamp)}"),
        _div("**现场的答复**｜" + "；".join(f"{k}＝{_flat(v)}" for k, v in facts.items())),
        _div(f"**现场实际情况**｜{_flat(discrepancy)}"),
        _div(f"**专家原结论（未入库）**｜根因：{_flat(cand['root_cause'])}"
             f"｜处置：{_flat(cand['disposition'])}"
             f"｜专家记录的复机验证：{_flat(cand.get('verification') or '（未填写）')}"),
        _div(f"知识库一个字都没写：候选案例状态已改为「{CAND_REJECTED}」。"
             f"读数对不上就优先解释差异，别沿用旧结论——修正后重新提交：{entry}"
             f"（确认码 {code}，一次性）"),
        {"tag": "hr"},
    ]
    if url:
        elements.append({"tag": "action", "actions": [
            {"tag": "button", "type": "primary", "url": url,
             "text": {"tag": "plain_text", "content": "🧑‍🔧 修正结论后重新提交"}}]})
    elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content":
        f"工单 {wo_id}｜候选案例 {cand['id']}｜状态 {cand['status']}"}]})
    return _card(f"↩️ 现场确认未通过｜{cand.get('alarm_code') or '无报警码'}｜{wo_id}",
                 "red", elements)


def _candidates(paths: Paths) -> list[dict[str, Any]]:
    if not paths.candidates.exists():
        return []
    return json.loads(paths.candidates.read_text(encoding="utf-8")).get("candidates", [])


def _write_candidates(items: list[dict[str, Any]], paths: Paths) -> None:
    paths.candidates.parent.mkdir(parents=True, exist_ok=True)
    paths.candidates.write_text(json.dumps({"candidates": items}, ensure_ascii=False, indent=2),
                                encoding="utf-8")


def _append_candidate(cand: dict[str, Any], paths: Paths) -> None:
    items = [c for c in _candidates(paths) if c.get("id") != cand["id"]]
    items.append(cand)
    _write_candidates(items, paths)


def _learned_cases(paths: Paths) -> list[dict[str, Any]]:
    if not paths.learned.exists():
        return []
    return json.loads(paths.learned.read_text(encoding="utf-8")).get("cases", [])
