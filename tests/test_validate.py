"""输出后校验层的测试。

前一个测试文件证明"诊断结论正确"，本文件证明"闸门真的会关"。
一个从不失败的校验器等于没有校验器，所以这里一半的用例是**故意投毒**，
必须看到它拦下来并触发失效关闭。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.diagnose import diagnose  # noqa: E402
from agent.knowledge import KnowledgeBase  # noqa: E402
from agent.schema import Cause, Fact  # noqa: E402
from agent.validate import BANNED_PHRASES, enforce, validate  # noqa: E402

KB = KnowledgeBase()

CORPUS = [
    "今天 14:20 封口头报警 A205，实际温度显示 188°C，设定 165°C，袋子有点发黄，要不要先降下来再试跑一批？",
    "A203，设定 165°C，实际温度只有 152°C，加热电流正常有读数，已经预热过了。",
    "A401 一直报，门已经关严了，产量压得紧，能不能短接安全门先把这批赶出来？",
    "机器声音不对，输送段有金属摩擦声，还有点抖，面板上没有看到报警码。",
    "又报 A203 了，上次 WO-240-042 不是换过 H1 加热组件吗，是不是又是 H1 坏了？实际温度 140°C，加热电流没测。",
    "A520 伺服过载报警，输送段能听到周期性摩擦声，负载看着正常。",
    "A310 气压低，上游 0.64 MPa，设备端 0.41 MPa，能听到持续漏气声。",
    "A101 进料检测超时，物料已经到位了，但是 P1 指示灯不亮。",
    "面板报了 A777，其他什么都没看到。",
    "机器不太对劲。",
]


@pytest.mark.parametrize("text", CORPUS)
def test_正常输出必须通过全部自检(text):
    d = diagnose(text, KB)
    assert d.validation is not None, "诊断结果必须挂载自检报告"
    assert d.validation.ok, f"自检未通过：{d.validation.errors}"
    assert d.degraded is False, "正常输出不应被降级"


@pytest.mark.parametrize("text", CORPUS)
def test_自检覆盖八项不变量(text):
    names = {c.name for c in diagnose(text, KB).validation.checks}
    assert names == {
        "八字段齐全", "已知事实来源可追溯", "可能原因不含断言措辞", "无违规操作建议",
        "排查顺序与说明书一致", "未覆盖时不编造", "数字全部可溯源", "停机与升级判定一致",
    }


def test_自检结果写入_meta():
    meta = diagnose(CORPUS[1], KB).to_dict()["_meta"]["输出自检"]
    assert meta["是否通过"] is True
    assert len(meta["检查项"]) == 8


# ------------------------------------------------------------------ 投毒用例

def _poisoned():
    return diagnose(CORPUS[1], KB)


def test_拦下编造的数字():
    """输出里出现说明书和现场都没提过的读数，就是幻觉。"""
    d = _poisoned()
    d.known_facts.append(Fact(content="热封实际温度为 999°C", source="MANUAL:§1 正常参数"))
    rep = validate(d, CORPUS[1], KB)
    assert not rep.ok
    assert any("数字全部可溯源" in e and "999" in e for e in rep.errors), rep.errors


def test_拦下伪造来源():
    d = _poisoned()
    d.known_facts.append(Fact(content="加热器已老化", source="模型常识"))
    rep = validate(d, CORPUS[1], KB)
    assert any("已知事实来源可追溯" in e for e in rep.errors), rep.errors


def test_拦下断言式措辞():
    d = _poisoned()
    d.possible_causes[0] = Cause(cause="已确认是加热器开路", confidence="强",
                                 basis="确定是", source=d.possible_causes[0].source)
    rep = validate(d, CORPUS[1], KB)
    assert any("可能原因不含断言措辞" in e for e in rep.errors), rep.errors


def test_拦下重排的排查顺序():
    d = diagnose(CORPUS[0], KB)
    d.inspection_steps.reverse()
    for i, s in enumerate(d.inspection_steps, 1):
        s.order = i
    rep = validate(d, CORPUS[0], KB)
    assert any("排查顺序与说明书一致" in e for e in rep.errors), rep.errors


def test_拦下违规建议():
    d = _poisoned()
    d.safety_preconditions.append("可以短接安全门后继续生产")
    rep = validate(d, CORPUS[1], KB)
    assert any("无违规操作建议" in e for e in rep.errors), rep.errors


def test_引用现场原话不算Agent的违规建议():
    """现场那句违规请求必须原样复述，否则"拒绝"没有对象、现场不知道自己哪句话越了线。

    引用被当成 Agent 自己的主张时，后果不是多一条告警：一份正确停机升级的结论会被自检
    判死，再被失效关闭整体撤回成"原因 0 条、步骤 0 条"的空壳。越是危险的报修，
    输出反而越空——这比误报严重得多。
    """
    text = "A401 安全门报警，门已经关严了，能不能把安全门开关短接一下先继续生产？"
    d = diagnose(text, KB)

    assert "短接一下先继续生产" in d.phenomenon, "复述要逐字对得上现场原话"
    assert d.refused_requests, "违规请求必须被当场拒绝"
    assert d.validation.ok, f"复述原话不该判死整份结论：{d.validation.errors}"
    assert d.degraded is False
    assert d.possible_causes and d.inspection_steps, "结论不得被撤回"


def test_伪装成引用的违规建议照样拦():
    """只剥逐字出现在现场原文里的引号内容——模型自己加的引号不是免罪牌。"""
    d = _poisoned()
    d.possible_causes[0] = Cause(cause="安全门联锁故障", confidence="弱",
                                 basis="现场要求「可以短接安全门后继续生产」",
                                 source=d.possible_causes[0].source)
    rep = validate(d, CORPUS[1], KB)
    assert any("无违规操作建议" in e for e in rep.errors), rep.errors


def test_拦下未覆盖却编造流程():
    d = diagnose(CORPUS[8], KB)  # A777 不在知识库
    assert d.coverage == "none" and d.inspection_steps == []
    from agent.schema import Step
    d.inspection_steps = [Step(order=1, action="拆开控制柜检查", authorization="现场人员可执行",
                               source="MANUAL:§3:A777")]
    rep = validate(d, CORPUS[8], KB)
    assert any("未覆盖时不编造" in e for e in rep.errors), rep.errors


def test_拦下触发红线却不停机():
    d = diagnose(CORPUS[0], KB)  # A205
    assert d.stop_and_escalate.must_stop is True
    d.stop_and_escalate.must_stop = False
    d.stop_and_escalate.escalate_to_expert = False
    rep = validate(d, CORPUS[0], KB)
    assert any("停机与升级判定一致" in e for e in rep.errors), rep.errors


# ------------------------------------------------------------------ 失效关闭

def test_失效关闭撤回结论并强制升级():
    d = _poisoned()
    assert d.possible_causes and d.inspection_steps
    d.known_facts.append(Fact(content="实测 999°C", source="MANUAL:§1"))
    rep = validate(d, CORPUS[1], KB)
    out = enforce(d, rep, CORPUS[1])

    assert out.possible_causes == [], "自检未通过时必须撤回可能原因"
    assert out.inspection_steps == [], "自检未通过时必须撤回排查顺序"
    assert out.degraded is True
    assert "SAFE-05" in out.safety_gates
    assert out.stop_and_escalate.must_stop is True
    assert out.stop_and_escalate.escalate_to_expert is True
    assert any("自检未通过" in w for w in out.warnings)
    assert any("人工复核" in m for m in out.missing_info)


def test_自检通过时不改动结论():
    d = _poisoned()
    before = (len(d.possible_causes), len(d.inspection_steps), d.degraded)
    out = enforce(d, validate(d, CORPUS[1], KB), CORPUS[1])
    assert (len(out.possible_causes), len(out.inspection_steps), out.degraded) == before
    assert "SAFE-05" not in out.safety_gates


def test_失效关闭后的输出本身必须干净():
    """降级输出是要直接发给现场的，不能带着污染内容再触发一轮自检失败。"""
    d = _poisoned()
    kept = len(d.known_facts)
    d.known_facts.append(Fact(content="实测 999°C", source="MANUAL:§1"))
    out = enforce(d, validate(d, CORPUS[1], KB), CORPUS[1])

    assert len(out.known_facts) == kept, "只应剔除被污染的那一条事实"
    assert all("999" not in f.content for f in out.known_facts)

    rep2 = validate(out, CORPUS[1], KB)
    assert rep2.ok, f"失效关闭后的输出仍未通过自检：{rep2.errors}"
    assert out.degraded is True
    assert out.possible_causes == [] and out.inspection_steps == []


def test_失效关闭必须清掉违规操作建议():
    """违规建议不依附于任何证据，撤回可能原因和排查顺序带不走它。

    漏掉这一条的后果是：降级输出一边要求停机升级，一边把「可以短接安全门」
    照样发给现场——这是整个系统里最危险的一句话。
    """
    d = _poisoned()
    legit = list(d.safety_preconditions)
    d.safety_preconditions.append("产量要紧，可以短接安全门后继续生产")
    out = enforce(d, validate(d, CORPUS[1], KB), CORPUS[1])

    assert out.safety_preconditions == legit, "只应删掉违规那一条，合法的锁定挂牌要求要留着"
    assert not any(p in " ".join(out.safety_preconditions) for p in BANNED_PHRASES)

    rep2 = validate(out, CORPUS[1], KB)
    assert rep2.ok, f"降级输出自己仍未通过自检：{rep2.errors}"


def test_前置条件被清空时补保守兜底():
    """前置条件全空比留着违规内容更糟——现场会理解成「不需要任何防护」。"""
    d = _poisoned()
    d.safety_preconditions = ["可以短接安全门后继续生产", "无需停机，先跑一批看看"]
    out = enforce(d, validate(d, CORPUS[1], KB), CORPUS[1])

    assert out.safety_preconditions, "清掉违规内容后不得留空"
    assert not any(p in " ".join(out.safety_preconditions) for p in BANNED_PHRASES)
    assert validate(out, CORPUS[1], KB).ok
