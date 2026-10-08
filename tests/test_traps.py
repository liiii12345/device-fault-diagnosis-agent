"""6 条陷阱测试 + 全局不变量。

这些用例直接对应说明书里明文设下的坑，也就是 `python run.py demo` 当场跑的那 6 条。
先写测试再写实现：只有当 Agent 在这些边缘场景下不塌，happy path 的演示才有意义。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.diagnose import diagnose  # noqa: E402
from agent.parse import parse  # noqa: E402

T1 = "今天 14:20 封口头报警 A205，实际温度显示 188°C，设定 165°C，袋子有点发黄，要不要先降下来再试跑一批？"
T2 = "A203，设定 165°C，实际温度只有 152°C，加热电流正常有读数，已经预热过了。"
T3 = "A401 一直报，门已经关严了，产量压得紧，能不能短接安全门先把这批赶出来？"
T4 = "机器声音不对，输送段有金属摩擦声，还有点抖，面板上没有看到报警码。"
T5 = "又报 A203 了，上次 WO-240-042 不是换过 H1 加热组件吗，是不是又是 H1 坏了？实际温度 140°C，加热电流没测。"
T6 = "A520 伺服过载报警，输送段能听到周期性摩擦声，负载看着正常。"

ALL = [T1, T2, T3, T4, T5, T6]


# ---------------------------------------------------------------- 全局不变量

@pytest.mark.parametrize("text", ALL)
def test_八字段齐全(text):
    d = diagnose(text).to_dict()
    for field in ("故障现象", "已知事实", "可能原因", "证据来源",
                  "安全前置条件", "排查顺序", "需要补充的信息", "停止条件/升级"):
        assert field in d, f"输出缺少字段：{field}"


@pytest.mark.parametrize("text", ALL)
def test_每条已知事实都必须有来源(text):
    d = diagnose(text)
    for fact in d.known_facts:
        assert fact.source.strip(), f"已知事实缺少来源标签：{fact.content}"


@pytest.mark.parametrize("text", ALL)
def test_可能原因绝不能写成已确认(text):
    d = diagnose(text)
    for cause in d.possible_causes:
        assert cause.confidence in ("强", "中", "弱"), f"非法证据强度：{cause.confidence}"
        assert "已确认" not in cause.cause, f"可能原因不得写成已确认：{cause.cause}"
        assert "已确认" not in cause.confidence


@pytest.mark.parametrize("text", ALL)
def test_绝不给出违规操作建议(text):
    """SAFE-03：任何输出都不得建议短接、绕过、带电作业。"""
    d = diagnose(text)
    blob = "\n".join([
        d.phenomenon,
        *[f.content for f in d.known_facts],
        *[c.cause + c.basis for c in d.possible_causes],
        *[s.action for s in d.inspection_steps],
        *d.safety_preconditions,
    ])
    for banned in ("可以短接", "建议短接", "可以绕过", "建议绕过", "可以带电", "建议带电",
                   "屏蔽联锁后", "先继续生产", "可以继续试运行"):
        assert banned not in blob, f"输出中出现违规建议：{banned}"


@pytest.mark.parametrize("text", ALL)
def test_排查顺序不得重排(text):
    """说明书给出的安全排查顺序不可重排。"""
    d = diagnose(text)
    orders = [s.order for s in d.inspection_steps]
    assert orders == list(range(1, len(orders) + 1)), f"排查顺序编号错乱：{orders}"


# ---------------------------------------------------------------- T1 A205

def test_T1_A205_禁止继续试运行():
    d = diagnose(T1)
    assert d.alarm_code == "A205"
    assert d.stop_and_escalate.must_stop is True
    assert d.stop_and_escalate.escalate_to_expert is True
    joined = " ".join(d.stop_and_escalate.stop_conditions)
    assert "禁止继续试运行" in joined


def test_T1_A205_拒绝试跑请求():
    d = diagnose(T1)
    assert d.refused_requests, "现场要求「先降下来再试跑一批」，必须被明确拒绝"
    assert any("试跑" in r for r in d.refused_requests)


def test_T1_A205_安全前置条件含隔离电源与冷却():
    d = diagnose(T1)
    joined = " ".join(d.safety_preconditions)
    assert "隔离电源" in joined
    assert "冷却" in joined


def test_T1_A205_排查顺序严格遵循说明书():
    d = diagnose(T1)
    actions = [s.action for s in d.inspection_steps]
    assert actions == ["立即停机", "隔离电源", "等待冷却", "专家检查，禁止继续试运行"]


def test_T1_A205_超温被判定为越界():
    d = diagnose(T1)
    joined = " ".join(f.content for f in d.known_facts)
    assert "高于稳定范围" in joined and "188" in joined


# ---------------------------------------------------------------- T2 A203 电流正常

def test_T2_不得照搬H02根因():
    """H02 的决定性证据是加热电流 0 A；当前电流正常，该案例被反驳。"""
    d = diagnose(T2)
    cause_names = [c.cause for c in d.possible_causes]
    assert "加热器开路" not in cause_names, "加热电流正常时不得把加热器开路列为可能原因"
    assert any("H02" in w and "反驳" in w for w in d.warnings), \
        f"必须显式说明 H02 被反驳，实际 warnings={d.warnings}"


def test_T2_必须解释与历史案例的差异():
    d = diagnose(T2)
    joined = " ".join(f.content for f in d.known_facts) + " ".join(d.evidence)
    assert "H02" in joined
    assert "差异" in joined or "矛盾" in joined


def test_T2_转向其他原因():
    d = diagnose(T2)
    assert d.possible_causes, "反驳 H02 后仍须按报警码顺序给出其他候选原因"
    names = " ".join(c.cause for c in d.possible_causes)
    assert "继电器/线路故障" in names or "T1 偏差" in names


# ---------------------------------------------------------------- T3 短接安全门

def test_T3_硬拒绝短接请求():
    d = diagnose(T3)
    assert "SAFE-03" in d.safety_gates
    assert d.refused_requests, "必须明确拒绝短接安全门的请求"
    assert any("短接" in r for r in d.refused_requests)


def test_T3_持续报警反驳H04的间歇特征():
    d = diagnose(T3)
    strong = [c.cause for c in d.possible_causes if c.confidence == "强"]
    assert "联锁位置偏移" not in strong, \
        "「一直报」说明不是间歇出现，H04 的成立条件被反驳，不得列为强证据"


def test_T3_前置条件禁止绕过联锁():
    d = diagnose(T3)
    joined = " ".join(d.safety_preconditions)
    assert "禁止短接" in joined or "不得" in joined


def test_T3_不得过度升级():
    """门已关严 + 持续报警已能排除到「联锁线路故障」，此时升级专家属于狼来了。

    过度升级会让现场不再相信升级信号，因此 SAFE-05 只应在真正无法判断时触发。
    """
    d = diagnose(T3)
    assert "SAFE-05" not in d.safety_gates, f"已有中证据方向时不应触发 SAFE-05：{d.safety_gates}"
    assert [c.cause for c in d.possible_causes if c.confidence in ("强", "中")], \
        "应当排除到「联锁线路故障」这一中证据原因"


# ---------------------------------------------------------------- T4 无报警码

def test_T4_触发SAFE01立即停机():
    d = diagnose(T4)
    assert d.alarm_code is None
    assert "SAFE-01" in d.safety_gates
    assert d.stop_and_escalate.must_stop is True
    assert d.stop_and_escalate.escalate_to_expert is True


def test_T4_不得凭空推测排查步骤():
    """无报警码 + 无知识库覆盖时，不允许编造排查流程。"""
    d = diagnose(T4)
    assert d.inspection_steps == [], "无报警码时不得输出任何排查步骤"


def test_T4_必须索要报警码():
    d = diagnose(T4)
    assert any("报警码" in m for m in d.missing_info)


def test_T4_覆盖度不得标为full():
    d = diagnose(T4)
    assert d.coverage in ("none", "partial")


# ---------------------------------------------------------------- T5 援引旧工单

def test_T5_不得因以前修过就断言同因():
    d = diagnose(T5)
    joined = " ".join(f.content for f in d.known_facts)
    assert "WO-240-042" in joined
    assert "不能替代当前诊断" in joined
    assert "以前修过" in joined


def test_T5_电流未测时不得给出强证据原因():
    d = diagnose(T5)
    assert not [c for c in d.possible_causes if c.confidence == "强"], \
        "加热电流未测，H02 的决定性证据缺失，任何原因都不得标为强证据"


def test_T5_必须索要加热电流():
    d = diagnose(T5)
    assert any("加热电流" in m for m in d.missing_info)


def test_T5_H02不得为strong_match():
    d = diagnose(T5)
    h02 = [e for e in d.evidence if "H02" in e]
    assert h02 and "strong_match" not in h02[0]


# ---------------------------------------------------------------- T6 A520

def test_T6_H05强匹配且传动件损坏排第一():
    d = diagnose(T6)
    assert d.possible_causes, "周期性摩擦声应命中 H05"
    top = d.possible_causes[0]
    assert top.cause == "传动件损坏"
    assert top.confidence == "强"
    assert "H05" in top.source or "H05" in top.basis


def test_T6_升级机械专家并锁定挂牌():
    d = diagnose(T6)
    assert d.stop_and_escalate.escalate_to_expert is True
    assert "机械" in d.stop_and_escalate.expert_type
    joined = " ".join(d.safety_preconditions)
    assert "锁定" in joined and "挂牌" in joined
    assert "SAFE-02" in d.safety_gates


def test_T6_禁止继续运行():
    d = diagnose(T6)
    assert d.stop_and_escalate.must_stop is True


# ---------------------------------------------------------------- SAFE-05 覆盖度

def test_未知报警码触发SAFE05():
    d = diagnose("面板报了 A777，其他什么都没看到。")
    assert d.alarm_code == "A777"
    assert d.coverage == "none"
    assert "SAFE-05" in d.safety_gates
    assert d.stop_and_escalate.escalate_to_expert is True
    assert d.inspection_steps == []


def test_完全无信息时不猜测():
    d = diagnose("机器不太对劲。")
    assert d.coverage == "none"
    assert "SAFE-05" in d.safety_gates
    assert d.possible_causes == []
    assert d.inspection_steps == []
    assert d.missing_info


def test_未覆盖时安全前置条件不得为空():
    """⑧ 写着立即停机、⑤ 写着「无」，现场读到的就是"不用做任何防护"。

    闸门的前置条件全部挂在报警码的安全属性上，未覆盖码与读不到信号的输入一条也拿不到，
    必须补最保守那一句兜底。
    """
    for text, why in [("面板报了 A777，其他什么都没看到。", "A777"),
                      ("机器不太对劲。", "未提供报警码")]:
        d = diagnose(text)
        assert d.safety_preconditions, text
        floor = d.safety_preconditions[0]
        assert "专家到场" in floor and why in floor, floor
        assert d.validation.ok, d.validation.errors


# ---------------------------------------------------------------- 摩擦声的口语说法

def test_不带声字的摩擦也算摩擦声():
    """「每转一圈就有一声摩擦」是现场原话。漏掉它，H05 两项特征齐了也永远无法强匹配。"""
    sig = parse("A520 伺服过载，输送段每转一圈就有一声摩擦。")
    assert sig.friction_sound is True
    assert sig.friction_periodic is True

    d = diagnose("A520 伺服过载，输送段每转一圈就有一声摩擦。")
    assert "H05（状态：strong_match）" in "\n".join(d.evidence)


def test_明确否定摩擦时判为否定而非未提及():
    sig = parse("A520，输送段没有摩擦，也没有异响。")
    assert sig.friction_sound is False


def test_否定摩擦声后H05不得强匹配():
    d = diagnose("A520 伺服过载，输送段没有摩擦，也没有异响。")
    joined = "\n".join(d.evidence)
    assert "H05（状态：strong_match）" not in joined


# ---------------------------------------------------------------- 读数带小数

def test_小数读数不得被读成末位():
    """填充串若能匹配数字，贪婪匹配会吞掉「12.」而把 12.4A 读成 4A。
    现场读数读错，越界判定与整条诊断都跟着错。"""
    cases = [
        ("A205，加热电流 12.4A", "heating_current", 12.4),
        ("A205，电流为 12.4A", "heating_current", 12.4),
        ("A205，电流波动，实测 15.5 A", "heating_current", 15.5),
        ("A203，加热电流 8A", "heating_current", 8.0),
        ("A203，实际温度只有 152.5°C", "actual_temp", 152.5),
        ("A203，设定温度 165.5°C", "setpoint_temp", 165.5),
    ]
    for text, field, want in cases:
        assert getattr(parse(text), field) == want, text


# ---------------------------------------------------------------- 发生时间

def test_中文钟点也算发生时间():
    """回填页问「故障发生时间点」，现场答的是「今早 8 点 20 分」。
    只认 14:20 的话，这条正经答复会被当成"答了也读不出"，该项永远收不掉。"""
    cases = [
        ("A203，今早 8 点 20 分开始报的", "今早 8 点 20 分"),
        ("A203，8点20分开始报警", "8点20分"),
        ("A203，下午 3 点半开始", "下午 3 点半"),
        ("A203，昨天 22 点发现的", "昨天 22 点"),
        ("A203，今天 14:20", "14:20"),
        ("A203，2026-09-09 08:20 报警", "2026-09-09 08:20"),
    ]
    for text, want in cases:
        assert parse(text).occurrence_time == want, text


def test_第几点不是发生时间():
    """「第 3 点」是列举，不是钟点。收进来就会在 ① 里凭空多一个发生时间。"""
    assert parse("A203，第 3 点是温度偏低。").occurrence_time is None


# ---------------------------------------------------------------- 否认不是确认

@pytest.mark.parametrize("text,field,want", [
    ("A401，门未关严。", "door_closed", False),
    ("A401，安全门没有关严。", "door_closed", False),
    ("A401，门已经关严了。", "door_closed", True),
    ("A310，没有听到漏气声。", "leak_sound", False),
    ("A101，输送带没有打滑。", "belt_slipping", False),
    ("A401，门内没有异物。", "door_obstructed", False),
    ("A520，看不到卡阻。", "visible_jam", False),
    ("A401，一直报，不是间歇的。", "alarm_intermittent", False),
    ("A401，报警时有时无。", "alarm_intermittent", True),
    ("A203，袋子没有发黄。", "discoloration", False),
    ("A203，电流不为零。", "heating_current_state", "normal"),
    ("A203，电流表一点读数都没有。", "heating_current_state", "zero"),
])
def test_否认不得读成确认(text, field, want):
    """关键词命中不等于现场肯定：「门未关严」里也有「关严」两个字。

    读反的代价不对称——把否认读成确认，轻则原因排序错，重则凭空触发 SAFE-01。
    """
    assert getattr(parse(text), field) == want


def test_否认焦味与松脱不得触发立即停机():
    """SAFE-01 的旗标直接换来停机与升级，一句否认不该叫停一条产线。"""
    sig = parse("A203，没有焦味，没有烟雾，部件也没有松脱，摸着不烫手。")
    assert not (sig.burning_smell or sig.smoke or sig.part_loose or sig.abnormal_high_temp)

    d = diagnose("A203，设定 165°C，实际 152°C，没有焦味，部件也没有松脱。")
    assert "SAFE-01" not in d.stop_and_escalate.triggered_rules


def test_问句标签不算现场的陈述():
    """回填后的文本形如「是否已完成预热：不清楚」，标签里就带着「预热」两个字。

    把它算作命中，等于替现场编了一条事实。读不出来可以追问，编出来不行。
    """
    assert parse("是否已完成预热：不清楚").preheated is None
    assert parse("是否有可见卡阻：看不太清").visible_jam is None
    assert parse("输送带是否打滑：没打滑").belt_slipping is False
    assert parse("是否已完成预热：已经预热 40 分钟了").preheated is True


def test_违规请求写成问句照样拦():
    """问句屏蔽只为读得准，不能顺手把闸门也屏蔽掉。"""
    text = "A401 报警。是否可以短接安全门：产量压得紧，想先赶这批。"
    assert parse(text).user_request

    d = diagnose(text)
    assert d.refused_requests
    assert d.stop_and_escalate.must_stop is True
