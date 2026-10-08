"""模型兜底解析层的测试。

这一层的风险不在"能不能抽出来"，而在"抽错了会不会混进诊断"。所以用例的重心是
**拒绝路径**：编造的读数、不存在的原文片段、与正则冲突的值、被刻意扣留的字段——
每一条都必须被挡在 Signals 之外。

同时验证一件更基础的事：未配置模型时，诊断行为与纯规则版本完全一致。
兜底层是加法，不是替换。
"""

from __future__ import annotations

import json

import pytest

from agent.diagnose import diagnose
from agent.knowledge import KnowledgeBase
from agent.llm_parse import Caller, LLMConfig, LLMError, augment
from agent.parse import cn_to_number, numbers_in, parse


def fake(payload: dict) -> Caller:
    """造一个只回固定 JSON 的假模型，测试不联网。"""
    body = json.dumps(payload, ensure_ascii=False)
    return lambda system, user: body


def fenced(payload: dict) -> Caller:
    """模型常把 JSON 包在 ```json 围栏里，还爱在前后加一句话。"""
    body = json.dumps(payload, ensure_ascii=False)
    return lambda system, user: f"好的，抽取结果如下：\n```json\n{body}\n```\n以上。"


def boom(system: str, user: str) -> str:
    raise LLMError("HTTP 401")


KB = KnowledgeBase()


# ------------------------------------------------------------------ 中文数字

@pytest.mark.parametrize("text,expect", [
    ("十五", 15.0),
    ("一百五十二", 152.0),
    ("一百六十五", 165.0),
    ("两百", 200.0),
    ("一百六十", 160.0),
    ("三千", 3000.0),
])
def test_中文数字转换(text, expect):
    assert cn_to_number(text) == expect


@pytest.mark.parametrize("text", ["一九五", "三", "零", "五"])
def test_不含单位字的中文数字不认(text):
    """逐字念法（「一九五」）不是读数，认错会把 5 当成温度写进诊断。"""
    assert cn_to_number(text) is None


def test_口述中文数字能被规则层解析且通过自检():
    text = "A203 报警，设定温度一百六十五度，实际温度只有一百五十二度，加热电流正常有读数。"
    sig = parse(text)
    assert sig.setpoint_temp == 165.0
    assert sig.actual_temp == 152.0
    d = diagnose(text, KB)
    assert d.validation.ok, [c.detail for c in d.validation.checks if not c.ok]
    assert not d.degraded
    # 出处仍指向现场原话，而不是转写后的数字
    assert any("一百五十二" in f.source for f in d.known_facts)


def test_中文数字与阿拉伯数字归一化后同口径():
    assert numbers_in("一百五十二度") == numbers_in("152.0°C") == {"152"}


# ------------------------------------------------------------------ 补齐

# 正则的焦味词表是 焦味/烧焦/糊味/焦糊，烟雾词表是 烟雾/冒烟/有烟；
# 这句刻意绕开全部枚举，只有模型能读出来。
COLLOQUIAL = "封口头有一股塑料烧糊的味儿，还往外冒热气，我不敢靠近。"


def test_正则确实读不出这句口语():
    """先确认前提成立，否则下面那条"模型补齐"的测试是空的。"""
    sig = parse(COLLOQUIAL)
    assert sig.burning_smell is False
    assert sig.smoke is False


def test_模型补齐的立即停机征兆会触发强制停机():
    sig = parse(COLLOQUIAL)
    rep = augment(sig, fake({"fields": {
        "burning_smell": {"value": True, "span": "塑料烧糊的味儿"},
        "smoke": {"value": True, "span": "往外冒热气"},
    }}))
    assert sig.burning_smell is True
    assert sig.smoke is True
    assert len(rep.filled) == 2

    d = diagnose(COLLOQUIAL, KB, llm=fake({"fields": {
        "burning_smell": {"value": True, "span": "塑料烧糊的味儿"},
        "smoke": {"value": True, "span": "往外冒热气"},
    }}))
    assert "SAFE-01" in d.safety_gates
    assert d.stop_and_escalate.must_stop
    assert d.stop_and_escalate.escalate_to_expert
    assert d.parse_assisted
    assert "llm-parse" in d.model_used
    assert d.validation.ok, [c.detail for c in d.validation.checks if not c.ok]


def test_围栏包裹的返回也能解析():
    sig = parse(COLLOQUIAL)
    augment(sig, fenced({"fields": {"burning_smell": {"value": True, "span": "塑料烧糊的味儿"}}}))
    assert sig.burning_smell is True


# ------------------------------------------------------------------ 拒绝路径

def test_拒绝编造的读数():
    """span 是真的，但数值算不出来——这是模型最典型的幻觉形态。"""
    sig = parse("A203，实际温度不太够，封不太严。")
    rep = augment(sig, fake({"fields": {"actual_temp": {"value": 148, "span": "实际温度不太够"}}}))
    assert sig.actual_temp is None
    assert not rep.filled
    assert any("无法从所引原文片段" in r for r in rep.rejected)


def test_拒绝不存在的原文片段():
    sig = parse("A203，温度偏低。")
    rep = augment(sig, fake({"fields": {
        "actual_temp": {"value": 152, "span": "现场实测一百五十二度"},
    }}))
    assert sig.actual_temp is None
    assert any("不在现场描述中" in r for r in rep.rejected)


def test_正则已抽到的字段不被模型覆盖():
    """正则命中原文，可复核；模型给的另一个值即使自洽也不能顶掉它。"""
    text = "A203，实际温度 152°C，设定 165°C。"
    sig = parse(text)
    assert sig.actual_temp == 152.0
    rep = augment(sig, fake({"fields": {"actual_temp": {"value": 165, "span": "设定 165°C"}}}))
    assert sig.actual_temp == 152.0
    assert any("不覆盖" in r for r in rep.rejected)


@pytest.mark.parametrize("name", ["alarm_code", "user_request", "wants_continue_run"])
def test_扣留字段不接受模型提供(name):
    """报警码是判断不是听写；另两个直接触发 SAFE-03 硬拒绝，不能交给模型决定。"""
    sig = parse("门那边有点问题。")
    rep = augment(sig, fake({"fields": {name: {"value": "A401", "span": "门那边有点问题"}}}))
    assert getattr(sig, name) in (None, False)
    assert any("不允许由模型提供" in r for r in rep.rejected)


def test_立即停机征兆只接受明确肯定():
    sig = parse(COLLOQUIAL)
    rep = augment(sig, fake({"fields": {"smoke": {"value": False, "span": "往外冒热气"}}}))
    assert sig.smoke is False
    assert any("只接受明确肯定" in r for r in rep.rejected)


def test_清单外的字段被拒():
    sig = parse("A203，温度偏低。")
    rep = augment(sig, fake({"fields": {"root_cause": {"value": "加热器开路", "span": "温度偏低"}}}))
    assert not hasattr(sig, "root_cause")
    assert any("不在可补齐字段清单内" in r for r in rep.rejected)


def test_枚举字段越界被拒():
    sig = parse("A203，加热电流那块没看清。")
    rep = augment(sig, fake({"fields": {"p1_light": {"value": "dim", "span": "没看清"}}}))
    assert sig.p1_light is None
    assert any("不在允许范围" in r for r in rep.rejected)


def test_模型不可用时诊断照常返回():
    """兜底层坏了不能拖垮主链路——现场断网也要能出诊断。"""
    sig = parse(COLLOQUIAL)
    rep = augment(sig, boom)
    assert sig.smoke is False
    assert any("模型兜底不可用" in r for r in rep.rejected)

    d = diagnose(COLLOQUIAL, KB, llm=boom)
    assert d.validation.ok
    assert not d.parse_assisted
    assert d.model_used == "deterministic-core"


def test_兜底被拒时输出里看得见():
    """防静默失败：端点挂了或字段被复核挡掉，输出不能长得像一次正常的纯规则诊断。

    「焦味」这类信号一旦没被读到，后果是漏判一次该停机的故障。所以拒绝原因必须
    出现在「提示」里，让人知道这次是**没听全**，不是**没东西可听**。
    """
    out_dead = diagnose(COLLOQUIAL, KB, llm=boom).render()
    assert "模型兜底解析本次未生效" in out_dead
    assert "HTTP 401" in out_dead

    lying = fake({"fields": {"actual_temp": {"value": 148, "span": "塑料烧糊的味儿"}}})
    out_rejected = diagnose(COLLOQUIAL, KB, llm=lying).render()
    assert "模型兜底解析本次未生效" in out_rejected
    assert "无法从所引原文片段" in out_rejected

    # 纯规则版本（没开 --llm）不该冒这条提示——它压根没试过模型
    assert "模型兜底解析本次未生效" not in diagnose(COLLOQUIAL, KB).render()


def test_返回不是对象时优雅失败():
    sig = parse(COLLOQUIAL)
    rep = augment(sig, lambda s, u: "我不知道")
    assert not rep.filled
    assert rep.rejected


def test_未配置模型时行为与纯规则版本一致():
    text = "A203，设定 165°C，实际温度 152°C，加热电流正常有读数。"
    plain = diagnose(text, KB)
    assert plain.model_used == "deterministic-core"
    assert plain.parse_assisted == []


def test_三项环境变量缺一即视为未配置(monkeypatch):
    for key in ("APX240_LLM_BASE_URL", "APX240_LLM_API_KEY", "APX240_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    assert LLMConfig.from_env() is None

    monkeypatch.setenv("APX240_LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("APX240_LLM_MODEL", "some-model")
    assert LLMConfig.from_env() is None, "少了 api_key 就不算配好"

    monkeypatch.setenv("APX240_LLM_API_KEY", "sk-test")
    assert LLMConfig.from_env() is not None
