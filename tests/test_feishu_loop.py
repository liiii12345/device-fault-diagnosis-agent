"""飞书报修闭环与数据飞轮的测试。

全部跑在临时目录里，不碰真实知识库——这样测试可以反复跑，真实知识库与线上工单表也不会被写脏。

这里最重要的两条断言不是"能沉淀"，而是：
  1. 专家确认之前，沉淀的案例**不得**参与诊断（否则 AI 就是在拿自己的猜测当证据）
  2. 沉淀案例的证据分量**低于**说明书原文（厂内经验不能压过厂商依据）
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.diagnose import diagnose  # noqa: E402
from agent.knowledge import KnowledgeBase  # noqa: E402
from feishu.client import Client, Config  # noqa: E402
from feishu import workflow as wf  # noqa: E402

# A205 在说明书里没有任何历史案例，用它做沉淀最能体现"知识库长出来了"
FAULT = "A205 热封温度高，实际温度 195°C，设定 165°C，封口头有焦味。"
CONFIRMED_CAUSE = "控制继电器触点粘连导致加热失控"


@pytest.fixture()
def sandbox(tmp_path: Path):
    """临时沙箱：复制一份知识库，工单与用量都落在 tmp 下。

    沉淀相关断言都以「厂内经验为零」为前提，所以复制完要把 cases_learned.json 清空。
    不清的话，只要真跑过一次 promote（真接飞书演示就会），这一组测试立刻变红——
    测试不该依赖知识库的当前状态，飞轮本来就该能反复真跑。
    """
    shutil.copytree(ROOT / "knowledge", tmp_path / "knowledge")
    (tmp_path / "knowledge" / "cases_learned.json").write_text(
        json.dumps({"cases": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    paths = wf.Paths(root=tmp_path)
    client = Client(Config(force_dry_run=True), var_dir=paths.var)
    kb = KnowledgeBase(root=paths.knowledge)
    return paths, client, kb


def test_报修建单并落工单(sandbox):
    paths, client, kb = sandbox
    t = wf.submit(FAULT, reporter="张三", device="APX-240", client=client, kb=kb, paths=paths)

    assert t.wo_id.startswith("RX-")
    wo = client.work_orders()[t.wo_id]
    assert wo["报警码"] == "A205"
    assert wo["报修人"] == "张三"
    assert wo["输出自检"] == "通过"
    assert wo["AI调用次数"] == 1
    assert wo["诊断全文"], "工单必须存完整证据链，不能只存结论"
    assert json.loads(wo["提取信号"])["actual_temp"] == 195.0


def test_高危报修路由到专家群(sandbox):
    paths, client, kb = sandbox
    t = wf.submit(FAULT, client=client, kb=kb, paths=paths)

    assert t.high_risk is True, "A205 属高风险，必须走专家通道"
    assert client.sent[-1]["to_experts"] is True
    assert client.sent[-1]["card"]["header"]["template"] == "red"
    assert client.work_orders()[t.wo_id]["状态"] == wf.STATUS_ESCALATED


def test_常规报修走报修群且不惊动专家(sandbox):
    paths, client, kb = sandbox
    t = wf.submit("A101 进料检测超时，物料已经到位了，P1 指示灯不亮。",
                  client=client, kb=kb, paths=paths)

    assert t.high_risk is False
    assert client.sent[-1]["to_experts"] is False
    assert client.sent[-1]["card"]["header"]["template"] == "blue"
    assert client.work_orders()[t.wo_id]["状态"] == wf.STATUS_NEW


def _body(card: dict) -> str:
    """卡片正文 = 所有 div 的文本拼起来（一段一个 div，层级是飞书自己画出来的）。"""
    return "\n".join(e["text"]["content"] for e in card["elements"] if e["tag"] == "div")


# 诊断卡上八个字段各自的 div 开头。卡片另外还挂着流程条目（全文指针／⑦ 回填入口／
# 双人确认入口），只有这几块是 render() 里同一份内容换了个排版。
FIELD_DIV = ("**①", "**②", "**③", "**④", "**⑤", "**⑥", "**⑦", "**⑧", "⚠️ **提示**")


def test_卡片正文已转成飞书语法(sandbox):
    """lark_md 不认 ## 标题与 --- 分隔线，直接发过去会显示成一堆井号。"""
    paths, client, kb = sandbox
    wf.submit(FAULT, client=client, kb=kb, paths=paths)
    body = _body(client.sent[-1]["card"])

    assert "##" not in body
    assert "**① 故障现象**" in body
    assert "195°C" in body


CARD_CASES = [
    FAULT,                                                  # A205 高危：停机 + 升级专家
    "A203 热封温度低，设定 165°C，实际只有 152°C，加热电流正常有读数。",  # ② 里有长案例逐条比对
]


def test_卡片一条明细都不省(sandbox):
    """卡片是全文的另一种排版，不是摘要：八个字段的每一条明细都必须原样出现在正文里。

    概览把条目删掉一半、只留个条数，读的人还得回去翻工单，卡片等于没做。
    密度靠"一条一行 + 剥掉模板腔"解决，不靠删内容解决。
    """
    paths, client, kb = sandbox
    for text in CARD_CASES:
        t = wf.submit(text, client=client, kb=kb, paths=paths)
        card = client.sent[-1]["card"]
        body = _body(card)
        d = t.diagnosis

        items = ([f.content for f in d.known_facts]
                 + [c.cause for c in d.possible_causes]
                 + [c.basis for c in d.possible_causes]
                 + list(d.evidence)
                 + list(d.safety_preconditions)
                 + [s.action for s in d.inspection_steps]
                 + list(d.missing_info)
                 + list(d.stop_and_escalate.stop_conditions)
                 + list(d.stop_and_escalate.triggered_rules)
                 + [d.stop_and_escalate.reason])
        for item in items:
            flat = " ".join(str(item).split())
            assert flat and flat in body, f"卡片漏了一条明细：{flat[:40]}"

        assert "见工单" not in body and not re.search(r"＋\s*\d+", body), "不许用条数占位代替明细"
        # 上界抓的是"一整块被折成一行"（十条事实拼起来上千字），不是"单条明细太长"：
        # 一条历史案例的逐条比对本来就有 190 字，超一行是允许的，折行交给飞书。
        longest = max(body.split("\n"), key=len)
        assert len(longest) <= 200, f"这一行塞了太多东西（{len(longest)} 字）：{longest[:40]}"
        assert len(card["elements"]) > 3, "一个字段一个 div，飞书才会排出间隔"
        assert f"{len(d.inspection_steps)} 步" in body, "⑥ 抬头报的步数必须等于实际列出的步数"
        # 只比八个字段的 div：卡片另外还挂着全文指针、⑦ 回填入口、双人确认入口这些流程条目，
        # 它们不在 render() 里，拿整张卡片比长度会把"多了一个按钮"误报成"排版变松了"。
        fields = "\n".join(e["text"]["content"] for e in card["elements"]
                           if e["tag"] == "div" and e["text"]["content"].startswith(FIELD_DIV))
        assert len(fields) < len(d.render()), "同一份内容的两种排版，卡片仍须比工单全文紧凑"
        assert "诊断全文" in body, "卡片要给出指针，说明全文在哪"
        assert client.work_orders()[t.wo_id]["诊断全文"].startswith("## 故障现象")

    assert "现场一律不动手" in _body(client.sent[0]["card"]), \
        "停机时也要把排查顺序列全，那几步是专家到场后的清单"


def test_六个字段一个不许少(sandbox):
    """现场必须拿到的六项输出：故障现象／可能原因／证据来源／排查顺序／需要补充的信息／是否升级专家。

    卡片按 ①~⑧ 分块，六项对应 ①③④⑥⑦⑧；②已知事实、⑤安全前置条件是本方案额外加的。
    字段与明细一条不少，工单里存的是同一份内容的完整版（出处标签原文与自检明细）。
    """
    paths, client, kb = sandbox
    wf.submit(FAULT, client=client, kb=kb, paths=paths)
    body = _body(client.sent[-1]["card"])

    required = ["故障现象", "已知事实", "可能原因", "证据来源",
                "安全前置条件", "排查顺序", "需要补充的信息", "是否升级专家"]
    for i, name in enumerate(required, 1):
        assert f"**{chr(0x2460 + i - 1)} {name}**" in body, f"卡片缺第 {i} 块：{name}"
    assert "①" in body and "⑧" in body


def test_卡片自带完整工单入口(sandbox):
    """卡片必须自带全文入口，但拼不出真链接时宁可不给——死链比没有链更糟。"""
    paths, client, kb = sandbox
    wf.submit(FAULT, client=client, kb=kb, paths=paths)
    card = client.sent[-1]["card"]

    assert client.cfg.live is False, "DryRun 下拿不到真 record_id"
    assert not [e for e in card["elements"] if e["tag"] == "action"], "不许发死链按钮"

    live = Client(Config(app_id="a", app_secret="b", chat_id="oc_x",
                         bitable_app_token="baskAAA", bitable_table_id="tblBBB"))
    assert live.wo_url("recCCC") == ("https://www.feishu.cn/base/baskAAA"
                                     "?table=tblBBB&record=recCCC")
    assert live.wo_url("") == "" and live.wo_url("local-RX-1") == ""


def test_判定写进卡头标题(sandbox):
    """扫一眼群列表就该知道这单要不要管，不用点开卡片。"""
    paths, client, kb = sandbox
    wf.submit(FAULT, client=client, kb=kb, paths=paths)
    assert "立即停机" in client.sent[-1]["card"]["header"]["title"]["content"]

    wf.submit("A203，设定 165°C，实际温度 152°C。", client=client, kb=kb, paths=paths)
    assert "可继续排查" in client.sent[-1]["card"]["header"]["title"]["content"]


def test_卡片只有红蓝两色_未覆盖也是红卡(sandbox):
    """覆盖度 none 必定触发 SAFE-05 强制升级，所以「知识库未覆盖」就是高危，
    红蓝之间不存在第三种状态。留一个永远发不出去的橙卡分支，等于在代码里写了句假话。"""
    paths, client, kb = sandbox
    cases = [(FAULT, "red"),
             ("A203，设定 165°C，实际温度 152°C。", "blue"),
             ("面板报了个 A777，说明书上翻不到这个码，机器还在响。", "red")]
    uncovered_wo = ""
    for text, color in cases:
        t = wf.submit(text, client=client, kb=kb, paths=paths)
        header = client.sent[-1]["card"]["header"]
        assert header["template"] == color, text
        assert ("高危" in header["title"]["content"]) is (color == "red")
        if "A777" in text:
            uncovered_wo = t.wo_id

    wo = client.work_orders()[uncovered_wo]
    assert wo["知识库覆盖度"] == "none" and wo["是否升级专家"] == "是"


def test_安全前置不许截成半句(sandbox):
    """「仅限授权人员执行」是责任边界本身，砍成「仅限授…」等于把免责写没了。"""
    paths, client, kb = sandbox
    wf.submit("A203，设定 165°C，实际温度 152°C，加热电流正常有读数。",
              client=client, kb=kb, paths=paths)
    body = _body(client.sent[-1]["card"])

    assert "仅限授权人员执行" in body
    assert "仅限授权…" not in body and "执行…" not in body


def test_未停机时不叫红线(sandbox):
    """SAFE-02/04 只是前置约束，写成「触发红线」会和「可继续排查」自相矛盾。"""
    paths, client, kb = sandbox
    wf.submit("A203，设定 165°C，实际温度 152°C，加热电流正常有读数。",
              client=client, kb=kb, paths=paths)
    body = _body(client.sent[-1]["card"])

    assert "命中安全规则" in body and "触发红线" not in body

    wf.submit(FAULT, client=client, kb=kb, paths=paths)
    assert "触发红线" in _body(client.sent[-1]["card"])


def test_专家名单在卡片上去重(sandbox):
    """两条规则可能点名同一拨专家，显示成「专家、专家（禁止现场自行处理）」很突兀。"""
    paths, client, kb = sandbox
    t = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    body = _body(client.sent[-1]["card"])

    assert "专家、专家" not in body
    assert "→ 专家（" in body
    # 去重只发生在显示层：工单里的诊断全文仍保留说明书原文的完整列表
    assert "专家、专家（禁止现场自行处理）" in client.work_orders()[t.wo_id]["诊断全文"]


def test_用量流水可支撑仪表盘(sandbox):
    paths, client, kb = sandbox
    wf.submit(FAULT, client=client, kb=kb, paths=paths)
    wf.submit("A203，设定 165°C，实际温度 152°C。", client=client, kb=kb, paths=paths)

    rows = [json.loads(x) for x in paths.usage.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert sum(r["calls"] for r in rows) == 2
    assert {r["coverage"] for r in rows} == {"full"}


# ------------------------------------------------------------------ 补充信息回填

T203 = "A203 热封温度低，设定 165°C，实际只有 152°C，加热电流正常有读数。"


def test_回填后另建关联工单而不改原单(sandbox):
    """第一轮的卡片已经进群、原单可能已经进了专家视野，改存档等于事后改口。"""
    paths, client, kb = sandbox
    t1 = wf.submit(T203, reporter="张三", client=client, kb=kb, paths=paths)
    t2 = wf.followup(t1.wo_id, {"是否已完成预热": "开机后已经预热 40 分钟了"},
                     client=client, kb=kb, paths=paths)
    wos = client.work_orders()

    assert t2.wo_id != t1.wo_id and t2.round == 2 and t2.prev_wo == t1.wo_id
    assert wos[t1.wo_id]["后续工单"] == t2.wo_id, "原单要能指向第二轮，否则查不到后续"
    assert wos[t2.wo_id]["关联工单"] == t1.wo_id
    assert wos[t2.wo_id]["诊断轮次"] == "2"
    assert wos[t1.wo_id]["现场描述"] == T203, "原单存档不许被第二轮改写"
    assert T203 in wos[t2.wo_id]["现场描述"]
    assert "已经预热 40 分钟" in wos[t2.wo_id]["现场描述"]


def test_补上的信息真的改变了结论(sandbox):
    """回填不是走个形式：读得出的信号要进证据链，被反驳的原因要下去。"""
    paths, client, kb = sandbox
    t1 = wf.submit(T203, client=client, kb=kb, paths=paths)
    assert "是否已完成预热" in t1.diagnosis.missing_info
    assert any(c.cause == "未预热" for c in t1.diagnosis.possible_causes)

    t2 = wf.followup(t1.wo_id, {"是否已完成预热": "开机后已经预热 40 分钟了",
                                "故障发生时间点": "今天 14:20"},
                     client=client, kb=kb, paths=paths)
    d = t2.diagnosis
    assert "是否已完成预热" not in d.missing_info
    assert "故障发生时间点" not in d.missing_info
    assert not any(c.cause == "未预热" for c in d.possible_causes), "预热已完成，这个原因就该下去"
    assert any("已预热" in f.content for f in d.known_facts)
    assert d.validation.ok and not d.degraded


def test_读不出的回答不许悄悄丢掉(sandbox):
    """「继电器是否粘连」解析层没有对应信号。答了就要认账：从 ⑦ 摘掉，
    同时写明本轮没有据此调整证据强度——假装判定过比读不出来严重得多。"""
    paths, client, kb = sandbox
    t1 = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    t2 = wf.followup(t1.wo_id, {"继电器是否粘连": "拆开看触点已经粘连发黑"},
                     client=client, kb=kb, paths=paths)
    d = t2.diagnosis

    assert "继电器是否粘连" not in d.missing_info
    assert any("未能从中读出可判定的信号" in w and "触点已经粘连发黑" in w for w in d.warnings)
    assert d.validation.ok, "动过 ⑦ 与提示之后必须重新过自检"


def test_答没测的项仍留在待补清单(sandbox):
    """「没测」不是这一项的内容，是这一项仍然缺。摘掉它再在卡上写「观测项都已给出」，
    等于告诉现场这项不必测了——而它恰恰还没测。同一轮里「今早 8 点 20 分」要真读得出来，
    否则一条正经答复会被误当成读不出。"""
    paths, client, kb = sandbox
    t1 = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    t2 = wf.followup(t1.wo_id, {"继电器是否粘连": "拆开看触点已经粘连发黑",
                                "T1 是否松脱": "没测",
                                "故障发生时间点": "今早 8 点 20 分"},
                     client=client, kb=kb, paths=paths)
    d = t2.diagnosis

    assert d.missing_info == ["T1 是否松脱"]
    assert d.unresolved_answers == ["继电器是否粘连"], "只有答了内容的才算读不出"
    assert any("未取得该项" in w and "没测" in w for w in d.warnings)
    assert d.validation.ok

    body = _body(client.sent[-1]["card"])
    assert "⑦ 需要补充的信息**｜1 项" in body
    assert "观测项现场都已给出" not in body


def test_回填入口照样过闸门(sandbox):
    """表单是另一个入口，不是绕过安全门控的侧门。"""
    paths, client, kb = sandbox
    t1 = wf.submit("A401 安全门报警，门已经关严了。", client=client, kb=kb, paths=paths)
    assert t1.diagnosis.refused_requests == []

    t2 = wf.followup(t1.wo_id,
                     {"报警是否间歇出现": "一直报。产量压得紧，能不能短接安全门先赶这批？"},
                     client=client, kb=kb, paths=paths)
    d = t2.diagnosis
    assert d.refused_requests, "违规请求从回填入口进来也要被明确拒绝"
    assert d.stop_and_escalate.must_stop is True
    assert d.validation.ok
    assert client.work_orders()[t2.wo_id]["已拒绝的违规请求"]


def test_一项都补不上就不建单(sandbox):
    """空答案跑出来的第二轮和第一轮一模一样，白烧一次调用还多一张卡。"""
    paths, client, kb = sandbox
    t1 = wf.submit(T203, client=client, kb=kb, paths=paths)

    with pytest.raises(ValueError):
        wf.followup(t1.wo_id, {"是否已完成预热": "   "}, client=client, kb=kb, paths=paths)
    with pytest.raises(ValueError):
        wf.followup(t1.wo_id, {"随便写一项": "有"}, client=client, kb=kb, paths=paths)
    assert len(client.work_orders()) == 1 and len(client.sent) == 1


def test_工单不存在时报错而不是凭空建单(sandbox):
    paths, client, kb = sandbox
    with pytest.raises(KeyError):
        wf.followup("RX-19700101-9999", {"是否已完成预热": "已预热"},
                    client=client, kb=kb, paths=paths)
    assert client.work_orders() == {}


def test_第二轮也发卡也记用量(sandbox):
    paths, client, kb = sandbox
    t1 = wf.submit(T203, client=client, kb=kb, paths=paths)
    t2 = wf.followup(t1.wo_id, {"是否已完成预热": "已经预热 40 分钟了"},
                     client=client, kb=kb, paths=paths)

    assert len(client.sent) == 2
    card = client.sent[-1]["card"]
    assert "第2轮" in card["header"]["title"]["content"]
    assert t1.wo_id in _body(card), "第二轮卡片要说清承接的是哪一单"
    rows = [json.loads(x) for x in paths.usage.read_text(encoding="utf-8").splitlines()]
    assert [r["wo_id"] for r in rows] == [t1.wo_id, t2.wo_id]
    assert sum(r["calls"] for r in rows) == 2, "一轮一次调用，第二轮不该把第一轮重烧一遍"


def test_卡片发不出去时第二轮结论仍要回去(sandbox, monkeypatch):
    """第二轮的请求人是刚填完表的那个人，群发失败不能把他的诊断一起吞掉。"""
    paths, client, kb = sandbox
    t1 = wf.submit(FAULT, client=client, kb=kb, paths=paths)

    def boom(*args, **kwargs):
        raise RuntimeError("im 接口 500")
    monkeypatch.setattr(client, "send_card", boom)

    t2 = wf.followup(t1.wo_id, {"继电器是否粘连": "触点已经粘连发黑"},
                     client=client, kb=kb, paths=paths)
    wo = client.work_orders()[t2.wo_id]

    assert t2.message_id == ""
    assert "im 接口 500" in wo["推送状态"]
    assert wo["输出自检"] == "通过" and wo["诊断全文"], "诊断已经成立，不能因为通知失败就丢掉"


def test_第三轮不把链条走成单行(sandbox):
    """共用一列存关联关系时，第三轮会把第二轮指回第一轮的指针覆盖掉。"""
    paths, client, kb = sandbox
    t1 = wf.submit(T203, client=client, kb=kb, paths=paths)
    t2 = wf.followup(t1.wo_id, {"是否已完成预热": "已经预热 40 分钟了"},
                     client=client, kb=kb, paths=paths)
    t3 = wf.followup(t2.wo_id, {"故障发生时间点": "今天 14:20"}, client=client, kb=kb, paths=paths)
    wos = client.work_orders()

    assert (t3.round, t3.prev_wo) == (3, t2.wo_id)
    assert wos[t2.wo_id]["关联工单"] == t1.wo_id, "第二轮仍要指得回第一轮"
    assert wos[t2.wo_id]["后续工单"] == t3.wo_id
    assert wos[t1.wo_id]["后续工单"] == t2.wo_id


def test_回填按钮只在有题可答且有真链接时出现(sandbox):
    """没有公网地址就不出按钮；⑦ 已空也不出——点进去是一张没有题目的空表单。"""
    paths, client, kb = sandbox

    def buttons(d, **kw):
        card = wf.build_card("RX-1", d, "张三", "APX-240", **kw)
        return [b["text"]["content"]
                for e in card["elements"] if e["tag"] == "action" for b in e["actions"]]

    t1 = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    d1 = t1.diagnosis
    assert d1.missing_info
    assert not any("补充信息" in b for b in buttons(d1)), "未配公网地址就不出死链按钮"
    assert any("补充信息" in b for b in buttons(d1, followup_url="https://x.dev/?wo=RX-1"))

    # 三个待补项全答完就是这条链的终点：说明书里每个报警码都有解析层读不出的观测项，
    # 所以第一轮描述凑不出空 ⑦，只有回填能把 ⑦ 收到零。
    t2 = wf.followup(t1.wo_id, {"继电器是否粘连": "触点已经粘连发黑",
                                "T1 是否松脱": "T1 接线端子紧固，没有松动",
                                "故障发生时间点": "今天 14:20"},
                     client=client, kb=kb, paths=paths)
    assert t2.diagnosis.missing_info == []
    assert t2.diagnosis.unresolved_answers == ["继电器是否粘连", "T1 是否松脱"]
    assert not any("补充信息" in b
                   for b in buttons(t2.diagnosis, followup_url="https://x.dev/?wo=RX-1"))
    body = _body(client.sent[-1]["card"])
    assert ("⑦ 需要补充的信息**｜无——本轮答复的 继电器是否粘连、T1 是否松脱 "
            "读不出可判定信号") in body
    assert "观测项现场都已给出" not in body, "⑦ 空是因为读不出的项被摘掉，不是都齐了"


def test_网页入口地址只看公网配置不看飞书凭据(sandbox):
    """回填页读的是本地工单存档，DryRun 下照样能填、能出第二轮。"""
    paths, client, kb = sandbox
    dry = Client(Config(force_dry_run=True, web_base_url="https://demo.dev/base/"))
    assert dry.followup_url("RX-1") == "https://demo.dev/base/?wo=RX-1"
    assert Client(Config(force_dry_run=True)).followup_url("RX-1") == ""


def test_待补项清单从工单里读得回来(sandbox):
    """回填页按这份清单出题，顺序要与卡片上 ⑦ 一致。"""
    paths, client, kb = sandbox
    t = wf.submit(T203, client=client, kb=kb, paths=paths)
    assert wf.missing_items(client.work_orders()[t.wo_id]) == t.diagnosis.missing_info

    legacy = {"需补充信息": "T1 接线状态、故障发生时间点"}
    assert wf.missing_items(legacy) == ["T1 接线状态", "故障发生时间点"]
    assert wf.missing_items({"需补充信息项": "[]", "需补充信息": ""}) == []


# ------------------------------------------------------------------ 飞轮

def test_闭环生成候选案例(sandbox):
    paths, client, kb = sandbox
    t = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    cand = wf.close(t.wo_id, CONFIRMED_CAUSE, "锁定挂牌后更换控制继电器",
                    "165°C 稳定 30 分钟试产合格", "专家（电气）", client=client, paths=paths)

    assert cand["status"] == wf.CAND_PENDING
    assert cand["match_signals"]["actual_temp"] == 195.0
    assert client.work_orders()[t.wo_id]["状态"] == wf.STATUS_CLOSED


def test_未确认的候选案例不得参与诊断(sandbox):
    """最关键的一条：AI 不能把自己的猜测写进知识库再拿它当证据。"""
    paths, client, kb = sandbox
    t = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    wf.close(t.wo_id, CONFIRMED_CAUSE, "更换控制继电器", "试产合格", "专家",
             client=client, paths=paths)

    fresh = KnowledgeBase(root=paths.knowledge)
    assert fresh.cases_learned == [], "候选案例还在 var/ 待确认，不得进入知识库"
    d = diagnose(FAULT, fresh)
    assert all(CONFIRMED_CAUSE not in c.cause for c in d.possible_causes)
    assert not any("LEARNED" in e for e in d.evidence)


def test_专家确认后案例入库并参与诊断(sandbox):
    paths, client, kb = sandbox
    t = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    wf.close(t.wo_id, CONFIRMED_CAUSE, "锁定挂牌后更换控制继电器", "试产合格", "专家",
             client=client, paths=paths)
    entry = wf.promote(t.wo_id, "专家", client=client, paths=paths)

    assert entry["id"] == "L01"
    assert entry["source_ref"] == "KB:LEARNED:L01"
    assert client.work_orders()[t.wo_id]["状态"] == wf.STATUS_PROMOTED

    fresh = KnowledgeBase(root=paths.knowledge)
    assert len(fresh.cases_learned) == 1
    d = diagnose(FAULT, fresh)
    assert any(c.cause == CONFIRMED_CAUSE for c in d.possible_causes), "沉淀案例应成为候选原因"
    assert any("KB:LEARNED:L01" in e for e in d.evidence), "证据出处必须标明是厂内沉淀而非说明书"


def test_沉淀案例的分量低于说明书(sandbox):
    """厂内经验最多给「中」，不能压过说明书依据的「强」。"""
    paths, client, kb = sandbox
    t = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    wf.close(t.wo_id, CONFIRMED_CAUSE, "更换控制继电器", "试产合格", "专家",
             client=client, paths=paths)
    wf.promote(t.wo_id, "专家", client=client, paths=paths)

    d = diagnose(FAULT, KnowledgeBase(root=paths.knowledge))
    learned = next(c for c in d.possible_causes if c.cause == CONFIRMED_CAUSE)
    assert learned.confidence == "中"
    assert "非说明书原文" in learned.basis


def test_依据栏说清这条案例过的是哪道签字闸门(sandbox):
    """双人签字入库的案例，与规则升级前只有专家一个署名的老案例，依据栏必须写成两种。

    混成一句「经专家确认入库」有两个坏处：新案例白攒了现场那一笔签字，
    老案例（真实知识库里的 L01 就是）则被说成也过了现场确认——它自己说不出这句话。
    """
    paths, client, kb = sandbox
    t = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    wf.close(t.wo_id, CONFIRMED_CAUSE, "更换控制继电器", "试产合格", "专家",
             client=client, paths=paths)
    wf.promote(t.wo_id, "张三", client=client, paths=paths)

    basis = next(c.basis for c in diagnose(FAULT, KnowledgeBase(root=paths.knowledge)).possible_causes
                 if c.cause == CONFIRMED_CAUSE)
    assert "报修人现场确认" in basis and "双人签字入库" in basis
    assert "早于双人确认规则" not in basis

    # 把现场署名从文件里拿掉，就是双人确认闸门上线前沉淀的那一类案例
    table = json.loads((paths.knowledge / "cases_learned.json").read_text(encoding="utf-8"))
    del table["cases"][0]["field_confirmed_by"]
    (paths.knowledge / "cases_learned.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")

    legacy = next(c.basis for c in diagnose(FAULT, KnowledgeBase(root=paths.knowledge)).possible_causes
                  if c.cause == CONFIRMED_CAUSE)
    assert "早于双人确认规则、没有现场署名" in legacy
    assert "双人签字入库" not in legacy, "不许替老案例补一个它没有的现场签字"


def test_沉淀案例被写进已知事实时标明出处性质(sandbox):
    paths, client, kb = sandbox
    t = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    wf.close(t.wo_id, CONFIRMED_CAUSE, "更换控制继电器", "试产合格", "专家",
             client=client, paths=paths)
    wf.promote(t.wo_id, "专家", client=client, paths=paths)

    d = diagnose(FAULT, KnowledgeBase(root=paths.knowledge))
    fact = next(f for f in d.known_facts if "L01" in f.source)
    assert fact.source.startswith("KB:LEARNED:")
    assert "非说明书" in fact.content


def test_沉淀后的输出仍须通过全部自检(sandbox):
    paths, client, kb = sandbox
    t = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    wf.close(t.wo_id, CONFIRMED_CAUSE, "更换控制继电器", "165°C 稳定 30 分钟试产合格", "专家",
             client=client, paths=paths)
    wf.promote(t.wo_id, "专家", client=client, paths=paths)

    d = diagnose(FAULT, KnowledgeBase(root=paths.knowledge))
    assert d.validation.ok, d.validation.errors
    assert d.degraded is False


# ------------------------------------------------------------------ 沉淀的准入约束

def test_缺根因或专家时不得闭环(sandbox):
    paths, client, kb = sandbox
    t = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    with pytest.raises(ValueError):
        wf.close(t.wo_id, "", "更换继电器", "试产合格", "专家", client=client, paths=paths)
    with pytest.raises(ValueError):
        wf.close(t.wo_id, CONFIRMED_CAUSE, "更换继电器", "试产合格", "", client=client, paths=paths)


def test_不存在的工单不能闭环(sandbox):
    paths, client, kb = sandbox
    with pytest.raises(KeyError):
        wf.close("RX-19700101-9999", "x", "y", "z", "专家", client=client, paths=paths)


def test_未闭环的工单不能提升(sandbox):
    paths, client, kb = sandbox
    t = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    with pytest.raises(KeyError):
        wf.promote(t.wo_id, "专家", client=client, paths=paths)


def test_不可重复提升(sandbox):
    paths, client, kb = sandbox
    t = wf.submit(FAULT, client=client, kb=kb, paths=paths)
    wf.close(t.wo_id, CONFIRMED_CAUSE, "更换继电器", "试产合格", "专家", client=client, paths=paths)
    wf.promote(t.wo_id, "专家", client=client, paths=paths)
    with pytest.raises(ValueError):
        wf.promote(t.wo_id, "专家", client=client, paths=paths)

    learned = json.loads(paths.learned.read_text(encoding="utf-8"))["cases"]
    assert len(learned) == 1, "重复提升会让同一条经验在知识库里出现两次"


def test_无报警码的工单不得沉淀(sandbox):
    """没有报警码就无从匹配，沉淀进去只会污染知识库。"""
    paths, client, kb = sandbox
    t = wf.submit("机器声音不对，有金属摩擦声。", client=client, kb=kb, paths=paths)
    cand = wf.close(t.wo_id, "传动轴承损坏", "更换轴承", "试产正常", "王工",
                    client=client, paths=paths)
    assert cand["status"].startswith("无法沉淀")


# ------------------------------------------------------------------ 双人确认入库
#
# 谁有权改知识库？两个人：技术结论由专家署名，现场可行性由报修人作证。
# 少一个签字，cases_learned.json 一个字都不写。这一节钉的就是这条边界。

EXPERT = {"expert": "专家（电气）", "root_cause": CONFIRMED_CAUSE,
          "action_taken": "断电锁定挂牌后更换控制继电器",
          "verification": "封口温度 165°C 稳定 30 分钟"}
READING = "封口温度 165°C 稳定 30 分钟，试产 200 袋合格"
NO_ALARM = "机器一直响，面板上翻不到码，说明书也没有这一条。"


def _escalated(sandbox, text: str = FAULT):
    """建一张会升级专家的单，并把工单上的一次性专家码取出来。"""
    paths, client, kb = sandbox
    t = wf.submit(text, reporter="张三", client=client, kb=kb, paths=paths)
    code = client.work_orders()[t.wo_id]["专家码"]
    assert len(code) == wf.CODE_LEN
    return t, code


def _expert_submit(sandbox, t, code: str, **over):
    paths, client, _ = sandbox
    a = {**EXPERT, **over}
    return wf.expert_close(t.wo_id, code, a["expert"], a["root_cause"], a["action_taken"],
                           a["verification"], client=client, paths=paths)


def _field_code(sandbox, t) -> str:
    return sandbox[1].work_orders()[t.wo_id]["现场码"]


def test_常规单不发确认码(sandbox):
    """常规单是现场自己排掉的：同一个人既写结论又确认"现场可行"，双人确认就成了自问自答。"""
    paths, client, kb = sandbox
    t = wf.submit("A101 进料检测超时，物料已经到位了，P1 指示灯不亮。",
                  client=client, kb=kb, paths=paths)
    wo = client.work_orders()[t.wo_id]
    assert not wo.get("专家码") and not wo.get("现场码")
    assert "🧑‍🔧" not in _body(client.sent[-1]["card"]), "蓝卡上不该出现处置结论入口"


def test_高危单的红卡带着一次性专家码(sandbox):
    paths, client, kb = sandbox
    t, code = _escalated(sandbox)
    body = _body(client.sent[-1]["card"])
    assert code in body and "两人签字才入库" in body


def test_专家提交后转给现场_知识库仍未被写(sandbox):
    paths, client, kb = sandbox
    t, code = _escalated(sandbox)
    res = _expert_submit(sandbox, t, code)
    wo = client.work_orders()[t.wo_id]

    assert res["stage"] == "expert" and res["promotable"] is True
    assert res["candidate"]["status"] == wf.CAND_PENDING
    assert wo["状态"] == wf.STATUS_AWAIT_FIELD
    assert wo["专家码"] == "", "码用过即作废：留着就能被第二次提交"
    assert wo["现场码"] and wo["现场码"] != code
    assert KnowledgeBase(root=paths.knowledge).cases_learned == [], "只签了一个字，不许入库"

    sent = client.sent[-1]
    assert sent["to_experts"] is False, "橙卡是给报修人的，不该再惊动专家群"
    assert sent["card"]["header"]["template"] == "orange"
    body = _body(sent["card"])
    for item in (EXPERT["expert"], EXPERT["root_cause"], EXPERT["action_taken"],
                 EXPERT["verification"], wo["现场码"]):
        assert item in body, "橙卡要把专家写的三条与现场码都摆出来，不能只给个链接"
    assert "不用你判断" in body, "得说清技术判断不归报修人"


def test_现场确认通过后入库_两张署名分开存(sandbox):
    paths, client, kb = sandbox
    t, code = _escalated(sandbox)
    _expert_submit(sandbox, t, code)
    res = wf.field_confirm(t.wo_id, _field_code(sandbox, t), "张三", "是", "是", READING,
                           client=client, paths=paths)

    assert res["passed"] is True and res["case_id"] == "L01"
    assert res["candidate"]["status"] == f"已提升为 {res['case_id']}", \
        "回执不能一边说已入库、一边把候选状态还显示成待现场确认"
    entry = json.loads(paths.learned.read_text(encoding="utf-8"))["cases"][0]
    assert entry["confirmed_by"] == EXPERT["expert"], "技术结论归专家"
    assert entry["field_confirmed_by"] == "张三", "现场可行性归报修人"
    assert entry["field_verification"]["复机关键读数"] == READING
    wo = client.work_orders()[t.wo_id]
    assert wo["状态"] == wf.STATUS_PROMOTED and wo["现场码"] == ""

    sent = client.sent[-1]
    assert sent["to_experts"] is True and sent["card"]["header"]["template"] == "green"
    body = _body(sent["card"])
    assert EXPERT["expert"] in body and "张三" in body and "L01" in body

    causes = diagnose(FAULT, KnowledgeBase(root=paths.knowledge)).possible_causes
    learned = next(c for c in causes if c.cause == CONFIRMED_CAUSE)
    assert learned.confidence == "中", "双人签字也不许让厂内经验压过说明书原文"


def test_现场说没恢复_知识库一个字都不写(sandbox):
    paths, client, kb = sandbox
    t, code = _escalated(sandbox)
    _expert_submit(sandbox, t, code)
    old = _field_code(sandbox, t)
    res = wf.field_confirm(t.wo_id, old, "张三", "是", "否", "复机 20 分钟又升到 188°C",
                           "换完继电器温度仍然爬升，怀疑还有第二处加热回路故障",
                           client=client, paths=paths)
    wo = client.work_orders()[t.wo_id]

    assert res["passed"] is False and res["case_id"] == ""
    assert json.loads(paths.learned.read_text(encoding="utf-8"))["cases"] == []
    assert wo["状态"] == wf.STATUS_REJECTED
    assert wo["现场分歧"] == "换完继电器温度仍然爬升，怀疑还有第二处加热回路故障"
    assert wo["现场码"] == "" and wo["专家码"] and wo["专家码"] != old
    cand = json.loads(paths.candidates.read_text(encoding="utf-8"))["candidates"][0]
    assert cand["status"] == wf.CAND_REJECTED
    assert cand["field_rejected"]["by"] == "张三"

    sent = client.sent[-1]
    assert sent["to_experts"] is True and sent["card"]["header"]["template"] == "red"
    body = _body(sent["card"])
    assert "知识库一个字都没写" in body and wo["专家码"] in body
    assert EXPERT["root_cause"] in body, "退回要把专家原结论带回去，他才知道该改哪一条"


def test_退回后专家改结论_可以再走一遍双人确认(sandbox):
    paths, client, kb = sandbox
    t, code = _escalated(sandbox)
    _expert_submit(sandbox, t, code)
    wf.field_confirm(t.wo_id, _field_code(sandbox, t), "张三", "是", "否",
                     "复机 20 分钟又升到 188°C", "温度仍然爬升", client=client, paths=paths)

    again = _expert_submit(sandbox, t, client.work_orders()[t.wo_id]["专家码"],
                           root_cause="继电器粘连＋温控仪 PID 参数漂移")
    wo = client.work_orders()[t.wo_id]
    assert again["promotable"] is True and wo["状态"] == wf.STATUS_AWAIT_FIELD
    assert len(json.loads(paths.candidates.read_text(encoding="utf-8"))["candidates"]) == 1, \
        "同一张工单只留一条候选案例，否则现场确认会先撞上那条已退回的旧记录"

    res = wf.field_confirm(t.wo_id, wo["现场码"], "张三", "是", "是", "165°C 稳定 60 分钟",
                           client=client, paths=paths)
    entry = json.loads(paths.learned.read_text(encoding="utf-8"))["cases"][0]
    assert res["case_id"] == entry["id"] == "L01"
    assert entry["root_cause"] == "继电器粘连＋温控仪 PID 参数漂移", "入库的必须是改过的那一版"


def test_说不可行却不写原因_退回不了(sandbox):
    """退回必须有下一步：只说一句"不行"，专家不知道该改哪一条，链就断在这儿。"""
    paths, client, kb = sandbox
    t, code = _escalated(sandbox)
    _expert_submit(sandbox, t, code)
    with pytest.raises(ValueError, match="写清实际情况"):
        wf.field_confirm(t.wo_id, _field_code(sandbox, t), "张三", "是", "否", "", "",
                         client=client, paths=paths)
    assert json.loads(paths.learned.read_text(encoding="utf-8"))["cases"] == []
    assert client.work_orders()[t.wo_id]["状态"] == wf.STATUS_AWAIT_FIELD, "退回没成立，别改状态"


def test_确认码是唯一凭证_错一次或用第二次都进不去(sandbox):
    paths, client, kb = sandbox
    t, code = _escalated(sandbox)
    with pytest.raises(wf.CodeRejected):
        _expert_submit(sandbox, t, "ZZZZZZ")
    # 拿专家码去走现场确认也不行：两个码各管一段，不能通用
    with pytest.raises(wf.CodeRejected):
        wf.field_confirm(t.wo_id, code, "张三", "是", "是", READING, client=client, paths=paths)

    _expert_submit(sandbox, t, code)
    with pytest.raises(wf.CodeRejected):
        _expert_submit(sandbox, t, code)   # 用过即作废
    assert KnowledgeBase(root=paths.knowledge).cases_learned == []


def test_阶段由码决定_页面说自己是谁不算数(sandbox):
    paths, client, kb = sandbox
    t, code = _escalated(sandbox)
    assert wf.close_form(t.wo_id, code, client=client, paths=paths)["stage"] == "expert"

    # 页面谎称自己是现场确认也没用：码属于专家阶段，就只走专家那一步
    res = wf.close_submit(t.wo_id, code, {**EXPERT, "stage": "field"},
                          client=client, paths=paths)
    assert res["stage"] == "expert"

    form = wf.close_form(t.wo_id, _field_code(sandbox, t), client=client, paths=paths)
    assert form["stage"] == "field"
    assert form["expert"] == EXPERT["expert"] and form["root_cause"] == EXPERT["root_cause"]
    assert form["questions"][0]["key"] == "reporter"
    assert form["questions"][0]["value"] == "张三", "报修人姓名从工单预填，不让现场再抄一遍"

    res = wf.close_submit(t.wo_id, _field_code(sandbox, t),
                          {"stage": "expert", "reporter": "张三", "executed": "是",
                           "recovered": "是", "reading": READING},
                          client=client, paths=paths)
    assert res["stage"] == "field" and res["case_id"] == "L01"


def test_现场确认只问现场看得见的事():
    """技术判断归专家。让报修人回答"根因对不对"，等于给案例盖一个假权威章。"""
    assert [q[0] for q in wf.FIELD_QUESTIONS] == [
        "reporter", "executed", "recovered", "reading", "discrepancy"]
    for _, label, kind, _ in wf.FIELD_QUESTIONS:
        assert kind in ("text", "yesno")
        assert not re.search(r"根因|原因|对不对|是否合理|是否正确", label), label
    assert sum(1 for q in wf.FIELD_QUESTIONS if q[2] == "yesno") == 2


def test_沉淀不了的单不去要现场确认(sandbox):
    """缺报警码就没有可机器复用的匹配特征，让报修人确认一个永远入不了库的案例没有意义。"""
    paths, client, kb = sandbox
    t, code = _escalated(sandbox, NO_ALARM)
    sent_before = len(client.sent)
    res = _expert_submit(sandbox, t, code)
    wo = client.work_orders()[t.wo_id]

    assert res["promotable"] is False
    assert res["candidate"]["status"].startswith("无法沉淀")
    assert wo["状态"] == wf.STATUS_CLOSED, "不会再有现场确认，挂着就是假待办"
    assert wo["专家码"] == "" and wo["现场码"] == ""
    assert len(client.sent) == sent_before, "这一步不发橙卡"
    assert wo["实际根因"] == EXPERT["root_cause"], "闭环本身照旧记录，只是不入知识库"


def test_网页回执不得把没入库说成已入库(sandbox):
    """页面上的话必须与 workflow 真做的事一致。这一步最容易被写成"已提交，感谢配合"。"""
    import webui

    paths, client, kb = sandbox
    t, code = _escalated(sandbox)
    page = webui._close_payload(_expert_submit(sandbox, t, code), live=False)
    assert page["headline"] == "技术结论已提交，等现场作证"
    assert any("不参与任何诊断" in line for line in page["lines"])
    assert not any("已写入" in f for f in page["facts"]), "候选阶段一个字都没进知识库"
    assert page["channel"].startswith("DryRun")

    bad = wf.field_confirm(t.wo_id, _field_code(sandbox, t), "张三", "是", "否",
                           "复机 20 分钟又升到 188°C", "温度仍然爬升",
                           client=client, paths=paths)
    page = webui._close_payload(bad, live=False)
    assert page["passed"] is False and "已入库" not in page["headline"]
    assert any("一个字都没写" in line for line in page["lines"])
    assert any("现场分歧" in f for f in page["facts"])
    assert not any("cases_learned" in f for f in page["facts"])

    t2, code2 = _escalated(sandbox)
    _expert_submit(sandbox, t2, code2)
    ok = wf.field_confirm(t2.wo_id, _field_code(sandbox, t2), "张三", "是", "是", READING,
                          client=client, paths=paths)
    page = webui._close_payload(ok, live=True)
    assert ok["case_id"] == "L01" and page["case_id"] == "L01" and "L01" in page["headline"]
    assert any("cases_learned.json" in f for f in page["facts"])
    assert any(EXPERT["expert"] in f for f in page["facts"])
    assert any("张三" in f for f in page["facts"])
    assert page["channel"] == "飞书 live：专家群已收到卡片"


# ------------------------------------------------------------------ 鉴权响应解析

def test_token_从响应顶层解析(monkeypatch, tmp_path):
    """飞书鉴权接口把 tenant_access_token 放在响应顶层，不像其它接口包在 data 里。
    曾误读成 data['tenant_access_token']，导致配了真凭据 live 仍永远 KeyError。"""
    class _Resp:
        def read(self):
            return json.dumps({"code": 0, "msg": "ok",
                               "tenant_access_token": "t-fake", "expire": 7200}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("feishu.client.urllib.request.urlopen", lambda *a, **k: _Resp())
    client = Client(Config(app_id="cli_x", app_secret="sec_x", chat_id="oc_y"), var_dir=tmp_path)
    assert client.token() == "t-fake"


# ------------------------------------------------------------------ 工单表写入

class _FakeBitable:
    """伪造多维表格的字段/记录接口，并记下所有调用。fields 传 {列名: 类型}。"""

    def __init__(self, fields: dict[str, int]):
        self.meta = {n: {"field_name": n, "type": t} for n, t in fields.items()}
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, method, path, payload=None, auth=True, unwrap=True):
        self.calls.append((method, path, payload))
        if "/fields" in path:
            if method == "GET":
                return {"items": list(self.meta.values()), "has_more": False}
            self.meta[payload["field_name"]] = {"field_name": payload["field_name"],
                                                "type": payload["type"]}
            return {}
        if method == "POST" and "/search" in path:
            return {"items": []}
        if method == "POST" and path.endswith("/records"):
            return {"record": {"record_id": "rec-fake"}}
        return {}

    def posted(self, method: str, suffix: str) -> dict:
        return next(c[2] for c in self.calls if c[0] == method and c[1].endswith(suffix))


def _live_client(monkeypatch, tmp_path, fake):
    client = Client(Config(app_id="cli_x", app_secret="sec_x", chat_id="oc_y",
                           bitable_app_token="bas_x", bitable_table_id="tbl_x"),
                    var_dir=tmp_path)
    monkeypatch.setattr(client, "_request", fake)
    return client


def test_缺列自动建列后再写记录(monkeypatch, tmp_path):
    """飞书的新增记录接口不会自动建列，缺一个整条 FieldNameNotFound、工单直接写不进去，
    所以补齐列必须发生在写记录之前。"""
    fake = _FakeBitable({"工单号": 1})
    client = _live_client(monkeypatch, tmp_path, fake)

    rid = client.upsert_work_order("RX-T-1", {
        "工单号": "RX-T-1", "提取信号": {"actual_temp": 152.0}, "AI调用次数": 1})

    assert rid == "rec-fake"
    assert [c[2]["field_name"] for c in fake.calls
            if c[0] == "POST" and "/fields" in c[1]] == ["提取信号", "AI调用次数"]


def test_文本列收到非字符串会被转成字符串(monkeypatch, tmp_path):
    """文本列只收字符串：int 直接发过去飞书会拒，dict 得先转 JSON。"""
    fake = _FakeBitable({"工单号": 1})
    client = _live_client(monkeypatch, tmp_path, fake)
    client.upsert_work_order("RX-T-2", {
        "工单号": "RX-T-2", "提取信号": {"actual_temp": 152.0}, "AI调用次数": 1})

    fields = fake.posted("POST", "/records")["fields"]
    assert fields["AI调用次数"] == "1"
    assert json.loads(fields["提取信号"])["actual_temp"] == 152.0


def test_已存在的列不重复创建且非文本列原样传值(monkeypatch, tmp_path):
    """只补空缺列：已存在的单选/数字列不能被改类型，值也不能被强行转成字符串。"""
    fake = _FakeBitable({"工单号": 1, "风险等级": 3})
    client = _live_client(monkeypatch, tmp_path, fake)
    client.upsert_work_order("RX-T-3", {"工单号": "RX-T-3", "风险等级": 2})

    assert not [c for c in fake.calls if c[0] == "POST" and "/fields" in c[1]]
    assert fake.posted("POST", "/records")["fields"]["风险等级"] == 2
