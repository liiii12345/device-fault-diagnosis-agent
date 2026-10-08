"""浏览器入口的测试。

这一层存在的理由很直接：只有命令行时，想确认输出长什么样的得先读代码。所以测试的重点不是页面好不好看，
而是两件事：

1. **网页壳没有改变任何判定**。同一条现场描述，走 HTTP 与走 `diagnose()` 必须给出逐字相同的结果——
   一旦这一层里混进了自己的判断逻辑，两道闸门就有了绕过路径。
2. **用户输入不会被当成 HTML 执行**。现场描述是外部输入，页面只要有一处 innerHTML 就是一个 XSS，
   所以这里直接对页面源码断言，把它钉死成回归项。
"""

from __future__ import annotations

import inspect
import json
import sys
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import webui  # noqa: E402
from agent.diagnose import diagnose  # noqa: E402
from agent.knowledge import KnowledgeBase  # noqa: E402
from agent.validate import BANNED_PHRASES  # noqa: E402
from feishu.workflow import CodeRejected  # noqa: E402

KB = KnowledgeBase()
SAMPLES = [("样例一", "A203，设定 165°C，实际温度只有 152°C，加热电流正常有读数。")]
NORMAL = SAMPLES[0][1]
VIOLATION = "A401 一直报，门已经关严了，产量压得紧，能不能短接安全门先把这批赶出来？"


@pytest.fixture(scope="module")
def base_url():
    """绑到随机端口起真实服务，测的是整条 HTTP 链路而不是 handler 函数。"""
    httpd = webui.build_server(KB, SAMPLES, None, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def _get(url: str) -> tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "")


def _post(url: str, body: bytes) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=body,
                                headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw, code = resp.read(), resp.status
    except urllib.error.HTTPError as exc:
        raw, code = exc.read(), exc.code
    return code, json.loads(raw.decode("utf-8"))


def _diagnose(base_url: str, text: str) -> dict:
    code, data = _post(f"{base_url}/api/diagnose",
                       json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"))
    assert code == 200, data
    return data


def _sections(rendered: str, *names: str) -> str:
    """从渲染文本里切出指定字段的正文，只看告诉现场「该做什么」的那几段。"""
    out, keep = [], False
    for line in rendered.split("\n"):
        if line.startswith("## "):
            keep = any(n in line for n in names)
        elif keep:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------- 页面本身

def test_首页可打开且声明utf8(base_url):
    code, body, ctype = _get(f"{base_url}/")
    assert code == 200
    assert "text/html" in ctype and "utf-8" in ctype
    assert "APX-240" in body.decode("utf-8")


def test_页面不得出现innerHTML():
    """现场描述是用户输入，回显时只要有一处 `.innerHTML =` 就是一个 XSS。

    这条断言看着像在测实现细节，其实是唯一能长期守住这一点的东西：
    以后谁为了让输出好看一点把 textContent 换成 innerHTML，测试当场就红。
    """
    assert ".innerHTML" not in webui.PAGE
    assert "document.write" not in webui.PAGE


def test_页面不引任何外部资源():
    """断网也要能打开。CDN 上的字体或脚本一旦加载失败，看到的就是白屏。"""
    for marker in ("http://", "https://", "//cdn", "@import"):
        assert marker not in webui.PAGE, f"页面里出现了外部依赖：{marker}"


def test_默认只监听本机():
    """默认绑 loopback。开放到局域网是显式动作，不该是默认值。"""
    params = inspect.signature(webui.serve).parameters
    assert params["host"].default == "127.0.0.1"


def test_首页响应带基本安全头(base_url):
    with urllib.request.urlopen(f"{base_url}/", timeout=10) as resp:
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"


# ---------------------------------------------------------------- 知识覆盖接口

def test_知识覆盖接口报告真实知识库(base_url):
    code, body, _ = _get(f"{base_url}/api/knowledge")
    assert code == 200
    data = json.loads(body.decode("utf-8"))
    assert data["device"] == KB.device["device_model"]
    assert data["alarm_count"] == len(KB.alarms)
    assert data["rule_count"] == len(KB.safety_rules)
    assert {a["code"] for a in data["alarms"]} == set(KB.alarms)


def test_样例接口原样回传(base_url):
    code, body, _ = _get(f"{base_url}/api/samples")
    assert code == 200
    assert json.loads(body.decode("utf-8")) == [
        {"label": label, "text": text} for label, text in SAMPLES]


# ---------------------------------------------------------------- 诊断接口

def test_网页结果与命令行逐字相同(base_url):
    """这一层不许有自己的判断。渲染文本、覆盖度、停机与升级判定全部对齐 diagnose()。"""
    got = _diagnose(base_url, NORMAL)
    want = diagnose(NORMAL, KB)

    assert got["rendered"] == want.render(verbose_check=True)
    assert got["coverage"] == want.coverage
    assert got["degraded"] is want.degraded
    assert got["stop_and_escalate"]["must_stop"] is want.stop_and_escalate.must_stop
    assert got["stop_and_escalate"]["escalate_to_expert"] is want.stop_and_escalate.escalate_to_expert
    assert got["stop_and_escalate"]["triggered_rules"] == want.stop_and_escalate.triggered_rules


def test_自检结果随响应一起给出(base_url):
    got = _diagnose(base_url, NORMAL)
    want = diagnose(NORMAL, KB)
    assert [(c["name"], c["ok"]) for c in got["validation"]] == [
        (c.name, c.ok) for c in want.validation.checks]
    assert all(c["ok"] for c in got["validation"]), got["validation"]


def test_违规请求经网页入口同样被拒绝(base_url):
    """闸门必须从浏览器这一侧也拦得住——否则换个入口就等于绕过了安全门控。"""
    got = _diagnose(base_url, VIOLATION)
    assert got["refused_requests"], "「短接安全门」这类请求应当被明确拒绝"
    assert got["stop_and_escalate"]["must_stop"] is True

    # 故障现象里照录现场原话「能不能短接安全门」是应该的，那不是建议；
    # 真正不许出现的是把违规做法写进告诉现场「该做什么」的那两段。
    actions = _sections(got["rendered"], "安全前置条件", "排查顺序")
    assert actions, "违规请求场景下仍应给出安全前置条件与排查顺序"
    hit = [p for p in BANNED_PHRASES if p in actions]
    assert not hit, f"可执行段落里出现了违规建议：{hit}"


def test_只约束作业方式的红线不算强制():
    """SAFE-02（断电锁定挂牌）与 SAFE-04（授权边界）不叫停作业。

    把它们和 SAFE-03 画成同一种红，「可继续排查」旁边就会糊着一片红，
    这一屏第一眼就是自相矛盾。分类依据取自 safety_rules.json 自身字段，
    换一台设备、换一张规则表，网页不会拿旧名单去解释新规则。
    """
    got = {r["id"]: r["hard"] for r in webui._rules(KB, [r["id"] for r in KB.safety_rules])}
    assert got["SAFE-02"] is False
    assert got["SAFE-04"] is False
    assert got["SAFE-01"] is True
    assert got["SAFE-03"] is True
    assert got["SAFE-05"] is True


def test_红线随响应带上规则原文(base_url):
    """页面上悬停要能看到规则原文，所以原文必须随响应给出，不能由前端自己编。"""
    rules = _diagnose(base_url, VIOLATION)["stop_and_escalate"]["rules"]
    by_id = {r["id"]: r for r in KB.safety_rules}
    assert rules
    for r in rules:
        assert r["requirement"] == by_id[r["id"]]["requirement"]
    assert any(r["id"] == "SAFE-03" and r["hard"] for r in rules)


def test_中文与摄氏度符号往返不丢字(base_url):
    got = _diagnose(base_url, "A205 热封温度高，实际 195°C，袋子发黄。")
    assert "195" in got["rendered"]
    assert "A205" in got["rendered"]


def test_未覆盖的报警码经网页入口也不编造(base_url):
    got = _diagnose(base_url, "面板报了个 A777，说明书上翻不到这个码，机器还在响。")
    assert got["coverage"] == "none"
    assert got["stop_and_escalate"]["escalate_to_expert"] is True
    assert "知识库未覆盖" in got["rendered"]


# ---------------------------------------------------------------- 错误路径

def test_空描述返回400(base_url):
    code, data = _post(f"{base_url}/api/diagnose", json.dumps({"text": "   "}).encode())
    assert code == 400
    assert "为空" in data["error"]


def test_非法json返回400(base_url):
    code, data = _post(f"{base_url}/api/diagnose", b"{not json")
    assert code == 400
    assert "JSON" in data["error"]


def test_非utf8字节返回400而不是500(base_url):
    code, data = _post(f"{base_url}/api/diagnose", b'{"text":"\xff\xfe\x80"}')
    assert code == 400
    assert "JSON" in data["error"]


def test_超长描述返回400(base_url):
    """请求体上限是唯一的长度闸门：JSON 解码只会让文本更短，不需要第二道检查。"""
    code, data = _post(f"{base_url}/api/diagnose",
                       json.dumps({"text": "A203 " * (webui.MAX_BODY // 2)}).encode())
    assert code == 400
    assert str(webui.MAX_BODY) in data["error"]


def test_刚好在上限内的描述照常诊断(base_url):
    """闸门只拦真正超长的，不能顺手把合法的长描述也挡掉。"""
    text = NORMAL + "补充说明。" * ((webui.MAX_BODY - 200) // 15)
    body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    assert len(body) <= webui.MAX_BODY
    assert _diagnose(base_url, text)["coverage"] in ("full", "partial", "none")


def test_未知路径返回404(base_url):
    assert _get(f"{base_url}/nope")[0] == 404
    assert _post(f"{base_url}/api/nope", b"{}")[0] == 404


# ---------------------------------------------------------------- 报修建单

def _fake_ticket(text: str, round_no: int = 1, prev_wo: str = "") -> SimpleNamespace:
    d = diagnose(text, KB)
    return SimpleNamespace(wo_id="RX-20260907-0007", diagnosis=d, high_risk=False,
                           record_id="rec-fake", message_id="om-fake",
                           round=round_no, prev_wo=prev_wo)


@contextmanager
def _reporting_server(report_fn):
    """另起一个服务，把建单这一步换成假实现：测试不能在客户表里留真工单。"""
    httpd = webui.build_server(KB, SAMPLES, None, "127.0.0.1", 0, report_fn=report_fn)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def _report(base_url: str, text: str) -> dict:
    code, data = _post(f"{base_url}/api/diagnose",
                       json.dumps({"text": text, "report": True}, ensure_ascii=False).encode("utf-8"))
    assert code == 200, data
    return data


def test_默认只诊断不建单(base_url):
    """勾不勾选是显式的：页面上随手点「诊断」不该在客户工单表里塞一行。"""
    assert "ticket" not in _diagnose(base_url, NORMAL)


def test_报修建单返回工单号与落库凭证():
    with _reporting_server(lambda t, r, kb, c: (_fake_ticket(t), True)) as base:
        ticket = _report(base, NORMAL)["ticket"]

    assert ticket["wo_id"] == "RX-20260907-0007"
    assert ticket["record_id"] == "rec-fake"
    assert ticket["message_id"] == "om-fake"
    assert "飞书 live" in ticket["channel"]


def test_没配凭据时如实标成DryRun():
    with _reporting_server(lambda t, r, kb, c: (_fake_ticket(t), False)) as base:
        ticket = _report(base, NORMAL)["ticket"]

    assert "DryRun" in ticket["channel"], "不能把本地落盘说成已推送飞书"


def test_报修分支的结论与纯诊断逐字相同():
    """壳可以加动作，不能加判断：建单前后必须是同一份结论，也不能多烧一次模型调用。"""
    calls = []

    def fake(text, reporter, kb, caller):
        calls.append(text)
        return _fake_ticket(text), True

    with _reporting_server(fake) as base:
        data = _report(base, NORMAL)

    assert calls == [NORMAL], "建单应复用同一次诊断，重复诊断等于多烧一次调用"
    assert data["rendered"] == diagnose(NORMAL, KB).render(verbose_check=True)


def test_建单失败仍给出可用诊断():
    """飞书不通、表没授权、网络抖动——都不能让页面变成 500，诊断本身是独立成立的。"""
    def boom(text, reporter, kb, caller):
        raise RuntimeError("bitable 未授权")

    with _reporting_server(boom) as base:
        data = _report(base, NORMAL)

    assert "bitable 未授权" in data["ticket"]["report_error"]
    assert data["rendered"] == diagnose(NORMAL, KB).render(verbose_check=True)


# ---------------------------------------------------------------- 补充信息回填

FOLLOW_FORM = {"wo_id": "RX-20260907-0007", "round": 2, "text": NORMAL, "reporter": "张三",
               "alarm_code": "A203", "verdict": "风险常规｜未升级专家｜自检通过",
               "items": ["是否已完成预热", "故障发生时间点"]}
ANSWERS = {"是否已完成预热": "开机后已经预热 40 分钟了"}
MERGED = NORMAL + "。补充信息：是否已完成预热：开机后已经预热 40 分钟了"


def _fake_followup(wo_id: str, answers: dict) -> tuple[SimpleNamespace, bool]:
    d = diagnose(MERGED, KB)
    return SimpleNamespace(wo_id="RX-20260907-0008", diagnosis=d, high_risk=False,
                           record_id="rec-2", message_id="om-2", round=2, prev_wo=wo_id), True


def _bomb(*args, **kwargs):
    raise AssertionError("这一分支不该走到第二轮")


def _fake_form(wo_id: str) -> dict:
    """与真实现同构：认不出的工单号抛 KeyError，否则 404 分支永远测不到。"""
    if wo_id != FOLLOW_FORM["wo_id"]:
        raise KeyError(wo_id)
    return dict(FOLLOW_FORM)


@contextmanager
def _follow_server(followup_fn=_bomb, form_fn=_fake_form):
    """另起一个服务，把回填表单与第二轮都换成假实现。

    表单默认实现会去读演示机上的真工单存档，测试跑一次就依赖一次现场状态——不行。
    """
    httpd = webui.build_server(KB, SAMPLES, None, "127.0.0.1", 0, followup_fn=followup_fn,
                               followup_form_fn=form_fn)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def _get_json(url: str) -> tuple[int, dict]:
    code, body, _ = _get(url)
    return code, json.loads(body.decode("utf-8"))


def test_回填页按工单自己出题目():
    """题目、原描述、轮次都取自工单存档，不再问现场要第二遍。"""
    with _follow_server() as base:
        code, data = _get_json(f"{base}/api/followup?wo=RX-20260907-0007")
    assert code == 200
    assert data == FOLLOW_FORM


def test_工单不存在时回填页报404():
    with _follow_server() as base:
        code, data = _get_json(f"{base}/api/followup?wo=RX-00000000-0000")
    assert code == 404
    assert "RX-00000000-0000" in data["error"]


def test_缺工单号时报404而不是500():
    with _follow_server() as base:
        code, data = _get_json(f"{base}/api/followup")
    assert code == 404
    assert "缺工单号" in data["error"]


def test_提交补充信息拿到第二轮完整结论():
    seen = []

    def fake(wo_id, answers, kb, caller):
        seen.append((wo_id, answers))
        return _fake_followup(wo_id, answers)

    with _follow_server(followup_fn=fake) as base:
        code, data = _post(f"{base}/api/followup", json.dumps(
            {"wo": "RX-20260907-0007", "answers": ANSWERS}, ensure_ascii=False).encode("utf-8"))

    assert code == 200
    assert seen == [("RX-20260907-0007", ANSWERS)], "壳只负责转交，不许自己改写答案"
    assert data["rendered"] == diagnose(MERGED, KB).render(verbose_check=True)
    assert data["ticket"]["round"] == 2
    assert data["ticket"]["prev_wo"] == "RX-20260907-0007"
    assert data["ticket"]["wo_id"] == "RX-20260907-0008"
    assert "飞书 live" in data["ticket"]["channel"]


def test_第二轮缺工单号返回400():
    with _follow_server() as base:
        code, data = _post(f"{base}/api/followup",
                           json.dumps({"answers": ANSWERS}).encode("utf-8"))
    assert code == 400
    assert "工单号" in data["error"]


def test_答案对不上待补项时返回400():
    def nope(wo_id, answers, kb, caller):
        raise ValueError("没有一条补充对得上工单 RX-1 的待补项")

    with _follow_server(followup_fn=nope) as base:
        code, data = _post(f"{base}/api/followup",
                           json.dumps({"wo": "RX-1", "answers": {"随便写一项": "有"}}).encode("utf-8"))
    assert code == 400
    assert "待补项" in data["error"]


def test_未送达时不许说群里已收到卡片():
    """通知失败要如实说：把未送达说成已推送，现场就会干等一张不会来的卡。"""
    t, _ = _fake_followup("RX-20260907-0007", ANSWERS)
    assert "群里已收到卡片" in webui._ticket(t, True)["channel"]

    t.message_id = ""
    assert "未送达" in webui._ticket(t, True)["channel"]
    assert "DryRun" in webui._ticket(t, False)["channel"]


# ---------------------------------------------------------------- 双人确认入库

CLOSE_WO = "RX-20260907-0007"
CLOSE_CODE = "7QB4DC"
EXPERT_FORM = {
    "wo_id": CLOSE_WO, "stage": "expert", "alarm_code": "A205", "device": "APX-240",
    "text": NORMAL, "reporter": "张三", "status": "已升级专家", "round": 1,
    "selfcheck": "通过", "risk": "高", "escalated": True, "rejected": {},
    "questions": [{"key": "expert", "label": "署名：谁下的这个技术结论", "kind": "text",
                   "required": True, "value": ""}],
}
EXPERT_ANSWERS = {"expert": "专家（电气）", "root_cause": "控制继电器触点粘连导致加热失控",
                  "action_taken": "断电锁定挂牌后更换控制继电器",
                  "verification": "封口温度 165°C 稳定 30 分钟"}
FIELD_RESULT = {
    "stage": "field", "passed": True, "promotable": True, "reporter": "张三",
    "expert": "专家（电气）", "case_id": "L01", "message_id": "om-fake",
    "facts": {"照专家处置执行": "是", "设备恢复正常生产": "是",
              "复机关键读数": "封口温度 165°C 稳定 30 分钟"},
    "candidate": {"id": f"CAND-{CLOSE_WO}", "status": "已提升为 L01", "alarm_code": "A205",
                  "source_work_order": CLOSE_WO, "root_cause": EXPERT_ANSWERS["root_cause"],
                  "disposition": EXPERT_ANSWERS["action_taken"],
                  "verification": EXPERT_ANSWERS["verification"],
                  "confirmed_by": "专家（电气）"},
}


@contextmanager
def _close_server(form_fn=None, close_fn=None):
    """另起一个服务，把双人确认的两个阶段都换成假实现。

    真实现会去读演示机上的工单存档、并真写 knowledge/cases_learned.json——
    测试跑一次就把知识库改一次，不行。
    """
    httpd = webui.build_server(KB, SAMPLES, None, "127.0.0.1", 0,
                               close_form_fn=form_fn, close_fn=close_fn)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def _close_get(base: str, wo: str = CLOSE_WO, code: str = CLOSE_CODE):
    return _get_json(f"{base}/api/close?wo={wo}&k={code}")


def _close_post(base: str, answers: dict, wo: str = CLOSE_WO, code: str = CLOSE_CODE):
    return _post(f"{base}/api/close",
                 json.dumps({"wo": wo, "k": code, "answers": answers},
                            ensure_ascii=False).encode("utf-8"))


def test_确认页的题目由服务端下发():
    """问哪几道题、哪些必填，都是服务端说了算——页面能改题目，"不问技术判断"就成了空话。"""
    seen = []

    def form(wo_id, code):
        seen.append((wo_id, code))
        return dict(EXPERT_FORM)

    with _close_server(form_fn=form) as base:
        code, data = _close_get(base)

    assert code == 200
    assert seen == [(CLOSE_WO, CLOSE_CODE)], "壳只负责转交工单号与确认码"
    assert data["stage"] == "expert"
    assert data["questions"][0]["key"] == "expert"


def test_确认码不对回403而不是400():
    """码不对是权限问题，不是"参数填错了"。混成 400，日志里就看不出有人在试别人的单。"""
    def rejected(wo_id, code):
        raise CodeRejected("确认码不对：请从群里的卡片重新点开，不要手抄地址")

    with _close_server(form_fn=rejected) as base:
        code, data = _close_get(base)
    assert code == 403
    assert "确认码不对" in data["error"]


def test_工单不存在回404且消息原样透传():
    def missing(wo_id, code):
        raise KeyError(f"工单 {wo_id} 不存在")

    with _close_server(form_fn=missing) as base:
        code, data = _close_get(base, wo="RX-00000000-0000")
    assert code == 404
    assert data["error"] == "工单 RX-00000000-0000 不存在", "str(KeyError) 会带上引号"


def test_专家还没提交时现场确认页回404():
    def no_candidate(wo_id, code):
        raise KeyError(f"工单 {wo_id} 没有待现场确认的候选案例：专家还没提交处置结论")

    with _close_server(form_fn=no_candidate) as base:
        code, data = _close_get(base)
    assert code == 404
    assert "专家还没提交处置结论" in data["error"]


def test_提交转交给workflow且回执给出案例编号():
    seen = []

    def close(wo_id, code, answers, kb, caller):
        seen.append((wo_id, code, answers))
        return dict(FIELD_RESULT), True

    with _close_server(close_fn=close) as base:
        code, data = _close_post(base, EXPERT_ANSWERS)

    assert code == 200
    assert seen == [(CLOSE_WO, CLOSE_CODE, EXPERT_ANSWERS)], "答案不许被壳改写"
    assert data["passed"] is True and data["case_id"] == "L01"
    assert "L01" in data["headline"]
    assert any("cases_learned.json" in f for f in data["facts"])
    assert any("专家（电气）" in f for f in data["facts"])
    assert any("张三" in f for f in data["facts"])
    assert data["channel"] == "飞书 live：专家群已收到卡片"


def test_缺确认码的提交回403():
    """没有码就没有这一步的权限：不能靠"忘了带参数"绕过去。"""
    with _close_server(close_fn=_bomb) as base:
        code, data = _post(f"{base}/api/close",
                           json.dumps({"wo": CLOSE_WO, "answers": EXPERT_ANSWERS}).encode())
    assert code == 403
    assert "确认码" in data["error"]


def test_提交时码已作废回403():
    def rejected(wo_id, code, answers, kb, caller):
        raise CodeRejected(f"工单 {wo_id} 当前没有待处理的确认码（可能已经处理过了）")

    with _close_server(close_fn=rejected) as base:
        code, data = _close_post(base, EXPERT_ANSWERS)
    assert code == 403
    assert "已经处理过了" in data["error"]


def test_内容填错回400():
    def bad(wo_id, code, answers, kb, caller):
        raise ValueError("现场与专家结论不符时必须写清实际情况，否则专家无从修正")

    with _close_server(close_fn=bad) as base:
        code, data = _close_post(base, {"executed": "是", "recovered": "否"})
    assert code == 400
    assert "写清实际情况" in data["error"]


def test_确认页也是同一条无外链无innerHTML的页面():
    """双人确认页与首页共用一份 PAGE：外部依赖与 XSS 的约束对它同样成立。"""
    for marker in ('id="close"', 'id="close-items"', "/api/close"):
        assert marker in webui.PAGE
    assert ".innerHTML" not in webui.PAGE
    assert "https://" not in webui.PAGE


def test_隐藏属性必须压过自带display的样式():
    """.samples 与 label.chk 自己写了 display，会盖掉 hidden 属性默认的 display:none——
    切到回填页/确认页后首页那排按钮和「报修建单」复选框还挂在屏幕上。样式表里必须有兜底。"""
    assert "[hidden] { display:none !important; }" in webui.PAGE
