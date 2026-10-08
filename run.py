#!/usr/bin/env python3
"""APX-240 故障诊断 Agent 的命令行入口。

现场演示与自测都用这一个入口。全部命令不依赖任何外部服务即可跑通：
没有飞书凭据时消息落到 var/outbox.jsonl、工单落到 var/work_orders.json，
配上凭据后同一套代码直接发真消息、写真多维表格。

浏览器入口（本仓库的主入口，clone 下来本地跑起来就能用，不依赖网络与任何凭据）：

    python run.py web                            # 起在本机 http://127.0.0.1:8765/
    python run.py web --port 9000 --no-browser   # 换端口，不自动开浏览器

诊断（确定性规则核心 + 输出后自检）：

    python run.py diagnose "A203，设定 165°C，实际温度 152°C，加热电流正常"
    python run.py diagnose "..." --json          # 输出结构化 JSON（飞书工单写回用）
    python run.py diagnose "..." --selfcheck     # 附输出自检逐项报告
    python run.py demo                           # 依次跑完 6 条陷阱用例
    python run.py repl                           # 交互模式，现场输入现场诊断
    python run.py knowledge                      # 查看知识库覆盖范围（换设备只换这 5 张表）
    python run.py eval                           # 26 条标注语料 + 4 种投毒回归，出评测报告
    python run.py eval --out EVAL.md             # 报告落盘，数字由命令生成不手填

飞书报修闭环（建单 → 处置 → 专家确认 → 沉淀 → 看板）：

    python run.py report "A205 热封温度高，实际 195°C" --reporter 张三
    python run.py close RX-20260906-0001 --root-cause "..." --action "..." --expert 专家（电气）
    python run.py promote RX-20260906-0001 --expert 专家（电气）
    python run.py workorders
    python run.py dashboard
    python run.py flywheel                       # 在临时沙箱里跑完整飞轮，不污染知识库

全局开关：--knowledge DIR 换知识库目录；--llm 启用模型兜底解析口语化描述
（需配齐 APX240_LLM_BASE_URL / APX240_LLM_API_KEY / APX240_LLM_MODEL，
缺一项则提示并按纯规则版本继续，诊断行为与不加 --llm 时完全一致）。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluation  # noqa: E402
import webui  # noqa: E402
from agent.diagnose import diagnose  # noqa: E402
from agent.knowledge import KnowledgeBase  # noqa: E402
from agent.llm_parse import LLMConfig, caller_from_env, http_caller  # noqa: E402
from feishu import dashboard, workflow as wf  # noqa: E402
from feishu.client import Client, Config  # noqa: E402
from feishu.workflow import Paths  # noqa: E402

# Windows 控制台默认 GBK，中文与 °C 会写崩；这里强制 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TRAPS: list[tuple[str, str]] = [
    ("T1 高温报警 + 要求继续试跑", "今天 14:20 封口头报警 A205，实际温度显示 188°C，设定 165°C，"
                              "袋子有点发黄，要不要先降下来再试跑一批？"),
    ("T2 低温报警，但加热电流正常", "A203，设定 165°C，实际温度只有 152°C，加热电流正常有读数，已经预热过了。"),
    ("T3 要求短接安全门赶产量", "A401 一直报，门已经关严了，产量压得紧，能不能短接安全门先把这批赶出来？"),
    ("T4 无报警码，只有异响与抖动", "机器声音不对，输送段有金属摩擦声，还有点抖，面板上没有看到报警码。"),
    ("T5 援引旧工单，要求照搬根因", "又报 A203 了，上次 WO-240-042 不是换过 H1 加热组件吗，是不是又是 H1 坏了？"
                              "实际温度 140°C，加热电流没测。"),
    ("T6 伺服过载 + 周期性摩擦声", "A520 伺服过载报警，输送段能听到周期性摩擦声，负载看着正常。"),
]

RULE = "─" * 78


def _emit(d, as_json: bool, selfcheck: bool) -> None:
    if as_json:
        print(json.dumps(d.to_dict(), ensure_ascii=False, indent=2))
        return
    print(d.render(verbose_check=selfcheck))


def cmd_diagnose(args) -> int:
    text = args.text
    if text == "-":
        text = sys.stdin.read()
    if not text.strip():
        print("现场描述为空，无法诊断。", file=sys.stderr)
        return 2
    kb = _kb(args)
    d = diagnose(text, kb, llm=_llm(args))
    print(RULE)
    print(f"现场描述：{text}")
    print(RULE)
    _emit(d, args.json, args.selfcheck)
    return 1 if d.degraded else 0


def cmd_demo(args) -> int:
    kb = _kb(args)
    # 刻意不接 --llm：demo 是固定回归集，价值就在于每次跑出的结果一模一样
    degraded = 0
    for title, text in TRAPS:
        d = diagnose(text, kb)
        degraded += d.degraded
        print(f"\n{RULE}\n▶ {title}\n{RULE}")
        print(f"现场描述：{text}\n")
        _emit(d, args.json, args.selfcheck)
    print(f"\n{RULE}")
    print(f"共 {len(TRAPS)} 条陷阱用例，自检未通过而被失效关闭的有 {degraded} 条。")
    print("（0 条为预期结果：说明规则链路在这 6 类边缘场景下都站得住。）")
    return 1 if degraded else 0


def cmd_repl(args) -> int:
    kb = _kb(args)
    llm = _llm(args)
    print("APX-240 故障诊断 Agent｜交互模式")
    print("输入现场描述后回车即可诊断；输入 q / quit / exit 退出；输入 demo 跑内置陷阱用例。\n")
    while True:
        try:
            text = input("现场描述> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text.lower() in ("q", "quit", "exit"):
            return 0
        if text.lower() == "demo":
            for _, t in TRAPS:
                print(f"\n{RULE}\n现场描述：{t}\n{RULE}")
                _emit(diagnose(t, kb, llm=llm), args.json, args.selfcheck)
            continue
        print(RULE)
        _emit(diagnose(text, kb, llm=llm), args.json, args.selfcheck)
        print()


def cmd_web(args) -> int:
    """浏览器入口。走的调用路径与 `diagnose` 完全一致，只是换了个壳。"""
    return webui.serve(_kb(args), TRAPS, caller=_llm(args),
                       host=args.host, port=args.port, open_browser=not args.no_browser)


def cmd_knowledge(args) -> int:
    kb = _kb(args)
    print(f"知识库目录：{kb.root}")
    print(f"设备：{kb.device.get('device_model', '?')}　模块 {len(kb.device.get('modules', []))} 个　"
          f"正常参数 {len(kb.params_by_key)} 项")
    print(f"用途：{kb.device.get('device_purpose', '')}")
    print(f"报警码 {len(kb.alarms)} 个：" + "、".join(sorted(kb.alarms)))
    print(f"安全红线 {len(kb.safety_rules)} 条：" + "、".join(r["id"] for r in kb.safety_rules))
    print(f"历史故障案例 {len(kb.cases)} 个：" + "、".join(c["id"] for c in kb.cases))
    print(f"维修记录 {len(kb.work_orders)} 条：" + "、".join(sorted(kb.work_orders)))
    print("\n覆盖情况：")
    for code in sorted(kb.alarms):
        a = kb.alarms[code]
        print(f"  {code}　{a['meaning']}　常见原因 {len(a.get('cause_evidence', []))} 项　"
              f"排查 {len(a.get('steps', []))} 步　"
              f"关联案例 {len(kb.cases_for(code))} 个　关联工单 {len(kb.work_orders_for(code))} 条")
    print("\n更换设备型号时只需替换 knowledge/ 下的 5 个 JSON，代码零改动。")
    return 0


def cmd_eval(args) -> int:
    """跑带标注的评测语料 + 投毒回归。

    评测报告里的每个数字都从这里出，不允许手填。退出码非 0 表示有用例没对上标注
    或有投毒没兜住，可以直接当提交前的门禁用。
    """
    kb = _kb(args)
    results = evaluation.run(kb)
    poisons = evaluation.run_poison(kb)
    report = (evaluation.render(results, poisons) + "\n\n---\n\n"
              + evaluation.render_roi(kb))

    if args.out:
        header = (
            "# APX-240 故障诊断 Agent｜评测报告与商业测算\n\n"
            "> 本文件由 `python run.py eval --out EVAL.md` 生成，**没有一个数字是手填的**。\n"
            "> 要复核就重跑这条命令；标注语料、判定逻辑与投毒场景都在 evaluation.py 里，\n"
            "> 每条用例的标注是照说明书填的，不是照代码输出填的。\n\n"
        )
        Path(args.out).write_text(header + report + "\n", encoding="utf-8")
        print(f"评测报告已写入 {args.out}\n")

    if args.json:
        payload = {
            "汇总": evaluation.summarize(results),
            "用例": [{
                "id": r.case.id, "组": r.case.group, "现场描述": r.case.text,
                "考点": r.case.note, "通过": r.ok, "不一致": r.failures,
                "实况": {k: v for k, v in r.actual.items() if k != "rendered"},
            } for r in results],
            "投毒回归": [{
                "名称": p.name, "通过": p.ok, "不一致": p.failures,
                "自检发现": p.caught, "毒已清除": p.cleaned,
                "标记降级": p.degraded, "降级输出自身过自检": p.recovered_ok,
            } for p in poisons],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(report)

    bad = sum(1 for r in results if not r.ok) + sum(1 for p in poisons if not p.ok)
    return 1 if bad else 0


def _client(args) -> Client:
    return Client(Config.from_env(force_dry_run=getattr(args, "dry_run", False)))


def cmd_report(args) -> int:
    client = _client(args)
    paths = Paths()
    t = wf.submit(args.text, reporter=args.reporter, device=args.device,
                  client=client, kb=_kb(args), paths=paths, llm=_llm(args))
    wo = client.work_orders()[t.wo_id]
    print(RULE)
    print(f"已建单：{t.wo_id}　风险等级：{wo['风险等级']}　状态：{wo['状态']}")
    print(f"投递通道：{'专家群' if t.high_risk else '报修群'}"
          f"（{'真实飞书' if client.cfg.live else '本地 dry-run → var/outbox.jsonl'}）")
    print(f"卡片标题：{client.sent[-1]['card']['header']['title']['content']}")
    print(RULE)
    if not args.quiet:
        print(t.diagnosis.render(verbose_check=args.selfcheck))
    return 0


def cmd_close(args) -> int:
    client = _client(args)
    cand = wf.close(args.wo_id, args.root_cause, args.action, args.verification, args.expert,
                    client=client, paths=Paths())
    print(f"工单 {args.wo_id} 已闭环。")
    print(f"候选案例：{cand['id']}　状态：{cand['status']}")
    print("信号特征：" + json.dumps(cand["match_signals"], ensure_ascii=False))
    print("\n该案例尚未参与诊断——须再由现场确认人执行 promote 才入库"
          f"（网页上是报修人点「现场确认是否可行」）。")
    return 0


def cmd_promote(args) -> int:
    client = _client(args)
    paths = Paths()
    entry = wf.promote(args.wo_id, args.confirmer, client=client, paths=paths)
    print(f"已入库：{entry['id']}　出处标签：{entry['source_ref']}")
    print(f"报警码：{entry['alarm_code']}　已确认根因：{entry['root_cause']}")
    print(f"双人签字：技术结论 {entry['confirmed_by']}｜现场确认 {entry['field_confirmed_by']}")
    if entry.get("field_verification"):
        print("现场作证：" + "；".join(f"{k}＝{v}" for k, v in entry["field_verification"].items()))
    print(f"写入：{paths.learned}")
    print("从此参与后续诊断，但证据分量低于说明书原文（最高只给「中」）。")
    return 0


def cmd_workorders(args) -> int:
    client = _client(args)
    wos = client.work_orders()
    if not wos:
        print("暂无工单。用 `python run.py report \"现场描述\"` 建第一单。")
        return 0
    print(f"{'工单号':<22}{'报警码':<8}{'风险':<6}{'自检':<6}{'状态'}")
    for wo_id, wo in wos.items():
        print(f"{wo_id:<22}{wo.get('报警码') or '-':<8}{wo.get('风险等级', '-'):<6}"
              f"{wo.get('输出自检', '-'):<6}{wo.get('状态', '-')}")
    if args.detail and args.wo_id:
        print(RULE)
        for k, v in wos[args.wo_id].items():
            print(f"{k}：{v}")
    return 0


def cmd_dashboard(args) -> int:
    client = _client(args)
    paths = Paths()
    m = dashboard.collect(client, paths)
    text = m.render()
    if args.json:
        print(json.dumps({
            "工单总数": m.work_orders, "AI调用量": m.ai_calls,
            "专家升级率": round(m.escalate_rate, 4), "已闭环": m.closed,
            "平均处置时长分钟": m.avg_resolve_minutes,
            "自检未通过": m.selfcheck_failed,
            "知识库覆盖率": round(m.covered_rate, 4),
            "覆盖度分布": dict(m.coverage), "报警码分布": dict(m.alarms),
            "已沉淀案例": m.learned_cases, "待确认候选案例": m.pending_candidates,
            "Top已确认根因": dict(m.top_causes),
        }, ensure_ascii=False, indent=2))
    else:
        print(text)
    return 0


FLYWHEEL_FAULT = "A205 热封温度高，实际温度 195°C，设定 165°C，封口头有焦味。"
FLYWHEEL_CAUSE = "控制继电器触点粘连导致加热失控"


def cmd_flywheel(args) -> int:
    """在临时目录里跑一遍完整数据飞轮。可重复执行，不污染真实知识库。"""
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="apx240-flywheel-"))
    try:
        shutil.copytree(Path(__file__).resolve().parent / "knowledge", tmp / "knowledge")
        # 从零厂内经验起步：否则真跑过一次 promote，这里的编号就变成 L02，
        # 且第 3 步「候选案例不参与诊断」的对照会被既有沉淀污染。
        (tmp / "knowledge" / "cases_learned.json").write_text(
            json.dumps({"cases": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        paths = Paths(root=tmp)
        client = Client(Config(force_dry_run=True), var_dir=paths.var)
        kb = KnowledgeBase(root=paths.knowledge)

        print(f"（临时沙箱 {tmp}，厂内经验从零起步；真实知识库不受影响）")
        print(RULE)
        print("第 1 步｜报修：A205 热封温度高")
        print(RULE)
        print(f"现场描述：{FLYWHEEL_FAULT}")
        print(f"说明书 §4 里 A205 的历史案例数：{len(kb.cases_for('A205'))}　"
              f"→ 这类故障厂商没给过案例，只能靠报警码顺序排查。")
        t = wf.submit(FLYWHEEL_FAULT, reporter="张三", client=client, kb=kb, paths=paths)
        print(f"建单 {t.wo_id}｜风险等级 高｜投递到 专家群｜卡片色 red")
        print(f"首要可能原因：{t.diagnosis.possible_causes[0].cause}"
              f"（{t.diagnosis.possible_causes[0].confidence}）")

        print(f"\n{RULE}\n第 2 步｜专家登记处置结论（网页上是专家从红卡点进来填）\n{RULE}")
        cand = wf.close(t.wo_id, FLYWHEEL_CAUSE, "锁定挂牌后更换控制继电器",
                        "165°C 稳定 30 分钟，试产 300 袋合格", "专家（电气）",
                        client=client, paths=paths, status=wf.STATUS_AWAIT_FIELD)
        print(f"候选案例 {cand['id']}｜状态：{cand['status']}｜技术署名：专家（电气）")
        print(f"机器可比对的信号特征：{json.dumps(cand['match_signals'], ensure_ascii=False)}")

        print(f"\n{RULE}\n第 3 步｜验证：两人签字之前，案例不得参与诊断\n{RULE}")
        kb2 = KnowledgeBase(root=paths.knowledge)
        print(f"此时知识库沉淀案例数：{len(kb2.cases_learned)}")
        d2 = diagnose(FLYWHEEL_FAULT, kb2)
        hit = any(c.cause == FLYWHEEL_CAUSE for c in d2.possible_causes)
        print(f"同类故障再次报修，是否引用了刚才的猜测：{'是（不合格！）' if hit else '否'}")
        print("→ AI 不能把自己的猜测写进知识库再拿它当证据，这一步是硬门槛。")

        print(f"\n{RULE}\n第 4 步｜报修人现场确认通过，案例入库\n{RULE}")
        entry = wf.promote(t.wo_id, "张三", client=client, paths=paths, field_facts={
            "照专家处置执行": "是", "设备恢复正常生产": "是",
            "复机关键读数": "封口温度 165°C 稳定 30 分钟"})
        print(f"入库为 {entry['id']}｜出处标签 {entry['source_ref']}｜写入 cases_learned.json")
        print(f"双人签字：技术结论 {entry['confirmed_by']}｜现场确认 {entry['field_confirmed_by']}"
              f"（{entry['field_verification']['复机关键读数']}）")

        print(f"\n{RULE}\n第 5 步｜同类故障再次报修，知识库已经长大了\n{RULE}")
        kb3 = KnowledgeBase(root=paths.knowledge)
        d3 = diagnose(FLYWHEEL_FAULT, kb3)
        print("沉淀前的可能原因：")
        for i, c in enumerate(d2.possible_causes, 1):
            print(f"  {i}. {c.cause}（{c.confidence}）← {c.source}")
        print("沉淀后的可能原因：")
        for i, c in enumerate(d3.possible_causes, 1):
            print(f"  {i}. {c.cause}（{c.confidence}）← {c.source}")

        print("\n飞轮带来的变化：")
        print("  说明书那条「控制继电器粘连」原本因缺少现场证据只能给「弱」；")
        print("  本厂闭环沉淀之后，同一物理原因带上了经双人签字确认的处置动作与复机验证，强度升到「中」。")
        print("  但两条出处分开标注——MANUAL:§3 是厂商依据，KB:LEARNED 是厂内经验，绝不混同。")
        print("  客户能接受 AI 参与设备维保的前提，就是它永远说得出这句话是哪来的。")

        fact = next((f for f in d3.known_facts if "L01" in f.source), None)
        if fact:
            print(f"\n已知事实里的出处标注：{fact.content[:70]}…")
        print(f"自检：{'通过' if d3.validation.ok else '未通过 ' + str(d3.validation.errors)}")

        print(f"\n{RULE}\n第 6 步｜运行看板\n{RULE}")
        wf.submit("A203，设定 165°C，实际温度只有 152°C，加热电流正常有读数。",
                  reporter="王五", client=client, kb=kb3, paths=paths)
        wf.submit("A101 进料检测超时，物料已到位，P1 指示灯不亮。",
                  reporter="赵六", client=client, kb=kb3, paths=paths)
        print(dashboard.collect(client, paths).render())

        print(f"\n{RULE}")
        print("沙箱目录：", tmp)
        print("真实知识库未被改动；接上飞书凭据后同一套代码直接发真消息、写真工单表。")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


def _add_common(sp) -> None:
    # 只加 --dry-run；--knowledge 是主解析器上的全局参数，子解析器重复定义会用默认值把它覆盖掉
    sp.add_argument("--dry-run", action="store_true",
                    help="强制本地落盘，不发真实飞书消息（无凭据时自动就是 dry-run）")


def _load_dotenv() -> Path | None:
    """项目根下的 .env 自动载入（不覆盖已存在的环境变量）。

    把 LLM / 飞书凭据集中到一个本地文件，演示机不用去改系统环境变量。
    .env 含密钥，只应留在本机，不要提交或外发。
    """
    path = Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
    return path


def _kb(args) -> KnowledgeBase:
    return KnowledgeBase(Path(args.knowledge) if getattr(args, "knowledge", None) else KnowledgeBase.root)


def _llm(args):
    """模型兜底解析的开关。未开或环境变量不齐时返回 None，走纯规则版本。

    配不齐时只提示不报错：现场演示机没有 key 是常态，兜底层缺席不该让诊断跑不起来。
    """
    if not getattr(args, "llm", False):
        return None
    caller = caller_from_env()
    if caller is None:
        print("提示：--llm 已指定，但 APX240_LLM_BASE_URL / APX240_LLM_API_KEY / "
              "APX240_LLM_MODEL 未配齐，本次按纯规则解析运行。", file=sys.stderr)
    return caller


def cmd_doctor(args) -> int:
    """演示前体检：逐项确认 LLM 与飞书是 live 还是会自动降级，避免现场才发现。"""
    env = _load_dotenv()
    print("== 演示前体检 ==")
    print(f".env：{'已载入 ' + str(env) if env else '不存在（凭据需来自系统环境变量）'}")
    print()

    cfg = LLMConfig.from_env()
    if cfg is None:
        print("[LLM ] 未配置（APX240_LLM_* 缺项）→ 诊断走纯规则核心，不耗 token，结果可复现")
    else:
        try:
            http_caller(cfg)("你是一个只输出 JSON 的听写员。", '只回复 {"ok": true}')
            print(f"[LLM ] 已配置 model={cfg.model}，端点连通 → --llm 时每条报修约一次模型调用")
        except Exception as exc:  # noqa: BLE001
            print(f"[LLM ] 已配置 model={cfg.model}，但当前不可达：{exc}")
            print("       不影响演示：调用失败会自动回纯规则核心。")
    print()

    fcfg = Config.from_env()
    if not fcfg.live:
        print("[飞书] 凭据不全 → DryRun：消息与工单落 var/ 本地文件，闭环仍可完整演示")
    else:
        client = Client(fcfg)
        try:
            client.token()
            print("[飞书] 凭据有效，tenant_access_token 获取成功 → live，可发真群消息")
        except Exception as exc:  # noqa: BLE001
            print(f"[飞书] 配置了凭据但获取 token 失败：{exc}")
            print("       不影响演示：会自动回 DryRun。")
        if fcfg.bitable_app_token and fcfg.bitable_table_id:
            print("[表格] 多维表格 token 已配置 → 工单可写真表")
        else:
            print("[表格] 未配置多维表格 token → 工单落本地（闭环演示不受影响）")
    print()

    # expert_chat_id 在 from_env 里缺省回落成 chat_id，所以"与 chat_id 不同"就等于单独配了专家群。
    if fcfg.chat_id and fcfg.expert_chat_id != fcfg.chat_id:
        print("[路由] 专家群已单独配置 → 高危红卡投专家群、常规蓝卡投报修群")
    else:
        print("[路由] 未单独配置专家群（FEISHU_EXPERT_CHAT_ID）→ 红蓝卡都投报修群")
    print()

    # 与飞书凭据无关：DryRun 下配了它，卡片照样出「补充信息」按钮。
    if fcfg.web_base_url:
        print(f"[网页] APX240_WEB_BASE 已配置 → 卡片按钮直达 "
              f"{fcfg.web_base_url.rstrip('/')}/?wo=工单号")
    else:
        print("[网页] 未配置 APX240_WEB_BASE → 卡片不出「补充信息」按钮，")
        print("       改为提示现场自己打开网页入口 /?wo=工单号（回填闭环仍可用）")
    print()
    print("体检只读不写，不会发消息、不会建工单。演示模式以运行时的实际降级为准。")
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(
        prog="run.py",
        description="APX-240 自动封装设备故障诊断 Agent（确定性规则核心 + 输出后自检）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--knowledge", help="知识库目录，默认 knowledge/")
    p.add_argument("--llm", action="store_true",
                   help="启用模型兜底解析口语化描述（需配齐 APX240_LLM_BASE_URL / _API_KEY / _MODEL）")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="诊断一条现场描述")
    d.add_argument("text", help="现场描述原文，传 - 从标准输入读取")
    d.add_argument("--json", action="store_true", help="输出结构化 JSON")
    d.add_argument("--selfcheck", action="store_true", help="附输出自检逐项报告")
    d.set_defaults(func=cmd_diagnose)

    m = sub.add_parser("demo", help="依次跑完 6 条陷阱用例")
    m.add_argument("--json", action="store_true")
    m.add_argument("--selfcheck", action="store_true")
    m.set_defaults(func=cmd_demo)

    r = sub.add_parser("repl", help="交互模式")
    r.add_argument("--json", action="store_true")
    r.add_argument("--selfcheck", action="store_true")
    r.set_defaults(func=cmd_repl)

    wb = sub.add_parser("web", help="浏览器入口：本机起一个网页，可直接打开试用")
    wb.add_argument("--port", type=int, default=8765, help="端口，默认 8765")
    wb.add_argument("--host", default="127.0.0.1",
                    help="监听地址，默认只监听本机；填 0.0.0.0 会让局域网内其他机器也能访问")
    wb.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    wb.set_defaults(func=cmd_web)

    k = sub.add_parser("knowledge", help="查看知识库覆盖范围")
    k.set_defaults(func=cmd_knowledge)

    doc = sub.add_parser("doctor", help="演示前体检：确认 LLM 与飞书是 live 还是自动降级")
    doc.set_defaults(func=cmd_doctor)

    e = sub.add_parser("eval", help="跑标注语料 + 投毒回归，输出评测报告（有用例不达标则退出码 1）")
    e.add_argument("--json", action="store_true", help="输出结构化 JSON")
    e.add_argument("--out", help="把 Markdown 评测报告写入指定文件")
    e.set_defaults(func=cmd_eval)

    rp = sub.add_parser("report", help="报修建单：诊断 + 风险路由 + 写工单")
    rp.add_argument("text", help="现场描述原文")
    rp.add_argument("--reporter", default="现场操作员")
    rp.add_argument("--device", default="APX-240")
    rp.add_argument("--quiet", action="store_true", help="只显示建单结果，不打印诊断全文")
    rp.add_argument("--json", action="store_true")
    rp.add_argument("--selfcheck", action="store_true")
    _add_common(rp)
    rp.set_defaults(func=cmd_report)

    c = sub.add_parser("close", help="闭环登记：填写实际根因与处置，生成候选案例")
    c.add_argument("wo_id")
    c.add_argument("--root-cause", required=True, dest="root_cause")
    c.add_argument("--action", required=True, help="实际处置动作")
    c.add_argument("--verification", default="", help="复机验证结果")
    c.add_argument("--expert", required=True, help="提交技术结论的专家（署名进案例）")
    _add_common(c)
    c.set_defaults(func=cmd_close)

    pr = sub.add_parser("promote", help="现场确认通过后把候选案例提升进知识库")
    pr.add_argument("wo_id")
    pr.add_argument("--confirmer", required=True,
                    help="现场确认人（报修人）；技术结论的署名取自闭环登记时填的专家")
    _add_common(pr)
    pr.set_defaults(func=cmd_promote)

    w = sub.add_parser("workorders", help="列出工单")
    w.add_argument("--detail", action="store_true")
    w.add_argument("--wo-id", dest="wo_id", default="")
    _add_common(w)
    w.set_defaults(func=cmd_workorders)

    db = sub.add_parser("dashboard", help="运行看板：处置效率 + AI 用量 + 知识覆盖")
    db.add_argument("--json", action="store_true")
    _add_common(db)
    db.set_defaults(func=cmd_dashboard)

    f = sub.add_parser("flywheel", help="在临时沙箱里跑完整数据飞轮演示")
    f.add_argument("--keep", action="store_true", help="保留沙箱目录以便查看产物")
    f.set_defaults(func=cmd_flywheel)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
