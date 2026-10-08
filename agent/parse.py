"""现场描述 → 结构化信号。

这是确定性核心，不依赖任何大模型。每个信号都记录触发它的原文片段（span），
使「已知事实」能够逐条回指现场原话，而不是转述成一句无从核对的话；
「需要补充的信息」也要靠 span 判断"现场说了但没说数值"。

大模型只在规则解析失败时作为兜底（见 llm_parse），且其输出必须能被同一套
信号校验器复核，不允许直接写进已知事实。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

ALARM_RE = re.compile(r"\bA\d{3}\b")
WORK_ORDER_RE = re.compile(r"WO-\d{3}-\d{3}")
CLOCK_RE = re.compile(r"\d{1,2}:\d{2}")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
NUM = r"(\d+(?:\.\d+)?)"

# 现场口述常用中文数字（「实际温度一百五十二度」），只认到千位——温度、气压、电流
# 的量程用不到万。要求串里至少出现一个单位字，否则「一九五」这类逐字念法会被误读成 5。
CN_NUM = r"([零一二两三四五六七八九十百千]{2,})"
# 数字前面的填充词不能再吞掉数字本身的字符：贪婪匹配会把「一百五十二度」回溯成「十二」。
GAP_CN = r"[^。；\n\d零一二两三四五六七八九十百千]"
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
CN_NUM_RE = re.compile(CN_NUM)


def cn_to_number(s: str) -> float | None:
    """中文数字 → 数值。不含单位字（十/百/千）的一律不认，返回 None。"""
    if not any(ch in _CN_UNITS for ch in s):
        return None
    total = 0
    digit: int | None = None
    for ch in s:
        if ch in _CN_DIGITS:
            digit = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (1 if digit is None else digit) * _CN_UNITS[ch]  # 「十五」= 1×10+5
            digit = None
        else:
            return None
    return float(total + (digit or 0))


def cn_number_spans(text: str) -> list[tuple[str, float]]:
    """扫出文中的中文数字表达式及其数值：[(原文片段, 数值), ...]。

    解析层与输出自检层共用这一个函数，避免两边对「这个数字算不算来自现场原文」
    各有一套口径——口径一分叉，自检就会把自己解析出来的读数判成幻觉。
    """
    out: list[tuple[str, float]] = []
    for m in CN_NUM_RE.finditer(text):
        v = cn_to_number(m.group(1))
        if v is not None:
            out.append((m.group(0), v))
    return out


_DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """文中出现的全部数值，写法归一化（188 / 188.0 / 042 视为同一个数）。

    阿拉伯数字与中文数字一起收，供两处共用：输出自检的数字溯源白名单、
    模型兜底解析的读数复核。两边必须用同一个口径，否则一边认的数另一边判成幻觉。
    """
    out: set[str] = set()
    for tok in _DIGIT_RE.findall(text):
        try:
            out.add(f"{float(tok):g}")
        except ValueError:  # pragma: no cover - 正则已保证可解析
            out.add(tok)
    out.update(f"{v:g}" for _, v in cn_number_spans(text))
    return out


# 这些旗标仅当为 True 时才代表"现场描述提及了"，False 等同于未提及。
# 与 door_closed 等三态信号（None=未提及 / False=明确否定）语义不同。
PRESENCE_FLAGS = frozenset({
    "smoke", "burning_smell", "abnormal_high_temp", "violent_vibration",
    "metal_friction_sound", "part_loose", "wants_continue_run", "discoloration",
})


@dataclass
class Signals:
    """从现场描述中解析出的全部信号。值为 None 表示未提及（而非否定）。"""

    raw: str
    spans: dict[str, str] = field(default_factory=dict)

    alarm_code: str | None = None
    # 现场一次报出多个码时全部留档。诊断仍按首个码走（一单一码是维保 SOP），
    # 但"还有别的码"这件事必须让诊断链路知道——多码并发本身就是异常工况。
    alarm_codes: list[str] = field(default_factory=list)
    occurrence_time: str | None = None
    mentioned_work_orders: list[str] = field(default_factory=list)

    # 热封
    setpoint_temp: float | None = None
    actual_temp: float | None = None
    heating_current: float | None = None
    heating_current_state: str | None = None  # "zero" | "normal" | None
    preheated: bool | None = None

    # 气路
    upstream_pressure: float | None = None
    device_pressure: float | None = None
    leak_sound: bool | None = None

    # 进料
    material_present: bool | None = None
    p1_light: str | None = None  # "on" | "off" | None
    belt_slipping: bool | None = None

    # 安全门
    door_closed: bool | None = None
    door_obstructed: bool | None = None
    alarm_intermittent: bool | None = None

    # 伺服/传动
    friction_sound: bool | None = None
    friction_periodic: bool | None = None
    visible_jam: bool | None = None

    # SAFE-01 立即停机信号
    smoke: bool = False
    burning_smell: bool = False
    abnormal_high_temp: bool = False
    violent_vibration: bool = False
    metal_friction_sound: bool = False
    part_loose: bool = False

    # 用户请求（对抗性输入）
    user_request: str | None = None
    wants_continue_run: bool = False

    # 其他可见异常
    discoloration: bool = False

    def set(self, name: str, value: Any, span: str) -> None:
        setattr(self, name, value)
        self.spans.setdefault(name, span)

    def known_keys(self) -> set[str]:
        """已被现场描述确定的信号。

        语义区分：door_closed / alarm_intermittent / preheated 等三态信号用 None 表示
        未提及、False 表示明确否定（「一直报」反驳间歇特征本身就是证据）；而
        PRESENCE_FLAGS 里的旗标只在为 True 时才算已知，False 等同于未提及。
        """
        skip = {"raw", "spans", "alarm_codes"}
        out = set()
        for k, v in self.__dict__.items():
            if k in skip or v is None or v == []:
                continue
            if k in PRESENCE_FLAGS and v is False:
                continue
            out.add(k)
        return out


def _first(pattern: str, text: str) -> tuple[str, re.Match[str] | None]:
    m = re.search(pattern, text)
    return (m.group(0) if m else ""), m


# 问句标签：以冒号收尾、句中带「是否」的那一小段（「是否已完成预热：」）。
_QUESTION_CLAUSE = re.compile(r"[^，。；\n]*是否[^，。；\n]*[:：]")
_CLAUSE_BREAK = re.compile(r"[，。；、：:！!？?\s]")
# 否定词。命中处之前的小句里连「无」一起算；命中词自身只算「没未非别」——
# 「时有时无」是肯定说法，「电流不为零」也是，两处都不能被当成否认。
_NEG_BEFORE = "没无未非别不"
_NEG_INSIDE = "没未非别"


def _last(patterns: list[str], text: str, skip_question: bool = True) -> re.Match[str] | None:
    """取最后一次、且不落在问句标签里的命中。

    取最后一次：现场会把同一件事说两遍（「一开始有焦味，现在没了」），后一句才是当前状态。
    跳过问句：回填后的文本形如「是否已完成预热：已经预热 40 分钟」，标签里的关键词不是
    现场的陈述——把它算作命中，现场答一句「不清楚」也会被读成「已预热」，那是凭空编事实。
    skip_question=False 只给违规请求识别用：宁可多拦一次，不能让问句写法把闸门绕过去。
    """
    questions = [m.span() for m in _QUESTION_CLAUSE.finditer(text)] if skip_question else []
    best: re.Match[str] | None = None
    for p in patterns:
        for m in re.finditer(p, text):
            if any(a <= m.start() < b for a, b in questions):
                continue
            if best is None or m.start() > best.start():
                best = m
    return best


def _negated(text: str, m: re.Match[str]) -> bool:
    """命中处是否被否定：「没打滑」「不是间歇的」「看不到卡阻」「门未关严」。

    只回看本小句——跨句回看会把上一句的「机器没问题」算到这一句头上。
    """
    head = _CLAUSE_BREAK.split(text[max(0, m.start() - 6):m.start()])[-1]
    return any(c in head for c in _NEG_BEFORE) or any(c in m.group(0) for c in _NEG_INSIDE)


def _search_any(patterns: list[str], text: str, skip_question: bool = True) -> str | None:
    m = _last(patterns, text, skip_question)
    return m.group(0) if m else None


def _num(patterns: list[str], text: str) -> tuple[float | None, str | None]:
    for p in patterns:
        m = re.search(p, text)
        if m:
            try:
                return float(m.group(1)), m.group(0)
            except (ValueError, IndexError):
                continue
    return None, None


def _num_cn(patterns: list[str], text: str) -> tuple[float | None, str | None]:
    """_num 的中文数字版本。span 仍返回原文，出处指的是操作员实际说的那句话。"""
    for p in patterns:
        m = re.search(p, text)
        if m:
            v = cn_to_number(m.group(1))
            if v is not None:
                return v, m.group(0)
    return None, None


def _flag(patterns: list[str], text: str, name: str, sig: Signals, value: Any = True,
          negatable: bool = False) -> bool:
    """命中即置 value；negatable 时，命中处被否定则置 False。

    三态信号里 False 是「现场明确否定」，PRESENCE_FLAGS 里 False 等同未提及——
    两种语义都要求否认不能被读成确认，否则一句「没有焦味」就能换来一次停机升级。
    """
    m = _last(patterns, text)
    if not m:
        return False
    sig.set(name, False if (negatable and _negated(text, m)) else value, m.group(0))
    return True


def parse(text: str) -> Signals:
    sig = Signals(raw=text)

    m = ALARM_RE.search(text)
    if m:
        sig.set("alarm_code", m.group(0), m.group(0))
    sig.alarm_codes = list(dict.fromkeys(ALARM_RE.findall(text)))

    sig.mentioned_work_orders = WORK_ORDER_RE.findall(text)
    if sig.mentioned_work_orders:
        sig.set("mentioned_work_orders", sig.mentioned_work_orders, sig.mentioned_work_orders[0])

    # 回填页上「故障发生时间点」这一题，现场多半用中文钟点答（「今早 8 点 20 分」），
    # 只认 14:20 会把一条真答复读成"答了也读不出"。裸「3 点」不收——「第 3 点」不是时间。
    daypart = r"(?:[今去昨前]天|今早|今晨|凌晨|早上|上午|中午|下午|傍晚|晚上)"
    # 逐条按优先级试，命中即止：整张表一次交给 _search_any 的话，它取起点最靠后的命中，
    # 「2026-09-09 08:20」会被只匹配到「08:20」的短模式抢走，日期就丢了。
    for pattern in (
        r"\d{4}-\d{2}-\d{2}\s*\d{1,2}:\d{2}",
        r"\d{1,2}:\d{2}",
        r"\d{4}-\d{2}-\d{2}",
        rf"{daypart}?\s*\d{{1,2}}\s*点\s*(?:\d{{1,2}}\s*分|半)",
        rf"{daypart}\s*\d{{1,2}}\s*点",
    ):
        m = _last([pattern], text)
        if m:
            sig.set("occurrence_time", m.group(0), m.group(0))
            break

    # --- 热封温度与加热电流 ---
    val, span = _num([
        rf"设定(?:温度)?[^。；\n\d]{{0,6}}{NUM}\s*(?:°C|℃|度)",
        rf"设定(?:温度)?\s*{NUM}",
        rf"{NUM}\s*(?:°C|℃|度)[^。；\n\d]{{0,4}}设定",
    ], text)
    if val is not None:
        sig.set("setpoint_temp", val, span or "")
    else:
        val, span = _num_cn([
            rf"设定(?:温度)?{GAP_CN}{{0,6}}{CN_NUM}\s*(?:°C|℃|度)",
        ], text)
        if val is not None:
            sig.set("setpoint_temp", val, span or "")

    val, span = _num([
        rf"(?:实际|实测|显示|当前|现在)[^。；\n\d]{{0,8}}{NUM}\s*(?:°C|℃|度)",
        rf"温度(?:只有|仅|为|是|降到|升到)[^。；\n\d]{{0,4}}{NUM}",
        rf"(?:只有|仅)\s*{NUM}\s*(?:°C|℃|度)",
    ], text)
    if val is not None:
        sig.set("actual_temp", val, span or "")
    else:
        val, span = _num_cn([
            rf"(?:实际|实测|显示|当前|现在){GAP_CN}{{0,8}}{CN_NUM}\s*(?:°C|℃|度)",
            rf"温度(?:只有|仅|为|是|降到|升到){GAP_CN}{{0,4}}{CN_NUM}\s*(?:°C|℃|度)?",
        ], text)
        if val is not None:
            sig.set("actual_temp", val, span or "")

    # 填充串必须排除数字：否则贪婪匹配会吞掉「12.」而把 12.4A 读成 4A
    val, span = _num([rf"(?:加热)?电流[^。；\n\d]{{0,6}}{NUM}\s*A(?![0-9])"], text)
    if val is not None:
        sig.set("heating_current", val, span or "")
        sig.set("heating_current_state", "zero" if val == 0 else "normal", span or "")
    else:
        span = _search_any([r"电流(?:为|是)?\s*0(?!\s*\d)", r"无电流", r"没有电流", r"电流\s*0\s*A",
                            r"没有读数", r"无读数", r"读数没有", r"一点读数都没"], text)
        if span:
            sig.set("heating_current", 0.0, span)
            sig.set("heating_current_state", "zero", span)
        else:
            m = _last([r"电流不为\s*0", r"电流不为零", r"电流正常", r"电流有", r"有电流", r"电流有读数"], text)
            if m and not _negated(text, m):
                sig.set("heating_current_state", "normal", m.group(0))

    # 现场不会照着信号名说话：答「已经预热 40 分钟了」「预热了半小时」都得认出来，
    # 否则回填了也读不到信号，第二轮等于白填。肯定分支带否定判定，「没有预热」仍是 False。
    if not _flag([r"已(?:经)?预热", r"预热(?:完成|过了|充分|好了|到位|了)",
                  r"预热\s*[\d半两一二三四五六七八九十]+\s*(?:分钟|小时|min)"],
                 text, "preheated", sig, True, negatable=True):
        _flag([r"未预热", r"没有预热", r"没预热", r"还没预热", r"冷机启动", r"刚开机没预热"],
              text, "preheated", sig, False)

    # --- 气路 ---
    val, span = _num([rf"上游[^。；\n\d]{{0,8}}{NUM}\s*MPa", rf"供气[^。；\n\d]{{0,8}}{NUM}\s*MPa"], text)
    if val is not None:
        sig.set("upstream_pressure", val, span or "")
    val, span = _num([rf"(?:设备端|设备入口|末端|机器端)[^。；\n\d]{{0,8}}{NUM}\s*MPa"], text)
    if val is not None:
        sig.set("device_pressure", val, span or "")
    _flag([r"漏气声", r"嘶嘶声", r"持续漏气", r"有漏气", r"听到漏气"], text, "leak_sound", sig,
          negatable=True)

    # --- 进料 ---
    if not _flag([r"物料已到位", r"物料到位", r"有物料", r"物料正常", r"料是到位的", r"料已到位"],
                 text, "material_present", sig, True, negatable=True):
        _flag([r"无物料", r"没有物料", r"物料没有到位", r"物料未到", r"缺料", r"没料", r"没有料", r"没上料"],
              text, "material_present", sig, False)
    if not _flag([r"P1\s*(?:指示)?灯不亮", r"指示灯不亮", r"灯不亮", r"P1\s*不亮", r"灯(?:没|未)(?:有)?亮"],
                 text, "p1_light", sig, "off"):
        _flag([r"P1\s*(?:指示)?灯亮", r"指示灯正常亮", r"灯是亮的", r"P1\s*灯亮"], text, "p1_light", sig, "on")
    _flag([r"输送带打滑", r"皮带打滑", r"打滑"], text, "belt_slipping", sig, negatable=True)

    # --- 安全门 ---
    if not _flag([r"门[^。；\n]{0,4}关(?:严|好|闭|上)", r"防护门[^。；\n]{0,6}关", r"门是关着的", r"关(?:严|好|闭|上)了"],
                 text, "door_closed", sig, True, negatable=True):
        _flag([r"门没关", r"门未关", r"门开着", r"门是开的", r"门没有关闭", r"门没关上",
               r"(?:没|未)关(?:严|好|闭|上)"], text, "door_closed", sig, False)
    _flag([r"门内有异物", r"有异物", r"异物挡住", r"异物卡"], text, "door_obstructed", sig, negatable=True)
    # 「一直报/持续报警」是对间歇特征的反驳，H04 的成立条件正是间歇出现
    if not _flag([r"间歇", r"偶发", r"时有时无", r"一阵一阵", r"偶尔出现"], text, "alarm_intermittent", sig, True,
                 negatable=True):
        _flag([r"一直报", r"持续报警", r"不停报", r"连续报警", r"一直在响"], text, "alarm_intermittent", sig, False)

    # --- 伺服/传动 ---
    # 「每转一圈就有一声摩擦」这类说法不带「声/音」字，但现场指的就是摩擦声。
    # 裸「摩擦」覆盖面宽，所以先判明确否定再判肯定。放宽是可接受的：该信号只影响
    # H05 的证据状态，不触发任何安全红线（SAFE-01 的金属摩擦声走独立信号
    # metal_friction_sound），误判方向是少认一个历史案例，不是漏停机。
    if not _flag([r"没有摩擦", r"无摩擦", r"未(?:出现|听到)摩擦", r"没听到摩擦", r"无异响", r"没有异响"],
                 text, "friction_sound", sig, False):
        _flag([r"摩擦声", r"金属摩擦", r"异响", r"摩擦音", r"摩擦"], text, "friction_sound", sig, negatable=True)
    _flag([r"周期性摩擦", r"有节奏的摩擦", r"周期性的(?:摩擦|异响)", r"规律性摩擦", r"每转一圈"], text,
          "friction_periodic", sig, negatable=True)
    _flag([r"卡阻", r"卡料", r"卡住", r"有异物卡在"], text, "visible_jam", sig, negatable=True)

    # --- SAFE-01 立即停机信号（默认 False，命中才置 True）---
    # 这几个旗标直接换来停机与升级，所以「没有焦味」「部件没有松脱」必须读成未提及：
    # 一句否认换来一条产线停产，是最贵的一种误判。
    for name, patterns in (
        ("smoke", [r"烟雾", r"冒烟", r"有烟"]),
        ("burning_smell", [r"焦味", r"烧焦", r"糊味", r"焦糊"]),
        ("abnormal_high_temp", [r"异常高温", r"异常发烫", r"烫手", r"温度异常偏高"]),
        ("violent_vibration", [r"剧烈振动", r"强烈振动", r"剧烈抖动", r"振动剧烈", r"抖得厉害"]),
        ("metal_friction_sound", [r"金属摩擦"]),
        ("part_loose", [r"部件松脱", r"零件松脱", r"松脱"]),
    ):
        _flag(patterns, text, name, sig, True, negatable=True)

    # --- 对抗性用户请求（问句写法也要看见：漏拦一次违规比多读一句问句严重得多）---
    sig.user_request = _search_any([
        r"短接[^。；\n]{0,12}", r"绕过报警[^。；\n]{0,8}", r"屏蔽(?:报警|联锁|安全门)[^。；\n]{0,8}",
        r"临时解除[^。；\n]{0,8}", r"解除联锁[^。；\n]{0,8}", r"带电(?:插拔|拆线)[^。；\n]{0,8}",
        r"徒手(?:触碰|接触)[^。；\n]{0,8}", r"赶产量[^。；\n]{0,8}", r"先跑起来[^。；\n]{0,8}",
        r"忽略报警[^。；\n]{0,8}", r"关掉报警[^。；\n]{0,8}", r"不用停机[^。；\n]{0,8}",
    ], text, skip_question=False)
    if sig.user_request:
        sig.spans["user_request"] = sig.user_request

    # 继续运行请求：与报警码的 forbid_continue_run 属性对撞，不属于 SAFE-03 违规请求
    span = _search_any([
        r"试跑[^。；\n]{0,10}", r"试产[^。；\n]{0,10}", r"先跑[^。；\n]{0,10}",
        r"继续(?:生产|运行|跑|开机)[^。；\n]{0,8}", r"能不能[^。；\n]{0,12}(?:跑|开机|生产)",
        r"先降下来[^。；\n]{0,10}", r"顶一下[^。；\n]{0,8}", r"先不管[^。；\n]{0,8}",
    ], text, skip_question=False)
    if span:
        sig.set("wants_continue_run", True, span)

    _flag([r"发黄", r"发黑", r"变色", r"焦化", r"烤糊", r"袋子[^。；\n]{0,4}黄"], text, "discoloration", sig,
          negatable=True)

    return sig


def param_deviation(sig: Signals, params: dict[str, dict[str, Any]]) -> list[tuple[str, str, str]]:
    """把现场读数与说明书正常参数比对，返回 (项目, 判定, 依据) 列表。

    这一步是纯确定性的：越界就是越界，不交给语言模型判断。
    """
    out: list[tuple[str, str, str]] = []

    if sig.actual_temp is not None:
        at = f"{sig.actual_temp:g}"
        rng = params.get("heat_seal_stable_range", {})
        lo, hi = rng.get("min"), rng.get("max")
        if lo is not None and hi is not None:
            if sig.actual_temp < lo:
                out.append(("热封温度", f"低于稳定范围（{at}°C < {lo}°C）",
                            f"说明书正常范围 {lo}–{hi}°C"))
            elif sig.actual_temp > hi:
                out.append(("热封温度", f"高于稳定范围（{at}°C > {hi}°C）",
                            f"说明书正常范围 {lo}–{hi}°C"))
            else:
                out.append(("热封温度", f"在稳定范围内（{at}°C）",
                            f"说明书正常范围 {lo}–{hi}°C"))

    if sig.device_pressure is not None:
        dp = f"{sig.device_pressure:g}"
        rng = params.get("inlet_air_pressure", {})
        lo, hi = rng.get("min"), rng.get("max")
        if lo is not None and hi is not None:
            if sig.device_pressure < lo:
                out.append(("设备入口气压", f"低于正常范围（{dp} MPa < {lo} MPa）",
                            f"说明书正常范围 {lo}–{hi} MPa"))
            elif sig.device_pressure > hi:
                out.append(("设备入口气压", f"高于正常范围（{dp} MPa > {hi} MPa）",
                            f"说明书正常范围 {lo}–{hi} MPa"))
            else:
                out.append(("设备入口气压", f"在正常范围内（{dp} MPa）",
                            f"说明书正常范围 {lo}–{hi} MPa"))

    return out
