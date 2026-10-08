"""口语化现场描述的兜底解析。

正则解析层是确定性核心，但它只认我枚举过的说法。现场口述千变万化
（「封口那儿摸着烫手」「袋子颜色不太对」），枚举不完——这是这类 Agent 在真实车间里
最常见的失效方式，比模型胡说更常见。

这一层把「听懂话」交给模型，但**模型的输出一个字段都不许直接进诊断**。每个字段必须
附带现场原文片段，并过三道复核：

1. 片段必须是现场描述的真实子串——模型不能引用一句没人说过的话；
2. 数值必须能从该片段里算出来（阿拉伯数字或中文数字均可）——模型不能编读数；
3. 规则解析已经抽到的字段一律不覆盖——能对上原文的正则优先于模型。

复核通过的字段才写进 Signals，出处仍标 `INPUT:「原文片段」`：那段话确实是现场说的，
只是由模型而不是正则读出来的。**模型在这里只负责把话听成结构，判断事实仍然是
确定性链路的事**——安全门控、证据匹配、输出自检全部照常执行，一步不减。

刻意不交给模型的三个字段：

- `alarm_code`：把现象映射成报警码是**判断**，不是听写。判断错了整条诊断链都会跟着错。
- `user_request` / `wants_continue_run`：这两个直接触发 SAFE-03 硬拒绝。
  让模型决定「这句话算不算违规请求」，等于把责任边界交到模型手上。

未配置模型时本模块不产生任何影响，诊断行为与纯规则版本逐字节一致。
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .evidence import SIGNAL_LABELS
from .parse import PRESENCE_FLAGS, Signals, numbers_in

Caller = Callable[[str, str], str]


class LLMError(RuntimeError):
    """模型不可用。兜底层失败不得影响诊断，规则解析的结果照常输出。"""


# 模型可以补齐的字段 → 复核方式。
FILLABLE: dict[str, str] = {
    # 读数：数值必须能从所引原文片段里算出来
    "setpoint_temp": "number",
    "actual_temp": "number",
    "heating_current": "number",
    "upstream_pressure": "number",
    "device_pressure": "number",
    # 枚举：只认固定取值
    "heating_current_state": "enum:zero,normal",
    "p1_light": "enum:on,off",
    # 三态布尔：None=未提及 / True / False，False 本身也是证据（「一直报」反驳间歇）
    "preheated": "bool",
    "door_closed": "bool",
    "door_obstructed": "bool",
    "alarm_intermittent": "bool",
    "material_present": "bool",
    "belt_slipping": "bool",
    "leak_sound": "bool",
    "friction_sound": "bool",
    "friction_periodic": "bool",
    "visible_jam": "bool",
    # SAFE-01 征兆：只接受 True。补出来的方向永远是"更保守"——只会触发停机升级，
    # 不会解除任何已成立的停机要求，所以放宽这一类是安全的。
    "smoke": "flag",
    "burning_smell": "flag",
    "abnormal_high_temp": "flag",
    "violent_vibration": "flag",
    "metal_friction_sound": "flag",
    "part_loose": "flag",
    "discoloration": "flag",
}

WITHHELD = ("alarm_code", "user_request", "wants_continue_run")

LABELS: dict[str, str] = {
    **SIGNAL_LABELS,
    "heating_current": "加热电流读数",
    "door_obstructed": "门内是否有异物",
    "belt_slipping": "输送带是否打滑",
    "visible_jam": "是否可见卡阻",
    "smoke": "烟雾",
    "burning_smell": "焦味",
    "abnormal_high_temp": "异常发烫",
    "violent_vibration": "剧烈振动",
    "metal_friction_sound": "金属摩擦声",
    "part_loose": "部件松脱",
    "discoloration": "产品变色",
}

_SYSTEM = """你在为设备故障诊断系统做"听写"，不是做诊断。

把现场描述里**明确说了**的信号抽成 JSON。只输出 JSON，不要任何解释。

规则：
1. 只抽现场明确说了的。没说的、需要你推断的、要靠常识补的，一律不输出该字段。
2. 每个字段必须附 "span"：现场描述中支撑该值的**原文连续片段**，逐字照抄，不得改写、不得拼接。
3. 数值字段的 value 必须能从 span 里直接读出来（原文写中文数字也可以）。
4. 禁止输出这三个字段：alarm_code、user_request、wants_continue_run。
5. 不要判断故障原因，不要给处置建议，不要补任何设备常识。

输出格式：
{"fields": {"字段名": {"value": 值, "span": "原文片段"}}}

没有可抽的字段就输出 {"fields": {}}"""


def _hint(kind: str) -> str:
    if kind == "number":
        return "数值"
    if kind == "bool":
        return "true 或 false"
    if kind == "flag":
        return "仅在明确提及时填 true"
    return "或".join(kind[5:].split(","))


@dataclass
class FillReport:
    """兜底解析的账本。filled 会写进输出提示，rejected 供调 prompt 与审计。"""

    filled: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.filled)


@dataclass
class LLMConfig:
    """OpenAI 兼容端点。三个值缺一个就视为未配置，诊断退回纯规则版本。"""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    # 兜底层要"快失败"：端点实测 1.4~2.3 秒，12 秒已是 5 倍余量；现场断网时一次可选的
    # 听写不该把交互卡住 20 秒。超时后退回纯规则结果，并在输出「提示」里显式留痕。
    timeout: float = 12.0

    @classmethod
    def from_env(cls) -> "LLMConfig | None":
        g = os.environ.get
        cfg = cls(
            base_url=g("APX240_LLM_BASE_URL", ""),
            api_key=g("APX240_LLM_API_KEY", ""),
            model=g("APX240_LLM_MODEL", ""),
        )
        return cfg if cfg.ready else None

    @property
    def ready(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


def http_caller(cfg: LLMConfig) -> Caller:
    """/chat/completions 调用，只用标准库——现场演示机装不了包也要能跑。"""
    url = cfg.base_url.rstrip("/") + "/chat/completions"

    def call(system: str, user: str) -> str:
        payload = json.dumps({
            "model": cfg.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        })
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMError(f"HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            raise LLMError(f"调用失败：{e}") from e
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("返回体缺少 choices[0].message.content") from e

    return call


def caller_from_env() -> Caller | None:
    cfg = LLMConfig.from_env()
    return http_caller(cfg) if cfg else None


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)
_WS_RE = re.compile(r"\s+")


def _extract_json(raw: str) -> dict[str, Any]:
    """模型常把 JSON 包在代码围栏里，或前后加一句话。只取第一个花括号块。"""
    m = _FENCE_RE.search(raw)
    text = m.group(1) if m else raw
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise LLMError("返回内容里没有 JSON 对象")
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise LLMError("返回的 JSON 不是对象")
    return obj


def _is_missing(sig: Signals, name: str) -> bool:
    """该字段是否还没被规则解析抽到。"""
    v = getattr(sig, name, None)
    # PRESENCE_FLAGS 默认 False 表示"未提及"，只有 True 才算已抽到
    return v is not True if name in PRESENCE_FLAGS else v is None


def _coerce(kind: str, value: Any, span: str) -> tuple[bool, Any]:
    """按字段类型复核模型给的值。第二个返回值在失败时是拒绝原因。"""
    if kind == "number":
        if isinstance(value, bool):
            return False, "值不是数字"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False, f"值 {value!r} 不是数字"
        if f"{v:g}" not in numbers_in(span):
            return False, f"读数 {value} 无法从所引原文片段「{span}」中得出"
        return True, v
    if kind.startswith("enum:"):
        opts = kind[5:].split(",")
        if value not in opts:
            return False, f"值 {value!r} 不在允许范围 {'/'.join(opts)}"
        return True, value
    if kind == "bool":
        if not isinstance(value, bool):
            return False, f"值 {value!r} 不是布尔"
        return True, value
    if kind == "flag":
        if value is not True:
            return False, "立即停机征兆只接受明确肯定，不接受推断"
        return True, True
    return False, f"未知字段类型 {kind}"  # pragma: no cover


def _ask(sig: Signals, missing: list[str]) -> str:
    lines = [f"现场描述：{sig.raw}", "", "可抽取的字段（只输出其中确有原文依据的）："]
    lines += [f"- {k}（{LABELS.get(k, k)}）：{_hint(FILLABLE[k])}" for k in missing]
    return "\n".join(lines)


def augment(sig: Signals, caller: Caller) -> FillReport:
    """就地把模型复核通过的字段补进 sig，返回补齐与拒绝的账本。

    任何异常都不向上抛：兜底层坏了，规则解析的结果照常可用。
    """
    rep = FillReport()
    missing = [k for k in FILLABLE if _is_missing(sig, k)]
    if not missing:
        return rep

    try:
        data = _extract_json(caller(_SYSTEM, _ask(sig, missing)))
    except (LLMError, json.JSONDecodeError) as e:
        rep.rejected.append(f"模型兜底不可用：{e}")
        return rep

    fields = data.get("fields")
    if not isinstance(fields, dict):
        rep.rejected.append("模型返回缺少 fields 对象")
        return rep

    haystack = _WS_RE.sub("", sig.raw)
    for name, item in fields.items():
        label = LABELS.get(name, name)
        if name in WITHHELD:
            rep.rejected.append(f"{label}：该字段不允许由模型提供")
            continue
        kind = FILLABLE.get(name)
        if kind is None:
            rep.rejected.append(f"{label}：不在可补齐字段清单内")
            continue
        if not isinstance(item, dict):
            rep.rejected.append(f"{label}：返回项不是对象")
            continue
        # 规则解析已经从原文抽到了，模型的读法再好也不覆盖
        if not _is_missing(sig, name):
            rep.rejected.append(f"{label}：规则解析已从原文抽到，不覆盖")
            continue

        span = str(item.get("span") or "").strip()
        if len(_WS_RE.sub("", span)) < 2 or _WS_RE.sub("", span) not in haystack:
            rep.rejected.append(f"{label}：所引原文片段不在现场描述中")
            continue

        ok, val = _coerce(kind, item.get("value"), span)
        if not ok:
            rep.rejected.append(f"{label}：{val}")
            continue

        sig.set(name, val, span)
        rep.filled.append(f"{label} ← 「{span}」")

    return rep
