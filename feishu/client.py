"""飞书通道：消息推送与工单表读写。

只用标准库 urllib，不引入 requests/httpx——现场演示机上装不了包也要能跑。

**没有凭据时自动落到 DryRun**：消息写进 var/outbox.jsonl，工单写进 var/work_orders.json。
这不是偷懒，而是一条设计约束——接手这套代码的人不该被要求先去申请一个飞书应用才能验证它：
闭环必须能在无网络、无账号的机器上完整走通，接上凭据后同一套代码直接发真消息。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE = "https://open.feishu.cn/open-apis"
VAR_DIR = Path(__file__).resolve().parent.parent / "var"


class FeishuError(RuntimeError):
    pass


@dataclass
class Config:
    app_id: str = ""
    app_secret: str = ""
    chat_id: str = ""            # 报修群
    expert_chat_id: str = ""     # 专家群，缺省复用报修群
    bitable_app_token: str = ""  # 工单多维表格
    bitable_table_id: str = ""
    web_base_url: str = ""       # 网页入口的公网地址，卡片上「补充信息」按钮指向它
    force_dry_run: bool = False

    @classmethod
    def from_env(cls, force_dry_run: bool = False) -> "Config":
        g = os.environ.get
        return cls(
            app_id=g("FEISHU_APP_ID", ""),
            app_secret=g("FEISHU_APP_SECRET", ""),
            chat_id=g("FEISHU_CHAT_ID", ""),
            expert_chat_id=g("FEISHU_EXPERT_CHAT_ID", "") or g("FEISHU_CHAT_ID", ""),
            bitable_app_token=g("FEISHU_BITABLE_APP_TOKEN", ""),
            bitable_table_id=g("FEISHU_BITABLE_TABLE_ID", ""),
            web_base_url=g("APX240_WEB_BASE", ""),
            force_dry_run=force_dry_run,
        )

    @property
    def live(self) -> bool:
        """只有拿到应用凭据与投递目标才算真连接。"""
        return bool(not self.force_dry_run and self.app_id and self.app_secret and self.chat_id)


class Client:
    """消息与工单的统一出口。live=False 时全部落到本地文件。"""

    def __init__(self, cfg: Config | None = None, var_dir: Path = VAR_DIR):
        self.cfg = cfg or Config.from_env()
        self.var_dir = var_dir
        self.var_dir.mkdir(parents=True, exist_ok=True)
        self._token: str = ""
        self._token_exp: float = 0.0
        self._fields_cache: dict[str, dict[str, Any]] | None = None
        self.sent: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ 底层

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None,
                 auth: bool = True, unwrap: bool = True) -> dict[str, Any]:
        url = f"{BASE}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json; charset=utf-8")
        if auth:
            req.add_header("Authorization", f"Bearer {self.token()}")
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                raise FeishuError(f"{method} {path} 返回 HTTP {e.code}：{e.read().decode('utf-8', 'replace')}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                # socket 读超时抛的是 TimeoutError 而非 URLError，不一起接住就会裸崩整场演示
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise FeishuError(f"{method} {path} 网络不可达或超时：{e}") from e
        if body.get("code") != 0:
            raise FeishuError(f"{method} {path} 业务失败 code={body.get('code')} msg={body.get('msg')}")
        return body.get("data", {}) if unwrap else body

    def token(self) -> str:
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        # 鉴权接口把 tenant_access_token 放在响应顶层，不像其它接口包在 data 里
        body = self._request(
            "POST", "/auth/v3/tenant_access_token/internal",
            {"app_id": self.cfg.app_id, "app_secret": self.cfg.app_secret},
            auth=False, unwrap=False,
        )
        self._token = body["tenant_access_token"]
        self._token_exp = time.time() + int(body.get("expire", 7200))
        return self._token

    # ------------------------------------------------------------------ 消息

    def send_card(self, card: dict[str, Any], to_experts: bool = False) -> str:
        """发交互卡片。返回消息 id；DryRun 下返回本地序号。"""
        chat_id = self.cfg.expert_chat_id if to_experts else self.cfg.chat_id
        entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "chat_id": chat_id or "(dry-run)",
                 "to_experts": to_experts, "card": card}
        self.sent.append(entry)
        if not self.cfg.live:
            self._append_jsonl("outbox.jsonl", entry)
            return f"dry-{len(self.sent):04d}"
        data = self._request(
            "POST", "/im/v1/messages?receive_id_type=chat_id",
            {"receive_id": chat_id, "msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)},
        )
        return data.get("message_id", "")

    # ------------------------------------------------------------------ 工单表

    def upsert_work_order(self, wo_id: str, fields: dict[str, Any]) -> str:
        """写入或更新工单。返回多维表格 record_id；DryRun 下写本地 JSON。"""
        if not self.cfg.live or not (self.cfg.bitable_app_token and self.cfg.bitable_table_id):
            return self._local_upsert(wo_id, fields)

        # 新增记录接口不会自动建列：先补齐缺失列，再按列类型转值，否则 FieldNameNotFound
        self._ensure_fields(list(fields))
        payload = self._coerce_text(fields)

        base = f"/bitable/v1/apps/{self.cfg.bitable_app_token}/tables/{self.cfg.bitable_table_id}/records"
        existing = self._local_index().get(wo_id, {}).get("record_id")
        if not existing:
            existing = self._find_record(wo_id)
        if existing:
            self._request("PUT", f"{base}/{existing}", {"fields": payload})
            rid = existing
        else:
            rid = self._request("POST", base, {"fields": payload}).get("record", {}).get("record_id", "")
        self._local_upsert(wo_id, {**fields, "record_id": rid})
        return rid

    def wo_url(self, record_id: str) -> str:
        """工单记录的直达链接。卡片上那个按钮就靠它，DryRun 下返回空串——
        本地假 record_id 拼出来的链接是死链，宁可不出按钮。"""
        if not (self.cfg.live and self.cfg.bitable_app_token and self.cfg.bitable_table_id):
            return ""
        if not record_id or record_id.startswith("local-"):
            return ""
        return (f"https://www.feishu.cn/base/{self.cfg.bitable_app_token}"
                f"?table={self.cfg.bitable_table_id}&record={record_id}")

    def followup_url(self, wo_id: str) -> str:
        """⑦ 需要补充的信息的回填入口。

        只看 APX240_WEB_BASE，不看飞书凭据：网页读的是本地工单存档，DryRun 下照样能填、
        能重出第二轮。没配公网地址就返回空串，卡片不出按钮——同 wo_url 的道理。
        """
        base = self.cfg.web_base_url.rstrip("/")
        return f"{base}/?wo={wo_id}" if base else ""

    def close_url(self, wo_id: str, code: str) -> str:
        """双人确认入库的入口。专家填写页与现场确认页共用，靠一次性确认码区分阶段。

        与 followup_url 同一个道理：只看 APX240_WEB_BASE，不看飞书凭据。
        没有码就没有入口——确认码是这条链上唯一的凭证，链接里不带它，
        等于把改写知识库的权限交给任何拿到公网地址的人。
        """
        base = self.cfg.web_base_url.rstrip("/")
        return f"{base}/?wo={wo_id}&k={code}" if base and code else ""

    def _fields_meta(self) -> dict[str, dict[str, Any]]:
        """列出表字段（field_name -> 字段定义），分页拉全后缓存。"""
        if self._fields_cache is None:
            base = f"/bitable/v1/apps/{self.cfg.bitable_app_token}/tables/{self.cfg.bitable_table_id}/fields"
            meta: dict[str, dict[str, Any]] = {}
            offset = 0
            while True:
                data = self._request("GET", f"{base}?page_size=100&offset={offset}")
                for it in data.get("items", []):
                    meta[it.get("field_name", "")] = it
                if not data.get("has_more"):
                    break
                offset += 100
            self._fields_cache = meta
        return self._fields_cache

    def _ensure_fields(self, names: list[str]) -> None:
        """表里没有的列，建成多行文本（type 1）。飞书不会在写记录时自动建列。"""
        meta = self._fields_meta()
        base = f"/bitable/v1/apps/{self.cfg.bitable_app_token}/tables/{self.cfg.bitable_table_id}/fields"
        for name in names:
            if name in meta:
                continue
            self._request("POST", base, {"field_name": name, "type": 1})
            meta[name] = {"field_name": name, "type": 1}

    def _coerce_text(self, fields: dict[str, Any]) -> dict[str, Any]:
        """文本列（type 1）只收字符串：dict/list 转 JSON，标量转 str；其它类型原样传。"""
        meta = self._fields_meta()
        out: dict[str, Any] = {}
        for name, val in fields.items():
            if meta.get(name, {}).get("type") == 1 and not isinstance(val, str):
                out[name] = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)
            else:
                out[name] = val
        return out

    def _find_record(self, wo_id: str) -> str:
        base = f"/bitable/v1/apps/{self.cfg.bitable_app_token}/tables/{self.cfg.bitable_table_id}/records"
        try:
            data = self._request("POST", f"{base}/search?page_size=1", {"filter": {
                "conjunction": "and",
                "conditions": [{"field_name": "工单号", "operator": "is", "value": [wo_id]}],
            }})
        except FeishuError:
            return ""
        items = data.get("items") or []
        return items[0].get("record_id", "") if items else ""

    # ------------------------------------------------------------------ 本地存储

    def _path(self, name: str) -> Path:
        return self.var_dir / name

    def _append_jsonl(self, name: str, entry: dict[str, Any]) -> None:
        with self._path(name).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _local_index(self) -> dict[str, dict[str, Any]]:
        p = self._path("work_orders.json")
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def _local_upsert(self, wo_id: str, fields: dict[str, Any]) -> str:
        store = self._local_index()
        prev = store.get(wo_id, {})
        store[wo_id] = {**prev, **fields, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self._path("work_orders.json").write_text(
            json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
        self._append_jsonl("work_order_log.jsonl", {"wo_id": wo_id, "fields": fields})
        return f"local-{wo_id}"

    def work_orders(self) -> dict[str, dict[str, Any]]:
        return self._local_index()

    def next_seq(self) -> int:
        return len(self._local_index()) + 1
