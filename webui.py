"""浏览器入口：把已经验证过的诊断链路摊开成一个能点开的页面。

只有一条 CLI 时，想确认它到底输出些什么的人得先读代码。这一层就是把 diagnose()
原样搬到浏览器上：

- 不做任何新的判断。页面只负责收发文本，两道闸门、四状态证据、失效关闭全在 agent/ 里，
  走的是与 `run.py diagnose` 完全相同的调用路径。
- `?wo=工单号` 让同一个页面变成 ⑦ 需要补充的信息的回填表单（飞书卡片上的按钮指向这里）。
  第二轮同样不加判断：只把答案并进原描述，再调一次同一条链路。
- `?wo=工单号&k=一次性确认码` 让同一个页面变成双人确认页：专家填处置结论、报修人确认现场可行。
  是哪一个阶段、问哪几道题、哪些必填，全部由服务端按码判定后给出，页面只负责画。
  确认码不是身份认证：它只证明"这个人拿得到群里那张卡"，转发链接等于转交这一步的权限
  （生产形态应换成飞书事件回调带 open_id，详见 feishu/workflow.py 的注释）。
- 只用标准库 http.server，无第三方依赖，断网可用（页面不引任何 CDN）。
- 服务端只回 JSON，所有文本在浏览器里用 textContent 落盘，不做 HTML 拼接——
  现场描述是用户输入，回显到页面时只要有一处 innerHTML 就是一个 XSS。
- 默认只绑 127.0.0.1。要开放给局域网另一台机器访问必须显式加 --host，并在启动时打印警告。
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs

from agent.diagnose import diagnose
from agent.knowledge import KnowledgeBase

# 现场描述可能很长，但再长也不该到 8KB；设上限是为了不让一个请求把内存吃掉。
MAX_BODY = 8192

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>APX-240 故障诊断 Agent</title>
<style>
:root { --line:#d9dde3; --dim:#5b6472; --bg:#f5f6f8; --bad:#b42318; --warn:#93370d; }
* { box-sizing:border-box; }
/* .samples / label.chk 自带 display，会盖掉 hidden 属性默认的 display:none，
   不写这条，切到回填页或确认页后首页那排按钮和「报修建单」复选框还挂在屏幕上。 */
[hidden] { display:none !important; }
body { margin:0; background:var(--bg); color:#16181d;
       font:15px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }
header { background:#fff; border-bottom:1px solid var(--line); padding:14px 22px; }
header h1 { margin:0; font-size:17px; }
header p { margin:4px 0 0; font-size:13px; color:var(--dim); }
main { max-width:960px; margin:0 auto; padding:20px 22px 60px; }
.card { background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px 18px; }
label { display:block; font-weight:600; font-size:13px; margin-bottom:6px; }
textarea { width:100%; min-height:96px; padding:10px 12px; font:14px/1.6 inherit;
           border:1px solid var(--line); border-radius:6px; resize:vertical; }
textarea:focus { outline:2px solid #2f6feb; border-color:#2f6feb; }
.orig { margin:0; padding:10px 12px; background:#f7f8fa; border:1px solid var(--line);
        border-radius:6px; font:13px/1.7 inherit; color:var(--dim);
        white-space:pre-wrap; word-break:break-word; }
.tip { margin:10px 0 0; font-size:13px; color:var(--dim); }
.gap { margin-top:12px; }
.gap input { width:100%; padding:8px 10px; font:14px/1.6 inherit;
             border:1px solid var(--line); border-radius:6px; }
.gap input:focus { outline:2px solid #2f6feb; border-color:#2f6feb; }
#follow-head, #close-head { margin:0 0 8px; font-size:14px; }
.gap textarea { width:100%; min-height:52px; padding:8px 10px; font:14px/1.6 inherit;
                border:1px solid var(--line); border-radius:6px; resize:vertical; }
.gap textarea:focus { outline:2px solid #2f6feb; border-color:#2f6feb; }
.gap .radios { display:flex; gap:16px; margin-top:2px; }
.gap .radios input { width:auto; }
.gap .radios label { display:inline-flex; align-items:center; gap:6px; margin:0;
                     font-weight:400; font-size:14px; }
.row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:12px; }
button { font:14px inherit; padding:8px 18px; border-radius:6px; cursor:pointer;
         border:1px solid var(--line); background:#fff; }
button.primary { background:#2f6feb; border-color:#2f6feb; color:#fff; font-weight:600; }
button.primary:disabled { background:#9db9ef; border-color:#9db9ef; cursor:default; }
.samples { display:flex; gap:8px; flex-wrap:wrap; margin:0 0 18px; padding:0; list-style:none; }
.samples button { font-size:13px; padding:6px 12px; color:var(--dim); }
.samples button:hover { border-color:#2f6feb; color:#2f6feb; }
.flag { display:inline-block; padding:3px 10px; border-radius:999px; font-size:13px;
        font-weight:600; border:1px solid; margin:0 8px 8px 0; }
.flag.stop { color:var(--bad); border-color:#f0b8b3; background:#fdf3f2; }
.flag.esc  { color:var(--warn); border-color:#f2cfb8; background:#fdf6f0; }
.flag.ok   { color:#14663a; border-color:#b6dcc6; background:#f1f9f4; }
.ticket { border:1px solid #b6dcc6; background:#f1f9f4; color:#14663a;
          border-radius:8px; padding:10px 14px; font-size:13px; line-height:1.8; }
.ticket.bad { border-color:#f0b8b3; background:#fdf3f2; color:var(--bad); }
.ticket b { font-weight:700; }
label.chk { display:inline-flex; align-items:center; gap:6px; font-size:13px;
            color:var(--dim); font-weight:400; }
section h2 { font-size:14px; margin:22px 0 8px; padding-bottom:5px;
             border-bottom:1px solid var(--line); }
pre { margin:0; white-space:pre-wrap; word-break:break-word; font:14px/1.7 inherit; }
#legend { margin:-2px 0 10px; font-size:12px; color:var(--dim); }
.checks { list-style:none; margin:0; padding:0; font-size:13px; }
.checks li { padding:3px 0; color:var(--dim); }
.checks li.bad { color:var(--bad); font-weight:600; }
.err { color:var(--bad); margin-top:12px; }
footer { margin-top:26px; font-size:12px; color:var(--dim); }
code { background:#f0f1f4; padding:1px 5px; border-radius:4px; font-size:13px; }
</style>
</head>
<body>
<header>
  <h1>APX-240 自动封装设备｜故障诊断 Agent</h1>
  <p>输入现场描述，输出 8 个字段。判断依据全部来自设备说明书与安全红线，不依赖模型常识；
     规则未覆盖时明确回答「无法判断」并升级专家。</p>
</header>
<main>
  <ul class="samples" id="samples"></ul>

  <div class="card">
    <div id="plain">
      <label for="text">现场故障描述（支持口语化说法与中文数字）</label>
      <textarea id="text" placeholder="例如：A203，设定 165°C，实际温度只有 152°C，加热电流正常有读数。"></textarea>
    </div>
    <div id="follow" hidden>
      <p id="follow-head"></p>
      <pre id="follow-orig" class="orig"></pre>
      <p class="tip">一项一行，用完整的话答，带上读数与单位（例：加热电流 12.4A／输送带没有打滑／14:20）。
        提交后系统用「原描述＋你的补充」重跑一次完整诊断，安全闸门与输出自检照常生效，
        另建一张关联工单并推送新一轮卡片。没把握的项可以留空。</p>
      <div id="follow-items"></div>
    </div>
    <div id="close" hidden>
      <p id="close-head"></p>
      <pre id="close-orig" class="orig"></pre>
      <div id="close-expert" hidden>
        <p class="tip">下面是专家提交的技术结论。你只回答现场看得见的事——
          <b>根因对不对由专家署名负责，不用你判断。</b></p>
        <ul class="checks" id="close-conclusion"></ul>
      </div>
      <div id="close-rejected" hidden>
        <p class="tip">这一单上一轮被现场确认退回。先看清分歧再改结论，读数对不上时优先解释差异。</p>
        <ul class="checks" id="close-rejected-items"></ul>
      </div>
      <p class="tip" id="close-note"></p>
      <div id="close-items"></div>
    </div>
    <div class="row">
      <button class="primary" id="go">诊断</button>
      <button id="clear">清空</button>
      <label class="chk" id="reportwrap" for="report"><input type="checkbox" id="report">
        报修建单（推送飞书卡片 + 写入工单表）</label>
      <span id="hint" style="font-size:13px;color:var(--dim)"></span>
    </div>
    <div class="err" id="err"></div>
  </div>

  <div id="out" hidden>
    <div class="ticket" id="ticket" hidden></div>
    <section>
      <h2>判定</h2>
      <div id="flags"></div>
      <p id="legend">强制红线＝触发即停机或当场拒绝；适用红线＝本次作业必须遵守的前置条件与授权边界。
        鼠标悬停在红线上可看规则原文。</p>
      <pre id="verdict"></pre>
    </section>
    <section id="secwrap"><h2>诊断输出</h2><div id="secs"></div></section>
    <section>
      <h2>输出自检（生成之后逐条核对，未通过即失效关闭）</h2>
      <ul class="checks" id="checks"></ul>
    </section>
  </div>

  <div id="close-out" hidden>
    <div class="ticket" id="close-res"></div>
    <section>
      <h2>这一步改了什么</h2>
      <ul class="checks" id="close-facts"></ul>
      <h2>接下来发生什么</h2>
      <ul class="checks" id="close-next"></ul>
    </section>
  </div>

  <footer>
    <div id="kb"></div>
    <p>证据来源标签：<code>MANUAL:§n</code> 说明书原文｜<code>KB:LEARNED</code> 本厂闭环沉淀（专家提交＋现场确认，两人都签字才入库）｜
       <code>INPUT</code> 现场描述原话。同一入口的命令行版本是 <code>python run.py diagnose "..."</code>。</p>
  </footer>
</main>

<script>
const $ = id => document.getElementById(id);

// 三种模式共用一个页面：无参数＝首页诊断；?wo=工单号＝卡片上「补充信息」按钮指向的
// 第二轮回填；?wo=工单号&k=一次性确认码＝双人确认页（专家填写处置结论／报修人现场确认）。
// 是专家还是现场由服务端按码判定，页面不问也不猜。
const Q = new URLSearchParams(location.search);
const FOLLOW = Q.get('wo');
const CODE = (Q.get('k') || '').trim();
const CLOSE = FOLLOW && CODE;

// 全部用 textContent 写入：现场描述是用户输入，任何一处 innerHTML 都是 XSS。
function el(tag, text, cls) {
  const n = document.createElement(tag);
  if (text != null) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + ' → HTTP ' + r.status);
  return r.json();
}

async function boot() {
  try {
    const kb = await getJSON('/api/knowledge');
    $('kb').textContent = '当前知识库：' + kb.root + '｜' + kb.device + '｜报警码 '
      + kb.alarm_count + ' 个｜安全红线 ' + kb.rule_count + ' 条｜历史案例 '
      + kb.case_count + ' 个（含现场沉淀 ' + kb.learned_count + ' 个）｜维修记录 '
      + kb.work_order_count + ' 条';
  } catch (e) { fail(e); return; }
  try {
    if (CLOSE) await bootClose(FOLLOW, CODE);
    else if (FOLLOW) await bootFollow(FOLLOW);
    else await bootSamples();
  } catch (e) { fail(e); }
}

async function bootSamples() {
  const samples = await getJSON('/api/samples');
  for (const s of samples) {
    const b = el('button', s.label);
    b.onclick = () => { $('text').value = s.text; $('text').focus(); };
    $('samples').appendChild(el('li', null)).appendChild(b);
  }
}

async function bootFollow(wo) {
  const f = await getJSON('/api/followup?wo=' + encodeURIComponent(wo));
  document.title = '补充信息｜工单 ' + f.wo_id;
  $('samples').hidden = true;
  $('plain').hidden = true;
  $('reportwrap').hidden = true;   // 第二轮必然建单发卡，不给"只诊断"这个选项
  $('follow').hidden = false;
  $('follow-head').textContent = '工单 ' + f.wo_id + '｜报警码 ' + (f.alarm_code || '无')
    + '｜报修人 ' + (f.reporter || '—') + '｜' + f.verdict;
  $('follow-orig').textContent = '第一轮现场描述：' + f.text;

  if (!f.items.length) {
    $('follow-items').appendChild(el('p', '这一单已无待补项，再跑一轮结论不变。要重新报修请打开首页。', 'tip'));
    $('go').disabled = true;
    return;
  }
  for (const item of f.items) {
    const box = el('div', null, 'gap');
    box.appendChild(el('label', item));
    const inp = document.createElement('input');
    inp.type = 'text';
    inp.placeholder = '用一句完整的话回答，带上读数与单位；没测就写"没测"';
    inp.dataset.item = item;
    box.appendChild(inp);
    $('follow-items').appendChild(box);
  }
  $('go').textContent = '生成第 ' + f.round + ' 轮诊断';
}

function fail(e) {
  $('err').textContent = '出错了：' + (e && e.message ? e.message : e);
}

async function bootClose(wo, code) {
  const f = await getJSON('/api/close?wo=' + encodeURIComponent(wo)
                          + '&k=' + encodeURIComponent(code));
  const expert = f.stage === 'expert';
  document.title = (expert ? '填写处置结论' : '现场确认') + '｜工单 ' + f.wo_id;
  $('samples').hidden = true;
  $('plain').hidden = true;
  $('reportwrap').hidden = true;   // 这一步不建新单，只把流程往下推
  $('close').hidden = false;
  $('close-head').textContent = (expert ? '🧑‍🔧 专家填写处置结论' : '✅ 报修人现场确认')
    + '｜工单 ' + f.wo_id + '｜报警码 ' + (f.alarm_code || '无') + '｜' + f.verdict
    + '｜当前状态 ' + (f.status || '—');
  $('close-orig').textContent = '现场描述：' + f.text;
  $('close-note').textContent = expert
    ? '提交后生成候选案例——此时它不参与任何诊断；要等报修人确认现场可行，两人都签字才写进知识库。'
    : '通过 → 案例入库、从此参与同类故障诊断；不通过 → 一个字都不写进知识库，分歧退回专家群。';

  if (expert) {
    // 被现场退回过的单，分歧必须摆在眼前：不看他就是在原地再交一份同样的结论。
    const r = f.rejected || {};
    if (r.discrepancy) {
      $('close-rejected').hidden = false;
      for (const line of ['退回人：' + (r.by || '—') + '　时间：' + (r.at || '—'),
                          '现场的答复：' + (r.facts || '—'),
                          '现场实际情况：' + r.discrepancy]) {
        $('close-rejected-items').appendChild(el('li', line));
      }
    }
  } else {
    $('close-expert').hidden = false;
    for (const row of [['专家署名', f.expert], ['实际根因', f.root_cause],
                       ['处置动作', f.disposition], ['专家记录的复机验证', f.verification]]) {
      $('close-conclusion').appendChild(el('li', row[0] + '：' + (row[1] || '（未填写）')));
    }
  }

  for (const q of f.questions) {
    const box = el('div', null, 'gap');
    box.dataset.key = q.key;
    box.dataset.kind = q.kind;
    if (q.required) box.dataset.required = '1';
    box.appendChild(el('label', q.label + (q.required ? '（必填）' : '')));
    if (q.kind === 'yesno') {
      const wrap = el('div', null, 'radios');
      for (const opt of ['是', '否']) {
        const lab = document.createElement('label');
        const radio = document.createElement('input');
        radio.type = 'radio';
        radio.name = q.key;
        radio.value = opt;
        lab.appendChild(radio);
        lab.appendChild(document.createTextNode(opt));
        wrap.appendChild(lab);
      }
      box.appendChild(wrap);
    } else {
      const ta = document.createElement('textarea');
      ta.value = q.value || '';
      box.appendChild(ta);
    }
    $('close-items').appendChild(box);
  }
  $('go').textContent = expert ? '提交技术结论（生成候选案例）' : '提交现场确认';
}

async function runClose() {
  const answers = {};
  let missing = '';
  for (const box of document.querySelectorAll('#close-items .gap')) {
    let v = '';
    if (box.dataset.kind === 'yesno') {
      const c = box.querySelector('input:checked');
      v = c ? c.value : '';
    } else {
      v = box.querySelector('textarea').value.trim();
    }
    if (!v) {
      // 必填项空着就在页面拦住：交给服务端只会得到一句"根因不得为空"，不如就地指出是哪一项。
      if (box.dataset.required) missing = missing || box.querySelector('label').textContent;
      continue;
    }
    answers[box.dataset.key] = v;
  }
  $('err').textContent = '';
  if (missing) { $('err').textContent = '这一项必填，不能空着：' + missing; return; }

  $('close-out').hidden = true;
  $('go').disabled = true;
  $('hint').textContent = '提交中…';
  try {
    const r = await fetch('/api/close', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({wo: FOLLOW, k: CODE, answers: answers})
    });
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || ('HTTP ' + r.status));
    paintClose(data);
  } catch (e) { fail(e); }
  finally { $('go').disabled = false; $('hint').textContent = ''; }
}

function paintClose(d) {
  $('close-out').hidden = false;
  const n = $('close-res');
  n.textContent = '';
  n.className = 'ticket' + (d.passed === false ? ' bad' : '');
  n.appendChild(el('b', d.headline));
  n.appendChild(document.createTextNode('　' + d.channel));

  // 逐条列全，不折叠：这一步动了知识库还是没动，看的人要能一眼数清楚。
  $('close-facts').textContent = '';
  for (const line of d.facts) $('close-facts').appendChild(el('li', line));
  $('close-next').textContent = '';
  for (const line of d.lines) $('close-next').appendChild(el('li', line));
  $('close-out').scrollIntoView({behavior: 'smooth', block: 'start'});
}

async function run() {
  if (CLOSE) return runClose();
  if (FOLLOW) return runFollow();
  const text = $('text').value.trim();
  $('err').textContent = '';
  if (!text) { $('err').textContent = '请先输入现场描述。'; return; }

  // 新请求一发出就把上一次的结果收起来：留着会让人把旧答案当成新答案。
  $('out').hidden = true;
  $('ticket').hidden = true;
  $('go').disabled = true;
  $('hint').textContent = $('report').checked ? '诊断并建单中…' : '诊断中…';
  try {
    const r = await fetch('/api/diagnose', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: text, report: $('report').checked})
    });
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || ('HTTP ' + r.status));
    paint(data);
  } catch (e) { fail(e); }
  finally { $('go').disabled = false; $('hint').textContent = ''; }
}

async function runFollow() {
  const answers = {};
  for (const inp of document.querySelectorAll('#follow-items input')) {
    const v = inp.value.trim();
    if (v) answers[inp.dataset.item] = v;
  }
  $('err').textContent = '';
  if (!Object.keys(answers).length) {
    $('err').textContent = '至少补一项：一项都不填，第二轮和第一轮一模一样，跑它没有意义。';
    return;
  }

  $('out').hidden = true;
  $('ticket').hidden = true;
  $('go').disabled = true;
  $('hint').textContent = '重跑诊断并建关联工单中…';
  try {
    const r = await fetch('/api/followup', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({wo: FOLLOW, answers: answers})
    });
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || ('HTTP ' + r.status));
    paint(data);
  } catch (e) { fail(e); }
  finally { $('go').disabled = false; $('hint').textContent = ''; }
}

function paintTicket(t) {
  const n = $('ticket');
  n.textContent = '';
  if (!t) { n.hidden = true; return; }
  n.hidden = false;
  n.className = 'ticket' + (t.report_error ? ' bad' : '');
  if (t.report_error) {
    n.appendChild(el('b', '建单失败：' + t.report_error));
    n.appendChild(document.createTextNode('　诊断结论不受影响，这一项是可降级的增强。'));
    return;
  }
  n.appendChild(el('b', '工单 ' + t.wo_id + ' 已建　'));
  n.appendChild(document.createTextNode(t.channel + '　record_id ' + t.record_id
      + '　message_id ' + t.message_id
      + (t.high_risk ? '　｜高危：已路由到专家群' : '　｜常规：报修群')
      + ((t.round || 1) > 1 ? '　｜第 ' + t.round + ' 轮，承接 ' + t.prev_wo : '')));

  // 本轮还有没补齐的项，就地给出下一轮入口：闭环不该断在"等下次有人想起来"。
  if ((t.missing || []).length) {
    const a = document.createElement('a');
    a.href = '/?wo=' + encodeURIComponent(t.wo_id);
    a.textContent = '✍️ 还有 ' + t.missing.length + ' 项待补（' + t.missing.join('、')
      + '）→ 补齐后重出第 ' + ((t.round || 1) + 1) + ' 轮';
    a.style.cssText = 'display:block;margin-top:6px;color:inherit';
    n.appendChild(a);
  }
}

function paint(d) {
  $('out').hidden = false;
  paintTicket(d.ticket);

  const flags = $('flags');
  flags.textContent = '';
  const esc = d.stop_and_escalate;
  flags.appendChild(el('span', d.degraded ? '已失效关闭（结论被撤回）'
      : (esc.must_stop ? '必须停止排查' : '可继续排查'),
      (d.degraded || esc.must_stop) ? 'flag stop' : 'flag ok'));
  if (esc.escalate_to_expert) {
    flags.appendChild(el('span', '升级专家：' + (esc.expert_type || '专家'), 'flag esc'));
  }
  // 红线分两类：一类触发即停机或拒绝（红），一类只约束这次作业该怎么做（琥珀）。
  // 全画成红色，「可继续排查」旁边就会糊着一片红，看着像自相矛盾。
  for (const r of esc.rules || []) {
    const b = el('span', (r.hard ? '强制红线 ' : '适用红线 ') + r.id,
                 r.hard ? 'flag stop' : 'flag esc');
    if (r.requirement) b.title = r.requirement;
    flags.appendChild(b);
  }
  flags.appendChild(el('span', '知识库覆盖度：' + d.coverage, 'flag ok'));
  if (d.refused_requests.length) {
    flags.appendChild(el('span', '已拒绝 ' + d.refused_requests.length + ' 条违规请求', 'flag stop'));
  }

  $('verdict').textContent = esc.reason || '';

  // 服务端给的是与命令行 --selfcheck 同一份渲染文本，这里按「## 」拆成块，方便逐字段核对。
  // 末尾的「## 输出自检」跳过不拆：下面有专门的自检面板，拆过来会重复显示一遍。
  const secs = $('secs');
  secs.textContent = '';
  let cur = null, body = null;
  for (const line of d.rendered.split('\\n')) {
    if (line.startsWith('## ')) {
      if (line.slice(3).indexOf('输出自检') === 0) { cur = body = null; continue; }
      if (cur) secs.appendChild(cur), secs.appendChild(body);
      cur = el('h3', line.slice(3));
      cur.style.cssText = 'font-size:13px;margin:14px 0 4px;color:var(--dim)';
      body = el('pre');
    } else if (!cur || line.trim() === '---') {
      continue;
    } else {
      body.textContent += (body.textContent ? '\\n' : '') + line;
    }
  }
  if (cur) secs.appendChild(cur), secs.appendChild(body);

  const checks = $('checks');
  checks.textContent = '';
  for (const c of d.validation || []) {
    checks.appendChild(el('li', (c.ok ? '✓ 通过　' : '✗ 未通过　') + c.name
      + (c.ok || !c.detail ? '' : '：' + c.detail), c.ok ? '' : 'bad'));
  }
  $('out').scrollIntoView({behavior: 'smooth', block: 'start'});
}

$('go').onclick = run;
$('clear').onclick = () => {
  $('text').value = '';
  document.querySelectorAll('#follow-items input').forEach(i => { i.value = ''; });
  document.querySelectorAll('#close-items textarea').forEach(t => { t.value = ''; });
  document.querySelectorAll('#close-items input').forEach(i => { i.checked = false; });
  $('out').hidden = true;
  $('close-out').hidden = true;
  $('err').textContent = '';
};
$('text').addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') run();
});
boot();
</script>
</body>
</html>
"""


def _key_msg(exc: KeyError, fallback: str) -> str:
    """workflow 抛的 KeyError 消息本身就是给人看的整句，直接透传。

    不能用 str(exc)：KeyError 的 str 会带上引号，页面上就成了 "'工单 RX-1 不存在'"。
    """
    return str(exc.args[0]) if exc.args else fallback


def _coverage(kb: KnowledgeBase) -> dict[str, Any]:
    return {
        "root": str(kb.root),
        "device": kb.device.get("device_model", "?"),
        "purpose": kb.device.get("device_purpose", ""),
        "alarm_count": len(kb.alarms),
        "rule_count": len(kb.safety_rules),
        "case_count": len(kb.all_cases),
        "learned_count": len(kb.cases_learned),
        "work_order_count": len(kb.work_orders),
        "alarms": [
            {"code": code, "meaning": a["meaning"],
             "causes": len(a.get("cause_evidence", [])), "steps": len(a.get("steps", [])),
             "cases": len(kb.cases_for(code)), "work_orders": len(kb.work_orders_for(code))}
            for code, a in sorted(kb.alarms.items())
        ],
    }


def _rules(kb: KnowledgeBase, ids: list[str]) -> list[dict[str, Any]]:
    """把触发的红线分成「强制停机/拒绝」与「只是约束怎么做」两类。

    分类依据取自安全规则表自身的字段（forbid_continue_run / escalate_to_expert / refuse），
    不在这一层另立一份红线名单——否则换一台设备、换一张 safety_rules.json，
    网页就会拿旧名单去解释新规则。判定本身仍然全部在 agent/ 里，这里只做展示分类。
    """
    hard_keys = ("forbid_continue_run", "escalate_to_expert", "refuse")
    by_id = {r["id"]: r for r in kb.safety_rules}
    out = []
    for rid in ids:
        rule = by_id.get(rid, {})
        out.append({
            "id": rid,
            "hard": any(rule.get(k) for k in hard_keys),
            "requirement": rule.get("requirement", ""),
        })
    return out


def _payload(d, kb: KnowledgeBase) -> dict[str, Any]:
    """页面要的东西：与命令行同一份渲染文本 + 结构化判定 + 自检逐项。"""
    return {
        "rendered": d.render(verbose_check=True),
        "coverage": d.coverage,
        "degraded": d.degraded,
        "refused_requests": d.refused_requests,
        "stop_and_escalate": {
            "must_stop": d.stop_and_escalate.must_stop,
            "escalate_to_expert": d.stop_and_escalate.escalate_to_expert,
            "expert_type": d.stop_and_escalate.expert_type,
            "triggered_rules": d.stop_and_escalate.triggered_rules,
            "rules": _rules(kb, d.stop_and_escalate.triggered_rules),
            "reason": d.stop_and_escalate.reason,
        },
        "validation": [{"name": c.name, "ok": c.ok, "detail": c.detail}
                       for c in (d.validation.checks if d.validation else [])],
    }


def _submit_report(text: str, reporter: str, kb: KnowledgeBase,
                   caller: Callable[[str, str], str] | None) -> tuple[Any, bool]:
    """走完整报修链路：诊断 → 路由 → 发卡 → 写工单 → 记用量。

    延迟导入：没配飞书凭据的机器也要能打开这个页面，而 workflow 会连带拉起飞书客户端。
    Client() 自己读 .env，缺凭据就落 DryRun，所以这条链路在没网、没有飞书账号的机器上照样跑通。
    """
    from feishu import workflow as wf
    from feishu.client import Client

    client = Client()
    t = wf.submit(text, reporter=reporter, client=client, kb=kb,
                  paths=wf.Paths(), llm=caller)
    return t, client.cfg.live


def _ticket(t: Any, live: bool) -> dict[str, Any]:
    """页面上要看得见「这一单真的落进客户系统了」，而不是一句"已处理"。"""
    return {
        "wo_id": t.wo_id,
        "record_id": t.record_id,
        "message_id": t.message_id,
        "high_risk": t.high_risk,
        "round": t.round,
        "prev_wo": t.prev_wo,
        "missing": list(t.diagnosis.missing_info),
        "channel": ("DryRun：卡片与工单落在本地 var/，未配飞书凭据" if not live
                    else "飞书 live：群里已收到卡片，工单表已多一行" if t.message_id
                    else "飞书 live：工单表已多一行，但卡片未送达，原因见工单「推送状态」"),
    }


def _followup_form(wo_id: str) -> dict[str, Any]:
    """回填页的预填内容：原描述与待补项都从工单存档里读，不再问现场要第二遍。"""
    from feishu import workflow as wf
    from feishu.client import Client

    wo = Client().work_orders().get(wo_id)
    if not wo:
        raise KeyError(wo_id)
    return {
        "wo_id": wo_id,
        "round": int(wo.get("诊断轮次") or 1) + 1,
        "text": wo.get("现场描述", ""),
        "reporter": wo.get("报修人", ""),
        "alarm_code": wo.get("报警码", ""),
        "verdict": (f"风险{wo.get('风险等级', '?')}"
                    f"｜{'已升级专家' if wo.get('是否升级专家') == '是' else '未升级专家'}"
                    f"｜自检{wo.get('输出自检', '?')}"),
        "items": wf.missing_items(wo),
    }


def _run_followup(wo_id: str, answers: dict[str, str], kb: KnowledgeBase,
                  caller: Callable[[str, str], str] | None) -> tuple[Any, bool]:
    """第二轮走完整报修链路：合并描述 → 重跑诊断 → 关联工单 → 发卡 → 记用量。"""
    from feishu import workflow as wf
    from feishu.client import Client

    client = Client()
    t = wf.followup(wo_id, answers, client=client, kb=kb, paths=wf.Paths(), llm=caller)
    return t, client.cfg.live


def _close_form(wo_id: str, code: str) -> dict[str, Any]:
    """双人确认页的预填内容。题目由 workflow 按确认码所属阶段给出，这一层只补一句判定摘要。"""
    from feishu import workflow as wf
    from feishu.client import Client

    form = wf.close_form(wo_id, code, client=Client(), paths=wf.Paths())
    form["verdict"] = (f"风险{form.get('risk') or '?'}"
                       f"｜{'已升级专家' if form.get('escalated') else '未升级专家'}"
                       f"｜自检{form.get('selfcheck') or '?'}")
    return form


def _run_close(wo_id: str, code: str, answers: dict[str, Any], kb: KnowledgeBase,
               caller: Callable[[str, str], str] | None) -> tuple[dict[str, Any], bool]:
    """转交双人确认的提交。走专家阶段还是现场阶段由码决定，页面报什么都不算。"""
    from feishu import workflow as wf
    from feishu.client import Client

    client = Client()
    return wf.close_submit(wo_id, code, answers, client=client, paths=wf.Paths()), client.cfg.live


def _close_payload(res: dict[str, Any], live: bool) -> dict[str, Any]:
    """双人确认这一步的回执。

    页面上要看得见"我这一下到底改了什么"：候选案例有没有生成、知识库有没有被写、
    卡片发去了哪个群。三件事全部取自 workflow 的返回值，页面不自己宣布结果——
    把没入库说成已入库，比不显示更糟。
    """
    cand = res.get("candidate") or {}
    stage, passed = res.get("stage"), res.get("passed")
    promotable = bool(res.get("promotable"))
    rejected = cand.get("field_rejected") or {}

    facts = [f"工单 {cand.get('source_work_order', '')}｜报警码 {cand.get('alarm_code') or '无'}",
             f"候选案例 {cand.get('id', '')}｜状态：{cand.get('status', '')}"]
    if res.get("expert"):
        facts.append(f"技术结论署名（专家）：{res['expert']}")
    for key, label in (("root_cause", "实际根因"), ("disposition", "处置动作"),
                       ("verification", "复机验证")):
        if key in cand:
            facts.append(f"{label}：{cand.get(key) or '（未填写）'}")
    if res.get("reporter"):
        facts.append(f"现场可行性作证（报修人）：{res['reporter']}")
    facts += [f"{k}＝{v}" for k, v in (res.get("facts") or {}).items()]
    if res.get("case_id"):
        facts.append(f"已写入 knowledge/cases_learned.json：{res['case_id']}"
                     f"｜出处标签 KB:LEARNED:{res['case_id']}")
    if rejected.get("discrepancy"):
        facts.append(f"现场分歧（已记在工单「现场分歧」上）：{rejected['discrepancy']}")
    if res.get("expert_code"):
        facts.append(f"重发的专家码：{res['expert_code']}（原来那张已失效）")

    if stage == "expert" and promotable:
        headline = "技术结论已提交，等现场作证"
        lines = [
            "候选案例已生成，状态「待现场确认」——此时它不参与任何诊断，AI 拿不到它当证据",
            "橙卡已发到报修群，请报修人回答四问：照做了吗／恢复了吗／复机读数多少／有无不符",
            "报修人确认通过后才写入 cases_learned.json，专家署名与现场作证一起存",
        ]
    elif stage == "expert":
        headline = "闭环已记录，但这一单沉淀不了"
        lines = [
            f"候选案例状态「{cand.get('status', '')}」：没有报警码就没有可机器复用的匹配特征，入不了知识库",
            "工单上的实际根因／处置动作／复机验证照旧记录，闭环时间与确认专家都已落表",
            "不再发橙卡：让现场确认一个永远入不了库的案例没有意义",
        ]
    elif passed:
        headline = f"两人签字齐了，案例 {res.get('case_id', '')} 已入库"
        lines = [
            "已写入 knowledge/cases_learned.json，从此参与同类故障诊断",
            "证据分量封顶「中」，出处标 KB:LEARNED，绝不与说明书原文混同",
            "绿卡已发到专家群，卡上两张署名都在：技术结论归专家，现场可行归报修人",
        ]
    else:
        headline = "现场确认未通过，已退回专家群"
        lines = [
            f"知识库一个字都没写：候选案例状态改为「{cand.get('status', '')}」",
            "现场的答复与分歧已记在工单上，事后查得出是谁说不可行、说的什么",
            f"红卡退回专家群，附上新的专家码 {res.get('expert_code', '')}",
            "专家改完结论重新提交，会再走一次现场确认——两人签字缺一不可",
        ]

    if not live:
        channel = "DryRun：卡片与工单落在本地 var/，未配飞书凭据"
    elif not res.get("message_id"):
        channel = ("这一阶段不发卡" if stage == "expert" and not promotable
                   else "飞书 live：卡片未送达，原因见工单「推送状态」")
    else:
        channel = f"飞书 live：{'专家群' if stage == 'field' else '报修群'}已收到卡片"

    return {"wo_id": cand.get("source_work_order", ""), "stage": stage, "passed": passed,
            "promotable": promotable, "case_id": res.get("case_id", ""),
            "headline": headline, "facts": facts, "lines": lines, "channel": channel}


def build_server(kb: KnowledgeBase, samples: list[tuple[str, str]],
                 caller: Callable[[str, str], str] | None = None,
                 host: str = "127.0.0.1", port: int = 8765,
                 report_fn: Callable | None = None,
                 followup_fn: Callable | None = None,
                 followup_form_fn: Callable | None = None,
                 close_form_fn: Callable | None = None,
                 close_fn: Callable | None = None) -> ThreadingHTTPServer:
    """只建服务不启动，便于测试绑到随机端口。

    report_fn / followup_fn / followup_form_fn / close_form_fn / close_fn 可注入：
    测试要能覆盖建单、第二轮与双人确认的分支，又不能每次跑测试都去读演示机上的
    真工单存档、或在客户表里多一行真工单。
    """
    # 延迟到建服务时才导入：只打开页面的场景不需要飞书那一层。
    from feishu.workflow import CodeRejected

    submit_report = report_fn or _submit_report
    run_followup = followup_fn or _run_followup
    followup_form = followup_form_fn or _followup_form
    close_form = close_form_fn or _close_form
    run_close = close_fn or _run_close

    class Handler(BaseHTTPRequestHandler):
        server_version = "APX240-FDE"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write("[web] %s\n" % (fmt % args))

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # 只在本机演示用，不需要被任何站点嵌进 iframe
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            path, _, query = self.path.partition("?")
            if path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/knowledge":
                self._json(200, _coverage(kb))
            elif path == "/api/samples":
                self._json(200, [{"label": t, "text": x} for t, x in samples])
            elif path == "/api/followup":
                wo_id = parse_qs(query).get("wo", [""])[0].strip()
                try:
                    self._json(200, followup_form(wo_id))
                except KeyError:
                    self._json(404, {"error": f"工单 {wo_id or '(缺工单号)'} 不存在，无法补充信息"})
            elif path == "/api/close":
                q = parse_qs(query)
                wo_id = q.get("wo", [""])[0].strip()
                code = q.get("k", [""])[0].strip()
                try:
                    self._json(200, close_form(wo_id, code))
                except CodeRejected as exc:
                    self._json(403, {"error": str(exc)})
                except KeyError as exc:
                    self._json(404, {"error": _key_msg(exc, f"工单 {wo_id or '(缺工单号)'} 不存在")})
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
            else:
                self._json(404, {"error": f"没有这个路径：{path}"})

        def _body(self) -> dict[str, Any] | None:
            """读请求体。不合法时已经回过响应，调用方只需 return。"""
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_BODY:
                self._json(400, {"error": f"请求体长度需在 1~{MAX_BODY} 字节之间"})
                return None
            try:
                req = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json(400, {"error": "请求体不是合法的 JSON"})
                return None
            if not isinstance(req, dict):
                self._json(400, {"error": "请求体不是合法的 JSON 对象"})
                return None
            return req

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/diagnose":
                self._diagnose()
            elif path == "/api/followup":
                self._followup()
            elif path == "/api/close":
                self._close()
            else:
                self._json(404, {"error": "没有这个接口"})

        def _diagnose(self) -> None:
            req = self._body()
            if req is None:
                return
            text = str(req.get("text", "")).strip()
            reporter = str(req.get("reporter", "")).strip() or "报修人（现场）"
            want_report = bool(req.get("report"))
            if not text:
                self._json(400, {"error": "现场描述为空，无法诊断。"})
                return

            # 诊断本身不该抛异常；真抛了就是代码 bug，把详情留在服务端控制台，
            # 页面上只给一句话——把 traceback 回显给浏览器没有意义，还会泄露路径。
            ticket: dict[str, Any] | None = None
            d = None
            if want_report:
                # submit 内部只做一次诊断，卡片、工单、用量都基于同一份结论，
                # 所以报修路径不再单独调 diagnose——否则会白跑一次模型调用。
                try:
                    t, live = submit_report(text, reporter, kb, caller)
                    d, ticket = t.diagnosis, _ticket(t, live)
                except Exception as exc:  # noqa: BLE001
                    # 飞书不通绝不能把页面一起拖崩：回落到纯诊断，建单失败单独标出来。
                    sys.stderr.write(f"[web] 报修建单失败：{exc!r}\n")
                    ticket = {"report_error": f"{type(exc).__name__}: {exc}"}
            if d is None:
                try:
                    d = diagnose(text, kb, llm=caller)
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(f"[web] 诊断失败：{exc!r}\n")
                    self._json(500, {"error": "诊断过程出错，详情见服务端控制台。"})
                    return

            payload = _payload(d, kb)
            if ticket is not None:
                payload["ticket"] = ticket
            self._json(200, payload)

        def _followup(self) -> None:
            req = self._body()
            if req is None:
                return
            wo_id = str(req.get("wo", "")).strip()
            raw = req.get("answers")
            answers = ({str(k): str(v) for k, v in raw.items()}
                       if isinstance(raw, dict) else {})
            if not wo_id:
                self._json(400, {"error": "缺少工单号，不知道要补哪一单。"})
                return

            try:
                t, live = run_followup(wo_id, answers, kb, caller)
            except KeyError:
                self._json(404, {"error": f"工单 {wo_id} 不存在，无法补充信息"})
                return
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"[web] 第二轮诊断失败：{exc!r}\n")
                self._json(500, {"error": "第二轮诊断出错，详情见服务端控制台。"})
                return

            payload = _payload(t.diagnosis, kb)
            payload["ticket"] = _ticket(t, live)
            self._json(200, payload)

        def _close(self) -> None:
            """专家提交处置结论／报修人现场确认，共用这一个接口，阶段由确认码判定。"""
            req = self._body()
            if req is None:
                return
            wo_id = str(req.get("wo", "")).strip()
            code = str(req.get("k", "")).strip()
            raw = req.get("answers")
            answers = ({str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {})
            if not wo_id:
                self._json(400, {"error": "缺少工单号，不知道要确认哪一单。"})
                return
            if not code:
                # 没有码就没有这一步的权限：这是 403，不是"参数没填对"的 400。
                self._json(403, {"error": "缺确认码：请从群里的卡片点开，不要手抄地址。"})
                return

            try:
                res, live = run_close(wo_id, code, answers, kb, caller)
            except CodeRejected as exc:
                self._json(403, {"error": str(exc)})
                return
            except KeyError as exc:
                self._json(404, {"error": _key_msg(exc, f"工单 {wo_id} 不存在")})
                return
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"[web] 双人确认失败：{exc!r}\n")
                self._json(500, {"error": "确认提交出错，详情见服务端控制台。"})
                return

            self._json(200, _close_payload(res, live))

    return ThreadingHTTPServer((host, port), Handler)


def serve(kb: KnowledgeBase, samples: list[tuple[str, str]],
          caller: Callable[[str, str], str] | None = None,
          host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True) -> int:
    """起本地服务并阻塞。Ctrl-C 退出，返回 0。"""
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"警告：正在监听 {host}，同一网络内的其他机器都能访问这个入口。\n"
              "      演示完请立刻 Ctrl-C 关掉；默认只监听 127.0.0.1。", file=sys.stderr)

    httpd = build_server(kb, samples, caller, host, port)
    bound = httpd.server_address
    shown = "127.0.0.1" if bound[0] in ("0.0.0.0", "::") else bound[0]
    url = f"http://{shown}:{bound[1]}/"
    print(f"APX-240 故障诊断 Agent 已启动：{url}")
    print(f"知识库：{kb.root}　设备：{kb.device.get('device_model', '?')}")
    # 卡片上的按钮指向 APX240_WEB_BASE，没配就不出按钮。这件事必须在启动时说清，
    # 不能等卡片已经发进真群、发现点不进去才回头找原因。
    from feishu.client import Client
    web_base = Client().cfg.web_base_url.rstrip("/")
    if web_base:
        print(f"补充信息入口：{web_base}/?wo=工单号")
        print(f"双人确认入口：{web_base}/?wo=工单号&k=一次性确认码"
              "（专家填处置结论／报修人确认现场可行）")
    else:
        print("未设 APX240_WEB_BASE：卡片上不会出现「补充信息」与「填写处置结论」按钮")
    print("按 Ctrl-C 停止。")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
    return 0
