"""Small FastAPI dashboard for watching DLCPD-25 training progress.

Run from the repo root:
    uvicorn dlcpd25_v2.web.progress:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from dlcpd25_v2.common import repo_root

TRAINING_ROOT = "artifacts/training/classification"
INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DLCPD-25 分类训练进度</title>
<style>
:root { --bg:#0f172a; --card:#1e293b; --line:#334155; --text:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; --ok:#34d399; --warn:#fbbf24; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans CJK SC",sans-serif; }
header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; align-items:center; }
h1 { font-size:20px; margin:0; }
.status { font-size:13px; padding:6px 12px; border-radius:999px; background:var(--line); }
.status.running { background:rgba(56,189,248,.15); color:var(--accent); }
.status.completed { background:rgba(52,211,153,.15); color:var(--ok); }
main { max-width:1080px; margin:20px auto; padding:0 16px; display:grid; gap:16px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px; }
.card .label { color:var(--muted); font-size:12px; margin-bottom:8px; }
.card .value { font-size:24px; font-weight:650; font-variant-numeric:tabular-nums; }
.card .sub { color:var(--muted); font-size:12px; margin-top:6px; }
.progress { height:10px; background:#0b1220; border-radius:999px; overflow:hidden; margin:10px 0 4px; }
.progress > span { display:block; height:100%; width:0; background:linear-gradient(90deg,var(--accent),var(--ok)); border-radius:999px; transition:width .5s; }
table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:14px; overflow:hidden; }
th,td { padding:10px 12px; text-align:left; font-size:13px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; }
svg { width:100%; height:240px; background:var(--card); border:1px solid var(--line); border-radius:14px; }
.tick { fill:var(--muted); font-size:10px; }
.legend { display:flex; gap:16px; color:var(--muted); font-size:12px; margin-bottom:8px; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
</style>
</head>
<body>
<header>
  <h1>DLCPD-25 · Plan-A 分类训练</h1>
  <div><span id="runId" style="color:var(--muted);margin-right:12px;"></span><span id="status" class="status">连接中…</span></div>
</header>
<main>
  <section class="cards" id="cards"></section>
  <section>
    <div class="legend"><span><span class="dot" style="background:#38bdf8"></span>训练 loss</span><span><span class="dot" style="background:#34d399"></span>val macro-F1</span></div>
    <svg id="chart" viewBox="0 0 800 240" preserveAspectRatio="none"></svg>
  </section>
  <section>
    <h2 style="font-size:15px;">每轮评估</h2>
    <div style="overflow:auto;"><table id="history"><thead><tr><th>epoch</th><th>train_loss</th><th>train_top1</th><th>val_loss</th><th>val_top1</th><th>val_top5</th><th>val_macro_f1</th></tr></thead><tbody></tbody></table></div>
  </section>
</main>
<script>
async function refresh(){
  try{
    const res = await fetch('/api/progress', {cache:'no-store'});
    const data = await res.json();
    const s = data.state || {};
    document.getElementById('status').textContent = s.status || '未知';
    document.getElementById('status').className = 'status ' + (s.status || '');
    document.getElementById('runId').textContent = s.run_id || '';
    const pct = (v,total) => total ? Math.max(0,Math.min(100,v/total*100)).toFixed(2) : 0;
    const epochPct = pct((s.epoch||0)-1 + (s.batch_in_epoch||0)/(s.steps_per_epoch||1), s.total_epochs||1);
    const totalPct = pct(s.global_step||0, s.total_steps||1);
    const fmt = v => v==null||v===undefined ? '—' : Number(v).toLocaleString(undefined,{maximumFractionDigits:4});
    const eta = v => v==null ? '—' : String(Math.floor(v/3600)).padStart(2,'0')+':'+String(Math.floor((v%3600)/60)).padStart(2,'0')+':'+String(Math.floor(v%60)).padStart(2,'0');
    document.getElementById('cards').innerHTML = [
      ['轮次 / 总轮次', (s.epoch||0)+' / '+ (s.total_epochs||0), '当前第 '+ (s.batch_in_epoch||0) +' / '+ (s.steps_per_epoch||0) +' 批'],
      ['轮次进度', epochPct+'%', '<div class="progress"><span style="width:'+epochPct+'%"></span></div>'],
      ['全局进度', totalPct+'%', '<div class="progress"><span style="width:'+totalPct+'%"></span></div>'],
      ['最近 loss', fmt(s.loss_recent), '本轮平均 '+fmt(s.avg_epoch_loss)],
      ['学习率', fmt(s.lr), 'EMA 模型用于验证'],
      ['最佳 macro-F1', fmt(s.best_metric), '选择指标 val_macro_f1'],
      ['已运行', eta(s.elapsed_seconds), 'GPU '+fmt(s.gpu_memory_mb)+' MB'],
      ['预计剩余', eta(s.eta_total_seconds), '本 epoch 剩余 '+eta(s.eta_epoch_seconds)],
    ].map(c=>'<div class="card"><div class="label">'+c[0]+'</div><div class="value">'+c[1]+'</div><div class="sub">'+c[2]+'</div></div>').join('');
    const hist = data.history || [];
    const body = document.querySelector('#history tbody');
    body.innerHTML = hist.slice().reverse().map(h=>'<tr><td>'+h.epoch+'</td><td>'+fmt(h.train_loss)+'</td><td>'+fmt(h.train_top1)+'</td><td>'+fmt(h.val_loss)+'</td><td>'+fmt(h.val_top1)+'</td><td>'+fmt(h.val_top5)+'</td><td>'+fmt(h.val_macro_f1)+'</td></tr>').join('');
    draw(hist);
  }catch(e){
    document.getElementById('status').textContent = '等待训练启动';
    document.getElementById('status').className = 'status';
  }
}
function draw(hist){
  const svg=document.getElementById('chart');
  svg.innerHTML='';
  if(!hist.length) return;
  const W=800,H=240,P=32;
  const losses=hist.map(h=>h.train_loss);
  const f1s=hist.map(h=>h.val_macro_f1);
  const all=losses.concat(f1s).filter(v=>v!=null);
  const min=Math.min(...all), max=Math.max(...all);
  const range=(max-min)||1;
  const x=i=>P+i*(W-2*P)/Math.max(hist.length-1,1);
  const y=v=>H-P-(v-min)/range*(H-2*P);
  function path(vals){
    return vals.map((v,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)).join(' ');
  }
  const ns='http://www.w3.org/2000/svg';
  const path1=document.createElementNS(ns,'path'); path1.setAttribute('d',path(losses)); path1.setAttribute('fill','none'); path1.setAttribute('stroke','#38bdf8'); path1.setAttribute('stroke-width','2'); svg.appendChild(path1);
  const path2=document.createElementNS(ns,'path'); path2.setAttribute('d',path(f1s)); path2.setAttribute('fill','none'); path2.setAttribute('stroke','#34d399'); path2.setAttribute('stroke-width','2'); svg.appendChild(path2);
  for(let i=0;i<=4;i++){
    const v=min+range*i/4;
    const yy=y(v);
    const line=document.createElementNS(ns,'line'); line.setAttribute('x1',P); line.setAttribute('x2',W-P); line.setAttribute('y1',yy); line.setAttribute('y2',yy); line.setAttribute('stroke','#334155'); line.setAttribute('stroke-width','1'); svg.appendChild(line);
    const text=document.createElementNS(ns,'text'); text.setAttribute('x',4); text.setAttribute('y',yy+3); text.setAttribute('class','tick'); text.textContent=v.toFixed(2); svg.appendChild(text);
  }
}
setInterval(refresh,3000);
refresh();
</script>
</body>
</html>
"""

app = FastAPI(title="DLCPD-25 Training Progress")


def _runs_dir() -> Path:
    return repo_root() / TRAINING_ROOT


def _latest_run() -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    runs_dir = _runs_dir()
    if not runs_dir.is_dir():
        return Path(), {}, []
    candidates = list(runs_dir.glob("*/state.json"))
    if not candidates:
        return Path(), {}, []
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    try:
        state = json.loads(latest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        state = {}
    history_path = latest.with_name("history.json")
    history: list[dict[str, Any]] = []
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        history = []
    return latest.parent, state, history


@app.get("/api/progress")
def api_progress() -> dict[str, Any]:
    run_dir, state, history = _latest_run()
    runs_dir = _runs_dir()
    return {
        "state": state,
        "history": history,
        "run_dir": str(run_dir),
        "updated_at": time.time(),
        "run_count": len(list(runs_dir.glob("*/state.json"))) if runs_dir.is_dir() else 0,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
