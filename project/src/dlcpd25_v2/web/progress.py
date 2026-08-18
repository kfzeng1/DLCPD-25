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

TRAINING_ROOTS = (
    "artifacts/training/classification",
    "artifacts/training/detection",
)
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
.card, .panel { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px; }
.card .label { color:var(--muted); font-size:12px; margin-bottom:8px; }
.card .value { font-size:24px; font-weight:650; font-variant-numeric:tabular-nums; }
.card .sub { color:var(--muted); font-size:12px; margin-top:6px; }
.progress { height:10px; background:#0b1220; border-radius:999px; overflow:hidden; margin:10px 0 4px; }
.progress > span { display:block; height:100%; width:0; background:linear-gradient(90deg,var(--accent),var(--ok)); border-radius:999px; transition:width .5s; }
.panel h2 { font-size:15px; margin:0 0 4px; }
.panel .hint { color:var(--muted); font-size:12px; margin:0 0 10px; }
.legend { display:flex; flex-wrap:wrap; gap:16px; color:var(--muted); font-size:12px; margin:8px 0 4px; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
svg { width:100%; height:auto; display:block; }
table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:14px; overflow:hidden; }
th,td { padding:10px 12px; text-align:left; font-size:13px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; }
.tick { fill:var(--muted); font-size:10px; }
.no-data { fill:var(--muted); font-size:14px; text-anchor:middle; }
</style>
</head>
<body>
<header>
  <h1 id="pageTitle">DLCPD-25 · Plan-A 训练进度</h1>
  <div><span id="runId" style="color:var(--muted);margin-right:12px;"></span><span id="status" class="status">连接中…</span></div>
</header>
<main>
  <section class="cards" id="cards"></section>

  <section class="panel">
    <h2 id="lossTitle">损失曲线</h2>
    <p class="hint">数值越低越好。两条线应整体下降。</p>
    <div class="legend">
      <span><span class="dot" style="background:#38bdf8"></span><span id="legTrainLoss">训练 loss</span></span>
      <span><span class="dot" style="background:#fb7185"></span><span id="legValLoss">验证 loss</span></span>
    </div>
    <svg id="lossChart" viewBox="0 0 800 270" preserveAspectRatio="xMidYMid meet"></svg>
  </section>

  <section class="panel">
    <h2 id="metricTitle">验证指标曲线</h2>
    <p class="hint">数值越高越好，最需要关注的是绿色 val macro-F1。</p>
    <div class="legend">
      <span><span class="dot" style="background:#34d399"></span><span id="legMacroF1">val macro-F1</span></span>
      <span><span class="dot" style="background:#38bdf8"></span><span id="legTop1">val Top-1</span></span>
      <span><span class="dot" style="background:#fbbf24"></span><span id="legTop5">val Top-5</span></span>
    </div>
    <svg id="metricChart" viewBox="0 0 800 270" preserveAspectRatio="xMidYMid meet"></svg>
  </section>

  <section class="panel">
    <h2>每轮评估</h2>
    <div style="overflow:auto;"><table id="history"><thead id="historyHead"></thead><tbody></tbody></table></div>
  </section>
</main>
<script>
async function refresh(){
  try{
    const res = await fetch('/api/progress', {cache:'no-store'});
    const data = await res.json();
    const s = data.state || {};
    const task = data.task || (s.task || 'classification');
    document.title = task === 'detection' ? 'IP102 检测训练进度' : 'DLCPD-25 分类训练进度';
    document.getElementById('pageTitle').textContent = task === 'detection' ? 'IP102 · Plan-A 检测训练' : 'DLCPD-25 · Plan-A 分类训练';
    document.getElementById('status').textContent = s.status || '未知';
    document.getElementById('status').className = 'status ' + (s.status || '');
    document.getElementById('runId').textContent = s.run_id || '';
    const pct = (v,total) => total ? Math.max(0,Math.min(100,v/total*100)).toFixed(2) : 0;
    const batchPct = pct(s.batch_in_epoch||0, s.steps_per_epoch||1);
    const globalPct = pct((s.epoch||1)-1 + (s.batch_in_epoch||0)/(s.steps_per_epoch||1), s.total_epochs||1);
    const fmt = v => {
      if(v==null||v===undefined) return '—';
      const n=Number(v);
      if(n!==0 && Math.abs(n)<0.001) return n.toExponential(3);
      return n.toLocaleString(undefined,{maximumFractionDigits:4});
    };
    const eta = v => v==null ? '—' : String(Math.floor(v/3600)).padStart(2,'0')+':'+String(Math.floor((v%3600)/60)).padStart(2,'0')+':'+String(Math.floor(v%60)).padStart(2,'0');
    document.getElementById('cards').innerHTML = [
      ['轮次 / 总轮次', (s.epoch||0)+' / '+ (s.total_epochs||0), '当前第 '+ (s.batch_in_epoch||0) +' / '+ (s.steps_per_epoch||0) +' 批'],
      ['本轮 batch 进度', batchPct+'%', '<div class="progress"><span style="width:'+batchPct+'%"></span></div>'],
      ['全局 epoch 进度', globalPct+'%', '<div class="progress"><span style="width:'+globalPct+'%"></span></div>'],
      ['最近 loss', fmt(s.loss_recent), '本轮平均 '+fmt(s.avg_epoch_loss)],
      ['学习率', fmt(s.lr), 'EMA 模型用于验证'],
      [task === 'detection' ? '最佳 mAP50:95' : '最佳 macro-F1', fmt(s.best_metric), task === 'detection' ? '选择指标 val_mAP_50_95' : '选择指标 val_macro_f1'],
      ['已运行', eta(s.elapsed_seconds), 'GPU '+fmt(s.gpu_memory_mb)+' MB'],
      ['预计剩余', eta(s.eta_total_seconds), '本 epoch 剩余 '+eta(s.eta_epoch_seconds)],
    ].map(c=>'<div class="card"><div class="label">'+c[0]+'</div><div class="value">'+c[1]+'</div><div class="sub">'+c[2]+'</div></div>').join('');
    const hist = data.history || [];
    const body = document.querySelector('#history tbody');
    const head = document.getElementById('historyHead');
    const last = hist.length ? hist[hist.length-1] : null;
    if(task === 'detection'){
      head.innerHTML = '<tr><th>epoch</th><th>train_loss</th><th>val_mAP50:95</th><th>val_AP50</th><th>val_AP75</th><th>val_AR100</th></tr>';
      body.innerHTML = hist.slice().reverse().map(h=>'<tr><td>'+h.epoch+'</td><td>'+fmt(h.train_loss)+'</td><td>'+fmt(h.val_mAP_50_95)+'</td><td>'+fmt(h.val_AP50)+'</td><td>'+fmt(h.val_AP75)+'</td><td>'+fmt(h.val_AR_100)+'</td></tr>').join('');
      document.getElementById('lossTitle').textContent = '训练损失曲线';
      document.getElementById('metricTitle').textContent = '验证 mAP 曲线';
      document.getElementById('legTrainLoss').textContent = last ? '训练 loss（最新 '+fmt(last.train_loss)+'）' : '训练 loss';
      document.getElementById('legValLoss').textContent = '';
      document.getElementById('legMacroF1').textContent = last ? 'val mAP50:95（最新 '+fmt(last.val_mAP_50_95)+'）' : 'val mAP50:95';
      document.getElementById('legTop1').textContent = last ? 'val AP50（最新 '+fmt(last.val_AP50)+'）' : 'val AP50';
      document.getElementById('legTop5').textContent = last ? 'val AP75（最新 '+fmt(last.val_AP75)+'）' : 'val AP75';
      drawChart('lossChart', [
        {name:'train_loss', color:'#38bdf8', values:hist.map(h=>h.train_loss)}
      ], v=>v.toFixed(3));
      drawChart('metricChart', [
        {name:'mAP50:95', color:'#34d399', values:hist.map(h=>h.val_mAP_50_95)},
        {name:'AP50', color:'#38bdf8', values:hist.map(h=>h.val_AP50)},
        {name:'AP75', color:'#fbbf24', values:hist.map(h=>h.val_AP75)}
      ], v=>v.toFixed(1)+'%');
    }else{
      head.innerHTML = '<tr><th>epoch</th><th>train_loss</th><th>train_top1</th><th>val_loss</th><th>val_top1</th><th>val_top5</th><th>val_macro_f1</th></tr>';
      body.innerHTML = hist.slice().reverse().map(h=>'<tr><td>'+h.epoch+'</td><td>'+fmt(h.train_loss)+'</td><td>'+fmt(h.train_top1)+'</td><td>'+fmt(h.val_loss)+'</td><td>'+fmt(h.val_top1)+'</td><td>'+fmt(h.val_top5)+'</td><td>'+fmt(h.val_macro_f1)+'</td></tr>').join('');
      document.getElementById('lossTitle').textContent = '损失曲线';
      document.getElementById('metricTitle').textContent = '验证指标曲线';
      document.getElementById('legTrainLoss').textContent = last ? '训练 loss（最新 '+fmt(last.train_loss)+'）' : '训练 loss';
      document.getElementById('legValLoss').textContent = last ? '验证 loss（最新 '+fmt(last.val_loss)+'）' : '验证 loss';
      document.getElementById('legMacroF1').textContent = last ? 'val macro-F1（最新 '+fmt(last.val_macro_f1)+'）' : 'val macro-F1';
      document.getElementById('legTop1').textContent = last ? 'val Top-1（最新 '+fmt(last.val_top1)+'）' : 'val Top-1';
      document.getElementById('legTop5').textContent = last ? 'val Top-5（最新 '+fmt(last.val_top5)+'）' : 'val Top-5';
      drawChart('lossChart', [
        {name:'train_loss', color:'#38bdf8', values:hist.map(h=>h.train_loss)},
        {name:'val_loss', color:'#fb7185', values:hist.map(h=>h.val_loss)}
      ], v=>v.toFixed(3));
      drawChart('metricChart', [
        {name:'macro_f1', color:'#34d399', values:hist.map(h=>h.val_macro_f1)},
        {name:'top1', color:'#38bdf8', values:hist.map(h=>h.val_top1)},
        {name:'top5', color:'#fbbf24', values:hist.map(h=>h.val_top5)}
      ], v=>v.toFixed(1)+'%');
    }
  }catch(e){
    document.getElementById('status').textContent = '等待训练启动';
    document.getElementById('status').className = 'status';
  }
}
function drawChart(id, series, yFmt){
  const svg=document.getElementById(id);
  svg.innerHTML='';
  const W=800,H=270,L=52,R=18,T=14,B=32;
  const ns='http://www.w3.org/2000/svg';
  const values=series.flatMap(s=>s.values).filter(v=>v!=null && Number.isFinite(Number(v)));
  if(!values.length){
    const text=document.createElementNS(ns,'text'); text.setAttribute('x',W/2); text.setAttribute('y',H/2); text.setAttribute('class','no-data'); text.textContent='暂无数据'; svg.appendChild(text); return;
  }
  const n=Math.max(...series.map(s=>s.values.length));
  let min=Math.min(...values), max=Math.max(...values);
  if(min===max){ min-=1; max+=1; }
  else { const pad=(max-min)*0.10; min-=pad; max+=pad; }
  const x=(i)=> n<=1 ? L+(W-L-R)/2 : L+i*(W-L-R)/(n-1);
  const y=(v)=> T+(max-Number(v))*(H-T-B)/(max-min);
  for(let t=0;t<=4;t++){
    const val=max-(max-min)*t/4;
    const yy=y(val);
    const line=document.createElementNS(ns,'line'); line.setAttribute('x1',L); line.setAttribute('x2',W-R); line.setAttribute('y1',yy); line.setAttribute('y2',yy); line.setAttribute('stroke','#334155'); line.setAttribute('stroke-width','1'); svg.appendChild(line);
    const text=document.createElementNS(ns,'text'); text.setAttribute('x',L-6); text.setAttribute('y',yy+3); text.setAttribute('text-anchor','end'); text.setAttribute('class','tick'); text.textContent=yFmt(val); svg.appendChild(text);
  }
  const step=Math.max(1,Math.ceil(n/12));
  for(let i=0;i<n;i+=step){
    const xx=x(i);
    const text=document.createElementNS(ns,'text'); text.setAttribute('x',xx); text.setAttribute('y',H-8); text.setAttribute('text-anchor','middle'); text.setAttribute('class','tick'); text.textContent=String(i+1); svg.appendChild(text);
  }
  const axis=document.createElementNS(ns,'line'); axis.setAttribute('x1',L); axis.setAttribute('x2',W-R); axis.setAttribute('y1',H-B); axis.setAttribute('y2',H-B); axis.setAttribute('stroke','#475569'); axis.setAttribute('stroke-width','1'); svg.appendChild(axis);
  series.forEach(s=>{
    const points=s.values.map((v,i)=>v==null?null:{x:x(i),y:y(v)});
    const path=points.map((p,i)=>(i===0?'M':'L')+p.x.toFixed(1)+' '+p.y.toFixed(1)).join(' ');
    const p=document.createElementNS(ns,'path'); p.setAttribute('d',path); p.setAttribute('fill','none'); p.setAttribute('stroke',s.color); p.setAttribute('stroke-width','2'); p.setAttribute('stroke-linejoin','round'); p.setAttribute('stroke-linecap','round'); svg.appendChild(p);
    if(n<=20){
      points.forEach(pt=>{
        const c=document.createElementNS(ns,'circle'); c.setAttribute('cx',pt.x); c.setAttribute('cy',pt.y); c.setAttribute('r','3'); c.setAttribute('fill',s.color); svg.appendChild(c);
      });
    }
  });
}
setInterval(refresh,3000);
refresh();
</script>
</body>
</html>
"""

app = FastAPI(title="DLCPD-25 Training Progress")


def _latest_run() -> tuple[str, Path, dict[str, Any], list[dict[str, Any]]]:
    candidates: list[Path] = []
    for relative_root in TRAINING_ROOTS:
        runs_dir = repo_root() / relative_root
        if runs_dir.is_dir():
            candidates.extend(runs_dir.glob("*/state.json"))
    if not candidates:
        return "", Path(), {}, []
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
    task = str(state.get("task") or latest.parent.parent.name)
    return task, latest.parent, state, history


@app.get("/api/progress")
def api_progress() -> dict[str, Any]:
    task, run_dir, state, history = _latest_run()
    run_count = 0
    for relative_root in TRAINING_ROOTS:
        runs_dir = repo_root() / relative_root
        if runs_dir.is_dir():
            run_count += len(list(runs_dir.glob("*/state.json")))
    return {
        "task": task,
        "state": state,
        "history": history,
        "run_dir": str(run_dir),
        "updated_at": time.time(),
        "run_count": run_count,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
