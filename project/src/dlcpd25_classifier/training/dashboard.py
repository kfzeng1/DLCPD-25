"""Serve a local, read-only dashboard for an A2 training run."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload(run_dir: Path) -> dict[str, Any]:
    history = _read_json(run_dir / "history.json", [])
    state = _read_json(run_dir / "run-state.json", {"status": "initializing"})
    final_metrics = _read_json(run_dir / "metrics.json", None)
    config_path = run_dir / "resolved-config.yaml"
    return {
        "run_id": run_dir.name,
        "state": state,
        "history": history,
        "final_metrics": final_metrics,
        "has_config": config_path.is_file(),
        "csv_path": str(run_dir / "metrics.csv"),
    }


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DLCPD-25 Training Monitor</title>
<style>
:root{color-scheme:light;--ink:#17202a;--muted:#667085;--line:#d9dee7;--soft:#f5f7fa;--blue:#2563eb;--red:#dc3f45;--green:#16845b;--amber:#a76500}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;letter-spacing:0}
header{border-bottom:1px solid var(--line);background:#fff}.header-inner,main{width:min(1440px,calc(100% - 40px));margin:auto}.header-inner{min-height:72px;display:flex;align-items:center;justify-content:space-between;gap:24px}
.brand{display:flex;align-items:baseline;gap:12px;min-width:0}.brand strong{font-size:19px}.brand span{color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.status{display:flex;align-items:center;gap:8px;font-weight:600;white-space:nowrap}.dot{width:9px;height:9px;border-radius:50%;background:var(--amber)}.dot.running{background:var(--green)}.dot.completed_pending_project_lead_acceptance{background:var(--blue)}
main{padding:24px 0 44px}.summary{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));border:1px solid var(--line);border-radius:6px;overflow:hidden}.metric{min-height:92px;padding:16px;border-right:1px solid var(--line)}.metric:last-child{border-right:0}.metric-label{color:var(--muted);font-size:12px}.metric-value{display:block;margin-top:8px;font-size:22px;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:24px}.panel{border:1px solid var(--line);border-radius:6px;padding:16px;min-width:0}.panel-head{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:12px}.panel h2{margin:0;font-size:15px}.legend{display:flex;gap:14px;color:var(--muted);font-size:12px}.key{display:inline-flex;align-items:center;gap:6px}.swatch{width:11px;height:3px;background:var(--blue)}.swatch.red{background:var(--red)}.swatch.green{background:var(--green)}.swatch.amber{background:var(--amber)}
.chart-wrap{height:280px;position:relative}canvas{display:block;width:100%;height:100%}.table-band{margin-top:24px}.table-head{display:flex;align-items:end;justify-content:space-between;gap:16px;margin-bottom:10px}.table-head h2{margin:0;font-size:15px}.updated{color:var(--muted);font-size:12px}.table-scroll{overflow:auto;border:1px solid var(--line);border-radius:6px}table{width:100%;border-collapse:collapse;min-width:1020px;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{position:sticky;top:0;background:var(--soft);color:#475467;font-size:12px}th:first-child,td:first-child{text-align:left}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#fafbfc}.empty{padding:52px 16px;text-align:center;color:var(--muted)}
@media(max-width:980px){.summary{grid-template-columns:repeat(3,1fr)}.metric:nth-child(3){border-right:0}.metric:nth-child(-n+3){border-bottom:1px solid var(--line)}.charts{grid-template-columns:1fr}}
@media(max-width:600px){.header-inner,main{width:min(100% - 24px,1440px)}.header-inner{min-height:64px}.brand{display:block}.brand span{display:block;margin-top:2px}.summary{grid-template-columns:repeat(2,1fr)}.metric{border-bottom:1px solid var(--line)}.metric:nth-child(odd){border-right:1px solid var(--line)}.metric:nth-child(even){border-right:0}.metric:nth-last-child(-n+2){border-bottom:0}.metric-value{font-size:19px}.chart-wrap{height:230px}.status span:last-child{display:none}}
</style>
</head>
<body>
<header><div class="header-inner"><div class="brand"><strong>DLCPD-25</strong><span id="run-name">Training Monitor</span></div><div class="status"><i id="status-dot" class="dot"></i><span id="status-text">正在连接</span><span id="epoch-text"></span></div></div></header>
<main>
<section class="summary" aria-label="训练摘要">
<div class="metric"><span class="metric-label">当前 Epoch</span><strong class="metric-value" id="m-epoch">-</strong></div>
<div class="metric"><span class="metric-label">学习率</span><strong class="metric-value" id="m-lr">-</strong></div>
<div class="metric"><span class="metric-label">Val Top-1</span><strong class="metric-value" id="m-acc">-</strong></div>
<div class="metric"><span class="metric-label">Val Macro-F1</span><strong class="metric-value" id="m-f1">-</strong></div>
<div class="metric"><span class="metric-label">Val Top-5</span><strong class="metric-value" id="m-top5">-</strong></div>
<div class="metric"><span class="metric-label">单轮耗时</span><strong class="metric-value" id="m-time">-</strong></div>
</section>
<section class="charts">
<div class="panel"><div class="panel-head"><h2>Loss</h2><div class="legend"><span class="key"><i class="swatch"></i>Train</span><span class="key"><i class="swatch red"></i>Val</span></div></div><div class="chart-wrap"><canvas id="loss-chart" aria-label="训练和验证 loss 曲线"></canvas></div></div>
<div class="panel"><div class="panel-head"><h2>Validation Scores</h2><div class="legend"><span class="key"><i class="swatch"></i>Top-1</span><span class="key"><i class="swatch green"></i>Macro-F1</span><span class="key"><i class="swatch amber"></i>Balanced</span></div></div><div class="chart-wrap"><canvas id="score-chart" aria-label="验证指标曲线"></canvas></div></div>
</section>
<section class="table-band"><div class="table-head"><h2>Epoch History</h2><span class="updated" id="updated">-</span></div><div class="table-scroll"><table><thead><tr><th>Epoch</th><th>LR</th><th>Train Loss</th><th>Train Acc</th><th>Val Loss</th><th>Val Top-1</th><th>Val Top-5</th><th>Macro-F1</th><th>Balanced Acc</th><th>Train img/s</th><th>耗时</th></tr></thead><tbody id="rows"><tr><td colspan="11" class="empty">等待首轮指标</td></tr></tbody></table></div></section>
</main>
<script>
const colors={blue:'#2563eb',red:'#dc3f45',green:'#16845b',amber:'#a76500',grid:'#d9dee7',muted:'#667085'};
const pct=v=>v==null?'-':(v*100).toFixed(2)+'%'; const num=(v,n=4)=>v==null?'-':Number(v).toFixed(n); const duration=s=>s==null?'-':`${Math.floor(s/60)}m ${Math.round(s%60)}s`;
function drawChart(canvas,labels,series,{percent=false}={}){const box=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1,w=Math.max(320,box.width),h=Math.max(220,box.height);canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,w,h);const p={l:48,r:14,t:14,b:32};const values=series.flatMap(s=>s.values).filter(Number.isFinite);if(!values.length){c.fillStyle=colors.muted;c.textAlign='center';c.fillText('等待更多数据',w/2,h/2);return}let min=percent?0:Math.min(...values),max=percent?1:Math.max(...values);if(max===min){max+=1}const x=i=>p.l+(labels.length===1?0:(i/(labels.length-1))*(w-p.l-p.r));const y=v=>p.t+(1-(v-min)/(max-min))*(h-p.t-p.b);c.strokeStyle=colors.grid;c.fillStyle=colors.muted;c.lineWidth=1;c.font='11px system-ui';for(let i=0;i<=4;i++){const yy=p.t+i*(h-p.t-p.b)/4;c.beginPath();c.moveTo(p.l,yy);c.lineTo(w-p.r,yy);c.stroke();const value=max-i*(max-min)/4;c.textAlign='right';c.fillText(percent?(value*100).toFixed(0)+'%':value.toFixed(2),p.l-8,yy+4)}series.forEach(s=>{c.strokeStyle=s.color;c.lineWidth=2;c.beginPath();s.values.forEach((v,i)=>{if(!Number.isFinite(v))return;const xx=x(i),yy=y(v);i?c.lineTo(xx,yy):c.moveTo(xx,yy)});c.stroke();s.values.forEach((v,i)=>{if(!Number.isFinite(v))return;c.fillStyle=s.color;c.beginPath();c.arc(x(i),y(v),3,0,Math.PI*2);c.fill()})});c.fillStyle=colors.muted;c.textAlign='center';labels.forEach((label,i)=>{if(labels.length<=8||i===0||i===labels.length-1||i%Math.ceil(labels.length/8)===0)c.fillText(label,x(i),h-10)})}
function render(data){const h=data.history||[],last=h.at(-1),state=data.state||{};document.getElementById('run-name').textContent=data.run_id;const status=state.status||'running';document.getElementById('status-text').textContent=status.startsWith('completed')?'训练完成':status==='running'?'训练中':'初始化';document.getElementById('status-dot').className='dot '+status;document.getElementById('epoch-text').textContent=last?`Epoch ${last.epoch}`:'';document.getElementById('m-epoch').textContent=last?last.epoch:'-';document.getElementById('m-lr').textContent=last?Number(last.learning_rate).toExponential(2):'-';document.getElementById('m-acc').textContent=last?pct(last.val.accuracy):'-';document.getElementById('m-f1').textContent=last?pct(last.val.macro_f1):'-';document.getElementById('m-top5').textContent=last?pct(last.val.top5_accuracy):'-';document.getElementById('m-time').textContent=last?duration(last.train.duration_seconds+last.val.duration_seconds):'-';const labels=h.map(r=>String(r.epoch));drawChart(document.getElementById('loss-chart'),labels,[{color:colors.blue,values:h.map(r=>r.train.loss)},{color:colors.red,values:h.map(r=>r.val.loss)}]);drawChart(document.getElementById('score-chart'),labels,[{color:colors.blue,values:h.map(r=>r.val.accuracy)},{color:colors.green,values:h.map(r=>r.val.macro_f1)},{color:colors.amber,values:h.map(r=>r.val.balanced_accuracy)}],{percent:true});const rows=document.getElementById('rows');rows.innerHTML=h.length?h.slice().reverse().map(r=>`<tr><td>${r.epoch}</td><td>${Number(r.learning_rate).toExponential(2)}</td><td>${num(r.train.loss)}</td><td>${pct(r.train.accuracy)}</td><td>${num(r.val.loss)}</td><td>${pct(r.val.accuracy)}</td><td>${pct(r.val.top5_accuracy)}</td><td>${pct(r.val.macro_f1)}</td><td>${pct(r.val.balanced_accuracy)}</td><td>${num(r.train.images_per_second,1)}</td><td>${duration(r.train.duration_seconds+r.val.duration_seconds)}</td></tr>`).join(''):'<tr><td colspan="11" class="empty">等待首轮指标</td></tr>';document.getElementById('updated').textContent='更新于 '+new Date().toLocaleTimeString()}
async function refresh(){try{const response=await fetch('/api/progress',{cache:'no-store'});if(!response.ok)throw new Error(response.status);render(await response.json())}catch(error){document.getElementById('status-text').textContent='连接中断'}}
addEventListener('resize',()=>refresh());refresh();setInterval(refresh,5000);
</script></body></html>"""


def make_handler(run_dir: Path) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/":
                body = PAGE.encode("utf-8")
                content_type = "text/html; charset=utf-8"
            elif route == "/api/progress":
                body = json.dumps(build_payload(run_dir), ensure_ascii=False).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return DashboardHandler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"run directory does not exist: {run_dir}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(run_dir))
    print(f"Training dashboard: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
