"""DLCPD-25 + IP102 Plan-A two-model web application."""

from __future__ import annotations

import argparse
import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps

from dlcpd25_v2.common import repo_path
from dlcpd25_v2.serving.joint_predictor import JointPredictor

INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DLCPD-25 病虫害与缺陷识别系统</title>
<style>
:root { --bg:#0f172a; --card:#1e293b; --line:#334155; --text:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; --ok:#34d399; --warn:#fbbf24; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans CJK SC",sans-serif; }
header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; align-items:center; }
h1 { font-size:20px; margin:0; }
main { max-width:1180px; margin:20px auto; padding:0 16px; display:grid; gap:16px; }
.panel { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px; }
.upload { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.drop { border:2px dashed var(--line); border-radius:14px; padding:28px; text-align:center; color:var(--muted); }
.preview { display:flex; justify-content:center; align-items:center; min-height:180px; background:#0b1220; border-radius:12px; overflow:hidden; }
.preview img { max-width:100%; max-height:340px; object-fit:contain; }
button { background:var(--accent); color:#04121f; border:0; border-radius:10px; padding:11px 18px; font-weight:650; cursor:pointer; }
button:disabled { opacity:.5; cursor:not-allowed; }
.results { display:grid; grid-template-columns:1.1fr .9fr; gap:16px; }
h2 { font-size:16px; margin:0 0 10px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th,td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; }
th { color:var(--muted); font-weight:600; }
.score { color:var(--ok); font-weight:650; }
.muted { color:var(--muted); font-size:12px; }
.tag { display:inline-block; padding:2px 8px; border-radius:999px; background:rgba(56,189,248,.15); color:var(--accent); font-size:12px; }
@media(max-width:900px){ .upload,.results{grid-template-columns:1fr;} }
</style>
</head>
<body>
<header><h1>DLCPD-25 农产品病虫害与缺陷识别</h1></header>
<main>
  <section class="panel">
    <h2>上传图片</h2>
    <div class="upload">
      <label class="drop" for="file">点击选择图片，或拖拽到此处<input id="file" type="file" accept="image/*" hidden></label>
      <div class="preview"><img id="preview" alt="预览"></div>
    </div>
    <div style="display:flex;gap:10px;align-items:center;margin-top:14px;">
      <button id="run">开始分析</button>
      <span id="status" class="muted">请选择图片</span>
    </div>
  </section>
  <section class="results" id="results" style="display:none;">
    <div class="panel">
      <h2>检测与标注</h2>
      <img id="annotated" style="width:100%;border-radius:10px;" alt="标注结果">
      <p id="detectCount" class="muted"></p>
    </div>
    <div class="panel">
      <h2>整图分类 Top-5</h2>
      <table><thead><tr><th>类别</th><th>宿主</th><th>属性</th><th>概率</th></tr></thead><tbody id="clsBody"></tbody></table>
      <h2 style="margin-top:16px;">害虫检测结果</h2>
      <table><thead><tr><th>害虫</th><th>宿主</th><th>置信度</th></tr></thead><tbody id="detBody"></tbody></table>
    </div>
  </section>
</main>
<script>
const fileInput=document.getElementById('file'), preview=document.getElementById('preview'), run=document.getElementById('run'), status=document.getElementById('status');
let file=null;
fileInput.addEventListener('change',e=>{file=e.target.files[0]||null; status.textContent=file?file.name:'请选择图片'; if(file) preview.src=URL.createObjectURL(file);});
run.addEventListener('click',async()=>{
  if(!file){status.textContent='请先选择图片';return;}
  run.disabled=true; status.textContent='分析中…';
  const fd=new FormData(); fd.append('file',file);
  try{
    const res=await fetch('/api/analyze',{method:'POST',body:fd});
    const data=await res.json();
    if(!res.ok) throw new Error(data.detail||'分析失败');
    document.getElementById('results').style.display='grid';
    document.getElementById('annotated').src=data.annotated_image;
    document.getElementById('detectCount').textContent='检测到 '+data.detection.count+' 个害虫目标';
    document.getElementById('clsBody').innerHTML=data.classification.top5.map(x=>'<tr><td>'+x.name+'</td><td>'+x.host_zh+'</td><td>'+x.category_zh+'</td><td class="score">'+(x.probability*100).toFixed(2)+'%</td></tr>').join('');
    document.getElementById('detBody').innerHTML=data.detection.detections.length?data.detection.detections.map(d=>'<tr><td>'+d.ip102_name+'</td><td>'+d.host_zh+'</td><td class="score">'+(d.score*100).toFixed(2)+'%</td></tr>').join('') : '<tr><td colspan="3" class="muted">未检测到害虫目标</td></tr>';
    status.textContent='分析完成';
  }catch(e){status.textContent='错误：'+e.message;}
  finally{run.disabled=false;}
});
</script>
</body>
</html>
"""


def _load_image_bytes(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法解析图片：{exc}") from exc


def create_app(config: dict[str, Any]) -> FastAPI:
    predictor: JointPredictor | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal predictor
        predictor = JointPredictor(
            classification_checkpoint=repo_path(config["classification_checkpoint"]),
            detection_checkpoint=repo_path(config["detection_checkpoint"]),
            classification_taxonomy=repo_path(config["classification_taxonomy"]),
            detection_class_map=repo_path(config["detection_class_map"]),
            detection_score_threshold=float(config.get("detection_score_threshold", 0.30)),
            detection_max_detections=int(config.get("detection_max_detections", 30)),
            annotated_max_side=int(config.get("annotated_max_side", 1200)),
        )
        yield
        predictor = None

    app = FastAPI(title="DLCPD-25 + IP102 Joint Inference", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/health")
    def health() -> dict[str, Any]:
        if predictor is None:
            return {"status": "loading"}
        return {"status": "ok", **predictor.model_info}

    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
        if predictor is None:
            raise HTTPException(status_code=503, detail="模型尚未加载完成")
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="上传文件为空")
        if len(data) > 30 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="图片不能超过 30MB")
        image = _load_image_bytes(data)
        classification = predictor.classify(image, top_k=int(config.get("classification_top_k", 5)))
        detections = predictor.detect(image)
        annotated = predictor.annotate(image, detections)
        buffer = io.BytesIO()
        annotated.save(buffer, format="JPEG", quality=90)
        annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "filename": file.filename,
            "width": image.width,
            "height": image.height,
            "classification": classification,
            "detection": {"count": len(detections), "detections": detections},
            "annotated_image": annotated_base64,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/plan-a/app.yaml"))
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    import yaml

    path = repo_path(args.config)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    host = args.host or config.get("host", "0.0.0.0")
    port = args.port or int(config.get("port", 7860))
    uvicorn.run(create_app(config), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
