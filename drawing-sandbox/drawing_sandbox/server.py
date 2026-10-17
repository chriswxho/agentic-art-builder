"""A constrained drawing tool with local editor, recorder, and live viewer."""
from __future__ import annotations

import base64
from dataclasses import asdict
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

from .policy import composition_constraints, current_instruction_set
from .recording import record_composition_step
from .report import summarize_render
from .renderer import check_scene, render, to_svg


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_RECORDINGS = Path(__file__).resolve().parent.parent / "recordings"


def render_drawing(scene, constraints=None):
    """The agent-facing tool, with the CMYK small-mark policy enforced."""
    checks = check_scene(scene, composition_constraints())
    if checks.valid and constraints is not None:
        requested = check_scene(scene, constraints)
        checks.hard_checks.update(requested.hard_checks)
        checks.metrics.update(requested.metrics)
        checks.violations.extend(requested.violations)
        checks.valid = checks.valid and requested.valid
    result = {"checks":asdict(checks)}
    result["summary"]=summarize_render(scene,result["checks"])
    if checks.valid:
        image=render(scene)
        buffer=io.BytesIO()
        image.save(buffer,format="PNG")
        result.update(svg=to_svg(scene),png_data_url="data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode())
    return result


class Handler(BaseHTTPRequestHandler):
    server_version="DrawingSandbox/0.1"

    def log_message(self,*args):
        pass

    def reply(self,status,body,mime="application/json; charset=utf-8"):
        if isinstance(body,dict):
            body=json.dumps(body,allow_nan=False).encode()
        elif isinstance(body,str):
            body=body.encode()
        self.send_response(status)
        self.send_header("Content-Type",mime)
        self.send_header("Content-Length",str(len(body)))
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):
            pass

    def local_request(self):
        hosts={"127.0.0.1:"+str(self.server.server_port),"localhost:"+str(self.server.server_port)}
        origin=self.headers.get("Origin")
        return self.headers.get("Host") in hosts and (origin is None or origin in {"http://"+host for host in hosts})

    def watch_run(self, query):
        values=parse_qs(query).get("run",[])
        if not values:
            runs=[]
            if _RECORDINGS.is_dir():
                runs=sorted(path.name for path in _RECORDINGS.iterdir()
                            if path.is_dir() and _RUN_ID.fullmatch(path.name)
                            and (path/"manifest.json").is_file())
            return self.reply(200,{"runs":runs})
        run_id=values[0]
        if not _RUN_ID.fullmatch(run_id):
            return self.reply(400,{"error":"Invalid run ID"})
        manifest_path=_RECORDINGS/run_id/"manifest.json"
        if not manifest_path.is_file():
            return self.reply(404,{"error":"Recording not found"})
        manifest=json.loads(manifest_path.read_text())
        steps=[]
        for step in manifest.get("steps",[]):
            provenance=step.get("provenance",{})
            steps.append({
                "index":step.get("index"),
                "label":step.get("label"),
                "recorded_at":step.get("recorded_at"),
                "final":step.get("final",False),
                "metrics":step.get("metrics",{}),
                "agent_id":provenance.get("agent_id"),
                "skill":provenance.get("skill"),
                "context_revision":provenance.get("context_revision",step.get("index")),
            })
        return self.reply(200,{
            "run_id":run_id,
            "title":manifest.get("composition",{}).get("title",run_id),
            "step_count":len(steps),
            "final_step":manifest.get("final_step"),
            "updated_at":manifest.get("updated_at"),
            "steps":steps,
        })

    def watch_frame(self, query):
        values=parse_qs(query)
        run_id=(values.get("run") or [""])[0]
        step_value=(values.get("step") or [""])[0]
        if not _RUN_ID.fullmatch(run_id):
            return self.reply(400,{"error":"Invalid run ID"})
        try:
            step_index=int(step_value)
        except ValueError:
            return self.reply(400,{"error":"Step must be an integer"})
        run_dir=(_RECORDINGS/run_id).resolve()
        manifest_path=run_dir/"manifest.json"
        if not manifest_path.is_file():
            return self.reply(404,{"error":"Recording not found"})
        manifest=json.loads(manifest_path.read_text())
        steps=manifest.get("steps",[])
        if not 0<=step_index<len(steps):
            return self.reply(404,{"error":"Step not found"})
        image_path=(run_dir/steps[step_index]["png"]).resolve()
        try:
            image_path.relative_to(run_dir)
        except ValueError:
            return self.reply(403,{"error":"Invalid frame path"})
        if not image_path.is_file():
            return self.reply(404,{"error":"Frame not found"})
        return self.reply(200,image_path.read_bytes(),"image/png")

    def do_GET(self):
        if not self.local_request():
            return self.reply(403,{"error":"Only local requests are accepted"})
        request=urlsplit(self.path)
        path=request.path
        if path=="/":
            return self.reply(200,(Path(__file__).parent/"web"/"index.html").read_bytes(),"text/html; charset=utf-8")
        if path=="/watch":
            return self.reply(200,(Path(__file__).parent/"web"/"watch.html").read_bytes(),"text/html; charset=utf-8")
        if path=="/api/examples":
            from .examples import examples
            return self.reply(200,{"examples":examples()})
        if path=="/api/dataset":
            from .dataset import task_card, tasks
            return self.reply(200,{"tasks":[task_card(task) for task in tasks()]})
        if path=="/api/instruction-set":
            return self.reply(200,{"instruction_set":current_instruction_set()})
        if path=="/api/watch":
            return self.watch_run(request.query)
        if path=="/api/watch/frame":
            return self.watch_frame(request.query)
        return self.reply(404,{"error":"Not found"})

    def do_POST(self):
        if not self.local_request():
            return self.reply(403,{"error":"Only local requests are accepted"})
        path=urlsplit(self.path).path
        if path not in {"/api/render","/api/record"}:
            return self.reply(404,{"error":"Not found"})
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0<=length<=2_000_000:
                return self.reply(413,{"error":"Drawing JSON is too large"})
            if self.headers.get("Content-Type","").split(";")[0]!="application/json":
                return self.reply(415,{"error":"Use application/json"})
            body=json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body,dict) or "scene" not in body:
                return self.reply(400,{"error":"Expected a JSON object with a scene field"})
            result=render_drawing(body["scene"],body.get("constraints"))
            if path=="/api/record" and result["checks"]["valid"]:
                result["recording"]=record_composition_step(
                    body["scene"],result,
                    run_id=body.get("run_id"),
                    prompt_set=body.get("prompt_set"),
                    composition=body.get("composition"),
                    step_label=body.get("step_label"),
                    final=body.get("final",False),
                    model_output=body.get("model_output"),
                )
            return self.reply(200,result)
        except (ValueError,TypeError) as error:
            return self.reply(400,{"error":str(error)})


def serve(port=8766,open_browser=True):
    server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    url="http://127.0.0.1:"+str(server.server_port)
    print("Drawing Sandbox · "+url,flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
