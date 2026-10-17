"""A stateless drawing tool and a small localhost editor."""
from __future__ import annotations

import base64
from dataclasses import asdict
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .policy import composition_constraints, current_instruction_set
from .recording import record_composition_step
from .report import summarize_render
from .renderer import check_scene, render, to_svg


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

    def do_GET(self):
        if not self.local_request():
            return self.reply(403,{"error":"Only local requests are accepted"})
        path=urlsplit(self.path).path
        if path=="/":
            return self.reply(200,(Path(__file__).parent/"web"/"index.html").read_bytes(),"text/html; charset=utf-8")
        if path=="/api/examples":
            from .examples import examples
            return self.reply(200,{"examples":examples()})
        if path=="/api/instruction-set":
            return self.reply(200,{"instruction_set":current_instruction_set()})
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
