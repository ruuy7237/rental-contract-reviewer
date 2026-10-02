# -*- coding: utf-8 -*-
"""租房合同审查 Agent —— 零依赖网页前端（标准库 http.server）。

启动：双击「双击我_网页版.bat」，浏览器会自动打开 http://127.0.0.1:8000
关闭那个黑色窗口即可停止服务。

设计：只用 Python 标准库，clone 下来 + 有 Python 就能跑，不装任何第三方包。
前端 index.html 也完全自包含（不依赖任何 CDN，国内网络也不会破图）。
"""
import os
import sys
import json
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # rental-contract-reviewer
WEB = os.path.join(ROOT, "web")
sys.path.insert(0, ROOT)

from src import agent, llm  # noqa: E402

PORT = int(os.getenv("PORT", "8000"))


def _read_body(req):
    length = int(req.headers.get("Content-Length", 0) or 0)
    raw = req.rfile.read(length) if length else b""
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            p = os.path.join(WEB, "index.html")
            try:
                html = open(p, encoding="utf-8").read()
            except Exception:
                self._send(404, {"error": "index.html 缺失"})
                return
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send(200, {"mode": llm.mode(), "model": llm.MODEL})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            data = _read_body(self)
            if self.path == "/api/review":
                text = (data.get("text") or "").strip()
                force_mock = bool(data.get("mock"))
                rep = agent.review(text, verbose=False, force_mock=force_mock)
                self._send(200, rep)
            elif self.path == "/api/chat":
                contract = data.get("contract") or ""
                question = data.get("question") or ""
                history = data.get("history") or []
                force_mock = bool(data.get("mock"))
                ans = agent.chat(contract, question, history, verbose=False, force_mock=force_mock)
                self._send(200, ans)
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            # 关键：出错也必须回包，否则浏览器只能看到 “Failed to fetch”，
            # 真正的原因（余额不足/限流/Key 错）会被藏起来。完整报错同时打印到启动窗口。
            traceback.print_exc()
            try:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
            except Exception:
                pass

    def log_message(self, *args):
        pass


def main():
    os.chdir(ROOT)  # 确保 .env（真 key）能被正确加载
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print("=" * 56)
    print("   租房合同审查 Agent（网页版）已启动")
    print("=" * 56)
    print(f"   请在浏览器打开： {url}")
    print("   关闭此窗口即可停止服务。")
    print("=" * 56)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
