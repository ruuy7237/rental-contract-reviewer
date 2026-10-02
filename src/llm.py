"""统一 LLM 客户端（与作品集其它项目同源，零第三方依赖）。

设计取舍：
- 只用标准库 urllib，clone 下来就能跑；
- 任何 OpenAI 兼容接口都能用（DeepSeek / 通义 / Kimi / OpenAI），改环境变量即可；
- 没有 API Key 时自动降级为 mock 模式，用来验证 agent loop 与评测流水线本身是否正确。

环境变量：
  LLM_BASE_URL  默认 https://api.deepseek.com/v1
  LLM_API_KEY   有值 -> 真实模型；为空 -> mock 模式
  LLM_MODEL     默认 deepseek-chat
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid


def _load_env():
    """可选：从项目根目录的 .env 读取 API Key，免得配系统环境变量。"""
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for cand in (os.path.join(root, ".env"), os.path.join(os.getcwd(), ".env")):
            if os.path.exists(cand):
                for line in open(cand, encoding="utf-8"):
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
                break
    except Exception:
        pass


_load_env()

BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")


def mode() -> str:
    return "mock" if not API_KEY else "api"


def chat(messages, tools=None, tool_choice="auto", temperature=0.0, retries=3):
    """返回标准化 assistant message dict。

    {"role": "assistant", "content": str|None, "tool_calls": [{"id","name","arguments"}]}
    """
    if mode() == "mock":
        return _mock_chat(messages, tools)

    url = BASE_URL.rstrip("/") + "/chat/completions"
    payload = {"model": MODEL, "messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    last_err = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "ignore")[:200]
            except Exception:
                pass
            hint = {
                401: "API Key 无效或未授权",
                402: "账户余额不足，请到 DeepSeek 平台充值后再试",
                429: "触发限流，请等几十秒再试",
                400: "请求参数被拒绝",
            }.get(e.code, "")
            last_err = f"HTTP {e.code} {hint}（{body or e.reason}）"
            # 429 限流：等久一点再试（连续审查时最容易触发）
            time.sleep(6.0 * (attempt + 1) if e.code == 429 else 2.0 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM 调用失败（已重试 {retries} 次）：{last_err}")


# --------------------------------------------------------------------------
# mock 模式：本项目不在这里实现合同审查逻辑（审查逻辑在 agent.py 的 _mock_review），
# 这里只负责让 tool-calling 的“收尾”这一步在没有 Key 时也能走通。
# --------------------------------------------------------------------------

def _last_user(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _tool_call(name, args):
    return {
        "id": "call_" + uuid.uuid4().hex[:8],
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _mock_chat(messages, tools=None):
    # 已经拿到工具结果 -> 收尾总结。保证 loop 能终止。
    if messages and messages[-1].get("role") == "tool":
        obs = [m.get("content", "") for m in messages if m.get("role") == "tool"]
        tail = " | ".join(o[:200] for o in obs[-3:])
        return {"role": "assistant", "content": f"（依据工具返回）{tail}", "tool_calls": []}

    text = _last_user(messages)
    # 默认：调用 list_redflag_types 了解红灯清单，再开始审查。
    if any(t.get("function", {}).get("name") for t in (tools or [])):
        return {"role": "assistant", "content": None,
                "tool_calls": [_tool_call("list_redflag_types", {})]}
    return {"role": "assistant",
            "content": f"[mock] 未启用工具，无法审查：{text[:80]}", "tool_calls": []}
