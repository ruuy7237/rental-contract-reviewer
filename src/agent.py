# -*- coding: utf-8 -*-
"""租房合同审查 Agent —— 一个真正会“检索 + 判断”的工具调用 Agent。

和作品集里 task-agent 同一个思路：不依赖任何 agent 框架，手写 plan -> 调工具 -> 校验 -> 收尾。
不同点在于这里只有两个工具，且任务目标非常明确：把合同里的“坑”一条条揪出来并附法律依据。

两个工具：
  - search_law(query)        ：检索知识库，给每条风险找法律依据（RAG 落地）
  - list_redflag_types()     ：列出本工具关注的“红灯”风险清单，避免漏查

护栏（面试能讲的点）：
  1) 每一步的法律依据都必须来自知识库检索，禁止模型凭空编法条；
  2) 输出强制带“非法律意见”免责声明；
  3) 步数护栏 max_steps 防止烧 token / 死循环；
  4) 没有 API Key 时自动降级为规则基线（_mock_review），保证流程可跑、可评测。
"""
import json
import os
import re

from . import llm
from .knowledge_base import KnowledgeBase

DISCLAIMER = (
    "⚠️ 本结果由 AI 基于通用法律常识自动生成，仅供信息参考，不构成法律意见，"
    "也不替代专业律师的判断。涉及重大权益请以正式法律意见为准。"
)

# --------------------------------------------------------------------------
# 红灯风险清单：mock 规则基线 与 真实 Agent 的 category 都从这里统一取值
# --------------------------------------------------------------------------
RED_FLAGS = [
    dict(id="deposit_no_refund", name="提前退租/违约不退押金", severity="高",
         pattern=r"不退.{0,6}押金|押金.{0,6}不(予|退)|无论.{0,12}押金|押金.{0,6}没收|押金.{0,6}概不",
         basis="民法典第496/497条（格式条款）：‘任何情况不退押金’可能剥夺租客解除权而无效；押金应依约退还。",
         suggestion="改为‘租赁期满、无违约且无损坏时，押金于 X 日内全额无息退还’。" ),
    dict(id="landlord_entry", name="房东可随时进入", severity="中",
         pattern=r"随时(进入|查看|上门)|可随时进入|房东.{0,8}随时|有权.{0,6}随时进入",
         basis="承租人享有对房屋的安静使用权与隐私，房东不得擅自进入。",
         suggestion="改为‘因维修等必要事由，须提前通知并征得承租人同意后方可进入’。" ),
    dict(id="repair_all_tenant", name="维修费全由租客承担", severity="高",
         pattern=r"(一切|所有|全部|任何).{0,6}维修|维修.{0,6}(一切|全部|均?由租客)|租客.{0,6}承担.{0,6}维修|维修费用.{0,6}由租客",
         basis="民法典第712条：出租人负有维修义务，不得通过条款完全排除。",
         suggestion="改为‘房屋固有/结构性维修由出租人负责，承租人使用损耗由承租人负责’。" ),
    dict(id="unequal_penalty", name="违约金不对等", severity="中",
         pattern=r"房东违约.{0,12}(不|无需|仅).{0,8}(赔偿|承担)|违约.{0,12}仅退(还?|租金)|租客违约.{0,12}扣(除|全)|仅退还租金",
         basis="违约金应双方对等，单向加重租客责任的条款可能不公平。",
         suggestion="补充‘出租人违约时承担对等赔偿责任’的条款。" ),
    dict(id="utility_markup", name="水电等费用加价", severity="中",
         pattern=r"电费.{0,6}[1-9]\.\d|水费.{0,6}加|加价|(1\.\d|2)\s*元/度|电费按.{0,4}[1-9]",
         basis="水电气按政府定价或实耗结算，出租人单方加价不合理。",
         suggestion="改为‘按政府定价/实耗结算，凭票据分摊’。" ),
    dict(id="no_objection", name="不得提出异议/放弃权利", severity="中",
         pattern=r"不得.{0,12}异议|不得提出.{0,6}(异议|意见)|放弃.{0,6}权利|不得.{0,6}主张",
         basis="民法典第497条：排除对方主要权利的格式条款可能无效。",
         suggestion="删除该条款，保留双方协商与异议、救济的权利。" ),
    dict(id="pet_visitor_ban", name="过度限制访客/宠物", severity="低",
         pattern=r"禁止.{0,6}访客|不得.{0,6}过夜|严禁.{0,6}宠物|任何访客|一律不得.{0,4}宠物",
         basis="合理约定居住人数/宠物可允许，但过度限制可能不合理且违背公序良俗。",
         suggestion="改为合理约定（如‘不得长期借住/转租’），而非一概禁止。" ),
    dict(id="early_takeback", name="房东提前收房无赔偿", severity="高",
         pattern=r"提前(收回|收房|终止)|无(需|理由).{0,6}收回|单方.{0,6}收回",
         basis="出租人无合法理由提前收回房屋，应承担违约责任并赔偿租客损失。",
         suggestion="补充‘提前收回的，退还剩余租金并赔偿搬家费、租金差价等损失’。" ),
]

RED_FLAG_BY_ID = {r["id"]: r for r in RED_FLAGS}


# --------------------------------------------------------------------------
# 工具定义（OpenAI 兼容的 function calling 格式）
# --------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_law",
            "description": "检索租房相关法律常识与霸王条款知识库，返回与查询最相关的法律要点（带来源文件）。用于对疑似问题条款找法律依据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要查证的法律问题，例如‘押金不退是否合法’、‘维修费谁承担’"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_redflag_types",
            "description": "列出本工具关注的所有‘红灯’风险类别清单（id + 名称 + 严重程度），帮助系统化排查合同，避免漏查。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


SYSTEM_PROMPT = """你是一位严谨的租房合同审查助手，不是聊天机器人。

你的工作流程：
1. 先阅读整份合同，识别哪些条款可能对承租人不公平或存在法律风险；
2. 对每一条疑似问题条款，调用 search_law 检索知识库，用真实法律要点作为依据，禁止凭空编造法条；
3. 需要时调用 list_redflag_types 对照红灯清单，确保覆盖主要风险类别；
4. 最后只输出一个 ```json 代码块（不要多余文字），结构如下：

{
  "summary": "一句话总评：这份合同整体对租客是否友好",
  "overall_risk": "高 / 中 / 低",
  "findings": [
    {
      "clause": "问题条款的原文摘录（尽量照抄）",
      "risk_level": "高 / 中 / 低",
      "category": "风险类别，必须使用 list_redflag_types 返回的中文名称（如‘房东可随时进入’），禁止使用英文 id",
      "legal_basis": "检索到的法律依据要点",
      "suggestion": "给租客的具体修改/应对建议"
    }
  ]
}

规则：
- findings 只写确实有问题的条款；没有问题的合同 findings 可以为空数组。
- 必须基于 search_law 的返回来写 legal_basis，不要杜撰法条编号。
- category 字段必须使用 list_redflag_types 返回的中文名称（如“房东可随时进入”），禁止使用英文 id。
- 若文本显然不是房屋租赁合同，请在 summary 中说明，findings 置空。
- 始终牢记：你提供的是信息参考，不是正式法律意见。"""


# --------------------------------------------------------------------------
# 工具执行
# --------------------------------------------------------------------------
def _execute_tool(name, args, kb):
    if name == "search_law":
        q = (args or {}).get("query", "")
        return kb.as_tool_result(q, topk=3)
    if name == "list_redflag_types":
        lines = [f"{r['id']} | {r['name']} | 严重程度：{r['severity']}" for r in RED_FLAGS]
        return "红灯风险清单：\n" + "\n".join(lines)
    return f"[未知工具] {name}"


def _normalize_call(call):
    fn = call.get("function", {})
    args = fn.get("arguments", "{}")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return None, "arguments 不是合法 JSON"
    return {"id": call.get("id") or "local", "name": fn.get("name"), "args": args}, None


# --------------------------------------------------------------------------
# 真实模型模式：工具调用循环（沿用 task-agent loop 的 400 修复：回填真实 tool_calls）
# --------------------------------------------------------------------------
def _run_real(contract_text, kb, verbose=False, max_steps=14):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请审查以下房屋租赁合同：\n\n<<<<合同开始>>>>\n{contract_text}\n<<<<合同结束>>>>"},
    ]
    steps, tool_calls, tool_errors = 0, 0, 0

    for step in range(1, max_steps + 1):
        steps = step
        msg = llm.chat(messages, TOOLS)
        calls = msg.get("tool_calls") or []

        if calls:
            assistant_msg = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}
            tool_messages = []
            for raw in calls:
                call, err = _normalize_call(raw)
                tool_calls += 1
                if err:
                    tool_errors += 1
                    tool_messages.append({
                        "role": "tool", "tool_call_id": raw.get("id") or "local",
                        "name": raw.get("function", {}).get("name", ""),
                        "content": f"[参数错误] {err}。请修正参数后重新调用。",
                    })
                    continue
                out = _execute_tool(call["name"], call["args"], kb)
                tool_messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": out})
                if verbose:
                    print(f"  [{step}] ✓ {call['name']}({str(call['args'])[:40]})")
            messages.append(assistant_msg)
            messages.extend(tool_messages)
            continue

        # 收尾：解析 JSON 报告
        answer = (msg.get("content") or "").strip()
        report = _parse_report(answer)
        report["steps"] = steps
        report["tool_calls"] = tool_calls
        report["tool_errors"] = tool_errors
        return report

    # 超过步数仍未产出结构化报告
    return {
        "summary": "审查未完成（超过最大步数）。",
        "overall_risk": "未知",
        "findings": [],
        "steps": steps, "tool_calls": tool_calls, "tool_errors": tool_errors,
        "disclaimer": DISCLAIMER,
    }


def _parse_report(text):
    # 优先提取 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        # 退而求其次：抓第一个 {...}
        m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"summary": (text[:200] or "（模型未返回结构化报告）"), "overall_risk": "未知", "findings": []}
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {"summary": (text[:200] or "（报告解析失败）"), "overall_risk": "未知", "findings": []}
    findings = []
    for f in data.get("findings", []) or []:
        findings.append({
            "clause": str(f.get("clause", "")).strip(),
            "risk_level": str(f.get("risk_level", "中")).strip(),
            "category": str(f.get("category", "其他风险")).strip(),
            "legal_basis": str(f.get("legal_basis", "")).strip(),
            "suggestion": str(f.get("suggestion", "")).strip(),
        })
    return {
        "summary": str(data.get("summary", "")).strip(),
        "overall_risk": str(data.get("overall_risk", "未知")).strip(),
        "findings": findings,
        "disclaimer": DISCLAIMER,
    }


# --------------------------------------------------------------------------
# mock 模式：规则基线。基于正则红线 + 几项特殊检查，产出结构化报告。
# 目的：没有 Key 时也能跑通整条流程，并作为评测的“规则基线”对照。
# --------------------------------------------------------------------------
def _extract_clause(text, start, end, span=50):
    a = max(0, start - span // 2)
    b = min(len(text), end + span // 2)
    snippet = text[a:b].replace("\n", " ").strip()
    return snippet


def _mock_review(contract_text):
    findings = []
    for rf in RED_FLAGS:
        for m in re.finditer(rf["pattern"], contract_text):
            clause = _extract_clause(contract_text, m.start(), m.end())
            findings.append({
                "id": rf["id"],
                "clause": clause,
                "risk_level": rf["severity"],
                "category": rf["name"],
                "legal_basis": rf["basis"],
                "suggestion": rf["suggestion"],
            })
            break  # 同一类别只报一次，避免刷屏

    # 特殊检查 1：租期超过 20 年
    for m in re.finditer(r"(?:租期|租赁期限|租赁期).{0,14}?(\d{1,2})\s*年", contract_text):
        yrs = int(m.group(1))
        if yrs > 20:
            findings.append({
                "id": "lease_over_20",
                "clause": _extract_clause(contract_text, m.start(), m.end()),
                "risk_level": "高",
                "category": "租期超过20年",
                "legal_basis": "民法典第705条：租赁期限不得超过二十年，超过部分无效。",
                "suggestion": "将租期改为 20 年以内；超出部分依法无效，请勿签署超长租期。",
            })
            break

    # 特殊检查 2：押金过高（押 N 付 M，N>=3 视为偏高；兼容中文数字）
    _CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    for m in re.finditer(r"押\s*([0-9一二三四五六七八九])\s*付\s*[0-9一二三四五六七八九]", contract_text):
        n = m.group(1)
        n = int(n) if n.isdigit() else _CN.get(n, 0)
        if n >= 3:
            findings.append({
                "id": "deposit_high",
                "clause": _extract_clause(contract_text, m.start(), m.end()),
                "risk_level": "中",
                "category": "押金过高",
                "legal_basis": "法律未规定押金上限，但明显超出合理范围（如超过两月租金）可能被认定不公平。",
                "suggestion": "协商将押金降至不超过两个月租金，并明确退还条件。",
            })
            break

    highs = sum(1 for f in findings if f["risk_level"] == "高")
    mids = sum(1 for f in findings if f["risk_level"] == "中")
    if highs >= 1:
        overall = "高"
    elif mids >= 1:
        overall = "中"
    elif findings:
        overall = "低"
    else:
        overall = "低"

    summary = (
        f"共发现 {len(findings)} 处风险点，整体风险等级【{overall}】。"
        + ("建议签约前与房东协商修改相关问题条款。" if findings else "未发现明显问题条款，但仍建议保留证据、看清附加条件。")
    )
    return {
        "summary": summary,
        "overall_risk": overall,
        "findings": findings,
        "steps": 1, "tool_calls": 0, "tool_errors": 0,
        "disclaimer": DISCLAIMER,
    }


# --------------------------------------------------------------------------
# 对外入口
# --------------------------------------------------------------------------
def review(contract_text, verbose=False, force_mock=False):
    """审查一份合同文本，返回结构化报告 dict。

    force_mock=True 时强制走规则基线（评测脚本用它算基线，不受 Key 有无影响）。
    """
    contract_text = (contract_text or "").strip()
    if not contract_text:
        return {"summary": "合同内容为空。", "overall_risk": "未知", "findings": [],
                "steps": 0, "tool_calls": 0, "tool_errors": 0, "disclaimer": DISCLAIMER}

    if force_mock or llm.mode() == "mock":
        return _mock_review(contract_text)

    kb = KnowledgeBase()
    if verbose:
        print("[模式] 真实模型（DeepSeek）")
    return _run_real(contract_text, kb, verbose=verbose)


# --------------------------------------------------------------------------
# 网页聊天模式：就一份合同进行一问一答（前端会带着合同全文来问）
# --------------------------------------------------------------------------
CHAT_SYSTEM_PROMPT = """你是一位严谨、口语化、对普通人友好的租房合同审查助手，正在和用户就一份具体的租房合同对话。
用户会针对合同里的具体条款提问，例如：
- “这条押金不退的条款合法吗？”
- “帮我改一下违约金条款”
- “房东可以随时进屋吗？”
- “这份合同整体有什么坑？”

你可以使用的工具：
- search_law(query)：检索租房法律常识与霸王条款知识库，返回最相关的法律要点（带来源文件）。必须用它为法律判断找依据，禁止凭空编造法条编号。
- list_redflag_types()：查看我们关注的所有红灯风险类别清单，帮助系统化排查。

回答要求：
1. 用中文、像朋友讲解一样口语化，但法律要点要准确、可操作；
2. 引用用户合同的具体条款时，尽量照抄原文句子；
3. 法律判断必须基于 search_law 的返回，不得杜撰法条；
4. 给出的建议要具体到“改成什么 wording”或“下一步做什么”；
5. 若用户的问题在合同里找不到对应条款，如实说明“合同里没写这句”，不要硬编；
6. 始终牢记：你提供的是信息参考，不是正式法律意见，重大权益请以律师意见为准。"""


# 离线模式下，用关键词把用户问题映射到红灯类别，从合同红线里挑相关内容回答
_QA_KEYWORDS = [
    ("押金", {"deposit_no_refund", "early_takeback"}),
    ("退租", {"deposit_no_refund", "early_takeback"}),
    ("收房", {"early_takeback"}),
    ("违约", {"unequal_penalty"}),
    ("赔偿", {"unequal_penalty", "early_takeback"}),
    ("维修", {"repair_all_tenant"}),
    ("进屋", {"landlord_entry"}),
    ("进入", {"landlord_entry"}),
    ("看房", {"landlord_entry"}),
    ("水电", {"utility_markup"}),
    ("电费", {"utility_markup"}),
    ("宠物", {"pet_visitor_ban"}),
    ("访客", {"pet_visitor_ban"}),
    ("异议", {"no_objection"}),
    ("权利", {"no_objection"}),
    ("租期", {"lease_over_20"}),
]


def _mock_chat_answer(contract_text, question):
    rep = _mock_review(contract_text)
    wanted = set()
    for kw, ids in _QA_KEYWORDS:
        if kw in question:
            wanted |= ids
    related = [f for f in rep["findings"] if f.get("id") in wanted] if wanted else []
    parts = ["（离线规则模式：未调用大模型，以下基于内置红线规则与知识库检索给出参考；仅供信息参考，不构成法律意见）"]
    if related:
        for f in related:
            parts.append(
                f"【{f['risk_level']}风险 · {f['category']}】\n"
                f"相关条款：{f['clause']}\n"
                f"依据：{f['legal_basis']}\n"
                f"建议：{f['suggestion']}"
            )
    else:
        kb = KnowledgeBase()
        res = kb.as_tool_result(question, topk=2)
        parts.append("未从合同中匹配到明确相关风险条款。知识库补充：\n" + res)
        if rep["findings"]:
            parts.append("不过这份合同里我们发现了以下风险点，供你参考：")
            for f in rep["findings"][:3]:
                parts.append(f"· 【{f['risk_level']}】{f['category']}：{f['clause']}")
        else:
            parts.append("合同里也未发现明显红线风险。可以换一种问法，或贴入更完整的合同文本再问。")
    return "\n\n".join(parts)


def _run_real_chat(contract_text, question, history, kb, verbose=False, max_steps=10):
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    if history:
        for h in history[-8:]:
            if isinstance(h, dict) and h.get("role") in ("user", "assistant"):
                messages.append({"role": h["role"], "content": str(h.get("content", ""))})
    messages.append({
        "role": "user",
        "content": (
            "这是用户上传的租房合同全文：\n"
            f"<<<<合同开始>>>>\n{contract_text}\n<<<<合同结束>>>>\n\n"
            f"用户的问题：{question}"
        ),
    })
    steps = 0
    for step in range(1, max_steps + 1):
        steps = step
        msg = llm.chat(messages, TOOLS)
        calls = msg.get("tool_calls") or []
        if calls:
            assistant_msg = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}
            tool_messages = []
            for raw in calls:
                call, err = _normalize_call(raw)
                if err:
                    tool_messages.append({
                        "role": "tool", "tool_call_id": raw.get("id") or "local",
                        "name": raw.get("function", {}).get("name", ""),
                        "content": f"[参数错误] {err}",
                    })
                    continue
                out = _execute_tool(call["name"], call["args"], kb)
                tool_messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": out})
                if verbose:
                    print(f"  [{step}] tool {call['name']}")
            messages.append(assistant_msg)
            messages.extend(tool_messages)
            continue
        answer = (msg.get("content") or "").strip()
        return {"answer": answer, "steps": steps}
    return {"answer": "（超过最大步数，未能生成回答）", "steps": steps}


def chat(contract_text, question, history=None, verbose=False, force_mock=False):
    """就一份合同进行一问一答（网页聊天用）。返回 {'answer': ..., 'steps': ...}。"""
    question = (question or "").strip()
    if not question:
        return {"answer": "请先输入你的问题，例如：这条押金不退的条款合法吗？", "steps": 0}
    if force_mock or llm.mode() == "mock":
        return {"answer": _mock_chat_answer(contract_text or "", question), "steps": 0, "mock": True}
    kb = KnowledgeBase()
    if verbose:
        print("[模式] 真实模型（DeepSeek）聊天")
    return _run_real_chat(contract_text or "", question, history, kb, verbose=verbose)


if __name__ == "__main__":
    sample = "租赁期内房东可随时进入房屋。无论何种原因提前退租，押金不予退还。房屋一切维修费用由租客承担。租期25年。"
    rep = review(sample, verbose=True)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
