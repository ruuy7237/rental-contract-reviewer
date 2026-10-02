# -*- coding: utf-8 -*-
"""评测：在预设合同上跑审查，计算“红灯条款”召回率。

两套指标（和 sql-analysis-agent 同一思路）：
  - 规则基线（force_mock）：用内置正则红线检测，结果完全确定，作为稳定的“成绩”；
  - 真实模型（若填了 Key）：用 DeepSeek 做语义审查，作为对比。

召回率 = 被成功识别的预期红线数 / 预期红线总数。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.agent import review
from src import llm

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(EVAL_DIR, "samples")
EXPECTED = json.load(open(os.path.join(EVAL_DIR, "expected.json"), encoding="utf-8"))


def norm(s):
    return "".join(s.split())


def matched(text, sigs):
    t = norm(text)
    return [s for s in sigs if norm(s) in t]


def _findings_text(rep):
    return " ".join((f.get("category", "") + " " + f.get("clause", "")) for f in rep.get("findings", []))


def main():
    api = llm.mode() == "api"
    print("=" * 64)
    print("   租房合同审查 Agent · 评测")
    print("=" * 64)
    print(f"模式：规则基线(必跑)" + (" + 真实模型(DeepSeek)" if api else "  [未填 Key，跳过真实模型]"))
    print("-" * 64)
    print(f"{'合同':<26}{'预期':>4}{'基线命中':>8}{'基线召回':>8}" + ("{:>10}".format("真实召回") if api else ""))

    base_total = base_hit = 0
    real_total = real_hit = 0
    rows = []

    for fn, sigs in EXPECTED.items():
        text = open(os.path.join(SAMPLES, fn), encoding="utf-8").read()
        rep_base = review(text, force_mock=True)
        bset = matched(_findings_text(rep_base), sigs)
        if sigs:
            base_total += len(sigs)
            base_hit += len(bset)

        real_recall = ""
        if api:
            rep_real = review(text)
            rset = matched(_findings_text(rep_real), sigs)
            if sigs:
                real_total += len(sigs)
                real_hit += len(rset)
            real_recall = f"{len(rset) / len(sigs) * 100:>8.0f}%"

        base_recall = f"{len(bset) / len(sigs) * 100:>8.0f}%" if sigs else "  clean"
        print(f"{fn:<26}{len(sigs):>4}{len(bset):>8}{base_recall:>8}" + (f"{real_recall:>10}" if api else ""))
        rows.append({"file": fn, "expected": len(sigs), "baseline_hit": len(bset),
                     "baseline_recall": (len(bset) / len(sigs) if sigs else None)})

    print("-" * 64)
    base_recall_pct = (base_hit / base_total * 100) if base_total else 100.0
    print(f"规则基线整体召回率：{base_recall_pct:.1f}%  （{base_hit}/{base_total}）")
    if api and real_total:
        real_recall_pct = real_hit / real_total * 100
        print(f"真实模型整体召回率：{real_recall_pct:.1f}%  （{real_hit}/{real_total}）")
    print("=" * 64)

    # 落盘（已被 .gitignore 忽略，仅本地记录）
    out = {
        "baseline_recall": base_recall_pct,
        "real_recall": (real_hit / real_total * 100 if (api and real_total) else None),
        "rows": rows,
    }
    with open(os.path.join(EVAL_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
