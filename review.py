# -*- coding: utf-8 -*-
"""命令行入口：审查一份房屋租赁合同。

用法：
  python review.py                  # 演示模式：用内置示例合同
  python review.py 我的合同.txt      # 审查你自己的合同（.txt / 拷贝自 PDF 的文字）
  python review.py 合同.pdf.txt      # 也支持把 PDF 文字拷出来存成 .txt

没有 API Key 时自动用规则基线（mock）；填了 .env 的 Key 后用真实模型。
"""
import os
import sys

from src.agent import review

DEMO = """房屋租赁合同

甲方（房东）：王某
乙方（租客）：李某

一、租赁标的：北京市朝阳区某小区 1 室。
二、租赁期限：租期25年，自签约之日起算。
三、租金与押金：月租金 3000 元，付款方式押三付一。
四、房屋使用：租赁期内，甲方可随时进入房屋查看情况。
五、提前退租：无论何种原因乙方提前退租，押金不予退还。
六、维修责任：房屋及设施的一切维修费用均由乙方承担。
七、其他：乙方不得对合同条款提出任何异议；电费按 1.5 元/度收取。
"""


def read_contract(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def print_report(rep):
    line = "=" * 60
    print(line)
    print("           租 房 合 同 审 查 报 告")
    print(line)
    print("总评：", rep.get("summary", ""))
    print("整体风险：", rep.get("overall_risk", ""))
    print("-" * 60)
    fs = rep.get("findings", [])
    if not fs:
        print("未发现明显问题条款。仍建议保留证据、看清附加条件。")
    for i, f in enumerate(fs, 1):
        print(f"\n[{i}] 【{f.get('risk_level', '?')}】{f.get('category', '')}")
        print("  问题条款：", f.get("clause", ""))
        print("  法律依据：", f.get("legal_basis", ""))
        print("  修改建议：", f.get("suggestion", ""))
    print("\n" + rep.get("disclaimer", ""))
    print(line)
    print(f"[统计] 风险点 {len(fs)} 个 | 工具调用 {rep.get('tool_calls', 0)} 次 | 步数 {rep.get('steps', 0)}")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.exists(path):
            print("文件不存在：", path)
            sys.exit(1)
        text = read_contract(path)
        print(f"[已读取] {path}（{len(text)} 字）\n")
    else:
        text = DEMO
        print("[演示模式] 未指定合同文件，使用内置示例合同。\n")

    rep = review(text, verbose=(os.getenv("VERBOSE") == "1"))
    print_report(rep)


if __name__ == "__main__":
    main()
