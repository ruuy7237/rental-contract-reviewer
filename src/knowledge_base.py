# -*- coding: utf-8 -*-
"""知识库检索层：把 knowledge/ 下的法律常识建成 BM25 索引。

合同审查 agent 通过 search_law(query) 在这里检索与某条条款相关的法律依据，
保证每一条风险提示都“有据可依”，而不是模型凭空编。
"""
import glob
import math
import os
import re
from collections import Counter

_KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge")

_TOKEN = re.compile(r"[a-zA-Z]+|\d+|[\u4e00-\u9fff]")
K1, B = 1.5, 0.75


def _tokenize(text):
    toks = [t.lower() for t in _TOKEN.findall(text or "")]
    han = re.findall(r"[\u4e00-\u9fff]", text or "")
    toks += [han[i] + han[i + 1] for i in range(len(han) - 1)]
    return toks


class BM25:
    def __init__(self, chunks):
        self.chunks = chunks
        self.docs = [_tokenize(c["text"]) for c in chunks]
        self.lens = [len(d) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens) if self.lens else 0
        self.tf = [Counter(d) for d in self.docs]
        self.df = Counter()
        for d in self.docs:
            for t in set(d):
                self.df[t] += 1
        self.n = len(self.docs)

    def search(self, query, topk=5):
        qt = _tokenize(query)
        scores = []
        for i in range(self.n):
            s = 0.0
            for t in qt:
                tf = self.tf[i].get(t, 0)
                if not tf:
                    continue
                idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * self.lens[i] / (self.avg or 1)))
            scores.append(s)
        order = sorted(range(self.n), key=lambda i: -scores[i])[:topk]
        return [(self.chunks[i], scores[i]) for i in order if scores[i] > 0]


def load_chunks():
    """把每篇文档的每一个要点拆成一条可检索片段，并带上文档标题作为上下文。"""
    chunks = []
    for fn in sorted(glob.glob(os.path.join(_KNOWLEDGE_DIR, "*.md"))):
        text = open(fn, encoding="utf-8").read()
        title = ""
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                continue
            if line.startswith("- "):
                chunks.append({
                    "id": f"{os.path.basename(fn)}#{len(chunks)}",
                    "text": f"{title}：{line[2:].strip()}",
                    "source": os.path.basename(fn),
                })
    return chunks


class KnowledgeBase:
    def __init__(self):
        self.chunks = load_chunks()
        self.bm25 = BM25(self.chunks)

    def search_law(self, query, topk=3):
        hits = self.bm25.search(query, topk=topk)
        out = []
        for c, score in hits:
            out.append(f"【来源：{c['source']}】{c['text']}")
        return out

    def as_tool_result(self, query, topk=3):
        res = self.search_law(query, topk=topk)
        return "\n".join(res) if res else "（知识库中未找到直接相关法律要点）"


if __name__ == "__main__":
    kb = KnowledgeBase()
    for q in ["押金不退", "房东可以随时进屋吗", "维修费谁出"]:
        print("\n== 查询:", q, "==")
        for line in kb.search_law(q):
            print(" -", line[:80])
