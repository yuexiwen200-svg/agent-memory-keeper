#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent Memory Keeper · 联想检索

把用户整句话丢进来，脚本自己拆词、算稀有度、排序。
设计要点：
  - 只保留纯实字片段（带"的/是/怎/到"这类虚字的一律丢弃），否则检索会被虚词带偏
  - 用 IDF 压掉高频片段，越常见越不值钱
  - 整句原话命中直接置顶

用法：
  python search_memory.py --src <trajectory.jsonl 或目录> "他刚才说的那句话"
  python search_memory.py --src <path> --top 12 "关键词"
"""
import argparse, json, math, os, re

VIRT = set("的了是吗怎着来去过说想要会能这那个我你他她它在就也都和跟把被给对不没还再呀吧呢啊很太最更到儿")

STOP = set("""的 了 吗 呢 吧 啊 呀 是 我 你 他 她 它 这 那 那个 这个 一个 咱们 我们 你们
就 也 都 和 跟 把 被 给 让 说 讲 对 不 没 有 会 能 要 想 看 去 来 到 着 过 还 再
怎么 什么 为什么 哪 哪个 多少 这样 那样 一下 现在 然后 但是 因为 所以 如果 就是
真 太 挺 特别 可以 应该 已经 一直 赶紧 随便 反正 其实 或者 而且 不过 时候
到底 来着 那个 这种 那种 一样 感觉 觉得 知道 东西 事情 有点""".split())


def ok(g):
    """只留纯实字片段"""
    if not g or len(g) < 2 or g in STOP:
        return False
    return not any(c in VIRT for c in g)


def grams(sentence):
    s = re.sub(r"[，。？！、；：,.?!;:\s“”\"'（）()《》\[\]【】]+", " ", sentence)
    out = {}
    for w in s.split(" "):
        w = w.strip()
        if not w or w in STOP:
            continue
        if len(w) <= 4:
            if ok(w):
                out[w] = max(out.get(w, 0), len(w) + 2)
        else:
            for n in (4, 3, 2):
                for i in range(len(w) - n + 1):
                    g = w[i:i + n]
                    if ok(g):
                        out[g] = max(out.get(g, 0), n)
    return out


def load_turns(path):
    """读取 jsonl（或目录下所有 jsonl），聚合成 (user, [assistant...]) 轮"""
    files = []
    if os.path.isdir(path):
        for n in sorted(os.listdir(path)):
            if n.endswith(".jsonl"):
                files.append(os.path.join(path, n))
    else:
        files = [path]
    turns = []
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                role = r.get("role")
                c = r.get("content") or ""
                if not isinstance(c, str):
                    c = ""
                if role == "user":
                    turns.append([c.strip(), []])
                elif role == "assistant" and turns:
                    if c.strip():
                        turns[-1][1].append(c.strip())
    return turns


def main():
    ap = argparse.ArgumentParser(description="联想检索会话轨迹")
    ap.add_argument("--src", required=True, help="trajectory.jsonl 或包含 jsonl 的目录")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("query", nargs="+")
    a = ap.parse_args()
    q = " ".join(a.query).strip()
    gs = grams(q)
    if not gs:
        print("这句里没抽出能搜的词，换个说法。")
        return
    turns = load_turns(a.src)
    texts = [re.sub(r"\s+", " ", u + " " + " ".join(x)) for u, x in turns]
    N = len(texts)

    df = {g: sum(1 for t in texts if g in t) for g in gs}
    keep = {g: w for g, w in gs.items() if 0 < df[g] <= N * 0.30} or gs

    scored = []
    qn = re.sub(r"\s+", "", q)
    for i, t in enumerate(texts, 1):
        hit, score = [], 0.0
        for g, w in keep.items():
            if g in t:
                hit.append(g)
                score += w * math.log((N + 1) / (df[g] + 1))
        if len(qn) >= 5 and qn in re.sub(r"\s+", "", t):
            score += 1000
        if score > 0:
            scored.append((score, i, hit, t))
    scored.sort(key=lambda x: -x[0])

    print("拆出 %d 片段，留下有辨识度的 %d 个；命中 %d 轮，前 %d 条：\n"
          % (len(gs), len(keep), len(scored), min(a.top, len(scored)))
)
    for score, i, hit, text in scored[:a.top]:
        key = max(hit, key=len) if hit else ""
        p = text.find(key) if key else 0
        seg = text[max(0, p - 80): p + 170]
        hs = "/".join(sorted(hit, key=len, reverse=True)[:5])
        print("[第%d轮|相关%.1f|命中:%s]" % (i, score, hs))
        print("   …%s…\n" % seg)
    if not scored:
        print("一条都没命中 → 按铁律：直说没聊过，不许编。")


if __name__ == "__main__":
    main()
