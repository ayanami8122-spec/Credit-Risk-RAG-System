"""流式输出冒烟测试：在 rag_chat_app 目录下运行 python verify_stream.py"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag_pipeline import get_rag_chain, iter_answer_deltas


def main() -> int:
    question = "什么是 PD？"
    print("加载 RAG 链…")
    chain = get_rag_chain()
    print(f"提问: {question}\n")

    tokens: list[str] = []
    t0 = time.perf_counter()
    first_token_at: float | None = None

    for token in iter_answer_deltas(chain, question):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        tokens.append(token)
        print(token, end="", flush=True)

    elapsed = time.perf_counter() - t0
    answer = "".join(tokens)
    print("\n")
    print("-" * 50)
    print(f"token 次数: {len(tokens)}")
    print(f"总字符数: {len(answer)}")
    print(f"总耗时: {elapsed:.2f}s")
    if first_token_at is not None:
        print(f"首 token 延迟: {first_token_at - t0:.2f}s")
    else:
        print("未收到任何 token")
        return 1

    if len(tokens) < 2:
        print("警告: token 过少，可能未真正流式输出")
        return 1

    if len(answer) < 10:
        print("警告: 回答过短")
        return 1

    print("流式冒烟测试通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

