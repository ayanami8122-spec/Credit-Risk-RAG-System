"""Backward-compatible import shim.

The RAG backend now lives in rag_pipeline.py. This file is kept so older
commands such as `python test.py` or imports from `test` still work.
"""

from rag_pipeline import get_rag_chain, iter_answer_deltas, main


if __name__ == "__main__":
    main()
