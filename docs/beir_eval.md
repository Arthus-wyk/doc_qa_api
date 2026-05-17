# BEIR Isolated Evaluation

This project provides an isolated BEIR evaluation runner that reuses the production retrieval logic while keeping evaluation data separate from business data.

## What is isolated

- Chroma index path is isolated by `DOC_QA_CHROMA_DIR`.
- BM25 chunk store path is isolated by `DOC_QA_CHUNK_STORE_PATH`.
- Output directory is `data/beir_eval/<dataset>_<run_name-or-timestamp>/`.
- Dataset cache is `data/beir_datasets/`.

No production `data/chroma` or production `data/chunks.jsonl` files are touched during evaluation.

## What is shared with production

- Query rewrite: `app.rag.query_transform.rewrite_query`
- Hybrid retrieval flow: `app.core.graph_notes.retrieve_docs_node`
- Vector retrieval backend: `app.rag.llamaindex_store.retrieve_documents`
- BM25 retriever builder: `app.rag.bm25.get_bm25_retriever`
- Reranking: `app.rag.reranker.rerank_documents`

The evaluator calls the same retrieval code path used in `/qa/ask`, but skips answer generation.

## Run

Install dependency:

```bash
pip install beir
```

Run evaluation:

```bash
python -m app.eval.beir_runner --dataset scifact --top-k 10 --run-name smoke --reset
```

Outputs:

- `data/beir_eval/<run>/metrics.json`
- console JSON summary (nDCG, MAP, Recall, Precision, MRR)

## Notes

- This runner uses the same LLM rewrite and rerank behavior as production; keep local model and API keys available.
- If `--run-name` points to an existing non-empty directory, the runner exits unless `--reset` is provided.
