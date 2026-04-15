"""
Purpose: apply the query dataset to 3 models on different query types. Also find out which route the semantic routing takes.
"""

import csv
import sys

from src.config import RAGConfig
from src.main import get_answer, parse_args
from src.generator import get_llama_model
from src.ranking.ranker import EnsembleRanker
from src.ranking.reranker import rerank
from src.retriever import load_artifacts, FAISSRetriever, BM25Retriever, IndexKeywordRetriever
from src.routing.semantic_router import SemanticRouter
import json


import time
import gc

def execute():
    model_names = [
        "models/Llama-3.2-1B-Instruct-Q8_0.gguf",
        "models/qwen2.5-0.5b-instruct-fp16.gguf",
        "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    ]

    models = [get_llama_model(model) for model in model_names]



    with open("data/query_dataset.json", "r") as f:
        dataset = json.load(f)

    # create semantic router
    semantic_router = SemanticRouter(embedding_model_path = "models/Qwen3-Embedding-4B-Q5_K_M.gguf")

    csv_data = [["Query", "Query type", "Model name", "Latency", "Quality", "Weighted Utility", "Route Taken"]]
    answers = [{}]

    # Replicating how main.py sets up the chat session so that get_answer can be used. Will write cleaner version later.
    cfg = RAGConfig.from_yaml("config/config.yaml")
    args = parse_args()


    try:
        artifacts_dir = cfg.get_artifacts_directory()
        faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(artifacts_dir, args.index_prefix)
        print(f"Loaded {len(chunks)} chunks and {len(sources)} sources from artifacts.")
        retrievers = [FAISSRetriever(faiss_idx, cfg.embed_model), BM25Retriever(bm25_idx)]
        if cfg.ranker_weights.get("index_keywords", 0) > 0:
            retrievers.append(IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path))
        
        ranker = EnsembleRanker(ensemble_method=cfg.ensemble_method, weights=cfg.ranker_weights, rrf_k=int(cfg.rrf_k))
        print("Loaded retrievers and initialized ranker.")
        artifacts = {"chunks": chunks, "sources": sources, "retrievers": retrievers, "ranker": ranker, "meta": meta}
    except Exception as e:
        print(f"ERROR: {e}. Run 'index' mode first.")
        sys.exit(1)

    for query in dataset:
        route = semantic_router.route(query["query"])
        print(f"Query: {query['query']}")
        print(f"Routed to model: {route}")
        for i, model in enumerate(models):
            print(i)
            start_time = time.time()
            answer, chunks, _ = get_answer(question=query["query"], cfg=cfg, args=args, logger=None, console=None, artifacts=artifacts, golden_chunks=None, is_test_mode=True, additional_log_info=None, model_path=model_names[i])
            end_time = time.time()
            # print("Answer:", answer)
            latency = end_time - start_time
            chunks = [chunk["content"] for chunk in chunks]
            chunks = rerank(query["query"], chunks, mode=cfg.rerank_mode, top_n=cfg.rerank_top_k)
            chunks = [chunk[0] for chunk in chunks]
            csv_data.append([query["query"], query["query_type"], model_names[i], latency, None, None, route])
            answers.append({"answer": answer, "chunks": chunks})
            
            # print(*(csv_data[-1]), sep=", ")
            
    # Below CSVs I will use in model_grading.py to use a larger model to grade
    print("\nSaving results to CSV...")
    with open("data/model_testing_results.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    with open("data/answers.json", "w", encoding="utf-8") as f:
        json.dump(answers, f)


    
# for model in models:
#     model.close()
    
# # for key in _EMBED_CACHE:
# #     _EMBED_CACHE[key].close()
# _EMBED_CACHE.clear()
# del semantic_router
    
# gc.collect()


if __name__ == "__main__":
    execute()