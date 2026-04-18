"""
Purpose: evaluate the 3 routing strategies (embedding, decoder, cross-encoder) against
         the query dataset and record chosen route + latency for each.
"""

import csv
import json
import time

from src.routing.semantic_router import ROUTES, SemanticRouter

_MODEL_PATH_TO_QUERY_TYPE = {r["model_path"]: r["query-type"] for r in ROUTES}


def model_path_to_query_type(model_path: str) -> str:
    return _MODEL_PATH_TO_QUERY_TYPE.get(model_path, "Unknown")


def execute():
    with open("routing_test_data/query_dataset.json", "r") as f:
        dataset = json.load(f)

    print("Initialising embedding router...")
    embedding_router = SemanticRouter(mode="embedder", embedding_model_path="models/Qwen3-Embedding-4B-Q5_K_M.gguf")

    print("Initialising decoder-small router...")
    decoder_router = SemanticRouter(mode="decoder", embedding_model_path="models/Qwen3-Embedding-4B-Q5_K_M.gguf", decoding_model_path="models/qwen2.5-0.5b-instruct-fp16.gguf")
    
    print("Initialising decoder-large router...")
    decoder_router_l = SemanticRouter(mode="decoder", embedding_model_path="models/Qwen3-Embedding-4B-Q5_K_M.gguf", decoding_model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf")

    print("Initialising cross-encoder router...")
    ce_router = SemanticRouter(mode="cross-encoder", embedding_model_path="models/Qwen3-Embedding-4B-Q5_K_M.gguf", ce_model_path="cross-encoder/ms-marco-MiniLM-L-6-v2")

    header = [
        "Query", "Query Type",
        "Embedding Route", "Embedding Latency (s)",
        "Decoder-small Route", "Decoder-small Latency (s)",
        "Decoder-large Route", "Decoder-large Latency (s)",
        "Cross-Encoder Route", "Cross-Encoder Latency (s)",
    ]
    rows = [header]

    for i, item in enumerate(dataset):
        query      = item["query"]
        query_type = item["query_type"]
        print(f"\n[{i+1}/{len(dataset)}] {query}")

        start = time.time()
        emb_path, emb_route = embedding_router.route(query)
        emb_latency = time.time() - start

        start = time.time()
        dec_path, dec_route = decoder_router.route(query)
        dec_latency = time.time() - start

        start = time.time()
        dec_path_l, dec_route_l = decoder_router_l.route(query)
        dec_latency_l = time.time() - start

        start = time.time()
        ce_path, ce_route = ce_router.route(query)
        ce_latency = time.time() - start

        rows.append([
            query, query_type,
            emb_route, f"{emb_latency:.4f}",
            dec_route, f"{dec_latency:.4f}",
            dec_route_l, f"{dec_latency_l:.4f}",
            ce_route,  f"{ce_latency:.4f}",
        ])

    print(f"\nSaving results to data/routing_testing_results.csv...")
    with open("routing_test_data/routing_testing_results.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print("Done.")


if __name__ == "__main__":
    execute()
