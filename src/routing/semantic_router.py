'''
File will have the 4 different types of queries mapped to 4 different models. Given an embedding of the query, use embedding model to choose the route, hence the model. Return model_path for inference later.
'''

from src.retriever import _get_embedder
import numpy as np


ROUTES = [
    {
        "query-type": "Basic retrieval",
        "description": "A query asking for explicit definitions, names, facts, and verbatim lookups. It needs the explanation to \"what\" something is in a concise manner.",
        "model_path": "models/qwen2.5-0.5b-instruct-fp16.gguf",
        "embedding": None
    },
    {
        "query-type": "Causal reasoning",
        "description": "A query asking how mechanisms work, why processes occur, and logical derivations. It needs the \"how\" and \"why\" explanations in detail",
        # "model_path": "models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
        "model_path": "models/qwen2.5-3b-instruct-q4_k_m.gguf",
        "embedding": None
    },
    {
        "query-type": "Synthesis",
        "description": "A query asking for comparisons between concepts, pros and cons, and summaries.",
        "model_path": "models/qwen2.5-0.5b-instruct-fp16.gguf",
        "embedding": None
    },
    {
        "query-type": "Extrapolation",
        "description": "A query applying textbook concepts to hypothetical scenarios not explicitly mentioned.",
        "model_path": "models/qwen2.5-3b-instruct-q4_k_m.gguf",
        "embedding": None
    },
]

class SemanticRouter:
    def __init__(self, embedding_model_path: str):
        # self.embedder = SentenceTransformer(embedding_model_path)
        self.embedder = _get_embedder(embedding_model_path)
        for route in ROUTES:
            # description = route["description"]
            # description = f"""
            #     Query type: {route["query-type"]}
            #     Query description: {route["description"]}
            #     Query examples: 
            #     - {"\n- ".join(route["examples"])}
            # """
            description = f"""
                Query type: {route["query-type"]}
                Query description: {route["description"]}
            """
            embedding = self.embedder.encode([description], convert_to_numpy=True)[0]
            route["embedding"] = embedding

    def route(self, query: str) -> str:
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        
        best_route = None
        best_similarity = -1
        
        for route in ROUTES:
            route_embedding = route["embedding"]
            cos_sim = np.dot(query_embedding, route_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(route_embedding))
            
            if cos_sim > best_similarity:
                best_similarity = cos_sim
                best_route = route
        
        print("Best route chosen:", best_route["query-type"])
        
        return best_route["model_path"]
    
    # def route(self, query: str) -> str:
        