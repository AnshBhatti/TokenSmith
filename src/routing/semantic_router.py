'''
File will have the 4 different types of queries mapped to 4 different models. Given an embedding of the query, use embedding model to choose the route, hence the model. Return model_path for inference later.
'''

from src.retriever import _get_embedder
from src.generator import get_llama_model
import numpy as np
import textwrap
from pydantic import BaseModel, field_validator
from typing import Optional
import json
from sentence_transformers import CrossEncoder

class RouteSelection(BaseModel):
    query_type: str

    @field_validator("query_type")
    @classmethod
    def must_be_valid_route(cls, v: str) -> str:
        valid = {r["query-type"] for r in ROUTES}
        if v not in valid:
            raise ValueError(f"'{v}' is not a valid query type. Must be one of: {valid}")
        return v


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
    def __init__(self, mode: str, embedding_model_path: str, decoding_model_path: Optional[str] = None, ce_model_path: Optional[str] = None):
        # self.embedder = SentenceTransformer(embedding_model_path)
        self.embedder = _get_embedder(embedding_model_path)
        self.decoder = None
        if mode not in ["embedder", "decoder", "cross-encoder"]:
            raise ValueError("Semantic router mode should be one of [\"embedder\", \"decoder\", \"cross-encoder\"]")
        self.mode = mode
        if mode == "cross-encoder" and not ce_model_path is None:
            self.ce_model = CrossEncoder(ce_model_path, max_length=512)
        if mode == "decoder" and not decoding_model_path is None:
            self.decoder = get_llama_model(decoding_model_path)
        for route in ROUTES:
            description = f"""
                Query type: {route["query-type"]}
                Query description: {route["description"]}
            """
            embedding = self.embedder.encode([description], convert_to_numpy=True)[0]
            route["embedding"] = embedding

    def route(self, query: str):
        if self.mode == "embedder":
            return self.route_embedding(query)
        elif self.mode == "decoder":
            return self.route_decoding(query)
        elif self.mode == "cross-encoder":
            return self.route_ce(query)

    def route_embedding(self, query: str) -> str:
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
        
        return best_route["model_path"], best_route["query-type"]
    
    def route_decoding(self, query: str) -> str:
        categories_text = "\n".join([f"- '{r['query-type']}': {r['description']}" for r in ROUTES])
        
        prompt = f"""You are a precise routing classification agent. 
Your task is to classify the user's query into exactly one of the following categories based on its intent:

{categories_text}

User Query: "{query}"

Output ONLY a JSON object with the key "query_type". Do not output markdown, reasoning, or any other text.
Example: {{"query_type": "Basic retrieval"}}
"""
        
        try:
            response = self.decoder.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a strict JSON-only routing agent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=30,
                response_format={"type": "json_object"} 
            )
            
            output_text = response["choices"][0]["message"]["content"].strip()
            
            parsed_json = json.loads(output_text)
            validated_route = RouteSelection(**parsed_json)
            
            for route in ROUTES:
                if route["query-type"] == validated_route.query_type:
                    print(f"LLM route chosen: {route['query-type']}")
                    return route["model_path"], route["query-type"]
                    
        except Exception as e:
            print(f"LLM routing failed ({e}). Falling back to embedding router...")
            return self.route(query)
    def route_ce(self, query: str) -> str:
        """
        Routes the query using a Cross-Encoder for high-accuracy intent classification.
        ce_model: An instantiated sentence_transformers.CrossEncoder model.
        """
        
        route_template = "Query type: {query_type}. Description: {description}"
        
        for route in ROUTES:
            print(route["query-type"])
        
        pairs = [[query, route_template.format(query_type = route["query-type"], description = route["description"])] for route in ROUTES]
        
        scores = self.ce_model.predict(pairs)
        
        best_idx = np.argmax(scores)
        best_route = ROUTES[best_idx]
        
        print(scores)
        
        print(f"Cross-Encoder route chosen: {best_route['query-type']} (Score: {scores[best_idx]:.4f})")
        
        return best_route["model_path"], best_route["query-type"]
        