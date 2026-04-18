import json
import random
import os
from typing import List
from pathlib import Path
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from time import time, sleep
from openai import OpenAI
import argparse

class QueryBatch(BaseModel):
    basic_retrieval: str = Field(
        description="A query asking for explicit definitions, names, facts, and verbatim lookups."
    )
    causal_reasoning: str = Field(
        description="A query asking how mechanisms work, why processes occur, and logical derivations."
    )
    synthesis: str = Field(
        description="A query asking for comparisons between concepts, pros and cons, and chapter summaries."
    )
    extrapolation: str = Field(
        description="A query applying textbook concepts to hypothetical scenarios not explicitly mentioned."
    )

class GeneratedQuery(BaseModel):
    query_type: str
    query: str

class QueryList(BaseModel):
    queries: list[GeneratedQuery]

class QueryDataset:
    """Dataset of queries for evaluation."""
    
    def __init__(self, query_count: int = 20):
        self.client_ = None
        self.model_name = "llama3.1"
        self.query_types = []
        self.initialized = False
        self.queries = []
        self.query_count = query_count
        self.chunks = []
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
    def _initialize(self) -> bool:
        try:
            # self.client_ = genai.Client()
            self.client = OpenAI(
                base_url="http://localhost:11434/v1", 
                api_key="ollama" # Could be anything for localhost
            )
            self.query_types = [
                "Basic retrieval: explicit definitions, names, facts, and verbatim lookups",
                "Causal reasoning: how mechanisms work, why processes occur, and logical derivations",
                "Synthesis: comparisons between concepts, pros and cons, and chapter summaries",
                "Extrapolation: applying textbook concepts to hypothetical scenarios not explicitly mentioned"
            ]
            self.chunks = self._load_chunks()
            self.initialized = True
            return True
        except Exception as e:
            print(f"QueryDataset - LLM initialization failed: {e}")
            return False
        
    def _load_chunks(self) -> List[str]:
        chunks = []
        filepath = self.project_root / "routing_test_data/extracted_sections.json"
        
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found.")
            return chunks

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    heading = item.get("heading", "")
                    content = item.get("content", "")
                    
                    if heading.strip().lower() != "introduction":
                        # Format as a clean string for the LLM context window
                        chunk_text = f"Heading: {heading}\nContent: {content}"
                        # chunks.append(chunk_text)
                        chunks.append([heading, content])
        except Exception as e:
            print(f"Error reading chunks: {e}")
            
        return chunks

    def generate(self) -> List[dict]:
        if not self._initialize():
            print("Failed to initialize LLM client. Cannot generate queries.")
            return []
            
        if len(self.chunks) < 5:
            print("Error: Not enough chunks loaded to sample 5 consecutive chunks.")
            return []

        dataset = []
        start_time = time()
        iterations = self.query_count // len(self.query_types)
        
        print(f"Generating {self.query_count} total queries across {iterations} batches...")
        
        types_str = "\n".join([f"- {qt}" for qt in self.query_types])
        for i in range(iterations):
            start_idx = random.randint(0, len(self.chunks) - 2)
            selected_chunks = self.chunks[start_idx : start_idx + 2]
            
            # Build the context block
            context = "\n\n---\n\n".join([f"Heading: {chunk[0]}\nContent: {chunk[1]}" for chunk in selected_chunks])
            
            
            prompt = f"""
            You are generating an evaluation dataset for a Retrieval-Augmented Generation (RAG) system.
            Based on the following textbook excerpts, generate exactly one query for EACH of the defined query types as a student looking to understand the material.
            
            Query Types:
            {types_str}
            
            Textbook Excerpts:
            {context}
            """
            
            try:
                # response = self.client_.models.generate_content(
                #     model='gemini-2.5-flash',
                #     contents=prompt,
                #     config=types.GenerateContentConfig(
                #         response_mime_type="application/json",
                #         response_schema=QueryList,
                #         temperature=0.7
                #     )
                # )
                # generated_data = json.loads(response.text)
                # for q in generated_data.get("queries", []):
                #     dataset.append({
                #         "query_type": q["query_type"],
                #         "query": q["query"]
                #     })
                
                response = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a precise dataset generator. You must output a JSON object with exactly four specific keys."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=QueryBatch,
                    temperature=0.3
                )
                

                batch = response.choices[0].message.parsed
                
                dataset.append({"query_type": "Basic retrieval", "query": batch.basic_retrieval})
                dataset.append({"query_type": "Causal reasoning", "query": batch.causal_reasoning})
                dataset.append({"query_type": "Synthesis", "query": batch.synthesis})
                dataset.append({"query_type": "Extrapolation", "query": batch.extrapolation})
                    
                print(f"Batch {i+1}/{iterations} complete. Time from start: {time() - start_time:.2f}s. Total queries so far: {len(dataset)}")
                    
            except Exception as e:
                print(f"Error generating queries for batch {i+1}: {e}")

        output_file = self.project_root / "routing_test_data/query_dataset.json"
        try:
            os.makedirs("data", exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=4)
            print(f"\nSuccessfully saved {len(dataset)} queries to {output_file}. Total time taken: {time() - start_time:.2f}s")
        except Exception as e:
            print(f"Error saving dataset to {output_file}: {e}")

        return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset of queries for RAG evaluation.")
    parser.add_argument("--query-count", type=int, default=20, help="Total number of queries to generate (should be a multiple of 4).")
    args = parser.parse_args()
    generator = QueryDataset(query_count=args.query_count)
    generated_dataset = generator.generate()