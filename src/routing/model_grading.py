import time
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Dict, List
import csv
import json

# Instead of grading each answer individually, it was more effective for the model to compare answers. Right now hardcoded to 3 models.
prompt = """You are an expert evaluator for a database textbook Q&A system. 

You are comparing three student answers to the same question based ONLY on the provided Reference Material.

**Reference Material:** {retrieved_chunks}
**Question:** {question}
**Query Type:** {query_type}: {query_description}

---
**Answer A:** {answer_a}
**Answer B:** {answer_b}
**Answer C:** {answer_c}
---

### **Step 1: Query-Type Specific Requirements**
You must adjust your definition of 'Complete' and 'Deep' based on the category:
- **Basic Retrieval**: Precision and brevity are king. A direct, accurate answer is a 10. If a model gives unnecessary info, it MUST be penalized in Relevance.
- **Causal Reasoning**: Requires explaining the 'why'. Identifying the outcome is only a 5/10. Explaining the internal mechanism (e.g., how a lock is acquired) is required for a 9-10.
- **Synthesis**: Must bridge multiple concepts. If the student lists facts separately without integrating them, the maximum Completeness score is 6.
- **Extrapolation**: Must apply textbook rules to the new scenario. Correct logical application is required for a 10.

### **Step 2: Comparison & Dominance Rules**
- **The Dominance Rule**: If one answer provides 'how' logic and another only provides a 'what' definition, the deeper answer MUST score at least 3 points higher in Technical Depth.
- **The Anti-Filler Rule**: If an answer repeats the same point multiple times or includes filler content (e.g., 'Here's an explanation'), it MUST be penalized in Relevance and Clarity.
- **Relative Ranking**: You must rank these answers. They cannot all be 'excellent.'

### **Step 3: Scoring Protocol (1-10)**
- **8-10**: Perfect, concise, and explains internal database logic.
- **5-7**: Accurate and complete, but lacks the highest level of technical nuance.
- **3-4**: Correct gist but misses the why or is too wordy/redundant.
- **1-2**: Significant omissions, inaccuracies, or irrelevant filler.

**Instructions:**
- Base scores on technical precision, not "niceness."
- Base your evaluation ONLY on the provided Reference Material.
- Provide specific, actionable feedback.
- Output strictly in JSON format."""


query_descriptions = {
    "Basic retrieval": "A query asking for explicit definitions, names, facts, and verbatim lookups.",
    "Causal reasoning": "A query asking how mechanisms work, why processes occur, and logical derivations.",
    "Synthesis": "A query asking for comparisons between concepts, pros and cons, and chapter summaries.",
    "Extrapolation": "A query applying textbook concepts to hypothetical scenarios not explicitly mentioned."
}
class ModelScore(BaseModel):
    model_id: str = Field(description="The ID of the answer being graded (e.g., 'Answer A')")
    accuracy: int = Field(ge=1, le=10, description="Are the facts correct according to the reference material?")
    completeness: int = Field(ge=1, le=10, description="Meets specific requirements for the Query Type.")
    technical_depth: int = Field(ge=1, le=10, description="Score 1-5 for 'What'; 6-10 for 'How' and 'Why'.")
    clarity: int = Field(ge=1, le=10, description="Answer is clear and answers all parts of the query.")
    relevance: int = Field(ge=1, le=10, description="Model's answer is on-track with what the question is asking and reference material is providing")
    feedback: str = Field(description="Strict technical justification for the score.")

class ComparisonResult(BaseModel):
    ranking: List[str] = Field(description="Ranking from best to worst, e.g., ['Answer B', 'Answer C', 'Answer A']")
    comparative_reasoning: str = Field(description="Why does the winner dominate? Why did the 'yapper' or 'surface-level' model lose?")
    score_a: ModelScore = Field(description="Answer A's score")
    score_b: ModelScore = Field(description="Answer B's score")
    score_c: ModelScore = Field(description="Answer C's score")

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", # Could be anything for localhost
)    

with open("routing_test_data/model_testing_results.csv", "r") as f:
    reader = csv.reader(f)
    csv_data = list(reader)
    
with open("routing_test_data/answers.json", "r") as f:
    chunk_collection = json.load(f)

# weights = [0.4, 0.3, 0.2, 0.1]
weights = [0.35, 0.25, 0.25, 0.1, 0.05]
assert(sum(weights) == 1)
# f = open("tmp.txt", 'w')


for i in range(1, len(csv_data), 3):
    # if csv_data[i][4]:
    #     continue
    # if i > 12:
    #     break
    score = None
    try:
        start = time.time()
        response = client.beta.chat.completions.parse(
            model="llama3.1", 
            messages=[
                {"role": "user", "content": prompt.format(
                    retrieved_chunks="\n\n---\n\n".join(chunk_collection[i]["chunks"]),
                    question=csv_data[i][0],
                    # generated_answer=chunk_collection[i]["answer"],
                    query_type=csv_data[i][1],
                    query_description=query_descriptions[csv_data[i][1]],
                    answer_a=chunk_collection[i]["answer"],
                    answer_b=chunk_collection[i + 1]["answer"],
                    answer_c=chunk_collection[i + 2]["answer"]
                )}
            ],
            temperature=0.3,
            response_format=ComparisonResult 
        )
        # f.write(chunk_collection[i]["answer"] + '. ANSWER DONE\n')
        parsed_result = response.choices[0].message.parsed
        # score = parsed_result.accuracy * weights[0] + parsed_result.completeness * weights[1] + parsed_result.technical_depth * weights[2] + parsed_result.clarity * weights[3] + parsed_result.relevance * weights[4]
        # print(f"{i}: Grading completed in {time.time() - start:.2f} seconds. Score: {score}. Query type: {csv_data[i][1]}. Model: {csv_data[i][2]}. Individual scores:", parsed_result.accuracy, parsed_result.completeness, parsed_result.technical_depth, parsed_result.clarity, parsed_result.relevance)
        for j, result in enumerate([parsed_result.score_a, parsed_result.score_b, parsed_result.score_c]):
            score = result.accuracy * weights[0] + result.completeness * weights[1] + result.technical_depth * weights[2] + result.clarity * weights[3] + result.relevance * weights[4]
            score = round(score, 2)
            csv_data[i + j][4] = score
            print(f"{i + j}: Grading completed in {time.time() - start:.2f} seconds. Score: {score}. Query type: {csv_data[i + j][1]}. Model: {csv_data[i + j][2]}. Individual scores:", result.accuracy, result.completeness, result.technical_depth, result.clarity, result.relevance, "Feedback: ", result.feedback)
            
            
    except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            print(f"Ollama grading failed: {e}")
        break
    # csv_data[i][4] = score
print("\nSaving final results with scores to CSV...")
with open("routing_test_data/model_testing_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)
print("Data saved to model_testing_results.csv")
# f.close()