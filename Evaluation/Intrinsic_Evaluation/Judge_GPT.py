import argparse
import json
import os
import re
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required.")
client = OpenAI(api_key=api_key)

COMPLETENESS_RUBRIC = (
    "Completeness Scoring Guide (0-10):\n"
    "- 10: Fully captures all Ground Truth facts with possibly helpful relevant detail.\n"
    "- 8-9: Covers most facts clearly with minor omissions or some additional context that does not contradict.\n"
    "- 6-7: Captures some key facts but misses several points or adds moderately extraneous/non-contradictory info.\n"
    "- 4-5: Partial coverage with many omissions or questionable additional info.\n"
    "- 1-3: Contains little of the Ground Truth facts.\n"
    "- 0: No relevant facts are present or answer is misleading."
)

ACCURACY_RUBRIC = (
    "Accuracy Scoring Guide (0-10):\n"
    "- 10: Fully accurate; no factual errors.\n"
    "- 8-9: Mostly accurate with minor trivial errors or consistent additions.\n"
    "- 6-7: Some factual inaccuracies or minor misinterpretations.\n"
    "- 4-5: Several incorrect points.\n"
    "- 1-3: Largely incorrect.\n"
    "- 0: Completely false or unrelated."
)

COMBINED_SYSTEM_PROMPT = (
    "You are a careful evaluator tasked with scoring the completeness and accuracy of extracted knowledge triplets, "
    "given a chunk of source text and a list of triplets supposedly extracted from it. "
    "Evaluate the triplets based on two metrics:\n"
    "1. **Completeness**: Reflects the recall of the triplets—how thoroughly they cover the important facts and distinct points in the chunk, ensuring minimal omissions. Allow consistent, useful additional context but focus on capturing key factual content.\n"
    "2. **Accuracy**: Reflects the precision of the triplets—how accurately they represent the facts in the document chunk, minimizing incorrect or unsupported information. Only consider what is present in the document text, not common knowledge.\n"
    "Use the provided scoring rubrics to assign a completeness score (0-10) and an accuracy score (0-10) for the triplets."
)

def build_scoring_prompt(chunk_text, triplets):
    triplet_lines = "\n".join(f"- ({t['subject']}, {t['predicate']}, {t['object']})" for t in triplets)
    return f"""
{COMBINED_SYSTEM_PROMPT}

Document chunk:
\"\"\"{chunk_text}\"\"\"

Triplets:
{triplet_lines}

Completeness Scoring Rubric:
{COMPLETENESS_RUBRIC}

Accuracy Scoring Rubric:
{ACCURACY_RUBRIC}

Provide scores in this JSON format:
{{
  "completeness": <number from 0 to 10>,
  "accuracy": <number from 0 to 10>
}}
""".strip()

def load_test_data(json_path):
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [(item["chunk_id"], item["file_id"], item["text"], item["triplets"]) for item in data]

def create_batch_requests(chunks):
    requests = []
    request_id_counter = 0
    
    for chunk_idx, (chunk_id, file_id, text, triplets) in enumerate(chunks):
        if not triplets:
            continue
        
        safe_chunk_id = re.sub(r'[^a-zA-Z0-9_.]', '_', chunk_id)
        custom_id = f"eval_{safe_chunk_id}_{request_id_counter}"
        prompt = build_scoring_prompt(text, triplets)
        requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": COMBINED_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 256
            }
        })
        request_id_counter += 1
    
    return requests

def save_batch_requests(requests, filename="batch_evaluation_requests.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")
    print(f"Saved {len(requests)} requests to {filename}")

def submit_batch_job(requests_file="batch_evaluation_requests.jsonl"):
    try:
        print("Uploading batch file...")
        with open(requests_file, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        
        print("Creating batch job...")
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Knowledge Triplet Combined Evaluation"}
        )
        
        print("Batch job created successfully!")
        print(f"   Batch ID: {batch.id}")
        print(f"   Status: {batch.status}")
        print(f"   Monitor at: https://platform.openai.com/batches")
        
        batch_info = {
            "batch_id": batch.id,
            "created_at": datetime.now().isoformat(),
            "input_file_id": batch_input_file.id,
            "status": batch.status
        }
        
        with open("evaluation_batch_info.json", "w") as f:
            json.dump(batch_info, f, indent=2)
        
        return batch.id
        
    except Exception as e:
        print(f"Error submitting batch: {e}")
        return None

def monitor_batch_status(batch_id):
    try:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch Status Report:")
        print(f"   ID: {batch.id}")
        print(f"   Status: {batch.status}")
        print(f"   Total requests: {batch.request_counts.total if batch.request_counts else 'N/A'}")
        print(f"   Completed: {batch.request_counts.completed if batch.request_counts else 'N/A'}")
        print(f"   Failed: {batch.request_counts.failed if batch.request_counts else 'N/A'}")
        
        if batch.status == "completed":
            print(f"   Output file ID: {batch.output_file_id}")
            print(f"   Error file ID: {batch.error_file_id}")
        
        return batch
    except Exception as e:
        print(f"Error checking batch status: {e}")
        return None

def download_batch_results(batch_id, output_file="batch_evaluation_results.jsonl"):
    try:
        batch = client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            print(f"Batch not completed yet. Status: {batch.status}")
            return None
        
        print("Downloading batch results...")
        
        result_file_content = client.files.content(batch.output_file_id)
        with open(output_file, "wb") as f:
            f.write(result_file_content.content)
        
        print(f"Results saved to {output_file}")
        
        if batch.error_file_id:
            print("Downloading error file...")
            error_file_content = client.files.content(batch.error_file_id)
            with open("batch_evaluation_errors.jsonl", "wb") as f:
                f.write(error_file_content.content)
            print("Errors saved to batch_evaluation_errors.jsonl")
        
        return output_file
        
    except Exception as e:
        print(f"Error downloading results: {e}")
        return None

def parse_custom_id(custom_id):
    match = re.match(r'eval_(.*)_(\d+)$', custom_id)
    if not match:
        raise ValueError(f"Invalid custom_id format: {custom_id}")
    return match.groups()

def process_batch_results(results_file="batch_evaluation_results.jsonl", chunks=None):
    try:
        results = []
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        
        print(f"Processing {len(results)} batch results...")
        
        organized_results = {}
        parsing_errors = []
        
        chunk_map = {re.sub(r'[^a-zA-Z0-9_.]', '_', c[0]): c for c in chunks}
        
        for result in results:
            if result["response"]["status_code"] != 200:
                continue
            
            try:
                safe_chunk_id, request_id = parse_custom_id(result["custom_id"])
                
                if safe_chunk_id not in organized_results:
                    organized_results[safe_chunk_id] = {"completeness": None, "accuracy": None}
                
                response_content = result["response"]["body"]["choices"][0]["message"]["content"]
                
                try:
                    parsed_result = json.loads(response_content)
                    completeness = float(parsed_result["completeness"])
                    accuracy = float(parsed_result["accuracy"])
                    
                    organized_results[safe_chunk_id]["completeness"] = completeness
                    organized_results[safe_chunk_id]["accuracy"] = accuracy
                    
                except Exception as e:
                    parsing_errors.append((result['custom_id'], f"Result parsing error: {e}"))
                    organized_results[safe_chunk_id]["completeness"] = None
                    organized_results[safe_chunk_id]["accuracy"] = None
                    
            except Exception as e:
                parsing_errors.append((result['custom_id'], f"ID parsing error: {e}"))
                continue
        
        if parsing_errors:
            print(f"Found {len(parsing_errors)} parsing errors")
        
        evaluation_results = []
        
        for safe_chunk_id, result_data in organized_results.items():
            if safe_chunk_id in chunk_map:
                chunk_id, file_id, text, triplets = chunk_map[safe_chunk_id]
                
                chunk_result = {
                    "chunk_id": chunk_id,
                    "file_id": file_id,
                    "text": text,
                    "triplets": triplets,
                    "evaluation": {
                        "completeness": result_data.get("completeness"),
                        "accuracy": result_data.get("accuracy")
                    }
                }
                evaluation_results.append(chunk_result)
            else:
                print(f"Missing original chunk data for {safe_chunk_id}")
        
        print(f"Processing Summary:")
        print(f"   Total chunks processed: {len(evaluation_results)}")
        print(f"   Total results organized: {len(organized_results)}")
        
        return evaluation_results
        
    except Exception as e:
        print(f"Error processing batch results: {e}")
        return None

def calculate_metrics(evaluation_results):
    completeness_scores = [res["evaluation"]["completeness"] for res in evaluation_results if isinstance(res["evaluation"]["completeness"], (int, float))]
    accuracy_scores = [res["evaluation"]["accuracy"] for res in evaluation_results if isinstance(res["evaluation"]["accuracy"], (int, float))]
    
    avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else None
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else None
    
    f1_score = None
    if avg_completeness and avg_accuracy:
        f1_score = 2 * (avg_completeness * avg_accuracy) / (avg_completeness + avg_accuracy)
    
    return avg_completeness, avg_accuracy, f1_score

def save_results_markdown(results, avg_completeness, avg_accuracy, f1_score, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Knowledge Triplet Evaluation Results\n\n")
        f.write("## Average Scores Across All Chunks\n\n")
        f.write(f"- **Completeness**: {avg_completeness:.2f}\n" if avg_completeness is not None else "- **Completeness**: N/A\n")
        f.write(f"- **Accuracy**: {avg_accuracy:.2f}\n" if avg_accuracy is not None else "- **Accuracy**: N/A\n")
        f.write(f"- **F1 Score**: {f1_score:.2f}\n" if f1_score is not None else "- **F1 Score**: N/A\n")
        f.write("\n---\n\n## Individual Evaluation Results\n")
        for res in results:
            f.write(f"\n### Chunk ID: {res['chunk_id']}\n")
            f.write(f"**File ID:** {res['file_id']}\n")
            f.write(f"**Text:** {res['text']}\n")
            f.write(f"**Triplets:**\n")
            for t in res['triplets']:
                f.write(f"- ({t['subject']}, {t['predicate']}, {t['object']})\n")
            f.write(f"**Evaluation:**\n")
            f.write(f"- Completeness Score: {res['evaluation']['completeness']}\n")
            f.write(f"- Accuracy Score: {res['evaluation']['accuracy']}\n")
            f.write("\n---\n")

def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch API Triplet Combined Evaluation Tool")
    parser.add_argument("--input", default="processed_data.json", 
                        help="Path to input JSON file (default: processed_data.json)")
    parser.add_argument("--output", default="triplet_evaluation_results.md",
                        help="Path to output Markdown file (default: triplet_evaluation_results.md)")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    
    print("OpenAI Batch API Triplet Combined Evaluation Tool")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Create and submit new batch job")
        print("2. Check batch status")
        print("3. Download and process results")
        print("4. Full workflow (create → submit → monitor)")
        print("5. Load existing batch info")
        print("6. Test custom_id parsing")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "1":
            print(f"Loading data from {input_file}...")
            chunks = load_test_data(input_file)
            if not chunks:
                print("No data loaded, please check your JSON file.")
                continue
            
            print(f"Loaded {len(chunks)} chunks.")
            
            print("Creating batch requests...")
            requests = create_batch_requests(chunks)
            save_batch_requests(requests)
            
            batch_id = submit_batch_job()
            if batch_id:
                print(f"Batch submitted successfully!")
                print(f"Use option 2
