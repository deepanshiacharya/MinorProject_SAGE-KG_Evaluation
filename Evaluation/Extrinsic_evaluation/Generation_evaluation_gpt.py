import argparse
import os
import json
import time
import re
from tqdm import tqdm
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required.")
client = OpenAI(api_key=api_key)

SCORING_RUBRICS = {
    "completeness": {
        "description": "whether the answer includes ALL important facts and distinct points from the ground truth, allowing consistent factual additions",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Fully captures all Ground Truth facts with possibly helpful relevant detail.\n"
            "- 8-9: Covers most facts clearly with minor omissions or some additional context that does not contradict.\n"
            "- 6-7: Captures some key facts but misses several points or adds moderately extraneous/non-contradictory info.\n"
            "- 4-5: Partial coverage with many omissions or questionable additional info.\n"
            "- 1-3: Contains little of the Ground Truth facts.\n"
            "- 0: No relevant facts are present or answer is misleading."
        )
    },
    "accuracy": {
        "description": "whether the answer is factually correct compared to ground truth, tolerating consistent elaborations",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Fully accurate; no factual errors.\n"
            "- 8-9: Mostly accurate with minor trivial errors or consistent additions.\n"
            "- 6-7: Some factual inaccuracies or minor misinterpretations.\n"
            "- 4-5: Several incorrect points.\n"
            "- 1-3: Largely incorrect.\n"
            "- 0: Completely false or unrelated."
        )
    },
    "knowledgeability": {
        "description": "whether the answer shows accurate domain knowledge consistent with the ground truth, allowing relevant expansions",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Fully matches domain knowledge with clarity.\n"
            "- 8-9: Mostly aligns with minor gaps or some relevant added detail.\n"
            "- 6-7: Exhibits some understanding but also gaps.\n"
            "- 4-5: Limited knowledge shown.\n"
            "- 1-3: Minimal or incorrect domain knowledge.\n"
            "- 0: No relevant domain knowledge."
        )
    },
    "relevance": {
        "description": "whether the answer stays on-topic using only ground truth facts or consistent relevant information",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Entirely relevant and on-topic.\n"
            "- 8-9: Mostly relevant; minimal off-topic content.\n"
            "- 6-7: Some minor digressions.\n"
            "- 4-5: Noticeable off-topic content.\n"
            "- 1-3: Barely related.\n"
            "- 0: Completely irrelevant."
        )
    },
    "logical_coherence": {
        "description": "whether the answer presents the ground truth facts clearly and logically, with possible well-integrated expansions",
        "rubric": (
            "Scoring Guide (0-10):\n"
            "- 10: Clear, well-structured, logically coherent.\n"
            "- 8-9: Mostly clear with minor flow issues.\n"
            "- 6-7: Some structure but less clear.\n"
            "- 4-5: Poorly organized.\n"
            "- 1-3: Very hard to follow.\n"
            "- 0: Completely incoherent."
        )
    }
}

def build_scoring_prompt(question, ground_truth, retrieved_answer, criterion, description, rubric):
    return f"""
You are an impartial evaluation judge.

You are given:

Question:
\"\"\"{question}\"\"\"

Ground Truth Answer:
\"\"\"{ground_truth}\"\"\"

Retrieved Answer:
\"\"\"{retrieved_answer}\"\"\"

Your task:
Evaluate how well the retrieved answer captures ALL relevant factual information in the Ground Truth Answer, considering the context of the Question.

- The retrieved answer should fully include every important fact from the Ground Truth Answer.
- Relevant facts present in the Question but not explicitly in the Ground Truth Answer may be included in the retrieved answer without penalty.
- The retrieved answer should not omit key facts from the Ground Truth Answer.
- The retrieved answer should not contain incorrect facts or contradictions relative to both the Ground Truth Answer and the Question.

Your evaluation must be based on the criterion: {criterion} — {description}

Scoring Rubric:
{rubric}

Provide output ONLY in this JSON format:
{{
  "retrieved": {{"score": <number from 0 to 10>}}
}}
""".strip()

def load_test_data(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = content.split('---')
    pairs = []
    
    for section in sections[1:]:
        section = section.strip()
        if not section:
            continue

        question_match = re.search(r'\*\*Question:\*\*\s*(.*?)(?=\n\s*\*\*|$)', section, re.DOTALL | re.IGNORECASE)
        gt_match = re.search(r'\*\*Ground Truth:\*\*\s*(.*?)(?=\n\s*\*\*|$)', section, re.DOTALL | re.IGNORECASE)
        retrieved_match = re.search(r'\*\*Retrieved Answer:\*\*\s*(.*?)(?=\n\s*\*\*Paths Explored:|$)', section, re.DOTALL | re.IGNORECASE)

        if question_match and gt_match and retrieved_match:
            question = question_match.group(1).strip()
            ground_truth = gt_match.group(1).strip()
            retrieved_answer = retrieved_match.group(1).strip()
            pairs.append((question, ground_truth, retrieved_answer))
    
    return pairs

def create_batch_requests(pairs):
    requests = []
    request_id = 0
    
    for pair_idx, (question, gt, retrieved_ans) in enumerate(pairs):
        for criterion, details in SCORING_RUBRICS.items():
            prompt = build_scoring_prompt(question, gt, retrieved_ans, 
                                        criterion, details["description"], details["rubric"])
            
            requests.append({
                "custom_id": f"pair_{pair_idx}_{criterion}_{request_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are an impartial evaluation judge."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0
                }
            })
            request_id += 1
    
    return requests

def save_batch_requests(requests, filename="batch_requests.jsonl"):
    with open(filename, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")
    print(f"Saved {len(requests)} requests to {filename}")

def submit_batch_job(requests_file="batch_requests.jsonl"):
    try:
        print("Uploading batch file...")
        with open(requests_file, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        
        print("Creating batch job...")
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Retrieval Evaluation"}
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
        
        with open("batch_info.json", "w") as f:
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

def download_batch_results(batch_id, output_file="batch_results.jsonl"):
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
            with open("batch_errors.jsonl", "wb") as f:
                f.write(error_file_content.content)
            print("Errors saved to batch_errors.jsonl")
        
        return output_file
        
    except Exception as e:
        print(f"Error downloading results: {e}")
        return None

def parse_custom_id(custom_id):
    parts = custom_id.split("_")
    pair_idx = int(parts[1])
    request_id = int(parts[-1])
    criterion_parts = parts[2:-1]
    criterion = "_".join(criterion_parts)
    return pair_idx, criterion, request_id

def process_batch_results(results_file="batch_results.jsonl", pairs=None):
    try:
        results = []
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        
        print(f"Processing {len(results)} batch results...")
        
        organized_results = {}
        parsing_errors = []
        
        for result in results:
            if result["response"]["status_code"] != 200:
                print(f"Error in result {result['custom_id']}: {result['response']}")
                continue
            
            try:
                pair_idx, criterion, request_id = parse_custom_id(result["custom_id"])
                
                if pair_idx not in organized_results:
                    organized_results[pair_idx] = {}
                
                response_content = result["response"]["body"]["choices"][0]["message"]["content"].strip()
                
                try:
                    if response_content.startswith("```json"):
                        response_content = response_content.strip("```json\n").strip("```")
                    elif response_content.startswith("```"):
                        response_content = response_content.strip("```\n").strip("```")
                    
                    parsed_scores = json.loads(response_content)
                    retrieved_score = float(parsed_scores["retrieved"]["score"])
                    
                    organized_results[pair_idx][criterion] = {
                        "retrieved_score": retrieved_score
                    }
                    
                except Exception as e:
                    parsing_errors.append((result['custom_id'], f"Score parsing error: {e}"))
                    organized_results[pair_idx][criterion] = {
                        "retrieved_score": None
                    }
                    
            except Exception as e:
                parsing_errors.append((result['custom_id'], f"ID parsing error: {e}"))
                continue
        
        if parsing_errors:
            print(f"Found {len(parsing_errors)} parsing errors")
        
        evaluation_results = []
        
        for pair_idx in sorted(organized_results.keys()):
            if pairs and pair_idx < len(pairs):
                question, gt, retrieved_ans = pairs[pair_idx]
                
                pair_result = {
                    "pair_index": pair_idx + 1,
                    "question": question,
                    "ground_truth": gt,
                    "retrieved_answer": retrieved_ans,
                    "scores": {"retrieved": {}}
                }
                
                for criterion in SCORING_RUBRICS.keys():
                    if criterion in organized_results[pair_idx]:
                        scores = organized_results[pair_idx][criterion]
                        pair_result["scores"]["retrieved"][criterion] = {"score": scores["retrieved_score"]}
                    else:
                        pair_result["scores"]["retrieved"][criterion] = {"score": None}
                
                evaluation_results.append(pair_result)
        
        print(f"Processing Summary:")
        print(f"   Total pairs processed: {len(evaluation_results)}")
        print(f"   Total results organized: {len(organized_results)}")
        
        for criterion in SCORING_RUBRICS.keys():
            count = sum(1 for pair_idx in organized_results 
                       for crit in organized_results[pair_idx] 
                       if crit == criterion)
            print(f"   '{criterion}' results found: {count}")
        
        return evaluation_results
        
    except Exception as e:
        print(f"Error processing batch results: {e}")
        return None

def save_results_markdown(results, avg_scores, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation Results\n\n")
        f.write("## Average Scores Across All Pairs\n\n")
        f.write("### Retrieved Answer\n")
        for criterion, avg in avg_scores["retrieved"].items():
            f.write(f"- **{criterion.capitalize()}**: {avg:.2f}\n" if avg is not None else f"- **{criterion.capitalize()}**: N/A\n")
        f.write("\n")

        f.write("\n---\n\n## Individual Evaluation Results\n")
        for res in results:
            f.write(f"\n### Pair {res['pair_index']}\n")
            f.write(f"**Question:** {res['question']}\n")
            f.write(f"**Ground Truth:** {res['ground_truth']}\n")
            f.write(f"**Retrieved Answer:** {res['retrieved_answer']}\n")

            f.write("**Scores:**\n")
            f.write("- Retrieved:\n")
            for crit, scr in res["scores"]["retrieved"].items():
                score = scr['score']
                score_str = f"{score:.2f}" if score is not None else "N/A"
                f.write(f"  - {crit.capitalize()}: {score_str}\n")
            f.write("\n---\n")

def calculate_average_scores(evaluation_results):
    avg_scores = {"retrieved": {}}
    for criterion in SCORING_RUBRICS.keys():
        values = [
            res["scores"]["retrieved"][criterion]["score"]
            for res in evaluation_results
            if res["scores"]["retrieved"][criterion]["score"] is not None
        ]
        avg_scores["retrieved"][criterion] = sum(values) / len(values) if values else None
    return avg_scores

def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch API Evaluation Tool for Retrieval")
    parser.add_argument("--input-file", default="input_data.md", 
                       help="Path to input Markdown file")
    parser.add_argument("--output-file", default="evaluation_results.md",
                       help="Path to output Markdown file")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    
    print("OpenAI Batch API Evaluation Tool - Retrieval Version")
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
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            print(f"Loading data from {input_file}...")
            test_pairs = load_test_data(input_file)
            if not test_pairs:
                print("No data loaded, please check your Markdown file.")
                continue
            
            print(f"Loaded {len(test_pairs)} QA sets.")
            
            print("Creating batch requests...")
            requests = create_batch_requests(test_pairs)
            save_batch_requests(requests)
            
            batch_id = submit_batch_job()
            if batch_id:
                print(f"Batch submitted successfully!")
                print(f"Use option 2 to monitor status with batch ID: {batch_id}")
        
        elif choice == "2":
            batch_id = input("Enter batch ID (or press Enter to load from batch_info.json): ").strip()
            
            if not batch_id:
                try:
                    with open("batch_info.json", "r") as f:
                        batch_info = json.load(f)
                        batch_id = batch_info["batch_id"]
                        print(f"Loaded batch ID: {batch_id}")
                except FileNotFoundError:
                    print("No batch_info.json found. Please enter batch ID manually.")
                    continue
            
            monitor_batch_status(batch_id)
        
        elif choice == "3":
            batch_id = input("Enter batch ID (or press Enter to load from batch_info.json): ").strip()
            
            if not batch_id:
                try:
                    with open("batch_info.json", "r") as f:
                        batch_info = json.load(f)
                        batch_id = batch_info["batch_id"]
                        print(f"Loaded batch ID: {batch_id}")
                except FileNotFoundError:
                    print("No batch_info.json found. Please enter batch ID manually.")
                    continue
            
            results_file = download_batch_results(batch_id)
            if results_file:
                print(f"Loading original data from {input_file}...")
                test_pairs = load_test_data(input_file)
                
                evaluation_results = process_batch_results(results_file, test_pairs)
                if evaluation_results:
                    avg_scores = calculate_average_scores(evaluation_results)
                    save_results_markdown(evaluation_results, avg_scores, output_file)
                    print("Results processed and saved to markdown!")
                    
                    print(f"Final Summary:")
                    print(f"   Retrieved averages:")
                    for criterion, avg in avg_scores["retrieved"].items():
                        status = f"{avg:.2f}" if avg is not None else "N/A"
                        print(f"     {criterion}: {status}")
        
        elif choice == "4":
            print("Starting full workflow...")
            print(f"Loading data from {input_file}...")
            test_pairs = load_test_data(input_file)
            if not test_pairs:
                print("No data loaded, please check your Markdown file.")
                continue
            
            print(f"Loaded {len(test_pairs)} QA sets.")
            
            requests = create_batch_requests(test_pairs)
            save_batch_requests(requests)
            batch_id = submit_batch_job()
            
            if not batch_id:
                print("Failed to submit batch. Stopping workflow.")
                continue
            
            print("Monitoring batch progress...")
            print("You can also monitor at: https://platform.openai.com/batches")
            
            while True:
                batch = monitor_batch_status(batch_id)
                if not batch:
                    break
                
                if batch.status == "completed":
                    print("Batch completed!")
                    
                    results_file = download_batch_results(batch_id)
                    if results_file:
                        evaluation_results = process_batch_results(results_file, test_pairs)
                        if evaluation_results:
                            avg_scores = calculate_average_scores(evaluation_results)
                            save_results_markdown(evaluation_results, avg_scores, output_file)
                            print("Full workflow completed successfully!")
                    break
                
                elif batch.status == "failed":
                    print("Batch failed!")
                    break
                
                elif batch.status in ["cancelled", "expired"]:
                    print(f"Batch {batch.status}!")
                    break
                
                else:
                    print(f"Batch still processing... Checking again in 30 seconds")
                    print("Press Ctrl+C to stop monitoring (batch will continue running)")
                    try:
                        time.sleep(30)
                    except KeyboardInterrupt:
                        print("Monitoring stopped. Batch continues running.")
                        print(f"Use option 2 to check status later with batch ID: {batch_id}")
                        break
        
        elif choice == "5":
            try:
                with open("batch_info.json", "r") as f:
                    batch_info = json.load(f)
                    print("Existing batch info:")
                    for key, value in batch_info.items():
                        print(f"   {key}: {value}")
                    
                    print("Checking current status...")
                    monitor_batch_status(batch_info["batch_id"])
                    
            except FileNotFoundError:
                print("No batch_info.json found.")
        
        elif choice == "6":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
