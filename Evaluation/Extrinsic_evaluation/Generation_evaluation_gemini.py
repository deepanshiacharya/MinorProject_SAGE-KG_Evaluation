import argparse
import os
import json
import time
import re
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required.")
genai.configure(api_key=api_key)
MODEL_NAME = "gemini-2.0-flash"

MAX_WORKERS = 10

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

def create_requests(pairs):
    requests = []
    request_id = 0
    
    for pair_idx, (question, gt, retrieved_ans) in enumerate(pairs):
        for criterion, details in SCORING_RUBRICS.items():
            prompt = build_scoring_prompt(question, gt, retrieved_ans, 
                                        criterion, details["description"], details["rubric"])
            
            requests.append({
                "custom_id": f"pair_{pair_idx}_{criterion}_{request_id}",
                "prompt": prompt
            })
            request_id += 1
    
    return requests

def process_single_request(req, model):
    try:
        system_prompt = "You are an impartial evaluation judge."
        full_prompt = system_prompt + "\n\n" + req["prompt"]
        response = model.generate_content(full_prompt)
        content = response.text.strip()
        return {
            "custom_id": req["custom_id"],
            "response": {"body": {"choices": [{"message": {"content": content}}]}}
        }
    except Exception as e:
        return {
            "custom_id": req["custom_id"],
            "response": {"error": str(e)}
        }

def process_requests_parallel(requests, output_file="results.jsonl", max_workers=MAX_WORKERS):
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=genai.types.GenerationConfig(temperature=0)
    )
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_req = {executor.submit(process_single_request, req, model): req 
                        for req in requests}
        
        with tqdm(total=len(requests), desc="Processing requests") as pbar:
            for future in as_completed(future_to_req):
                result = future.result()
                results.append(result)
                pbar.update(1)
                
                if len(results) % 50 == 0:
                    with open(output_file, "w", encoding="utf-8") as f:
                        for r in results:
                            f.write(json.dumps(r) + "\n")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"Saved {len(results)} results to {output_file}")
    return output_file

def process_requests_batch(requests, output_file="results.jsonl", batch_size=50):
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=genai.types.GenerationConfig(temperature=0)
    )
    
    results = []
    total_batches = (len(requests) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(requests), batch_size), desc="Processing batches", total=total_batches):
        batch = requests[i:i + batch_size]
        
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batch))) as executor:
            futures = [executor.submit(process_single_request, req, model) for req in batch]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        if i + batch_size < len(requests):
            time.sleep(0.5)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"Saved {len(results)} results to {output_file}")
    return output_file

def parse_custom_id(custom_id):
    parts = custom_id.split("_")
    pair_idx = int(parts[1])
    request_id = int(parts[-1])
    criterion_parts = parts[2:-1]
    criterion = "_".join(criterion_parts)
    return pair_idx, criterion, request_id

def process_results(results_file="results.jsonl", pairs=None):
    try:
        results = []
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        
        print(f"Processing {len(results)} results...")
        
        organized_results = {}
        parsing_errors = []
        
        for result in results:
            try:
                pair_idx, criterion, request_id = parse_custom_id(result["custom_id"])
                
                if pair_idx not in organized_results:
                    organized_results[pair_idx] = {}
                
                if "error" in result["response"]:
                    organized_results[pair_idx][criterion] = {"retrieved_score": None}
                    continue
                
                response_content = result["response"]["body"]["choices"][0]["message"]["content"]
                
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
        
        return evaluation_results
        
    except Exception as e:
        print(f"Error processing results: {e}")
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
    parser = argparse.ArgumentParser(description="Optimized Gemini Evaluation Tool")
    parser.add_argument("--input-file", default="input_data.md", 
                       help="Path to input Markdown file")
    parser.add_argument("--output-file", default="evaluation_results.md",
                       help="Path to output Markdown file")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS,
                       help="Maximum parallel workers (default: 10)")
    parser.add_argument("--batch-mode", action="store_true",
                       help="Use batch processing mode instead of full parallel")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for batch mode (default: 50)")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    
    print("Optimized Gemini Evaluation Tool")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Max workers: {args.max_workers}")
    print("=" * 50)
    
    print(f"Loading data from {input_file}...")
    test_pairs = load_test_data(input_file)
    if not test_pairs:
        print("No data loaded")
        return
    
    print(f"Loaded {len(test_pairs)} QA sets.")
    
    print("Creating requests...")
    requests = create_requests(test_pairs)
    print(f"Total requests to process: {len(requests)}")
    
    start_time = time.time()
    
    if args.batch_mode:
        print(f"Processing requests in batch mode (batch size: {args.batch_size})...")
        results_file = process_requests_batch(requests, batch_size=args.batch_size)
    else:
        print("Processing requests in parallel mode...")
        results_file = process_requests_parallel(requests, max_workers=args.max_workers)
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Average time per request: {elapsed_time/len(requests):.2f} seconds")
    
    if results_file:
        evaluation_results = process_results(results_file, test_pairs)
        if evaluation_results:
            avg_scores = calculate_average_scores(evaluation_results)
            save_results_markdown(evaluation_results, avg_scores, output_file)
            print("Results saved to markdown!")
            
            print(f"Final Summary:")
            print(f"   Retrieved averages:")
            for criterion, avg in avg_scores["retrieved"].items():
                status = f"{avg:.2f}" if avg is not None else "N/A"
                print(f"     {criterion}: {status}")

if __name__ == "__main__":
    main()
