import argparse
import json
import re
import google.generativeai as genai
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

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
    "1. **Completeness** – how thoroughly they cover the important facts and distinct points in the chunk.\n"
    "2. **Accuracy** – how precisely they represent the facts in the text without adding unsupported information.\n"
    "Use the rubrics provided below and respond ONLY in valid JSON."
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

Provide scores in strict JSON:
{{
  "completeness": <number from 0 to 10>,
  "accuracy": <number from 0 to 10>
}}
""".strip()

def load_test_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(item["chunk_id"], item["file_id"], item["text"], item["triplets"]) for item in data]

def evaluate_chunk(chunk_text, triplets):
    prompt = build_scoring_prompt(chunk_text, triplets)
    try:
        response = model.generate_content(prompt)
        raw_output = response.text.strip()
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if not match:
            return None, None
        parsed = json.loads(match.group())
        return parsed.get("completeness"), parsed.get("accuracy")
    except Exception:
        return None, None

def process_all(chunks):
    results = []
    for chunk_id, file_id, text, triplets in tqdm(chunks, desc="Evaluating chunks"):
        if not triplets:
            continue
        completeness, accuracy = evaluate_chunk(text, triplets)
        results.append({
            "chunk_id": chunk_id,
            "file_id": file_id,
            "text": text,
            "triplets": triplets,
            "evaluation": {
                "completeness": completeness,
                "accuracy": accuracy
            }
        })
    return results

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
        f.write("# Knowledge Triplet Evaluation Results (Gemini)\n\n")
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
            for t in res["triplets"]:
                f.write(f"- ({t['subject']}, {t['predicate']}, {t['object']})\n")
            f.write(f"**Evaluation:**\n")
            f.write(f"- Completeness Score: {res['evaluation']['completeness']}\n")
            f.write(f"- Accuracy Score: {res['evaluation']['accuracy']}\n")
            f.write("\n---\n")

def main():
    parser = argparse.ArgumentParser(description="Gemini Knowledge Triplet Evaluation Tool")
    parser.add_argument("--input-file", default="input_data.json", help="Path to input JSON file")
    parser.add_argument("--output-file", default="evaluation_results.md", help="Path to output Markdown file")
    args = parser.parse_args()

    chunks = load_test_data(args.input_file)
    print(f"Loaded {len(chunks)} chunks for evaluation")

    results = process_all(chunks)

    avg_completeness, avg_accuracy, f1_score = calculate_metrics(results)

    save_results_markdown(results, avg_completeness, avg_accuracy, f1_score, args.output_file)

    print("Evaluation complete!")
    print(f"Avg Completeness: {avg_completeness:.2f}")
    print(f"Avg Accuracy: {avg_accuracy:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
