import argparse
import string
import re
from typing import List, Tuple

def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split()).strip()

def exact_match(test_pairs: List[Tuple[str, str]]) -> float:
    N = len(test_pairs)
    matches = sum(1 for gt, pred in test_pairs if normalize(gt) == normalize(pred))
    return matches * 100 / N if N > 0 else 0.0

def extract_pairs_from_markdown(content: str) -> List[Tuple[str, str]]:
    pairs = []
    gt, pred = None, None
    
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("**Ground Truth:**"):
            gt = line.replace("**Ground Truth:**", "").strip()
        elif line.startswith("**Retrieved Answer:**"):
            pred = line.replace("**Retrieved Answer:**", "").strip()
            if gt and pred:
                pairs.append((gt, pred))
                gt, pred = None, None
        i += 1
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Exact Match Evaluation")
    parser.add_argument("--input-file", default="input_data.md", help="Path to input Markdown file")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        content = f.read()

    test_pairs = extract_pairs_from_markdown(content)
    em_score = exact_match(test_pairs)
    print(f"Exact Match Score: {em_score:.4f} %")

if __name__ == "__main__":
    main()
