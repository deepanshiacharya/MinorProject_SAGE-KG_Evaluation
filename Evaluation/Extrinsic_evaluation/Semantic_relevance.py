import argparse
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def load_test_pairs_from_markdown(md_file_path: str) -> List[tuple]:
    with open(md_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pairs = []
    gt, rt = None, None
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("**Ground Truth:**"):
            gt = line.replace("**Ground Truth:**", "").strip()
        elif line.startswith("**Retrieved Answer:**"):
            rt = line.replace("**Retrieved Answer:**", "").strip()
            if gt and rt:
                pairs.append((gt, rt))
                gt, rt = None, None
        i += 1
    return pairs

def embed_sentences(sentences: List[str], model) -> np.ndarray:
    return model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

def retrieval_similarity(retrieved: List[str], gold: List[str], model, verbose: bool = False) -> float:
    assert len(retrieved) == len(gold), "Input lists must be of same length"

    retr_embs = embed_sentences(retrieved, model)
    gold_embs = embed_sentences(gold, model)

    sims = [
        cosine_similarity(retr_embs[i].reshape(1, -1), gold_embs[i].reshape(1, -1)).item()
        for i in range(len(retrieved))
    ]

    if verbose:
        for idx, score in enumerate(sims, start=1):
            print(f"Pair {idx} similarity: {score:.4f}")

    return float(np.mean(sims))

def main():
    parser = argparse.ArgumentParser(description="Retrieval Similarity Evaluation")
    parser.add_argument("--input-file", default="input_data.md", help="Path to input Markdown file")
    parser.add_argument("--verbose", action="store_true", help="Print individual similarities")
    args = parser.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    test_pairs = load_test_pairs_from_markdown(args.input_file)

    if not test_pairs:
        print("No Ground Truth / Retrieved Answer pairs found.")
        return

    gold_texts = [gt for gt, _ in test_pairs]
    retrieved_texts = [rt for _, rt in test_pairs]

    rs_score = retrieval_similarity(retrieved_texts, gold_texts, model, args.verbose) * 100.0
    print(f"Average Retrieval Similarity (R-S): {rs_score:.4f} %")

if __name__ == "__main__":
    main()
