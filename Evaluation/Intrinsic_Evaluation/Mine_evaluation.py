import json
import sys
import os
from typing import Literal

# Add the src directory to Python path to import from source code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kg_gen.kg_gen import KGGen
import networkx as nx
import numpy as np
from openai import OpenAI
from datasets import load_dataset

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def gpt_evaluate_response(correct_answer: str, context: str) -> int:
    # TODO: migrate to dspy, use dspy and gpt-5-nano or smth similar
    prompt = f"""
    Context:
    {context}

    Correct Answer:
    {correct_answer}

    Task:
    Determine whether the context contains the information stated in the correct answer. \\
    Respond with "1" if yes, and "0" if no. Do not provide any explanation, just the number.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an evaluator that checks if the correct answer can be deduced from the information in the context.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1,
        temperature=0.0,
    )
    return int(response.choices[0].message.content.strip())


def evaluate_accuracy(
    kggen: KGGen,
    queries: list[dict],
    node_embeddings: dict[str, np.ndarray],
    graph: nx.DiGraph,
    output_file: str,
):
    print(
        f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
    )
    correct = 0
    results = []

    for query in queries:
        *_, context_text = kggen.retrieve(query, node_embeddings, graph)
        evaluation = gpt_evaluate_response(query, context_text)
        result = {
            "correct_answer": query,
            "retrieved_context": context_text,
            "evaluation": evaluation,
        }
        results.append(result)
        correct += evaluation

    accuracy = correct / len(queries)
    results.append({"accuracy": f"{accuracy * 100:.2f}%"})

    # Save results to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


def main(evaluation_model: Literal["kggen", "graphrag", "openie"] = "kggen"):
    # Load data from Hugging Face (with local fallback)
    dataset = load_dataset("josancamon/kg-gen-MINE-evaluation-dataset")["train"]
    queries = [item["generated_queries"] for item in dataset.to_list()]
    if evaluation_model == "kggen":
        kg_data = [item["kggen"] for item in dataset.to_list()]
    elif evaluation_model == "graphrag":
        kg_data = [item["graphrag_kg"] for item in dataset.to_list()]
    elif evaluation_model == "openie":
        kg_data = [item["openie_kg"] for item in dataset.to_list()]

    kggen = KGGen(retrieval_model="all-MiniLM-L6-v2")
    valid_pairs = [
        (kg, queries) for kg, queries in zip(kg_data, queries) if kg is not None
    ]

    for i, (kg, queries) in enumerate(valid_pairs):
        output_file = f"experiments/MINE/results/{evaluation_model}/results_{i}.json"
        try:
            nxGraph = kggen.to_nx(kggen.from_dict(kg))
            node_embeddings, _ = kggen.generate_embeddings(nxGraph)
            evaluate_accuracy(
                kggen,
                queries,
                node_embeddings,
                nxGraph,
                output_file,
            )
        except Exception as e:
            print(f"Error processing file {output_file}: {str(e)}, skipping...")


if __name__ == "__main__":
    main()
