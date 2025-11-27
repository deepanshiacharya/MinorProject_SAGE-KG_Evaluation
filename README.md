# SAGE-KG: A Comprehensive Evaluation of Agentic Knowledge Graph Generation

This repository contains the official implementation, evaluation pipeline, and results for SAGE-KG, an agentic, schema-free, open-source Knowledge Graph (KG) generation framework designed for robust factual extraction and multi-hop reasoning.

SAGE-KG uses small open-source LLMs (Qwen2.5 series) and a three-stage agentic workflow to convert unstructured text into high-quality (subject, predicate, object) triplets without supervision or predefined schemas.
It includes a powerful retrieval module—Enhanced Graph Traversal (EGT)—and a grounded answer generator that significantly improves multi-hop QA performance.

## Key Features
Agentic KG Construction
Three specialized agents—Fact Extractor, Schema Planner, Triplet Creator—work together to produce precise and complete triplets.

Schema-Free & Supervision-Free
No fixed ontology or fine-tuning required.
Optimized Multi-Hop Retrieval (EGT)
Combining:
1. semantic chunk retrieval
2. hybrid seed entity selection (dense + BM25)
3. bounded multi-hop traversal
4. reasoning path discovery
5. cross-encoder reranking

Superior Intrinsic & Extrinsic Performance
Outperforms OpenIE, GraphRAG, KGGen, and Zero-shot GraphRAG on:
1. HotpotQA
2. MuSiQue
3. 2WikiMultiHopQA

State-of-the-art MINE Benchmark Score
SAGE-KG-14B achieves 92.95% factual capture, the highest among all evaluated systems
SAGE-KG converts text into high-fidelity knowledge graphs using:
Agentic extraction
Dynamic schema induction
Triplet creation
Evidence-grounded retrieval
LLM-guided answer synthesis
All components are built using Qwen2.5 models (3B, 7B, 14B) running locally via Ollama (Q4_K_M quantization).

## The framework supports:
1. Automatic KG creation
2. Multi-hop question answering
3. Evidence tracing
4. Cross-encoder reranking for relevance


## Datasets
We use 500 samples each from:
HotpotQA	Sentence-level multi-hop QA
MuSiQue	Compositional multi-step QA
2WikiMultiHopQA	Cross-article reasoning
MINE	Manually verified factual benchmark
## Baselines
Triplet Extractors
Stanford OpenIE
Zero-Shot GraphRAG (Qwen2.5-14B)
KGGen (GPT-4o supervised workflow)

## Retrieval & QA Systems
Standard RAG (BGE-large)
Microsoft GraphRAG (Local & Global)
Think-on-Graph v1
SAGE-KG (EGT)

## Intrinsic Evaluation
Evaluated using GPT-4o-mini and Gemini 2.0 Flash as independent judges.
Metrics:
Precision (P)
Recall (R)
F1 (harmonic mean)

Key highlight:
SAGE-KG-14B achieves F1 ≈ 8.85–8.90, outperforming all unsupervised baselines and rivaling supervised KGGen.

## Extrinsic Evaluation
Metrics:
Exact Match (EM)
Semantic Relevance (SR)
Generation Evaluation
(Completeness, Accuracy, Knowledgeability, Relevance, Coherence)

## SAGE-KG-14B is the best-performing model across HotpotQA, MuSiQue, and 2WikiMultiHopQA.

It beats:
Microsoft GraphRAG
Standard RAG
KGGen
Zero-shot GraphRAG

Judge Agreement

Quadratic weighted Cohen’s Kappa:
0.9355 (HotpotQA & 2Wiki)
0.8773 (MuSiQue)

This indicates excellent inter-judge consistency.

## MINE Benchmark
Method	Score (%)
SAGE-KG (14B)	92.95
SAGE-KG (7B)	84.76
SAGE-KG (3B)	72.06
KGGen	66.07
Zero-shot GraphRAG	47.80
OpenIE	29.84

SAGE-KG sets a new state-of-the-art for factual completeness and correctness in unsupervised KG extraction.


## Work submitted by 
202418012 Yashraj Chudasama
202418015 Deepanshi Acharya
