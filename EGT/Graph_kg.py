import argparse
import re
import os
import json
import numpy as np
import pickle
import joblib
import networkx as nx
from collections import defaultdict
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def build_context_documents(triplets):
    combined_text = "\n".join(triplets)
    return [Document(text=combined_text)]

def answer_question_with_llm(query_text, docs, llm, embedding_model):
    custom_prompt = PromptTemplate("""
You are an expert analyst given a set of factual triplets extracted from reliable sources.
Your task is to carefully analyze these facts and provide a clear, concise, short to the point answer to the question.
Answer the question as factual type, just the fact, with no description.

Factual context from multi-approach graph traversal:
{context_str}

Question: {query_str}
Answer: """)
    
    index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model)
    engine = index.as_query_engine(llm=llm, text_qa_template=custom_prompt)
    return engine.query(query_text)

def read_sample_file(file_path):
    qa_pairs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    current_question = None
    current_answer = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('**Question:**'):
            if current_question and current_answer:
                qa_pairs.append((current_question, current_answer))
            current_question = line.replace('**Question:**', '').strip()
            current_answer = None
            
        elif line.startswith('**Answer:**'):
            current_answer = line.replace('**Answer:**', '').strip()
    
    if current_question and current_answer:
        qa_pairs.append((current_question, current_answer))
    
    return qa_pairs

def save_to_markdown(queries, ground_truths, results, output_file):
    output_content = "# Retrieval Results\n\n"
    output_content += "---\n\n"
    
    for i, (query, truth) in enumerate(zip(queries, ground_truths), 1):
        output_content += f"\n### Pair {i}\n"
        output_content += f"**Question:** {query}\n"
        output_content += f"**Ground Truth:** {truth}\n"
        output_content += f"**Retrieved Answer:** {results.get(query, 'No answer available')}\n\n"
        output_content += "---\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)

class NetworkXEnhancedChunkHybridRetriever:
    def __init__(self, max_triplets_per_hop=30, max_hop_depth=4):
        self.max_triplets_per_hop = max_triplets_per_hop
        self.max_hop_depth = max_hop_depth
        self.chunk_embeddings = {}
        self.chunk_ids = []
        self.embedding_matrix = None
        self.chunk_metadata = {}
        self.llm = None
        
        self._load_graph_data()
        self._load_chunk_data()
        self._load_tfidf_data()
        self._prepare_embeddings()

    def set_llm(self, llm):
        self.llm = llm

    def _load_graph_data(self, graph_file):
        try:
            with open(graph_file, "rb") as f:
                self.G = pickle.load(f)
        except Exception:
            self.G = nx.MultiDiGraph()

    def _load_chunk_data(self, chunk_file):
        try:
            with open(chunk_file, "rb") as f:
                data = pickle.load(f)
                self.chunk_triplet_mapping = data["chunk_triplet_mapping"]
                self.chunk_embeddings = data["chunk_embeddings"]
        except Exception:
            self.chunk_triplet_mapping = {}
            self.chunk_embeddings = {}

    def _load_tfidf_data(self, tfidf_file):
        try:
            tfidf_data = joblib.load(tfidf_file)
            self.vectorizer = tfidf_data["vectorizer"]
            self.tfidf_matrix = tfidf_data["tfidf_matrix"]
            self.tfidf_entity_list = tfidf_data["entity_list"]
        except Exception:
            self.vectorizer = None
            self.tfidf_matrix = None
            self.tfidf_entity_list = []

    def _prepare_embeddings(self):
        try:
            self.chunk_ids = list(self.chunk_embeddings.keys())
            self.chunk_metadata = {}
            
            for chunk_id in self.chunk_ids:
                chunk_info = self.chunk_triplet_mapping[chunk_id]
                self.chunk_metadata[chunk_id] = {
                    'triplet_count': chunk_info['triplet_count'],
                    'file_id': chunk_info['file_id']
                }
            
            if self.chunk_embeddings:
                embeddings_list = []
                for chunk_id in self.chunk_ids:
                    emb = self.chunk_embeddings[chunk_id]
                    if isinstance(emb, list):
                        emb = np.array(emb, dtype=np.float32)
                    embeddings_list.append(emb)
                
                self.embedding_matrix = np.vstack(embeddings_list)
            else:
                self.embedding_matrix = np.empty((0, 1024), dtype=np.float32)
                
        except Exception:
            self.embedding_matrix = np.empty((0, 1024), dtype=np.float32)

    def _select_chunks(self, query_text, top_k=5, embedding_model=None):
        if self.embedding_matrix.shape[0] == 0:
            return []
        
        query_emb = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
        query_emb = query_emb.reshape(1, -1)
        
        query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        embeddings_norm = self.embedding_matrix / np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        
        similarities = cosine_similarity(query_norm, embeddings_norm)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        selected_chunks = []
        for idx in top_indices:
            chunk_id = self.chunk_ids[idx]
            selected_chunks.append(chunk_id)
        
        return selected_chunks

    def _extract_keywords(self, query_text):
        stop_words = {
            'what', 'did', 'how', 'the', 'and', 'to', 'between', 'it', 'they', 
            'do', 'is', 'was', 'were', 'a', 'an', 'for', 'that', 'pay', 'amount'
        }
        words = re.findall(r'\b\w+\b', query_text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords

    def _vector_search_within_chunks(self, query_text, selected_chunks, top_k=5, embedding_model=None):
        if self.G.number_of_nodes() == 0:
            return []
        
        query_vector = np.array(embedding_model.get_text_embedding(query_text), dtype=np.float32)
        query_vector = query_vector.reshape(1, -1)
        query_norm = query_vector / np.linalg.norm(query_vector)
        
        entity_scores = []
        
        for node in self.G.nodes(data=True):
            entity_name = node[0]
            entity_data = node[1]
            
            entity_chunk_ids = set(entity_data.get('chunk_ids', []))
            if not entity_chunk_ids.intersection(set(selected_chunks)):
                continue
            
            entity_embedding = entity_data.get('embedding')
            if entity_embedding is None:
                continue
            
            if isinstance(entity_embedding, list):
                entity_embedding = np.array(entity_embedding, dtype=np.float32)
            
            entity_embedding = entity_embedding.reshape(1, -1)
            entity_norm = entity_embedding / np.linalg.norm(entity_embedding)
            
            similarity = cosine_similarity(query_norm, entity_norm)[0][0]
            entity_scores.append((entity_name, similarity, list(entity_chunk_ids)))
        
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        return entity_scores[:top_k]

    def _bm25_search_within_chunks(self, query_text, selected_chunks, top_k=5):
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []
        
        keywords = self._extract_keywords(query_text)
        if not keywords:
            return []
        
        query_vector = self.vectorizer.transform([" ".join(keywords)])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        entity_scores = []
        for i, entity_name in enumerate(self.tfidf_entity_list):
            if entity_name in self.G.nodes:
                entity_data = self.G.nodes[entity_name]
                entity_chunk_ids = set(entity_data.get('chunk_ids', []))
                
                if entity_chunk_ids.intersection(set(selected_chunks)):
                    score = similarities[i]
                    entity_scores.append((entity_name, score, list(entity_chunk_ids)))
        
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        return entity_scores[:top_k]

    def _get_query_relevant_seed_entities(self, query_text, selected_chunks, max_entities=10, embedding_model=None):
        vector_results = self._vector_search_within_chunks(query_text, selected_chunks, top_k=8, embedding_model=embedding_model)
        bm25_results = self._bm25_search_within_chunks(query_text, selected_chunks, top_k=8)
        
        vector_entities = [name for name, score, chunk_ids in vector_results]
        bm25_entities = [name for name, score, chunk_ids in bm25_results]
        
        combined_entities = list(dict.fromkeys(vector_entities + bm25_entities))[:max_entities]
        
        return combined_entities, vector_entities, bm25_entities

    def _multi_hop_traversal_from_seeds(self, seed_entities, selected_chunks, hop_depth):
        all_paths = []
        
        for entity_name in seed_entities:
            if entity_name not in self.G.nodes:
                continue
            
            current_entities = {entity_name}
            visited_edges = set()
            
            for current_hop in range(1, hop_depth + 1):
                next_entities = set()
                hop_paths = []
                
                for current_entity in current_entities:
                    if len(hop_paths) >= self.max_triplets_per_hop:
                        break
                    
                    for neighbor in self.G.neighbors(current_entity):
                        for edge_key, edge_data in self.G[current_entity][neighbor].items():
                            edge_id = (current_entity, neighbor, edge_key)
                            if edge_id in visited_edges:
                                continue
                            
                            visited_edges.add(edge_id)
                            next_entities.add(neighbor)
                            
                            chunk_relevance_weight = 2.0 if edge_data.get('chunk_id') in selected_chunks else 1.0
                            
                            path_info = {
                                'subject': current_entity,
                                'predicate': edge_data.get('predicate', 'UNKNOWN'),
                                'object': neighbor,
                                'hop_distance': current_hop,
                                'chunk_id': edge_data.get('chunk_id', 'N/A'),
                                'chunk_relevance_weight': chunk_relevance_weight,
                                'source_type': 'multihop',
                                'seed_entity': entity_name
                            }
                            hop_paths.append(path_info)
                            
                            if len(hop_paths) >= self.max_triplets_per_hop:
                                break
                
                all_paths.extend(hop_paths)
                current_entities = next_entities
        
        return all_paths

    def get_contextual_subgraph(self, start_entities, selected_chunks, depth=2):
        subgraph_triplets = []
        entities_in_subgraph = set(start_entities)
        
        for current_depth in range(1, depth + 1):
            entity_list = list(entities_in_subgraph) if current_depth > 1 else list(start_entities)
            
            for current_entity in entity_list[:10]:
                if current_entity not in self.G.nodes:
                    continue
                
                triplet_count = 0
                for neighbor in self.G.neighbors(current_entity):
                    if triplet_count >= self.max_triplets_per_hop:
                        break
                    
                    entities_in_subgraph.add(neighbor)
                    
                    for edge_key, edge_data in self.G[current_entity][neighbor].items():
                        chunk_relevance_weight = 2.0 if edge_data.get('chunk_id') in selected_chunks else 1.0
                        
                        triplet = {
                            'subject': current_entity,
                            'predicate': edge_data.get('predicate', 'UNKNOWN'),
                            'object': neighbor,
                            'hop_distance': current_depth,
                            'chunk_id': edge_data.get('chunk_id', 'N/A'),
                            'chunk_relevance_weight': chunk_relevance_weight,
                            'source_type': 'subgraph'
                        }
                        subgraph_triplets.append(triplet)
                        triplet_count += 1
        
        return subgraph_triplets

    def get_reasoning_paths(self, query_text, start_entities, selected_chunks, max_path_length=3, embedding_model=None):
        reasoning_triplets = []
        
        vector_targets = self._vector_search_within_chunks(query_text, selected_chunks, top_k=3, embedding_model=embedding_model)
        keyword_targets = self._bm25_search_within_chunks(query_text, selected_chunks, top_k=3)
        
        target_entities = [name for name, _, _ in vector_targets + keyword_targets][:3]
        
        for start in start_entities[:3]:
            for target in target_entities:
                if start == target or start not in self.G.nodes or target not in self.G.nodes:
                    continue
                
                try:
                    if nx.has_path(self.G, start, target):
                        path = nx.shortest_path(self.G, start, target, weight=None)
                        
                        if len(path) > max_path_length + 1:
                            continue
                        
                        for i in range(len(path) - 1):
                            current_node = path[i]
                            next_node = path[i + 1]
                            
                            if next_node in self.G[current_node]:
                                edge_data = list(self.G[current_node][next_node].values())[0]
                                
                                chunk_relevance_weight = 2.0 if edge_data.get('chunk_id') in selected_chunks else 1.0
                                
                                path_info = {
                                    'subject': current_node,
                                    'predicate': edge_data.get('predicate', 'UNKNOWN'),
                                    'object': next_node,
                                    'hop_distance': len(path) - 1,
                                    'chunk_id': edge_data.get('chunk_id', 'N/A'),
                                    'chunk_relevance_weight': chunk_relevance_weight,
                                    'source_type': 'reasoning_path'
                                }
                                reasoning_triplets.append(path_info)
                                
                except nx.NetworkXNoPath:
                    continue
                except Exception:
                    continue
        
        return reasoning_triplets

    def _normalize_triplet_for_deduplication(self, subject, predicate, obj):
        entities = sorted([subject, obj])
        return (entities[0], predicate, entities[1])

    def _rerank_triplets_with_cross_encoder(self, query_text, triplets, top_k=50, cross_encoder=None):
        if not triplets:
            return []
        
        pairs = [(query_text, triplet) for triplet in triplets]
        scores = cross_encoder.predict(pairs)
        
        scored_triplets = sorted(zip(triplets, scores), key=lambda x: x[1], reverse=True)
        
        return [triplet for triplet, _ in scored_triplets[:top_k]]

    def _deduplicate_triplets(self, triplets):
        seen = set()
        unique_triplets = []
        
        for triplet in triplets:
            core_triplet = triplet.split(') [')[0] + ')'
            if core_triplet not in seen:
                seen.add(core_triplet)
                unique_triplets.append(triplet)
        
        return unique_triplets

    def _single_step_retrieval(self, query_text, hop_depth=None, embedding_model=None, cross_encoder=None):
        hop_depth = hop_depth or self.max_hop_depth
        
        if self.embedding_matrix.shape[0] == 0:
            return [], []
        
        selected_chunks = self._select_chunks(query_text, top_k=5, embedding_model=embedding_model)
        
        seed_entities, vector_entities, bm25_entities = self._get_query_relevant_seed_entities(
            query_text, selected_chunks, max_entities=12, embedding_model=embedding_model
        )
        
        if not seed_entities:
            return [], []
        
        multihop_paths = self._multi_hop_traversal_from_seeds(seed_entities, selected_chunks, hop_depth)
        subgraph_paths = self.get_contextual_subgraph(seed_entities, selected_chunks, depth=2)
        reasoning_paths = self.get_reasoning_paths(query_text, seed_entities, selected_chunks, max_path_length=3, embedding_model=embedding_model)
        all_path_info = multihop_paths + subgraph_paths + reasoning_paths
        
        seen_normalized = set()
        triplets = []
        
        for path_info in all_path_info:
            normalized_key = self._normalize_triplet_for_deduplication(
                path_info['subject'], 
                path_info['predicate'], 
                path_info['object']
            )
            
            if normalized_key not in seen_normalized:
                seen_normalized.add(normalized_key)
                
                chunk_relevance = "★" if path_info.get('chunk_relevance_weight', 1.0) > 1.0 else ""
                triplet_str = (
                    f"({path_info['subject']} --[{path_info['predicate']}]-> {path_info['object']}) "
                    f"[Chunk-{path_info.get('chunk_id', 'N/A')}{chunk_relevance}, "
                    f"Hop-{path_info['hop_distance']}, {path_info['source_type']}]"
                )
                triplets.append(triplet_str)
        
        ranked_triplets = self._rerank_triplets_with_cross_encoder(query_text, triplets, top_k=50, cross_encoder=cross_encoder)
        
        return triplets, ranked_triplets

    def get_related_triplets(self, query_text, hop_depth=None, embedding_model=None, cross_encoder=None):
        return self._single_step_retrieval(query_text, hop_depth, embedding_model, cross_encoder)

def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Question Answering Retriever")
    parser.add_argument("--qa-file", default="questions.md", help="Input QA Markdown file")
    parser.add_argument("--output-file", default="output.md", help="Output results Markdown file")
    parser.add_argument("--graph-file", default="knowledge_graph.pickle", help="Input graph pickle file")
    parser.add_argument("--chunk-file", default="chunk_data.pickle", help="Input chunk data pickle file")
    parser.add_argument("--tfidf-file", default="tfidf_data.joblib", help="Input TF-IDF joblib file")
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5", help="Embedding model name")
    parser.add_argument("--llm-model", default="qwen2.5:14b", help="Ollama LLM model")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2", help="Cross-encoder model")
    args = parser.parse_args()
    
    embedding = HuggingFaceEmbedding(model_name=args.embedding_model)
    llm = Ollama(model=args.llm_model, base_url="http://localhost:11434", request_timeout=60.0)
    cross_encoder = CrossEncoder(args.cross_encoder_model)
    
    qa_pairs = read_sample_file(args.qa_file)
    
    if not qa_pairs:
        return
        
    queries = [q for q, _ in qa_pairs]
    ground_truths = [a for _, a in qa_pairs]
    
    retriever = NetworkXEnhancedChunkHybridRetriever(max_triplets_per_hop=25)
    retriever._load_graph_data(args.graph_file)
    retriever._load_chunk_data(args.chunk_file)
    retriever._load_tfidf_data(args.tfidf_file)
    retriever.set_llm(llm)
    results = {}
    
    try:
        for user_query in queries:
            raw_triplets, ranked_triplets = retriever.get_related_triplets(user_query, embedding_model=embedding, cross_encoder=cross_encoder)
            
            answer = "No answer available"
            if ranked_triplets:
                docs = build_context_documents(ranked_triplets)
                answer = str(answer_question_with_llm(user_query, docs, llm, embedding))
            
            results[user_query] = answer
        
        save_to_markdown(queries, ground_truths, results, args.output_file)
        
    except Exception:
        pass

if __name__ == "__main__":
    main()
