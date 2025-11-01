---
title: RAG Concepts and Implementation
created: 2025-11-01
tags: [rag, embeddings, vector-search, ai]
---

# RAG Concepts and Implementation

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by providing them with relevant context retrieved from a knowledge base before generating a response.

### The RAG Pipeline

```
User Question
    ↓
Embed Question → Vector Search → Retrieve Top-K Chunks
                                        ↓
                                 Relevant Context
                                        ↓
                    LLM (with context) → Generate Answer
```

## Why RAG?

### Problems RAG Solves

1. **Hallucinations**: LLMs can make up facts. RAG grounds responses in real documents.
2. **Outdated Knowledge**: LLMs have a training cutoff date. RAG uses current documents.
3. **Domain-Specific Knowledge**: LLMs lack your personal notes and private information.
4. **Attribution**: RAG can cite sources for claims.

### Advantages Over Fine-Tuning

- **No retraining required**: Update knowledge by updating documents
- **Cost-effective**: No GPU time for training
- **Transparent**: You can see what context was used
- **Dynamic**: Add new information instantly

## Key Components

### 1. Document Chunking

Breaking documents into smaller pieces (chunks) for indexing and retrieval.

**Why chunk?**
- Embedding models have max input lengths
- Smaller chunks = more precise matching
- Easier to fit relevant context in LLM prompt

**Toronto AI approach:**
- Character-based chunking (2400 chars ≈ 600 tokens)
- 320 character overlap between chunks
- Preserve heading hierarchy for context

### 2. Embeddings

Converting text into high-dimensional vectors that capture semantic meaning.

**Properties of embeddings:**
- Similar meanings → similar vectors
- Can be compared using cosine similarity or L2 distance
- Model-specific dimensions (e.g., mxbai-embed-large uses 1024 dimensions)

**Example:**
```
"machine learning" → [0.23, -0.15, 0.89, ..., 0.44]  (1024 numbers)
"artificial intelligence" → [0.21, -0.18, 0.91, ..., 0.48]  (similar!)
"cooking recipes" → [-0.62, 0.34, -0.12, ..., -0.33]  (different!)
```

### 3. Vector Storage

Efficiently storing and searching millions of vectors.

**FAISS (Facebook AI Similarity Search):**
- Fast approximate nearest neighbor search
- Multiple index types (flat, IVF, HNSW)
- Toronto AI uses IndexFlatL2 (exact search, simple, works for <100k vectors)

### 4. Retrieval

Finding the most relevant chunks for a query.

**Process:**
1. Embed the user's question
2. Search FAISS index for top-K similar vectors
3. Retrieve corresponding text chunks from SQLite
4. Rank by relevance (optional reranking)
5. Return top results

### 5. Context Injection

Adding retrieved context to the LLM prompt.

**Example prompt structure:**
```
System: You are a helpful assistant. Use the following context to answer questions.

Context:
[Chunk 1 from notes/project.md]
[Chunk 2 from notes/architecture.md]
[Chunk 3 from notes/rag.md]

User: How does RAG work?