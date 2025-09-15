# RAG Experiments

This repository is a sandbox for experimenting with **Retrieval-Augmented Generation (RAG)** techniques.
The main goal is to explore different retrieval strategies, indexing methods, and generation pipelines to improve **accuracy, reliability, and efficiency** in knowledge-augmented systems.

## Goals

* Test various retrieval approaches (dense, sparse, hybrid).
* Compare algorithms and techniques for re-ranking, query expansion, and context selection.
* Evaluate the impact of retrieval strategies on answer quality, coherence, and computational cost.

## Repository Structure

* `notebooks/` → Exploratory tests and quick prototypes.
* `src/` → Core implementations of retrieval, indexing, and RAG utilities.
* `experiments/` → Scripts and configurations for running RAG experiments.
* `results/` → Metrics, logs, and analysis of outcomes.

## Status

This project is in an exploratory phase. Initial results will focus on **comparative testing of retrieval and generation pipelines**, with special emphasis on how retrieval strategies influence the performance of RAG systems.

## Future Work

* Integrate standard RAG benchmarks (e.g., Natural Questions, HotpotQA).
* Explore lightweight fine-tuning methods (LoRA, adapters, etc.) to adapt generation models for retrieval-augmented tasks.
* Analyze trade-offs between retrieval complexity, latency, and answer quality.
  
📌 Work in progress.
