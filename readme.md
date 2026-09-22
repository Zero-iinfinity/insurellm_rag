# InsureLLM RAG Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-green.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Chroma-VectorStore-orange.svg)](https://www.trychroma.com/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-red.svg)](https://gradio.app/)

An end-to-end Retrieval-Augmented Generation (RAG) system built for **InsureLLM**, a fictional insurance technology company used as the knowledge domain. The project ingests internal company documents, embeds them locally with a HuggingFace sentence-transformer model, retrieves relevant context with ChromaDB, and generates grounded answers through an LLM. It also includes a self-contained retrieval + generation evaluation routine.

The current implementation lives entirely in a single notebook, `rag_pipeline.ipynb`, organized into four sections described below.

## Key Features

- **Document Ingestion:** Loads every Markdown file under `knowledge-base/<category>/`, tags each document with its folder name as `doc_type` metadata, and splits it into overlapping chunks with `RecursiveCharacterTextSplitter`.
- **Local Embeddings + Vector Store:** Uses the lightweight `all-MiniLM-L6-v2` HuggingFace embedding model and persists vectors to a local Chroma database (`vector_db/`), so no embedding API key is required.
- **RAG Question Answering:** Retrieves the top-*k* relevant chunks for a query and feeds them to an LLM (`gpt-4.1-nano` by default) with a system prompt that keeps answers grounded in the retrieved context.
- **Gradio Chat UI:** A one-line `gr.ChatInterface` wrapper turns `answer_question` into a browser-based chat app.
- **Evaluation Suite:** Combines a retrieval metric (Mean Reciprocal Rank) with an LLM-as-a-judge that scores generated answers against a reference answer on accuracy, completeness, and relevance.

## How the Notebook Is Organized

1. **Section 1 — Ingestion (`build_database`):** Reads Markdown files from `knowledge-base/` → chunks them → embeds them → stores them in a persistent Chroma collection. Run once, or whenever the source documents change.
2. **Section 2 — Retrieval & Generation (`answer_question`):** Takes a question → retrieves matching chunks from Chroma → builds a system prompt with that context → calls the LLM → returns the answer and the retrieved documents.
3. **Section 3 — Gradio Interface (`chat_handler`):** Wraps `answer_question` so it can be launched as an interactive chat app with `gr.ChatInterface(chat_handler).launch()`.
4. **Section 4 — Evaluation (`evaluate_pipeline`):** Runs a question through the pipeline, scores retrieval quality, and asks the LLM to grade the generated answer against a reference.

## Evaluation Methodology

The notebook evaluates two separate parts of the pipeline rather than judging the final answer alone:

**1. Retrieval quality — Mean Reciprocal Rank (MRR)**
For a given question, you supply a list of expected keywords (e.g. names, terms that should appear in a correct source chunk). `calculate_mrr` scans the retrieved chunks in ranked order and records `1 / rank` for the first chunk that contains each keyword, or `0` if it never appears. Averaging this across keywords gives a score between 0 and 1 that rewards the retriever for surfacing the right chunk near the top, rather than merely somewhere in the result set.

**2. Generation quality — LLM-as-a-judge**
The generated answer and a human-written reference answer are both handed to the LLM with instructions to score **Accuracy**, **Completeness**, and **Relevance** on a 1-5 scale and return the result as JSON. This avoids needing exact-match or ROUGE-style string comparison, which tends to penalize correct answers that are just phrased differently from the reference.

`evaluate_pipeline` returns both scores together, e.g.:

```json
{
  "retrieval": { "mrr": 0.5, "keywords_found": 2, "total_keywords": 3 },
  "generation": { "accuracy": 5, "completeness": 4, "relevance": 5, "feedback": "..." }
}
```

Note: this notebook currently implements MRR only. If you want a second retrieval metric such as nDCG (useful to also credit lower-ranked but still relevant chunks), see the enhancement ideas below.

## Setup

```bash
pip install langchain langchain-chroma langchain-huggingface langchain-community langchain-text-splitters langchain-openai gradio python-dotenv sentence-transformers
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Place your source Markdown files under `knowledge-base/<category>/`, e.g. `knowledge-base/hr/`, `knowledge-base/products/`. Then run the ingestion cell once (uncomment `vectorstore = build_database()`).

## Using a Free OpenRouter API Key Instead of OpenAI

The notebook currently calls OpenAI directly through `ChatOpenAI(model_name=MODEL)`, which needs a paid `OPENAI_API_KEY`. OpenRouter exposes an OpenAI-compatible endpoint with several free-tier models, so you can swap providers with a small change and no LangChain code rewrite.

1. Get a free API key at `https://openrouter.ai/` (Sign in → Keys). Add it to `.env`:
   ```
   OPENROUTER_API_KEY=sk-or-...
   ```
2. Pick a currently available free model from `https://openrouter.ai/models?max_price=0` (free-tier model names and availability change over time, so check there rather than hardcoding one long-term).
3. Update the LLM initialization in Section 2:
   ```python
   import os
   from langchain_openai import ChatOpenAI

   MODEL = "meta-llama/llama-3.1-8b-instruct:free"  # or any current free model id

   llm = ChatOpenAI(
       model_name=MODEL,
       openai_api_key=os.getenv("OPENROUTER_API_KEY"),
       openai_api_base="https://openrouter.ai/api/v1",
       temperature=0,
   )
   ```
4. Everything else (`answer_question`, `chat_handler`, `evaluate_pipeline`) works unchanged, since they only depend on the `llm` object's `.invoke()` interface, not on which provider it points to.

Free OpenRouter models typically have lower rate limits and weaker instruction-following than `gpt-4.1-nano`, so expect the LLM-as-a-judge JSON output to occasionally need the existing try/except fallback in `evaluate_pipeline`.

## Repository Structure 

```text
insurellm-rag/
├── knowledge-base/          # Markdown source documents (create your own categories)
├── vector_db/                # Persisted Chroma vector store (generated by build_database)
├── rag_pipeline.ipynb        # Ingestion, RAG, Gradio UI, and evaluation - all in one notebook
├── .env                       # OPENAI_API_KEY or OPENROUTER_API_KEY
└── README.md
```
