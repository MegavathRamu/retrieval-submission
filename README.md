OVERVIEW:
This project implements a semantic search engine that retrieves the top 100 most relevant documents for a given natural language query. Here's a breakdown of how it works:


LOGIC AND IMPLEMENTATION:
approch and logic:


Data Loading

Connects to a MongoDB database.

Fetches documents and combines text fields like title, description, and about into a single string per document.

Text Preprocessing

Cleans and standardizes the text (removing special characters, lowercasing, etc.).

Embedding Generation

Each document's text is passed through a pre-trained embedding model (e.g., OpenAI, Cohere, HuggingFace).

This produces a fixed-size embedding vector (e.g., 1024 dimensions).

All document embeddings are saved for later use.

Building the FAISS Index

FAISS (Facebook AI Similarity Search) is used to build a fast index of all document embeddings.

This allows efficient approximate nearest neighbor search.

Query Embedding and Retrieval

When a user submits a query:

The query is embedded using the same model.

The FAISS index is searched for the 100 nearest document embeddings.

The IDs of these top-100 documents are returned.

Evaluation

The returned document IDs are sent to an evaluation API.

The evaluation returns an overallScore (e.g., Recall@100).

These scores are collected and saved for 10 public queries.




results:
