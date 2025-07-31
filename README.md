OVERVIEW:
This project implements a semantic search engine that retrieves the top 100 most relevant documents for a given natural language query. Here's a breakdown of how it works:




LOGIC AND IMPLEMENTATION:
approch and logic:


1.Data Loading

2.Connects to a MongoDB database.

3.Fetches documents and combines text fields like title, description, and about into a single string per document.

4.Text Preprocessing

5.Cleans and standardizes the text (removing special characters, lowercasing, etc.).

6.Embedding Generation

7.Each document's text is passed through a pre-trained embedding model (e.g., OpenAI, Cohere, HuggingFace).

8.This produces a fixed-size embedding vector (e.g., 1024 dimensions).

9.All document embeddings are saved for later use.

10.Building the FAISS Index

11.FAISS (Facebook AI Similarity Search) is used to build a fast index of all document embeddings.

12.This allows efficient approximate nearest neighbor search.

13:Query Embedding and Retrieval

When a user submits a query:

The query is embedded using the same model.

The FAISS index is searched for the 100 nearest document embeddings.

The IDs of these top-100 documents are returned.

Evaluation

The returned document IDs are sent to an evaluation API.

The evaluation returns an overallScore (e.g., Recall@100).

These scores are collected and saved for 10 public queries.





