import os

import openai
from dotenv import load_dotenv
from haystack import Document, Pipeline
#from haystack.components.embedders import OllamaTextEmbedder #, OpenAITextEmbedder
from haystack.utils.auth import Secret
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

# from haystack.schema import Filter
from neo4j import GraphDatabase
from neo4j_haystack import (
    Neo4jClientConfig,
    Neo4jDocumentStore,
    Neo4jEmbeddingRetriever,
)
import gradio as gr

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# OpenAI API key
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#openai.api_key = OPENAI_API_KEY

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# Step 1: Fetch related movies using multi-hop reasoning from Neo4j
def fetch_multi_hop_related_movies(title):
    query = """
    MATCH (m:Movie {title: $title})<-[:DIRECTED]-(d:Director)-[:DIRECTED]->(related:Movie)
    RETURN related.title AS related_movie, related.overview AS overview
    """
    with driver.session() as session:
        result = session.run(query, title=title)
        documents = [
            Document(
                content=record["overview"], meta={"title": record["related_movie"]}
            )
            for record in result
        ]
    return documents


# Initialize Neo4j Document Store and Haystack Components
client_config = Neo4jClientConfig(
    url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD
)

document_store = Neo4jDocumentStore(
    client_config=client_config,
    index="overview_embeddings",  # The name of the Vector Index in Neo4j
    node_label="Movie",  # Label to Neo4j nodes which store Documents
    embedding_dim=1024,  # Dimension of embeddings (for OpenAI ADA it's 1536)
    # The mxbai-embed-large model available through Ollama produces embeddings with a dimension of 1024.
    embedding_field="embedding",
    similarity="cosine",  # Cosine similarity for vector search
    verify_connectivity=True,
)

# Initialize Haystack's OpenAITextEmbedder for creating embeddings
#text_embedder = OpenAITextEmbedder(
#    api_key=Secret.from_env_var("OPENAI_API_KEY"), model="text-embedding-ada-002"
#)

text_embedder = OllamaTextEmbedder(model="mxbai-embed-large")


# Step 2: Context-Aware Search with Multi-Hop Reasoning
def perform_semantic_search_with_multi_hop(query, movie_title, parts):
    # Fetch multi-hop related movies from Neo4j
    multi_hop_docs = fetch_multi_hop_related_movies(movie_title)

    if not multi_hop_docs:
        print(f"No related movies found for {movie_title}")
        parts.append(f"No related movies found for {movie_title}\n")
        return

    # Write these documents to the document store
    document_store.write_documents(multi_hop_docs)

    # Generate embedding for the search query (e.g., "time travel")
    query_embedding = text_embedder.run(query).get("embedding")

    if query_embedding is None:
        print("Query embedding not created successfully.")
        parts.append("Query embedding not created successfully.\n")
        return

    # Perform vector search only on the multi-hop related movies
    similar_docs = document_store.query_by_embedding(query_embedding, top_k=3)

    if not similar_docs:
        print("No similar documents found.")
        parts.append("No similar documents found.\n")
        return

    for doc in similar_docs:
        title = doc.meta.get("title", "N/A")
        overview = doc.meta.get("overview", "N/A")
        score = doc.score
        print(f"Title: {title}\nOverview: {overview}\nScore: {score:.2f}\n{'-' * 40}")
        parts.append(f"Title: {title}\nOverview: {overview}\nScore: {score:.2f}\n{'-' * 40}")
    print("\n\n")
    parts.append("\n\n")


# Step 3: Dynamic Filtering
def perform_filtered_search(query, parts):
    pipeline = Pipeline()
    text_embedder = OllamaTextEmbedder(model="mxbai-embed-large")
    pipeline.add_component("query_embedder", text_embedder)
    # pipeline.add_component("retriever", retriever)
    pipeline.add_component(
        "retriever", Neo4jEmbeddingRetriever(document_store=document_store)
    )
    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")

    result = pipeline.run(
        data={
            "query_embedder": {"text": query},
            "retriever": {
                "top_k": 5,
                "filters": {
                    "field": "release_date",
                    "operator": ">=",
                    "value": "1995-11-17",
                },
            },
        }
    )

    # Extracting documents from the retriever results
    documents = result["retriever"]["documents"]

    for doc in documents:
        # Extract title and overview from document metadata
        title = doc.meta.get("title", "N/A")
        overview = doc.meta.get("overview", "N/A")
        release_date = doc.meta.get("release_date", "N/A")

        # Extract score from the document (not from meta)
        score = getattr(doc, "score", None)

        # Format score if it exists, else show "N/A"
        score_display = f"{score:.2f}" if score is not None else "N/A"

        # Print the title, overview, and score (or N/A for missing score)
        print(
            f"Title: {title}\nOverview: {overview}\nReleased Date:{release_date}\nScore: {score_display}\n{'-' * 40}\n"
        )
        parts.append(f"Title: {title}\nOverview: {overview}\nReleased Date:{release_date}\nScore: {score_display}\n{'-' * 40}\n")


# Step 4: Optimized Search for Recommendations
def perform_optimized_search(query, top_k, parts):
    # Perform optimized search by adjusting top_k
    optimized_results = document_store.query_by_embedding(
        query_embedding=text_embedder.run(query).get("embedding"), top_k=top_k
    )

    for doc in optimized_results:
        title = doc.meta["title"]
        overview = doc.meta.get("overview", "N/A")
        print(f"Title: {title}\nOverview: {overview}\n{'-' * 40}")
        parts.append(f"Title: {title}\nOverview: {overview}\n{'-' * 40}\n")


# Main function to execute all use cases
def main():
    parts = []
    #movie_title = "Jurassic Park"
    movie_title = "Mission to Mir"
    #search_query = "Find movies about dinosaurs"
    search_query = "Find movies about cold-war"

    parts.append("=== Context-Aware Search with Multi-Hop Reasoning ===")
    print("=== Context-Aware Search with Multi-Hop Reasoning ===")
    perform_semantic_search_with_multi_hop(search_query, movie_title, parts)

    parts.append("=== Dynamic Filtered Search ===")
    print("=== Dynamic Filtered Search ===")
    #perform_filtered_search("Movies about space exploration", parts)
    perform_filtered_search(search_query, parts)

    print("=== Optimized Search for Recommendations ===")
    perform_optimized_search("Recommend movies about time travel", 10, parts)


from haystack.components.builders import PromptBuilder
# ... (import other necessary haystack components)

# Assume 'rag_pipeline' is your defined Haystack pipeline
# from your_haystack_script import rag_pipeline

def predict(movie_title_value,find_movie_value, recommend_movie_value):
    parts = []
    # movie_title = "Jurassic Park"
    movie_title = movie_title_value
    # search_query = "Find movies about dinosaurs"
    search_query = find_movie_value

    parts.append("=== Context-Aware Search with Multi-Hop Reasoning ===\n")
    perform_semantic_search_with_multi_hop(search_query, movie_title, parts)

    parts.append("=== Dynamic Filtered Search ===\n")
    perform_filtered_search(search_query, parts)

    parts.append("=== Optimized Search for Recommendations ===\n")
    perform_optimized_search(recommend_movie_value, 10, parts)

    final_string = "".join(parts)
    result = {"replies": [final_string]}
    return result["replies"][0]

# Create the Gradio interface
gr_face = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=1, placeholder="Movie title",value="Mission to Mir",label="Movie title"),
        gr.Textbox(lines=2, placeholder="Find movies about ...",value="Find movies about cold-war",label="Find"),
        gr.Textbox(lines=1, placeholder="Recommend movies about ...",value="Recommend movies about time travel",label="Recommend"),
    ],
    outputs= gr.Textbox(lines=20, placeholder="text"),
    title="Haystack RAG Movie Recommendation",
    description="Ask questions based on a custom document store using a Haystack pipeline."
)

if __name__ == "__main__":
    #main()
    gr_face.launch()
