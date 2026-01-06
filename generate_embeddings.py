import os
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# Initialize Haystack with OpenAI for text embeddings
def initialize_haystack():
    embedder = OllamaTextEmbedder(model="mxbai-embed-large")
    return embedder


# Retrieve movie plots and titles from Neo4j
def retrieve_movie_plots():
    query = """
    MATCH (m:Movie)
    WHERE m.embedding IS NULL
    RETURN m.tmdbId AS tmdbId, m.title AS title, m.overview AS overview
    """
    with driver.session() as session:
        results = session.run(query)
        movies = [
            {
                "tmdbId": row["tmdbId"],
                "title": row["title"],
                "overview": row["overview"]
            }
            for row in results
        ]
    return movies


# Store the embedding in Neo4j (runs in the main thread)
def store_embedding_in_neo4j(tmdbId, embedding):
    query = """
    MATCH (m:Movie {tmdbId: $tmdbId})
    SET m.embedding = $embedding
    """
    with driver.session() as session:
        session.run(query, tmdbId=tmdbId, embedding=embedding)
    print(f"✅ Stored embedding for TMDB ID: {tmdbId}")


# Parallel embedding generation with ThreadPoolExecutor
def generate_and_store_embeddings(embedder, movies, max_workers=10):
    results_to_store = []

    def process_movie(movie):
        title = movie.get("title", "Unknown Title")
        overview = str(movie.get("overview", "")).strip()
        tmdbId = movie.get("tmdbId")

        if not overview:
            print(f"⚠️ Skipping {title} — No overview available.")
            return None

        try:
            print(f"🔄 Generating embedding for: {title}")
            embedding_result = embedder.run(overview)
            embedding = embedding_result.get("embedding")
            if embedding:
                return (tmdbId, embedding)
            else:
                print(f"❌ No embedding generated for: {title}")
        except Exception as e:
            print(f"❌ Error processing {title}: {e}")
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_movie, movie) for movie in movies]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results_to_store.append(result)

    # Store all embeddings after parallel processing
    for tmdbId, embedding in results_to_store:
        store_embedding_in_neo4j(tmdbId, embedding)


# Verify a few embeddings from Neo4j
def verify_embeddings():
    query = """
    MATCH (m:Movie)
    WHERE m.embedding IS NOT NULL
    RETURN m.title AS title, m.embedding AS embedding
    LIMIT 10
    """
    with driver.session() as session:
        results = session.run(query)
        for record in results:
            title = record["title"]
            embedding = np.array(record["embedding"])[:5]
            print(f"🎬 {title}: {embedding}...")

# Create or drop the vector index in Neo4j AuraDB
def create_or_reset_vector_index():
    with driver.session() as session:
        try:
            # Drop the existing vector index if it exists
            session.run("DROP INDEX overview_embeddings IF EXISTS ")
            print("Old index dropped")
        except:
            print("No index to drop")

        # Create a new vector index on the embedding property
        print("Creating new vector index")
        query_index = """
        CREATE VECTOR INDEX overview_embeddings IF NOT EXISTS
        FOR (m:Movie) ON (m.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1024,  
            `vector.similarity_function`: 'cosine'}}
        """
        session.run(query_index)
        print("Vector index created successfully")


# Main function
def main():
    embedder = initialize_haystack()
    movies = retrieve_movie_plots()

    if not movies:
        print("No movies found with missing embeddings.")
        return

    generate_and_store_embeddings(embedder, movies, max_workers=20)
    verify_embeddings()
    create_or_reset_vector_index()

if __name__ == "__main__":
    main()