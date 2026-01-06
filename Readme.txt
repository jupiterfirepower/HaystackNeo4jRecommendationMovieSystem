docker compose up -d
or docker compose -f path_to_fole up -d

->/neo4j_home directory structure
    /conf
    /data
    /import
    /logs
    /plugins

Download file and copy to ->/neo4j_home/plugins
wget https://github.com/neo4j/apoc/releases/download/2025.11.2/apoc-2025.11.2-core.jar
# wget https://github.com/neo4j/apoc/releases/download/5.26.19/apoc-5.26.19-core.jar

Copy ./data/normalized_data to  /neo4j_home/import/normalized_data

Download Neo4jDesktop from https://neo4j.com/docs/desktop/current/installation/    AppImage for Linux

clone github repository

git clone https://github.com/jupiterfirepower/HaystackNeo4jRecommendationMovieSystem.git

uv venv
source .venv/bin/activate  # On macOS/Linux
uv add -r requirements.txt
uv sync

run graph_build.py from Dev IDE or terminal (Developing and tested in Python3.14)
uv run graph_build.py

Install ollama and pull models
curl -fsSL https://ollama.com/install.sh | sh
wget https://ollama.com/install.sh
./install.sh
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64/cuda-fedora42.repo
sudo dnf update
# for Nvidia
sudo dnf install cuda-toolkit

sudo systemctl status ollama.service

# ollama pull deepseek-r1:1.5b
# ollama serve
# ollama stop
# ollama stop deepseek-r1:1.5b
ollama pull mxbai-embed-large
# ollama pull nomic-embed-text

uv run generate_embeddings.py
It's may continue 1-1.5 hours or less depend on hardware(processor etc)

uv run graph_rag_recommend_search.py





