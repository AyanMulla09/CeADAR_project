
# CeADAR Research Agents Project

## Overview
This project is a modular research pipeline developed for CeADAR (Centre for Applied Data Analytics Research). It automates the process of searching, downloading, filtering, and analyzing research papers using a set of intelligent agents. The pipeline is designed for extensibility, reproducibility, and ease of use for data science research tasks.

## Features
- Automated search and download of research papers
- PDF extraction and caching
- Embedding and indexing for semantic search
- Full-text and metadata filtering
- Gap analysis and description generation
- Modular agent-based architecture
- CSV and JSON export of results

## Project Structure
```
agents/           # Core agent modules (search, filter, download, etc.)
cache/            # Temporary cache files
output/           # Final analysis and results (CSV, JSON)
pdf_cache/        # Downloaded PDF files
utils/            # Utility scripts for export and processing
tools/            # Research tools and helpers
main.py           # Entry point for running the pipeline
config.py         # Configuration settings
requirements.txt  # Python dependencies
README.md         # Project documentation
```


## Installation
1. Clone the repository:
  ```sh
  git clone https://github.com/AyanMulla09/CeADAR_project.git
  ```
2. Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```

## Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed for local LLM inference
- [Docker](https://www.docker.com/) for running ElasticSearch

## Setting Up ElasticSearch with Docker
To run ElasticSearch locally using Docker:
```sh
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.8.0
```
You can stop ElasticSearch with:
```sh
docker stop elasticsearch
```

## Setting Up Ollama and Loading Models
Install Ollama and pull your desired model (e.g., Llama3):
```sh
# Install Ollama (see https://ollama.ai/download)
ollama pull llama3
ollama serve
```
To use a different model, replace `llama3` with your preferred model name (e.g., `mistral`, `phi3`, etc.):
```sh
ollama pull mistral
```
Then update the model name in `config.py`:
```python
LLM_MODEL = "mistral"
```

## Usage
Run the main pipeline with a research topic:
```sh
python main.py "Your Research Topic Here"
```
Or run interactively:
```sh
python main.py
```

Configure your research topic, model, and other parameters in `config.py`.

### Example: Adjusting to Use Other Models
1. Pull the model with Ollama:
  ```sh
  ollama pull phi3
  ```
2. Update `LLM_MODEL` in `config.py`:
  ```python
  LLM_MODEL = "phi3"
  ```
3. Restart Ollama server if needed:
  ```sh
  ollama serve
  ```

### Example: Running ElasticSearch on a Different Port
Change the port in the Docker command and update `ELASTICSEARCH_HOST` in `config.py`:
```sh
docker run -d --name elasticsearch -p 9300:9200 -e "discovery.type=single-node" elasticsearch:8.8.0
```
```python
ELASTICSEARCH_HOST = "localhost:9300"
```

### Output
Results are exported to CSV and JSON formats in the `output/` folder.


## How It Works
- The orchestrator agent coordinates the workflow.
- Agents perform tasks such as searching for papers, downloading PDFs, extracting text, filtering, and generating descriptions.
- Results are exported to CSV and JSON formats in the `output/` folder.

## Troubleshooting
- If Ollama or ElasticSearch are not running, the pipeline will not work. Ensure both are started before running the code.
- For model errors, verify the model name in `config.py` matches the one pulled with Ollama.
- For ElasticSearch errors, check Docker container status and port configuration.

## Contributing
Feel free to fork the repository and submit pull requests for improvements or new features.

## License
This project is for academic and research purposes at CeADAR.

## Contact
For questions or collaboration, contact [Ayan Mulla](mailto:ayan.mulla09@gmail.com).
