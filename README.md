# ReACT AI Research Pipeline

A comprehensive research paper analysis pipeline using ReACT (Reasoning and Acting) agents and LangChain. This system automates the entire research paper discovery, filtering, analysis, and gap identification process.

## 🌟 Features

- **Multi-Agent Architecture**: Specialized ReACT agents for each stage of the research pipeline
- **Comprehensive Paper Discovery**: Intelligent search query generation and ArXiv integration
- **Advanced Filtering**: Two-stage filtering (abstract-level and full-text analysis)
- **Gap Analysis**: Automated identification of research gaps that papers address
- **RAG System**: ElasticSearch integration with semantic embeddings for retrieval
- **Parallel Processing**: Concurrent execution for improved performance
- **Comprehensive Logging**: Detailed logging and error handling

## 🏗️ Architecture

### Agents Overview

1. **ReACT Orchestrator**: Coordinates the entire pipeline execution
2. **Research Topic Agent**: Elaborates topics and generates search queries
3. **Paper Search Agent**: Searches ArXiv for relevant papers
4. **Filter Agent**: Performs initial relevance filtering on abstracts
5. **Description Gen Agent**: Generates detailed descriptions and categorizations
6. **Download/Extract Agent**: Downloads PDFs and extracts text content
7. **Full Text Filter Agent**: Performs detailed relevance analysis on full papers
8. **Gap Generation Agent**: Analyzes research gaps that papers address
9. **Embedding/Indexing Agent**: Indexes papers to ElasticSearch with embeddings

### Pipeline Flow

```
Research Topic → Query Generation → Paper Search → Abstract Filter → 
PDF Download → Text Extraction → Full-text Filter → Gap Analysis → 
ElasticSearch Indexing → RAG System Ready
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama** with Llama3.1 model installed
3. **ElasticSearch** running on localhost:9200 (optional, for RAG features)

### Installation

1. Clone or copy the project to your desired location
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and start Ollama with Llama3.1:
```bash
# Install Ollama (follow instructions for your OS)
ollama pull llama3.1
```

4. (Optional) Start ElasticSearch for RAG features:
```bash
# Using Docker
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.8.0
```

### Usage

#### Command Line Interface

```bash
# Run full pipeline
python main.py "Sustainable AI for Climate Change Mitigation"

# Interactive mode
python main.py
```

#### Programmatic Usage

```python
from react_agents import run_research_pipeline

# Run the complete pipeline
results = run_research_pipeline(
    topic="Machine Learning for Healthcare",
    max_per_query=5
)

# Access results
papers = results["pipeline_results"]["gap_generation_agent"]["papers_with_gap_analysis"]
```

## 📁 Project Structure

```
react_agents/
├── __init__.py                 # Package initialization
├── main.py                     # Main application entry point
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── agents/                     # ReACT agents
│   ├── __init__.py
│   ├── base_agent.py          # Base agent class
│   ├── orchestrator.py        # Pipeline orchestrator
│   ├── research_topic.py      # Topic analysis agent
│   ├── paper_search.py        # Paper search agent
│   ├── filter.py              # Abstract filtering agent
│   ├── description_gen.py     # Description generation agent
│   ├── download_extract.py    # PDF download/extract agent
│   ├── full_text_filter.py    # Full-text filtering agent
│   ├── gap_generation.py      # Gap analysis agent
│   └── embedding_indexing.py  # ElasticSearch indexing agent
├── tools/                      # LangChain tools
│   ├── __init__.py
│   └── research_tools.py      # Research-specific tools
└── utils/                      # Utility functions
    └── __init__.py
```

## ⚙️ Configuration

The system is configured through the `config.py` file. Key settings include:

```python
class Config:
    # LLM Configuration
    LLM_MODEL = "llama3.1"
    LLM_TEMPERATURE = 0.1
    
    # Processing Configuration
    MAX_CHUNK_WORDS = 1500
    OVERLAP_WORDS = 300
    NUM_THREADS = 4
    
    # ElasticSearch Configuration
    ELASTICSEARCH_HOST = "localhost:9200"
    ELASTICSEARCH_INDEX = "research_papers"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

## 🔧 Advanced Usage

### Running Individual Agents

```python
from react_agents.agents import create_research_topic_agent

# Create and use individual agents
topic_agent = create_research_topic_agent()
result = topic_agent.execute({
    "topic": "Quantum Machine Learning"
})
```

### Custom Agent Development

Extend the `BaseReACTAgent` class to create custom agents:

```python
from react_agents.agents.base_agent import BaseReACTAgent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

class CustomAgent(BaseReACTAgent):
    def __init__(self):
        tools = [
            Tool(
                name="custom_tool",
                description="Description of custom tool",
                func=self._custom_function
            )
        ]
        
        super().__init__(
            name="custom_agent",
            description="Custom agent description",
            tools=tools
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        # Define your custom prompt template
        pass
    
    def execute(self, input_data):
        # Implement your custom execution logic
        pass
```

## 📊 Output Formats

The pipeline generates several output files:

1. **Pipeline Results**: Complete execution results in JSON format
2. **Research Papers CSV**: Structured data with gap analysis
3. **Logs**: Detailed execution logs in `research_pipeline.log`

### Sample Output Structure

```json
{
  "status": "completed",
  "pipeline_results": {
    "research_topic_agent": {
      "original_topic": "AI for Healthcare",
      "topic_elaboration": "...",
      "search_queries": ["...", "..."]
    },
    "gap_generation_agent": {
      "papers_with_gap_analysis": [
        {
          "title": "Paper Title",
          "research_gap": "Gap description",
          "gap_direction": "Research direction",
          "gap_solution": "Solution approach"
        }
      ]
    }
  }
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Verify model is available: `ollama list`

2. **ElasticSearch Connection Error**
   - Check ElasticSearch is running on port 9200
   - Verify with: `curl http://localhost:9200`

3. **PDF Download Failures**
   - Check internet connection
   - Verify ArXiv URLs are accessible
   - Some papers may have restricted access

4. **Memory Issues**
   - Reduce `NUM_THREADS` in config
   - Decrease `MAX_CHUNK_WORDS` for processing
   - Process smaller batches of papers

### Performance Optimization

1. **Parallel Processing**: Adjust `NUM_THREADS` based on your system
2. **Chunk Size**: Optimize `MAX_CHUNK_WORDS` for your LLM capacity
3. **Caching**: The system automatically caches LLM responses
4. **ElasticSearch**: Use SSD storage for better indexing performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

### Development Guidelines

- Follow the existing code structure and patterns
- Use type hints for better code clarity
- Add docstrings for all public methods
- Implement proper error handling
- Update the README for new features

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for agent orchestration
- Uses [Ollama](https://ollama.ai/) for local LLM inference
- Integrates [ElasticSearch](https://www.elastic.co/) for semantic search
- PDF processing via [PyMuPDF](https://pymupdf.readthedocs.io/)
- Embeddings from [Sentence Transformers](https://www.sbert.net/)

## 📚 Further Reading

- [ReACT Paper](https://arxiv.org/abs/2210.03629) - Original ReACT methodology
- [LangChain Documentation](https://docs.langchain.com/) - Agent development guide
- [ElasticSearch Guide](https://www.elastic.co/guide/) - Search and analytics platform

---

For more information or support, please refer to the documentation or create an issue in the repository.
