
# ReACT AI Research Pipeline

## Overview
This is a comprehensive research paper analysis pipeline developed for CeADAR (Centre for Applied Data Analytics Research) using ReACT (Reasoning and Acting) agents and LangChain. The system automates the entire research paper discovery, filtering, analysis, and gap identification process with intelligent agents working collaboratively to provide comprehensive research insights.

## Key Features

- **Multi-Agent ReACT Architecture**: Specialized agents for each pipeline stage using reasoning and acting paradigm
- **Comprehensive Paper Discovery**: Intelligent search query generation and ArXiv integration
- **Advanced Two-Stage Filtering**: Abstract-level and full-text analysis for precision
- **Gap Analysis**: Automated identification of research gaps that papers address
- **RAG System**: ElasticSearch integration with semantic embeddings for retrieval
- **Parallel Processing**: Concurrent execution for improved performance
- **Interactive & Command-Line Modes**: Flexible usage options
- **Comprehensive Export**: JSON and CSV output formats
- **Detailed Logging**: Complete pipeline execution tracking

## Architecture

### Agent Pipeline
```
Research Topic → Query Generation → Paper Search → Abstract Filter → 
PDF Download → Text Extraction → Full-text Filter → Gap Analysis → 
ElasticSearch Indexing → RAG System Ready
```

### Core Agents
1. **ReACT Orchestrator**: Coordinates the entire pipeline execution
2. **Research Topic Agent**: Elaborates topics and generates search queries
3. **Paper Search Agent**: Searches ArXiv for relevant papers
4. **Filter Agent**: Performs initial relevance filtering on abstracts
5. **Description Gen Agent**: Generates detailed descriptions and categorizations
6. **Download/Extract Agent**: Downloads PDFs and extracts text content
7. **Full Text Filter Agent**: Performs detailed relevance analysis on full papers
8. **Gap Generation Agent**: Analyzes research gaps that papers address
9. **Embedding/Indexing Agent**: Indexes papers to ElasticSearch with embeddings

## Project Structure
```
react_agents/
├── agents/                     # ReACT agent implementations
│   ├── base_agent.py          # Base ReACT agent class
│   ├── orchestrator.py        # Pipeline orchestrator
│   ├── research_topic.py      # Topic analysis agent
│   ├── paper_search.py        # ArXiv search agent
│   ├── filter.py              # Abstract filtering agent
│   ├── description_gen.py     # Description generation agent
│   ├── download_extract.py    # PDF download/extract agent
│   ├── full_text_filter.py    # Full-text filtering agent
│   ├── gap_generation.py      # Gap analysis agent
│   └── embedding_indexing.py  # ElasticSearch indexing agent
├── tools/                      # LangChain research tools
│   └── research_tools.py      # ArXiv, PDF, ElasticSearch tools
├── utils/                      # Utility functions
│   ├── csv_export.py          # Comprehensive CSV export
│   └── simple_csv_export.py   # Simple CSV export
├── cache/                      # LLM response cache
├── output/                     # Analysis results (JSON, CSV)
├── pdf_cache/                  # Downloaded PDF files
├── main.py                     # Main application entry point
├── config.py                   # Configuration settings
└── requirements.txt            # Python dependencies
```

## Quick Start

### Prerequisites
- **Python 3.8+**
- **[Ollama](https://ollama.ai/)** for local LLM inference
- **[Docker](https://www.docker.com/)** for ElasticSearch (optional)

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/AyanMulla09/CeADAR_project.git
   cd CeADAR_project
   ```

2. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Setting Up Ollama
1. Install Ollama from [https://ollama.ai/download](https://ollama.ai/download)
2. Pull and start the default model:
   ```sh
   ollama pull llama3.1
   ollama serve
   ```

### Setting Up ElasticSearch (Optional for RAG features)
Run ElasticSearch using Docker:
```sh
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.8.0
```

Stop ElasticSearch when done:
```sh
docker stop elasticsearch
```

## 📋 Usage

### Command Line Mode
```sh
# Run full pipeline with a research topic
python main.py "Machine Learning for Climate Change Mitigation"

# Or specify topic in quotes
python main.py "Sustainable AI and Energy Efficiency"
```

### Interactive Mode
```sh
# Run without arguments for interactive menu
python main.py
```

Interactive options include:
- Run full pipeline
- Test individual agents
- View agent status
- Export existing results to CSV
- Pipeline configuration

### Configuration
Customize the pipeline by editing `config.py`:

```python
class Config:
    # LLM Configuration
    LLM_MODEL = "llama3.1"          # Change to your preferred model
    LLM_TEMPERATURE = 0.1
    MAX_TOKENS = 4000
    
    # Pipeline Settings
    MAX_PER_QUERY = 2               # Papers per search query
    NUM_THREADS = 4                 # Parallel processing threads
    
    # ElasticSearch Configuration
    ELASTICSEARCH_HOST = "localhost:9200"
    ELASTICSEARCH_INDEX = "research_papers"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

## Advanced Configuration

### Using Different LLM Models
1. Pull a different model with Ollama:
   ```sh
   ollama pull mistral
   # or
   ollama pull phi3
   ```

2. Update `config.py`:
   ```python
   LLM_MODEL = "mistral"  # or "phi3"
   ```

3. Restart Ollama if needed:
   ```sh
   ollama serve
   ```

### Custom ElasticSearch Setup
For different ports or configurations:
```sh
# Run on different port
docker run -d --name elasticsearch -p 9300:9200 -e "discovery.type=single-node" elasticsearch:8.8.0
```

Update `config.py`:
```python
ELASTICSEARCH_HOST = "localhost:9300"
```

## Output and Results

### Generated Files
The pipeline creates several output files in the `output/` directory:

1. **`pipeline_results_[topic].json`**: Complete execution results with agent outputs
2. **`papers_analysis_[topic].csv`**: Structured CSV with research gap analysis
3. **`research_pipeline.log`**: Detailed execution logs

### Sample Output Structure
```json
{
  "status": "completed",
  "pipeline_results": {
    "research_topic_agent": {
      "original_topic": "Machine Learning for Healthcare",
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

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```sh
   # Ensure Ollama is running
   ollama serve
   
   # Verify model is available
   ollama list
   ```

2. **ElasticSearch Connection Error**
   ```sh
   # Check ElasticSearch status
   curl http://localhost:9200
   
   # Restart if needed
   docker restart elasticsearch
   ```

3. **PDF Download Failures**
   - Check internet connection
   - Some ArXiv papers may have access restrictions
   - Verify ArXiv URLs are accessible

4. **Memory Issues**
   - Reduce `NUM_THREADS` in config
   - Decrease `MAX_CHUNK_WORDS` for processing
   - Process smaller batches with lower `MAX_PER_QUERY`

### Performance Optimization

- **Parallel Processing**: Adjust `NUM_THREADS` based on your system (default: 4)
- **Chunk Size**: Optimize `MAX_CHUNK_WORDS` for your LLM capacity (default: 1500)
- **Caching**: The system automatically caches LLM responses in `cache/`
- **ElasticSearch**: Use SSD storage for better indexing performance
## Contributing

### Development Guidelines
- Follow the existing ReACT agent patterns in `agents/base_agent.py`
- Use type hints for better code clarity
- Add comprehensive docstrings for all public methods
- Implement proper error handling and logging
- Test agents individually before integration

### Adding New Agents
1. Extend `BaseReACTAgent` class
2. Implement required abstract methods
3. Add agent to orchestrator pipeline
4. Update configuration if needed

### Example Agent Implementation
```python
from agents.base_agent import BaseReACTAgent
from langchain.tools import Tool

class CustomAgent(BaseReACTAgent):
    def __init__(self):
        tools = [
            Tool(
                name="custom_tool",
                description="Description of custom functionality",
                func=self._custom_function
            )
        ]
        
        super().__init__(
            name="custom_agent",
            description="Custom agent for specific research tasks",
            tools=tools
        )
    
    def _create_prompt_template(self):
        return PromptTemplate(
            input_variables=["input", "chat_history", "agent_scratchpad"],
            template="Your custom prompt template here..."
        )
```

## Research Workflow

### Typical Research Session
1. **Topic Analysis**: Agent elaborates your research topic
2. **Query Generation**: Creates multiple search strategies
3. **Paper Discovery**: Searches ArXiv with generated queries
4. **Initial Filtering**: Filters papers based on abstracts
5. **Content Extraction**: Downloads and extracts PDF content
6. **Deep Analysis**: Performs full-text relevance analysis
7. **Gap Identification**: Analyzes research gaps addressed
8. **Knowledge Indexing**: Creates searchable knowledge base
9. **Results Export**: Generates CSV and JSON outputs

### Use Cases
- **Literature Reviews**: Comprehensive topic analysis
- **Research Gap Analysis**: Identify opportunities for new research
- **Knowledge Discovery**: Find related work and methodologies
- **Academic Research**: Systematic paper collection and analysis
- **Industry Research**: Track developments in specific domains

## Technical Details

### ReACT Agent Architecture
Each agent follows the ReACT (Reasoning and Acting) paradigm:
- **Reasoning**: Analyzes the current state and available information
- **Acting**: Takes specific actions using available tools
- **Observation**: Processes results and updates understanding

### LLM Integration
- Uses Ollama for local LLM inference
- Supports multiple models (Llama, Mistral, Phi, etc.)
- Implements response caching for efficiency
- Configurable temperature and token limits

### Data Processing Pipeline
- **Concurrent Processing**: Multi-threaded PDF processing
- **Chunking Strategy**: Intelligent text segmentation
- **Error Recovery**: Robust handling of failures
- **Progress Tracking**: Real-time pipeline monitoring

## Performance Metrics

The pipeline tracks comprehensive metrics:
- **Collection**: Papers found per query
- **Filtering**: Relevance rates at each stage
- **Processing**: Download success rates
- **Analysis**: Gap analysis completion rates
- **Indexing**: ElasticSearch indexing statistics

##  Example Workflows

### Climate Change Research
```sh
python main.py "Machine Learning for Climate Change Mitigation"
```

### Healthcare AI
```sh
python main.py "AI Applications in Personalized Medicine"
```

### Sustainable Computing
```sh
python main.py "Energy-Efficient Deep Learning Algorithms"
```

## Additional Resources

### Dependencies
- **[LangChain](https://langchain.com/)**: Agent orchestration framework
- **[Ollama](https://ollama.ai/)**: Local LLM inference
- **[ElasticSearch](https://www.elastic.co/)**: Search and analytics
- **[PyMuPDF](https://pymupdf.readthedocs.io/)**: PDF processing
- **[Sentence Transformers](https://www.sbert.net/)**: Text embeddings

### Related Research
- [ReACT Paper](https://arxiv.org/abs/2210.03629) - Original ReACT methodology
- [LangChain Documentation](https://docs.langchain.com/) - Agent development
- [ArXiv API](https://arxiv.org/help/api) - Paper search capabilities

## License
This project is developed for academic and research purposes at CeADAR (Centre for Applied Data Analytics Research).

## Contact
**Ayan Mulla**  
 [ayan.mulla09@gmail.com](mailto:ayan.mulla09@gmail.com)  
 CeADAR - Centre for Applied Data Analytics Research  
 [GitHub Repository](https://github.com/AyanMulla09/CeADAR_project)

---

*For technical support, feature requests, or contributions, please create an issue in the GitHub repository.*
