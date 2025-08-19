
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

## Usage
Run the main pipeline:
```sh
python main.py
```

Configure your research topic and parameters in `config.py`.

## How It Works
- The orchestrator agent coordinates the workflow.
- Agents perform tasks such as searching for papers, downloading PDFs, extracting text, filtering, and generating descriptions.
- Results are exported to CSV and JSON formats in the `output/` folder.

## Contributing
Feel free to fork the repository and submit pull requests for improvements or new features.

## License
This project is for academic and research purposes at CeADAR.

## Contact
For questions or collaboration, contact [Ayan Mulla](mailto:ayan.mulla09@gmail.com).
