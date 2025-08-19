"""
Main application for the ReACT AI Research Pipeline
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import agents - using absolute imports to avoid relative import issues
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.orchestrator import create_orchestrator
from agents.research_topic import create_research_topic_agent
from agents.paper_search import create_paper_search_agent
from agents.filter import create_filter_agent
from agents.description_gen import create_description_gen_agent
from agents.download_extract import create_download_extract_agent
from agents.full_text_filter import create_fulltext_filter_agent
from agents.gap_generation import create_gap_generation_agent
from agents.embedding_indexing import create_embedding_indexing_agent

from config import Config
from utils.simple_csv_export import export_papers_simple_csv


def initialize_all_agents():
    """Initialize all agents in the pipeline"""
    logger.info("Initializing all agents...")
    
    agents = {
        "orchestrator": create_orchestrator(),
        "research_topic_agent": create_research_topic_agent(),
        "paper_search_agent": create_paper_search_agent(),
        "filter_agent": create_filter_agent(),
        "description_gen_agent": create_description_gen_agent(),
        "download_extract_agent": create_download_extract_agent(),
        "fulltext_filter_agent": create_fulltext_filter_agent(),
        "gap_generation_agent": create_gap_generation_agent(),
        "embedding_indexing_agent": create_embedding_indexing_agent()
    }
    
    logger.info(f"Initialized {len(agents)} agents successfully")
    return agents


def save_results(results: Dict[str, Any], filename: str, topic: str = ""):
    """Save pipeline results to file and export to CSV"""
    try:
        # Save JSON results
        output_path = Path(Config.OUTPUT_DIR) / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"JSON results saved to {output_path}")
        
        # Export to CSV files
        if topic:
            try:
                csv_file = export_papers_simple_csv(results, topic, Config.OUTPUT_DIR)
                if csv_file:
                    logger.info(f"Simple CSV exported: {csv_file}")
                    print(f"\n📊 CSV Export: {Path(csv_file).name}")
                else:
                    logger.warning("No CSV file was created")
                    
            except Exception as e:
                logger.error(f"CSV export failed: {e}")
                print(f"⚠️  CSV export failed: {e}")
                
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def run_research_pipeline(topic: str, max_per_query: int | None = None) -> Dict[str, Any]:
    """Run the complete research pipeline"""
    
    if max_per_query is None:
        max_per_query = Config.MAX_PER_QUERY
    
    logger.info(f"Starting research pipeline for topic: {topic}")
    
    # Initialize all agents
    agents = initialize_all_agents()
    orchestrator = agents["orchestrator"]
    
    # Validate pipeline
    validation = orchestrator.validate_pipeline()
    if not validation["valid"]:
        logger.error(f"Pipeline validation failed: {validation}")
        return {"error": "Pipeline validation failed", "details": validation}
    
    # Prepare input data
    input_data = {
        "topic": topic,
        "max_per_query": max_per_query,
        "config": Config.to_dict()
    }
    
    # Execute the pipeline
    try:
        results = orchestrator.execute(input_data)
        
        # Save results
        save_results(results, f"pipeline_results_{topic.replace(' ', '_')}.json", topic)
        
        # Print summary
        print_pipeline_summary(results)
        
        return results
        
    except Exception as e:
        error_msg = f"Pipeline execution failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def print_pipeline_summary(results: Dict[str, Any]):
    """Print a summary of pipeline results"""
    print("\n" + "="*80)
    print("RESEARCH PIPELINE SUMMARY")
    print("="*80)
    
    status = results.get("status", "unknown")
    print(f"Pipeline Status: {status.upper()}")
    
    if results.get("errors"):
        print(f"Errors Encountered: {len(results['errors'])}")
        for error in results["errors"]:
            print(f"  - {error}")
    
    pipeline_results = results.get("pipeline_results", {})
    
    # Topic Analysis Summary
    if "research_topic_agent" in pipeline_results:
        topic_results = pipeline_results["research_topic_agent"]
        print(f"\n📚 TOPIC ANALYSIS:")
        print(f"  - Original Topic: {topic_results.get('original_topic', 'N/A')}")
        print(f"  - Search Queries Generated: {topic_results.get('num_queries_generated', 0)}")
    
    # Paper Collection Summary
    if "paper_search_agent" in pipeline_results:
        search_results = pipeline_results["paper_search_agent"]
        print(f"\n🔍 PAPER COLLECTION:")
        print(f"  - Total Papers Found: {search_results.get('total_papers_collected', 0)}")
        print(f"  - Queries Executed: {search_results.get('successful_queries', 0)}")
    
    # Filtering Summary
    if "filter_agent" in pipeline_results:
        filter_results = pipeline_results["filter_agent"]
        filter_stats = filter_results.get("filtering_statistics", {})
        print(f"\n📋 ABSTRACT FILTERING:")
        print(f"  - Papers Retained: {filter_stats.get('relevant_count', 0)}/{filter_stats.get('total_papers', 0)}")
        print(f"  - Relevance Rate: {filter_stats.get('relevance_rate', 0):.1%}")
    
    # Download/Extract Summary
    if "download_extract_agent" in pipeline_results:
        download_results = pipeline_results["download_extract_agent"]
        final_stats = download_results.get("final_stats", {})
        print(f"\n📥 PDF PROCESSING:")
        print(f"  - PDFs Downloaded: {final_stats.get('downloaded_pdfs', 0)}")
        print(f"  - Text Extracted: {final_stats.get('extracted_papers', 0)}")
        print(f"  - Total Words: {final_stats.get('total_content_words', 0):,}")
    
    # Full-text Filtering Summary
    if "fulltext_filter_agent" in pipeline_results:
        fulltext_results = pipeline_results["fulltext_filter_agent"]
        fulltext_stats = fulltext_results.get("fulltext_filtering_stats", {})
        print(f"\n🔍 FULL-TEXT FILTERING:")
        print(f"  - Papers Retained: {fulltext_stats.get('relevant_count', 0)}/{fulltext_stats.get('total_papers', 0)}")
        print(f"  - Relevance Rate: {fulltext_stats.get('relevance_rate', 0):.1%}")
    
    # Gap Analysis Summary
    if "gap_generation_agent" in pipeline_results:
        gap_results = pipeline_results["gap_generation_agent"]
        gap_stats = gap_results.get("gap_analysis_stats", {})
        print(f"\n🔬 GAP ANALYSIS:")
        print(f"  - Papers Analyzed: {gap_stats.get('successful', 0)}/{gap_stats.get('total_papers', 0)}")
        print(f"  - Success Rate: {gap_stats.get('success_rate', 0):.1%}")
    
    # Indexing Summary
    if "embedding_indexing_agent" in pipeline_results:
        index_results = pipeline_results["embedding_indexing_agent"]
        print(f"\n💾 ELASTICSEARCH INDEXING:")
        print(f"  - Papers Indexed: {index_results.get('unique_papers_indexed', 0)}")
        print(f"  - Document Chunks: {index_results.get('total_chunks', 0)}")
        print(f"  - RAG System Ready: {'✅' if index_results.get('rag_system_ready') else '❌'}")
    
    print("\n" + "="*80)


def run_interactive_mode():
    """Run the pipeline in interactive mode"""
    print("ReACT AI Research Pipeline")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Run full pipeline")
        print("2. Run single agent")
        print("3. View agent status")
        print("4. Export existing results to CSV")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            topic = input("Enter research topic: ").strip()
            if topic:
                max_per_query = input(f"Max papers per query (default {Config.MAX_PER_QUERY}): ").strip()
                max_per_query = int(max_per_query) if max_per_query.isdigit() else Config.MAX_PER_QUERY
                
                print(f"\n🚀 Starting pipeline for: {topic}")
                results = run_research_pipeline(topic, max_per_query)
                
                if "error" not in results:
                    print("✅ Pipeline completed successfully!")
                    print("📊 Both JSON and CSV files have been created in the output directory")
                else:
                    print(f"❌ Pipeline failed: {results['error']}")
        
        elif choice == "2":
            # Initialize agents
            agents = initialize_all_agents()
            orchestrator = agents["orchestrator"]
            
            print("\nAvailable agents:")
            available_agents = orchestrator.get_pipeline_status()["available_agents"]
            for i, agent_name in enumerate(available_agents, 1):
                print(f"{i}. {agent_name}")
            
            try:
                agent_choice = int(input("\nSelect agent number: ")) - 1
                if 0 <= agent_choice < len(available_agents):
                    agent_name = available_agents[agent_choice]
                    test_input = input(f"Enter test input for {agent_name}: ")
                    
                    result = orchestrator.run_single_step(agent_name, {"topic": test_input})
                    print(f"\nResult: {json.dumps(result, indent=2)}")
                else:
                    print("Invalid selection")
            except ValueError:
                print("Invalid input")
        
        elif choice == "3":
            agents = initialize_all_agents()
            orchestrator = agents["orchestrator"]
            status = orchestrator.get_pipeline_status()
            
            print(f"\nPipeline Status:")
            print(f"Available Agents: {len(status['available_agents'])}")
            print(f"Missing Agents: {status['missing_agents']}")
            print(f"Pipeline Valid: {'✅' if len(status['missing_agents']) == 0 else '❌'}")
        
        elif choice == "4":
            # Export existing results to CSV
            output_dir = Path(Config.OUTPUT_DIR)
            json_files = list(output_dir.glob("pipeline_results_*.json"))
            
            if not json_files:
                print("❌ No pipeline results found in output directory")
            else:
                print(f"\nFound {len(json_files)} result files:")
                for i, file_path in enumerate(json_files, 1):
                    print(f"{i}. {file_path.name}")
                
                try:
                    file_choice = int(input("\nSelect file number: ")) - 1
                    if 0 <= file_choice < len(json_files):
                        selected_file = json_files[file_choice]
                        
                        # Load the JSON results
                        with open(selected_file, 'r', encoding='utf-8') as f:
                            results = json.load(f)
                        
                        # Extract topic from filename
                        topic = selected_file.stem.replace("pipeline_results_", "").replace("_", " ")
                        
                        print(f"\n📊 Exporting {selected_file.name} to CSV...")
                        
                        # Export to CSV
                        csv_file = export_papers_simple_csv(results, topic, Config.OUTPUT_DIR)
                        
                        if csv_file:
                            print("✅ CSV export completed!")
                            print(f"CSV file: {Path(csv_file).name}")
                        else:
                            print("❌ CSV export failed")
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Invalid input")
                except Exception as e:
                    print(f"❌ Error during CSV export: {e}")
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Command line mode
        topic = " ".join(sys.argv[1:])
        results = run_research_pipeline(topic)
        
        if "error" in results:
            sys.exit(1)
    else:
        # Interactive mode
        run_interactive_mode()


if __name__ == "__main__":
    main()
