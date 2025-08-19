"""
Streamlit Frontend for ReACT AI Research Pipeline
"""
import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
import time
from datetime import datetime
import threading
import queue

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import run_research_pipeline, initialize_all_agents
from config import Config

# Page configuration
st.set_page_config(
    page_title="ReACT AI Research Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #1f77b4, #ff7f0e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}

.metric-card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #1f77b4;
}

.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

.error-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

.info-box {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None
if 'progress_queue' not in st.session_state:
    st.session_state.progress_queue = queue.Queue()
if 'stage_outputs' not in st.session_state:
    st.session_state.stage_outputs = {}
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = None
if 'pipeline_status' not in st.session_state:
    st.session_state.pipeline_status = "idle"

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 ReACT AI Research Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("**CeADAR Centre for Applied Data Analytics Research**")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # Model Selection
        available_models = ["llama3.1", "llama3", "mistral", "phi3", "codellama"]
        selected_model = st.selectbox(
            "Select LLM Model",
            available_models,
            index=0,
            help="Choose the Ollama model to use for analysis"
        )
        
        # Update config if model changed
        if selected_model != Config.LLM_MODEL:
            Config.LLM_MODEL = selected_model
            st.success(f"Model updated to {selected_model}")
        
        # Pipeline Parameters
        st.subheader("📊 Pipeline Parameters")
        max_per_query = st.slider(
            "Max Papers per Query",
            min_value=1,
            max_value=10,
            value=Config.MAX_PER_QUERY,
            help="Maximum number of papers to collect per search query"
        )
        
        num_threads = st.slider(
            "Processing Threads",
            min_value=1,
            max_value=8,
            value=Config.NUM_THREADS,
            help="Number of parallel processing threads"
        )
        
        # Update config
        Config.MAX_PER_QUERY = max_per_query
        Config.NUM_THREADS = num_threads
        
        # System Status
        st.subheader("🔍 System Status")
        check_system_status()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Run Pipeline", "📊 Results", "📈 Analytics", "📋 History"])
    
    with tab1:
        run_pipeline_tab()
    
    with tab2:
        results_tab()
    
    with tab3:
        analytics_tab()
    
    with tab4:
        history_tab()

def check_system_status():
    """Check and display system status"""
    
    # Check Ollama
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model=Config.LLM_MODEL)
        test_response = llm.invoke("Test")
        ollama_status = "✅ Connected"
        ollama_color = "green"
    except Exception as e:
        ollama_status = "❌ Not Available"
        ollama_color = "red"
    
    # Check ElasticSearch
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch([f"http://{Config.ELASTICSEARCH_HOST}"], verify_certs=False, request_timeout=5)
        if es.ping():
            es_status = "✅ Connected"
            es_color = "green"
        else:
            es_status = "❌ Not Connected"
            es_color = "red"
    except Exception:
        es_status = "❌ Not Available"
        es_color = "red"
    
    st.markdown(f"**Ollama ({Config.LLM_MODEL}):** {ollama_status}")
    st.markdown(f"**ElasticSearch:** {es_status}")
    
    # Configuration display
    with st.expander("🔧 Current Configuration"):
        st.json({
            "LLM_MODEL": Config.LLM_MODEL,
            "MAX_PER_QUERY": Config.MAX_PER_QUERY,
            "NUM_THREADS": Config.NUM_THREADS,
            "ELASTICSEARCH_HOST": Config.ELASTICSEARCH_HOST,
            "EMBEDDING_MODEL": Config.EMBEDDING_MODEL
        })

def run_pipeline_tab():
    """Tab for running the research pipeline"""
    
    st.header("🚀 Research Pipeline Execution")
    
    # Initialize run_button to avoid UnboundLocalError
    run_button = False
    
    # Pipeline status indicator
    if st.session_state.pipeline_status != "idle":
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.session_state.current_stage:
                st.markdown(f"**Current Stage:** {st.session_state.current_stage}")
        with col2:
            if st.session_state.pipeline_status == "running":
                st.markdown("🔄 **Status:** Running")
            elif st.session_state.pipeline_status == "completed":
                st.markdown("✅ **Status:** Completed")
            elif st.session_state.pipeline_status == "error":
                st.markdown("❌ **Status:** Error")
        with col3:
            if st.session_state.stage_outputs:
                completed_stages = sum(1 for stage in st.session_state.stage_outputs.values() 
                                     if stage.get('status') == 'completed')
                total_stages = len(st.session_state.stage_outputs)
                st.markdown(f"**Progress:** {completed_stages}/{total_stages}")
    
    # Research topic input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        research_topic = st.text_input(
            "Research Topic",
            placeholder="e.g., Machine Learning for Climate Change Mitigation",
            help="Enter your research topic. Be specific for better results.",
            disabled=st.session_state.pipeline_running
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.pipeline_running:
            if st.button("� Stop Pipeline", type="secondary"):
                st.session_state.pipeline_running = False
                st.session_state.pipeline_status = "stopped"
                st.session_state.current_stage = "Pipeline Stopped"
                st.rerun()
        else:
            run_button = st.button("🔬 Run Analysis", type="primary")
    
    # Control buttons
    if not st.session_state.pipeline_running:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            if st.button("🧹 Clear Results"):
                st.session_state.pipeline_results = None
                st.session_state.stage_outputs = {}
                st.session_state.pipeline_status = "idle"
                st.session_state.current_stage = None
                st.rerun()
        
        with col_btn2:
            if st.session_state.stage_outputs:
                if st.button("🔄 Refresh Display"):
                    st.rerun()
    
    # Example topics
    st.markdown("**💡 Example Topics:**")
    example_topics = [
        "Sustainable AI and Energy Efficiency",
        "Machine Learning for Healthcare Diagnostics", 
        "Quantum Machine Learning Applications",
        "AI Ethics in Autonomous Systems",
        "Deep Learning for Climate Modeling"
    ]
    
    cols = st.columns(len(example_topics))
    for i, topic in enumerate(example_topics):
        with cols[i]:
            if st.button(topic, key=f"example_{i}", disabled=st.session_state.pipeline_running):
                research_topic = topic
                st.session_state.selected_topic = topic
                st.rerun()
    
    # Use selected example topic
    if 'selected_topic' in st.session_state:
        research_topic = st.session_state.selected_topic
        st.text_input("Research Topic", value=research_topic, key="topic_display", disabled=True)
    
    # Run pipeline
    if run_button and research_topic and not st.session_state.get('pipeline_running', False):
        st.session_state.selected_topic = research_topic  # Store topic for tracking
        run_pipeline_async(research_topic, Config.MAX_PER_QUERY)
    
    # Display progress
    display_pipeline_progress()

def run_pipeline_with_stages(topic: str, max_per_query: int):
    """Simplified pipeline runner - delegates to main pipeline function"""
    # This function is kept for compatibility but now just calls the main pipeline
    return run_research_pipeline(topic, max_per_query)
    
    try:
        agents = initialize_all_agents()
        orchestrator = agents["orchestrator"]
        
        st.session_state.stage_outputs["initialization"] = {
            "status": "completed",
            "message": f"✅ Successfully initialized {len(agents)} agents",
            "details": list(agents.keys())
        }
        
        # Validate pipeline
        st.session_state.current_stage = "Validating Pipeline"
        validation = orchestrator.validate_pipeline()
        
        if not validation["valid"]:
            st.session_state.stage_outputs["validation"] = {
                "status": "error",
                "message": "❌ Pipeline validation failed",
                "details": validation
            }
            st.session_state.pipeline_status = "error"
            return {"error": "Pipeline validation failed", "details": validation}
        
        st.session_state.stage_outputs["validation"] = {
            "status": "completed",
            "message": "✅ Pipeline validation successful",
            "details": validation
        }
        
        # Prepare input data
        input_data = {
            "topic": topic,
            "max_per_query": max_per_query,
            "config": Config.to_dict()
        }
        
        # Execute pipeline stages one by one
        pipeline_stages = [
            ("research_topic_agent", "🔍 Analyzing Research Topic", "Elaborating topic and generating search queries"),
            ("paper_search_agent", "📚 Searching for Papers", "Searching ArXiv for relevant papers"),
            ("filter_agent", "🔽 Filtering Abstracts", "Filtering papers based on abstract relevance"),
            ("description_gen_agent", "📝 Generating Descriptions", "Creating detailed paper descriptions"),
            ("download_extract_agent", "⬇️ Downloading PDFs", "Downloading and extracting PDF content"),
            ("fulltext_filter_agent", "� Full-text Analysis", "Performing detailed relevance analysis"),
            ("gap_generation_agent", "🔬 Gap Analysis", "Analyzing research gaps addressed by papers"),
            ("embedding_indexing_agent", "💾 Indexing Papers", "Creating searchable knowledge base")
        ]
        
        results = {"status": "running", "pipeline_results": {}}
        
        for stage_name, stage_title, stage_description in pipeline_stages:
            st.session_state.current_stage = stage_title
            st.session_state.stage_outputs[stage_name] = {
                "status": "running",
                "message": f"🔄 {stage_description}...",
                "details": []
            }
            
            try:
                # Run the individual stage
                stage_result = orchestrator.run_single_step(stage_name, input_data)
                
                if stage_result and "error" not in stage_result:
                    st.session_state.stage_outputs[stage_name] = {
                        "status": "completed",
                        "message": f"✅ {stage_description} completed",
                        "details": stage_result,
                        "summary": extract_stage_summary(stage_name, stage_result)
                    }
                    results["pipeline_results"][stage_name] = stage_result
                    
                    # Update input_data with results for next stage
                    input_data.update(stage_result)
                else:
                    st.session_state.stage_outputs[stage_name] = {
                        "status": "error",
                        "message": f"❌ {stage_description} failed",
                        "details": stage_result.get("error", "Unknown error") if stage_result else "No result returned"
                    }
                    
            except Exception as e:
                st.session_state.stage_outputs[stage_name] = {
                    "status": "error",
                    "message": f"❌ {stage_description} failed: {str(e)}",
                    "details": str(e)
                }
        
        # Final results
        results["status"] = "completed"
        st.session_state.pipeline_status = "completed"
        st.session_state.current_stage = "Pipeline Completed"
        
        return results
        
    except Exception as e:
        st.session_state.pipeline_status = "error"
        st.session_state.current_stage = f"Error: {str(e)}"
        return {"error": str(e)}

def extract_stage_summary(stage_name: str, stage_result: dict):
    """Extract key metrics from stage results for display"""
    summary = {}
    
    if stage_name == "research_topic_agent":
        summary = {
            "topic": stage_result.get("original_topic", "N/A"),
            "queries_generated": stage_result.get("num_queries_generated", 0),
            "elaboration_length": len(stage_result.get("topic_elaboration", ""))
        }
    elif stage_name == "paper_search_agent":
        summary = {
            "papers_found": stage_result.get("total_papers_collected", 0),
            "successful_queries": stage_result.get("successful_queries", 0),
            "failed_queries": stage_result.get("failed_queries", 0)
        }
    elif stage_name == "filter_agent":
        stats = stage_result.get("filtering_statistics", {})
        summary = {
            "papers_processed": stats.get("total_papers", 0),
            "papers_relevant": stats.get("relevant_count", 0),
            "relevance_rate": f"{stats.get('relevance_rate', 0):.1%}"
        }
    elif stage_name == "download_extract_agent":
        final_stats = stage_result.get("final_stats", {})
        summary = {
            "pdfs_downloaded": final_stats.get("downloaded_pdfs", 0),
            "texts_extracted": final_stats.get("extracted_papers", 0),
            "total_words": f"{final_stats.get('total_content_words', 0):,}"
        }
    elif stage_name == "fulltext_filter_agent":
        stats = stage_result.get("fulltext_filtering_stats", {})
        summary = {
            "papers_analyzed": stats.get("total_papers", 0),
            "papers_relevant": stats.get("relevant_count", 0),
            "relevance_rate": f"{stats.get('relevance_rate', 0):.1%}"
        }
    elif stage_name == "gap_generation_agent":
        stats = stage_result.get("gap_analysis_stats", {})
        summary = {
            "papers_analyzed": stats.get("total_papers", 0),
            "successful_analysis": stats.get("successful", 0),
            "success_rate": f"{stats.get('success_rate', 0):.1%}"
        }
    elif stage_name == "embedding_indexing_agent":
        summary = {
            "papers_indexed": stage_result.get("unique_papers_indexed", 0),
            "document_chunks": stage_result.get("total_chunks", 0),
            "rag_ready": "✅" if stage_result.get("rag_system_ready") else "❌"
        }
    
    return summary

def run_pipeline_async(topic: str, max_per_query: int):
    """Run the pipeline asynchronously with real-time updates"""
    
    def pipeline_runner():
        try:
            # Don't use session state directly in thread
            results = run_research_pipeline(topic, max_per_query)
            
            # Use a simple flag file to signal completion
            output_dir = Path(Config.OUTPUT_DIR)
            signal_file = output_dir / f"pipeline_complete_{topic.replace(' ', '_')}.json"
            
            with open(signal_file, 'w') as f:
                json.dump({
                    "status": "completed",
                    "results": results,
                    "timestamp": time.time()
                }, f)
                
        except Exception as e:
            # Signal error
            output_dir = Path(Config.OUTPUT_DIR)
            signal_file = output_dir / f"pipeline_complete_{topic.replace(' ', '_')}.json"
            
            with open(signal_file, 'w') as f:
                json.dump({
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }, f)
    
    # Start pipeline in a separate thread (only if not already running)
    if not st.session_state.get('pipeline_running', False):
        thread = threading.Thread(target=pipeline_runner)
        thread.daemon = True  # Daemon thread will die when main program exits
        thread.start()
        
        # Update session state to indicate pipeline started
        st.session_state.pipeline_running = True
        st.session_state.pipeline_status = "running"
        st.session_state.current_stage = "Pipeline Starting..."

def display_pipeline_progress():
    """Display real-time pipeline progress with stage outputs"""
    
    if st.session_state.pipeline_status == "idle":
        return
    
    # Check for completion signal file if pipeline is running
    if st.session_state.pipeline_running and 'selected_topic' in st.session_state:
        topic = st.session_state.selected_topic.replace(' ', '_')
        signal_file = Path(Config.OUTPUT_DIR) / f"pipeline_complete_{topic}.json"
        
        if signal_file.exists():
            try:
                with open(signal_file, 'r') as f:
                    signal_data = json.load(f)
                
                if signal_data["status"] == "completed":
                    st.session_state.pipeline_results = signal_data["results"]
                    st.session_state.pipeline_running = False
                    st.session_state.pipeline_status = "completed"
                    st.session_state.current_stage = "Pipeline Completed Successfully"
                    # Clean up signal file
                    signal_file.unlink()
                    st.rerun()
                    
                elif signal_data["status"] == "error":
                    st.session_state.pipeline_running = False
                    st.session_state.pipeline_status = "error"
                    st.session_state.current_stage = f"Error: {signal_data['error']}"
                    # Clean up signal file
                    signal_file.unlink()
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error reading pipeline status: {e}")
    
    # Current status
    st.markdown("### 🔄 Pipeline Progress")
    
    if st.session_state.current_stage:
        st.markdown(f"**Current Stage:** {st.session_state.current_stage}")
    
    # Progress indicator
    if st.session_state.pipeline_status == "running":
        st.markdown('<div class="info-box">🔄 Pipeline is running... Please wait for completion.</div>', 
                   unsafe_allow_html=True)
        # Auto-refresh every 3 seconds when running
        time.sleep(3)
        st.rerun()
    elif st.session_state.pipeline_status == "completed":
        st.markdown('<div class="success-box">✅ Pipeline completed successfully!</div>', 
                   unsafe_allow_html=True)
    elif st.session_state.pipeline_status == "error":
        st.markdown('<div class="error-box">❌ Pipeline encountered an error.</div>', 
                   unsafe_allow_html=True)
    
    # Stage-by-stage output
    if st.session_state.stage_outputs:
        st.markdown("### 📋 Stage Details")
        
        for stage_name, stage_data in st.session_state.stage_outputs.items():
            with st.expander(f"{stage_data['message']}", expanded=(stage_data['status'] == 'running')):
                
                # Status indicator
                if stage_data['status'] == 'running':
                    st.markdown("🔄 **Status:** Running")
                    st.spinner("Processing...")
                elif stage_data['status'] == 'completed':
                    st.markdown("✅ **Status:** Completed")
                elif stage_data['status'] == 'error':
                    st.markdown("❌ **Status:** Error")
                    st.error(f"Error: {stage_data.get('details', 'Unknown error')}")
                
                # Summary metrics
                if 'summary' in stage_data and stage_data['summary']:
                    st.markdown("**📊 Summary:**")
                    cols = st.columns(len(stage_data['summary']))
                    for i, (key, value) in enumerate(stage_data['summary'].items()):
                        with cols[i]:
                            st.metric(key.replace('_', ' ').title(), value)
                
                # Detailed output
                if stage_data['status'] == 'completed' and 'details' in stage_data:
                    with st.expander("🔍 Detailed Output", expanded=False):
                        if isinstance(stage_data['details'], dict):
                            # Show key information in a more readable format
                            for key, value in stage_data['details'].items():
                                if key in ['papers', 'papers_with_gap_analysis'] and isinstance(value, list):
                                    st.markdown(f"**{key.replace('_', ' ').title()}:** {len(value)} items")
                                    if value and len(value) > 0:
                                        # Show first few items
                                        for i, item in enumerate(value[:3]):
                                            if isinstance(item, dict) and 'title' in item:
                                                st.markdown(f"• {item['title']}")
                                        if len(value) > 3:
                                            st.markdown(f"... and {len(value) - 3} more")
                                elif isinstance(value, (str, int, float, bool)):
                                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                        else:
                            st.json(stage_data['details'])
    
    # Auto-refresh for running pipeline
    if st.session_state.pipeline_status == "running":
        time.sleep(2)
        st.rerun()

def results_tab():
    """Tab for displaying pipeline results"""
    
    st.header("📊 Pipeline Results")
    
    if st.session_state.pipeline_results is None:
        st.info("🔍 No results available. Please run the pipeline first.")
        return
    
    results = st.session_state.pipeline_results
    
    if "error" in results:
        st.markdown(f'<div class="error-box">❌ Pipeline Error: {results["error"]}</div>', 
                   unsafe_allow_html=True)
        return
    
    # Success message
    st.markdown('<div class="success-box">✅ Pipeline completed successfully!</div>', 
               unsafe_allow_html=True)
    
    # Results summary
    display_results_summary(results)
    
    # Detailed results
    with st.expander("📋 Detailed Results", expanded=False):
        st.json(results)
    
    # Download options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download JSON Results"):
            download_json_results(results)
    
    with col2:
        if st.button("📊 Download CSV Results"):
            download_csv_results(results)

def display_results_summary(results: dict):
    """Display a summary of pipeline results"""
    
    pipeline_results = results.get("pipeline_results", {})
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    # Topic Analysis
    if "research_topic_agent" in pipeline_results:
        topic_results = pipeline_results["research_topic_agent"]
        with col1:
            st.metric(
                "Search Queries Generated",
                topic_results.get("num_queries_generated", 0),
                help="Number of search queries generated from the topic"
            )
    
    # Paper Collection
    if "paper_search_agent" in pipeline_results:
        search_results = pipeline_results["paper_search_agent"]
        with col2:
            st.metric(
                "Papers Found",
                search_results.get("total_papers_collected", 0),
                help="Total number of papers collected from ArXiv"
            )
    
    # Filtering
    if "filter_agent" in pipeline_results:
        filter_results = pipeline_results["filter_agent"]
        filter_stats = filter_results.get("filtering_statistics", {})
        with col3:
            relevance_rate = filter_stats.get("relevance_rate", 0)
            st.metric(
                "Relevance Rate",
                f"{relevance_rate:.1%}",
                help="Percentage of papers deemed relevant after filtering"
            )
    
    # Gap Analysis
    if "gap_generation_agent" in pipeline_results:
        gap_results = pipeline_results["gap_generation_agent"]
        gap_stats = gap_results.get("gap_analysis_stats", {})
        with col4:
            st.metric(
                "Gap Analysis Success",
                f"{gap_stats.get('success_rate', 0):.1%}",
                help="Success rate of gap analysis generation"
            )
    
    # Papers table
    if "gap_generation_agent" in pipeline_results:
        gap_results = pipeline_results["gap_generation_agent"]
        papers = gap_results.get("papers_with_gap_analysis", [])
        
        if papers:
            st.subheader("📚 Analyzed Papers")
            
            # Convert to DataFrame
            df = pd.DataFrame(papers)
            
            # Display table with key columns
            display_columns = ["title", "research_gap", "gap_direction", "gap_solution"]
            available_columns = [col for col in display_columns if col in df.columns]
            
            if available_columns:
                st.dataframe(
                    df[available_columns],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)

def analytics_tab():
    """Tab for analytics and visualizations"""
    
    st.header("📈 Pipeline Analytics")
    
    if st.session_state.pipeline_results is None:
        st.info("🔍 No data available for analytics. Please run the pipeline first.")
        return
    
    results = st.session_state.pipeline_results
    pipeline_results = results.get("pipeline_results", {})
    
    # Agent performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Agent Performance")
        agent_metrics = []
        
        for agent_name, agent_results in pipeline_results.items():
            if isinstance(agent_results, dict):
                # Extract performance metrics
                if "filtering_statistics" in agent_results:
                    stats = agent_results["filtering_statistics"]
                    agent_metrics.append({
                        "Agent": agent_name.replace("_", " ").title(),
                        "Success Rate": stats.get("relevance_rate", 0),
                        "Processed": stats.get("total_papers", 0)
                    })
                elif "gap_analysis_stats" in agent_results:
                    stats = agent_results["gap_analysis_stats"]
                    agent_metrics.append({
                        "Agent": agent_name.replace("_", " ").title(),
                        "Success Rate": stats.get("success_rate", 0),
                        "Processed": stats.get("total_papers", 0)
                    })
        
        if agent_metrics:
            df_metrics = pd.DataFrame(agent_metrics)
            fig = px.bar(
                df_metrics,
                x="Agent",
                y="Success Rate",
                title="Agent Success Rates",
                color="Success Rate",
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Pipeline Flow")
        
        # Create pipeline flow visualization
        if "paper_search_agent" in pipeline_results and "filter_agent" in pipeline_results:
            search_results = pipeline_results["paper_search_agent"]
            filter_results = pipeline_results["filter_agent"]
            
            flow_data = {
                "Stage": ["Papers Found", "After Abstract Filter", "After Full-text Filter", "Gap Analysis"],
                "Count": [
                    search_results.get("total_papers_collected", 0),
                    filter_results.get("filtering_statistics", {}).get("relevant_count", 0),
                    pipeline_results.get("fulltext_filter_agent", {}).get("fulltext_filtering_stats", {}).get("relevant_count", 0),
                    pipeline_results.get("gap_generation_agent", {}).get("gap_analysis_stats", {}).get("successful", 0)
                ]
            }
            
            df_flow = pd.DataFrame(flow_data)
            fig = px.funnel(
                df_flow,
                x="Count",
                y="Stage",
                title="Pipeline Flow Analysis"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Paper analysis
    if "gap_generation_agent" in pipeline_results:
        gap_results = pipeline_results["gap_generation_agent"]
        papers = gap_results.get("papers_with_gap_analysis", [])
        
        if papers:
            st.subheader("📚 Paper Analysis")
            
            # Gap direction analysis
            df_papers = pd.DataFrame(papers)
            
            if "gap_direction" in df_papers.columns:
                gap_counts = df_papers["gap_direction"].value_counts()
                
                fig = px.pie(
                    values=gap_counts.values,
                    names=gap_counts.index,
                    title="Research Gap Directions"
                )
                st.plotly_chart(fig, use_container_width=True)

def history_tab():
    """Tab for viewing historical results"""
    
    st.header("📋 Results History")
    
    # Get all result files
    output_dir = Path(Config.OUTPUT_DIR)
    json_files = list(output_dir.glob("pipeline_results_*.json"))
    
    if not json_files:
        st.info("🔍 No historical results found.")
        return
    
    st.write(f"Found {len(json_files)} result files:")
    
    # Display files as cards
    for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
        with st.expander(f"📄 {json_file.name}", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # File info
                file_stats = json_file.stat()
                st.write(f"**Created:** {datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Size:** {file_stats.st_size / 1024:.1f} KB")
            
            with col2:
                if st.button("👁️ View", key=f"view_{json_file.name}"):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    st.session_state.pipeline_results = results
                    st.success("Results loaded!")
            
            with col3:
                if st.button("📥 Download", key=f"download_{json_file.name}"):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    st.download_button(
                        "💾 Download JSON",
                        content,
                        file_name=json_file.name,
                        mime="application/json"
                    )

def download_json_results(results: dict):
    """Download results as JSON"""
    json_content = json.dumps(results, indent=2, ensure_ascii=False, default=str)
    
    st.download_button(
        "💾 Download JSON Results",
        json_content,
        file_name=f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def download_csv_results(results: dict):
    """Download results as CSV"""
    pipeline_results = results.get("pipeline_results", {})
    
    if "gap_generation_agent" in pipeline_results:
        gap_results = pipeline_results["gap_generation_agent"]
        papers = gap_results.get("papers_with_gap_analysis", [])
        
        if papers:
            df = pd.DataFrame(papers)
            csv_content = df.to_csv(index=False)
            
            st.download_button(
                "📊 Download CSV Results",
                csv_content,
                file_name=f"papers_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No paper data available for CSV export.")
    else:
        st.warning("No gap analysis results available for CSV export.")

if __name__ == "__main__":
    main()
