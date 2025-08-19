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
        es = Elasticsearch([Config.ELASTICSEARCH_HOST])
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
    
    # Research topic input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        research_topic = st.text_input(
            "Research Topic",
            placeholder="e.g., Machine Learning for Climate Change Mitigation",
            help="Enter your research topic. Be specific for better results."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("🔬 Run Analysis", type="primary", disabled=st.session_state.pipeline_running)
    
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
            if st.button(topic, key=f"example_{i}"):
                st.session_state.selected_topic = topic
                st.rerun()
    
    # Use selected example topic
    if 'selected_topic' in st.session_state:
        research_topic = st.session_state.selected_topic
        st.text_input("Research Topic", value=research_topic, key="topic_display", disabled=True)
    
    # Run pipeline
    if run_button and research_topic:
        st.session_state.pipeline_running = True
        run_pipeline_async(research_topic, Config.MAX_PER_QUERY)
    
    # Display progress
    if st.session_state.pipeline_running:
        display_pipeline_progress()

def run_pipeline_async(topic: str, max_per_query: int):
    """Run the pipeline asynchronously"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def pipeline_runner():
        try:
            # Update progress
            progress_bar.progress(10)
            status_text.text("🔄 Initializing agents...")
            
            # Run the pipeline
            results = run_research_pipeline(topic, max_per_query)
            
            progress_bar.progress(100)
            status_text.text("✅ Pipeline completed successfully!")
            
            # Store results
            st.session_state.pipeline_results = results
            st.session_state.pipeline_running = False
            
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.session_state.pipeline_running = False
            st.error(f"❌ Pipeline failed: {str(e)}")
    
    # Start pipeline in a separate thread
    thread = threading.Thread(target=pipeline_runner)
    thread.start()

def display_pipeline_progress():
    """Display real-time pipeline progress"""
    
    st.markdown('<div class="info-box">🔄 Pipeline is running... Please wait for completion.</div>', 
                unsafe_allow_html=True)
    
    # Auto-refresh every 5 seconds
    time.sleep(5)
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
