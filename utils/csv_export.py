"""
CSV Export Module for ReACT Agents Pipeline Results
"""
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class CSVExporter:
    """Export pipeline results to various CSV formats"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_pipeline_results(self, results: Dict[str, Any], topic: str) -> Dict[str, str]:
        """Export complete pipeline results to multiple CSV files"""
        
        exported_files = {}
        base_filename = f"pipeline_results_{topic.replace(' ', '_').replace('--', '_')}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. Summary CSV - High-level pipeline metrics
            summary_file = self._export_summary(results, f"{base_filename}_summary")
            if summary_file:
                exported_files["summary"] = summary_file
            
            # 2. Papers CSV - Individual paper details with gap analysis
            papers_file = self._export_papers(results, f"{base_filename}_papers")
            if papers_file:
                exported_files["papers"] = papers_file
            
            # 3. Search Queries CSV - Query performance metrics
            queries_file = self._export_search_queries(results, f"{base_filename}_queries")
            if queries_file:
                exported_files["queries"] = queries_file
            
            # 4. Keywords CSV - Research keywords extracted
            keywords_file = self._export_keywords(results, f"{base_filename}_keywords")
            if keywords_file:
                exported_files["keywords"] = keywords_file
            
            # 5. Pipeline Steps CSV - Agent execution details
            steps_file = self._export_pipeline_steps(results, f"{base_filename}_steps")
            if steps_file:
                exported_files["steps"] = steps_file
            
            logger.info(f"Exported {len(exported_files)} CSV files: {list(exported_files.keys())}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return {}
    
    def _export_summary(self, results: Dict[str, Any], filename: str) -> Optional[str]:
        """Export pipeline summary to CSV"""
        try:
            pipeline_results = results.get("pipeline_results", {})
            
            # Collect summary metrics
            summary_data = {
                "pipeline_status": results.get("status", "unknown"),
                "original_topic": results.get("input_data", {}).get("topic", ""),
                "execution_timestamp": datetime.now().isoformat(),
                "total_errors": len(results.get("errors", [])),
            }
            
            # Topic analysis metrics
            if "research_topic_agent" in pipeline_results:
                topic_results = pipeline_results["research_topic_agent"]
                summary_data.update({
                    "search_queries_generated": topic_results.get("num_queries_generated", 0),
                })
            
            # Paper collection metrics
            if "paper_search_agent" in pipeline_results:
                search_results = pipeline_results["paper_search_agent"]
                summary_data.update({
                    "total_papers_found": search_results.get("total_papers_collected", 0),
                    "successful_queries": search_results.get("successful_queries", 0),
                })
            
            # Filtering metrics
            if "filter_agent" in pipeline_results:
                filter_results = pipeline_results["filter_agent"]
                filter_stats = filter_results.get("filtering_statistics", {})
                summary_data.update({
                    "abstract_filtered_papers": filter_stats.get("relevant_count", 0),
                    "abstract_filter_rate": filter_stats.get("relevance_rate", 0),
                })
            
            # Download/Extract metrics
            if "download_extract_agent" in pipeline_results:
                download_results = pipeline_results["download_extract_agent"]
                final_stats = download_results.get("final_stats", {})
                summary_data.update({
                    "pdfs_downloaded": final_stats.get("downloaded_pdfs", 0),
                    "texts_extracted": final_stats.get("extracted_papers", 0),
                    "total_words": final_stats.get("total_content_words", 0),
                })
            
            # Full-text filtering metrics
            if "fulltext_filter_agent" in pipeline_results:
                fulltext_results = pipeline_results["fulltext_filter_agent"]
                fulltext_stats = fulltext_results.get("fulltext_filtering_stats", {})
                summary_data.update({
                    "fulltext_filtered_papers": fulltext_stats.get("relevant_count", 0),
                    "fulltext_filter_rate": fulltext_stats.get("relevance_rate", 0),
                })
            
            # Gap analysis metrics
            if "gap_generation_agent" in pipeline_results:
                gap_results = pipeline_results["gap_generation_agent"]
                gap_stats = gap_results.get("gap_analysis_stats", {})
                summary_data.update({
                    "gap_analyzed_papers": gap_stats.get("successful", 0),
                    "gap_analysis_success_rate": gap_stats.get("success_rate", 0),
                })
            
            # Indexing metrics
            if "embedding_indexing_agent" in pipeline_results:
                index_results = pipeline_results["embedding_indexing_agent"]
                summary_data.update({
                    "papers_indexed": index_results.get("unique_papers_indexed", 0),
                    "document_chunks": index_results.get("total_chunks", 0),
                    "rag_system_ready": index_results.get("rag_system_ready", False),
                    "embedding_model": index_results.get("embedding_model", ""),
                    "elasticsearch_index": index_results.get("elasticsearch_index", ""),
                })
            
            # Write to CSV
            csv_file = self.output_dir / f"{filename}.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=summary_data.keys())
                writer.writeheader()
                writer.writerow(summary_data)
            
            return str(csv_file)
            
        except Exception as e:
            logger.error(f"Error exporting summary CSV: {e}")
            return None
    
    def _export_papers(self, results: Dict[str, Any], filename: str) -> Optional[str]:
        """Export papers with gap analysis to CSV"""
        try:
            papers_data = []
            
            # Get papers with gap analysis
            pipeline_results = results.get("pipeline_results", {})
            if "gap_generation_agent" in pipeline_results:
                gap_results = pipeline_results["gap_generation_agent"]
                papers_with_gaps = gap_results.get("papers_with_gap_analysis", [])
                
                for paper in papers_with_gaps:
                    paper_data = {
                        "title": paper.get("title", ""),
                        "url": paper.get("url", ""),
                        "research_gap": paper.get("research_gap", ""),
                        "gap_direction": paper.get("gap_direction", ""),
                        "gap_solution": paper.get("gap_solution", ""),
                        "gap_keywords": "; ".join(paper.get("gap_keywords", [])),
                        "text_length": len(paper.get("extracted_text", "")),
                        "has_extracted_text": bool(paper.get("extracted_text", "")),
                    }
                    papers_data.append(paper_data)
            
            # Fallback: get papers from other stages if gap analysis not available
            if not papers_data:
                # Try fulltext filtered papers
                if "fulltext_filter_agent" in pipeline_results:
                    papers = pipeline_results["fulltext_filter_agent"].get("relevant_papers", [])
                elif "download_extract_agent" in pipeline_results:
                    papers = pipeline_results["download_extract_agent"].get("papers_with_content", [])
                elif "filter_agent" in pipeline_results:
                    papers = pipeline_results["filter_agent"].get("relevant_papers", [])
                else:
                    papers = []
                
                for paper in papers:
                    paper_data = {
                        "title": paper.get("title", ""),
                        "url": paper.get("url", ""),
                        "research_gap": "Not analyzed",
                        "gap_direction": "Not analyzed", 
                        "gap_solution": "Not analyzed",
                        "gap_keywords": "",
                        "text_length": len(paper.get("extracted_text", "")),
                        "has_extracted_text": bool(paper.get("extracted_text", "")),
                    }
                    papers_data.append(paper_data)
            
            if not papers_data:
                logger.warning("No papers data found for CSV export")
                return None
            
            # Write to CSV
            csv_file = self.output_dir / f"{filename}.csv"
            if papers_data:
                df = pd.DataFrame(papers_data)
                df.to_csv(csv_file, index=False, encoding='utf-8')
                return str(csv_file)
            
            return None
            
        except Exception as e:
            logger.error(f"Error exporting papers CSV: {e}")
            return None
    
    def _export_search_queries(self, results: Dict[str, Any], filename: str) -> Optional[str]:
        """Export search query performance to CSV"""
        try:
            queries_data = []
            
            pipeline_results = results.get("pipeline_results", {})
            if "paper_search_agent" in pipeline_results:
                search_results = pipeline_results["paper_search_agent"]
                query_details = search_results.get("query_details", [])
                
                for i, query_detail in enumerate(query_details, 1):
                    query_data = {
                        "query_number": i,
                        "query_text": query_detail.get("query", ""),
                        "papers_found": query_detail.get("count", 0),
                        "unique_papers": query_detail.get("unique_count", 0),
                        "success": query_detail.get("count", 0) > 0,
                        "execution_time": query_detail.get("execution_time", 0),
                    }
                    queries_data.append(query_data)
            
            if not queries_data:
                logger.warning("No query data found for CSV export")
                return None
            
            # Write to CSV
            csv_file = self.output_dir / f"{filename}.csv"
            df = pd.DataFrame(queries_data)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            return str(csv_file)
            
        except Exception as e:
            logger.error(f"Error exporting queries CSV: {e}")
            return None
    
    def _export_keywords(self, results: Dict[str, Any], filename: str) -> Optional[str]:
        """Export research keywords to CSV"""
        try:
            keywords_data = []
            
            pipeline_results = results.get("pipeline_results", {})
            if "gap_generation_agent" in pipeline_results:
                gap_results = pipeline_results["gap_generation_agent"]
                papers_with_gaps = gap_results.get("papers_with_gap_analysis", [])
                
                for paper in papers_with_gaps:
                    paper_title = paper.get("title", "Unknown")
                    keywords = paper.get("gap_keywords", [])
                    
                    for keyword in keywords:
                        keyword_data = {
                            "paper_title": paper_title,
                            "keyword": keyword,
                            "paper_url": paper.get("url", ""),
                        }
                        keywords_data.append(keyword_data)
            
            if not keywords_data:
                logger.warning("No keywords data found for CSV export")
                return None
            
            # Write to CSV
            csv_file = self.output_dir / f"{filename}.csv"
            df = pd.DataFrame(keywords_data)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            return str(csv_file)
            
        except Exception as e:
            logger.error(f"Error exporting keywords CSV: {e}")
            return None
    
    def _export_pipeline_steps(self, results: Dict[str, Any], filename: str) -> Optional[str]:
        """Export pipeline step execution details to CSV"""
        try:
            steps_data = []
            
            pipeline_results = results.get("pipeline_results", {})
            step_order = [
                "research_topic_agent",
                "paper_search_agent", 
                "filter_agent",
                "description_gen_agent",
                "download_extract_agent",
                "fulltext_filter_agent",
                "gap_generation_agent",
                "embedding_indexing_agent"
            ]
            
            for i, step_name in enumerate(step_order, 1):
                if step_name in pipeline_results:
                    step_result = pipeline_results[step_name]
                    
                    step_data = {
                        "step_number": i,
                        "agent_name": step_name,
                        "status": "completed" if step_result else "failed",
                        "has_error": "error" in str(step_result),
                        "execution_order": i,
                    }
                    
                    # Add specific metrics based on agent type
                    if step_name == "research_topic_agent":
                        step_data["output_metric"] = step_result.get("num_queries_generated", 0)
                        step_data["metric_type"] = "queries_generated"
                    elif step_name == "paper_search_agent":
                        step_data["output_metric"] = step_result.get("total_papers_collected", 0)
                        step_data["metric_type"] = "papers_found"
                    elif step_name == "filter_agent":
                        stats = step_result.get("filtering_statistics", {})
                        step_data["output_metric"] = stats.get("relevant_count", 0)
                        step_data["metric_type"] = "papers_filtered"
                    elif step_name == "download_extract_agent":
                        stats = step_result.get("final_stats", {})
                        step_data["output_metric"] = stats.get("extracted_papers", 0)
                        step_data["metric_type"] = "papers_extracted"
                    elif step_name == "fulltext_filter_agent":
                        stats = step_result.get("fulltext_filtering_stats", {})
                        step_data["output_metric"] = stats.get("relevant_count", 0)
                        step_data["metric_type"] = "papers_fulltext_filtered"
                    elif step_name == "gap_generation_agent":
                        stats = step_result.get("gap_analysis_stats", {})
                        step_data["output_metric"] = stats.get("successful", 0)
                        step_data["metric_type"] = "papers_analyzed"
                    elif step_name == "embedding_indexing_agent":
                        step_data["output_metric"] = step_result.get("unique_papers_indexed", 0)
                        step_data["metric_type"] = "papers_indexed"
                    else:
                        step_data["output_metric"] = 0
                        step_data["metric_type"] = "unknown"
                    
                    steps_data.append(step_data)
            
            if not steps_data:
                logger.warning("No pipeline steps data found for CSV export")
                return None
            
            # Write to CSV
            csv_file = self.output_dir / f"{filename}.csv"
            df = pd.DataFrame(steps_data)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            return str(csv_file)
            
        except Exception as e:
            logger.error(f"Error exporting pipeline steps CSV: {e}")
            return None
    
    def export_single_csv(self, results: Dict[str, Any], topic: str) -> Optional[str]:
        """Export all results to a single comprehensive CSV file"""
        try:
            all_data = []
            
            pipeline_results = results.get("pipeline_results", {})
            if "gap_generation_agent" in pipeline_results:
                gap_results = pipeline_results["gap_generation_agent"]
                papers_with_gaps = gap_results.get("papers_with_gap_analysis", [])
                
                for i, paper in enumerate(papers_with_gaps, 1):
                    # Flatten all data into a single row
                    row_data = {
                        # Paper identifiers
                        "paper_number": i,
                        "title": paper.get("title", ""),
                        "url": paper.get("url", ""),
                        
                        # Gap analysis
                        "research_gap": paper.get("research_gap", ""),
                        "gap_direction": paper.get("gap_direction", ""),
                        "gap_solution": paper.get("gap_solution", ""),
                        "gap_keywords": "; ".join(paper.get("gap_keywords", [])),
                        
                        # Text metrics
                        "text_length": len(paper.get("extracted_text", "")),
                        "has_extracted_text": bool(paper.get("extracted_text", "")),
                        
                        # Pipeline metadata
                        "original_topic": results.get("input_data", {}).get("topic", ""),
                        "pipeline_status": results.get("status", ""),
                        "export_timestamp": datetime.now().isoformat(),
                    }
                    
                    # Add pipeline summary metrics to each row
                    if "paper_search_agent" in pipeline_results:
                        search_results = pipeline_results["paper_search_agent"]
                        row_data["total_papers_found"] = search_results.get("total_papers_collected", 0)
                    
                    if "embedding_indexing_agent" in pipeline_results:
                        index_results = pipeline_results["embedding_indexing_agent"]
                        row_data["papers_indexed"] = index_results.get("unique_papers_indexed", 0)
                        row_data["rag_system_ready"] = index_results.get("rag_system_ready", False)
                    
                    all_data.append(row_data)
            
            if not all_data:
                logger.warning("No data available for single CSV export")
                return None
            
            # Write to CSV
            filename = f"pipeline_complete_{topic.replace(' ', '_').replace('--', '_')}"
            csv_file = self.output_dir / f"{filename}.csv"
            df = pd.DataFrame(all_data)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            return str(csv_file)
            
        except Exception as e:
            logger.error(f"Error exporting single CSV: {e}")
            return None


def export_results_to_csv(results: Dict[str, Any], topic: str, output_dir: str = "output") -> Dict[str, str]:
    """Convenience function to export pipeline results to CSV"""
    exporter = CSVExporter(output_dir)
    return exporter.export_pipeline_results(results, topic)


def export_single_csv_file(results: Dict[str, Any], topic: str, output_dir: str = "output") -> Optional[str]:
    """Convenience function to export all results to a single CSV file"""
    exporter = CSVExporter(output_dir)
    return exporter.export_single_csv(results, topic)
