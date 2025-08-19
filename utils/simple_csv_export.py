"""
Simple CSV Export Module for ReACT Agents Pipeline Results
Creates a single CSV with the exact columns from the JSON structure
"""
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def export_papers_simple_csv(results: Dict[str, Any], topic: str, output_dir: str = "output") -> Optional[str]:
    """Export papers with gap analysis to a simple CSV with exact JSON columns"""
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        papers_data = []
        
        # Get papers with gap analysis from the results
        pipeline_results = results.get("pipeline_results", {})
        if "gap_generation_agent" in pipeline_results:
            gap_results = pipeline_results["gap_generation_agent"]
            papers_with_gaps = gap_results.get("papers_with_gap_analysis", [])
            
            for paper in papers_with_gaps:
                # Extract exactly the columns from the JSON structure
                paper_data = {
                    "title": paper.get("title", ""),
                    "url": paper.get("url", ""),
                    "research_gap": paper.get("research_gap", ""),
                    "gap_direction": paper.get("gap_direction", ""),
                    "gap_solution": paper.get("gap_solution", ""),
                    "analysis_complete": paper.get("analysis_complete", False),
                    "gap_keywords": "; ".join(paper.get("gap_keywords", []))  # Convert list to semicolon-separated string
                }
                papers_data.append(paper_data)
        
        if not papers_data:
            logger.warning("No papers data found for CSV export")
            return None
        
        # Create filename
        safe_topic = topic.replace(' ', '_').replace('--', '_').replace('/', '_').replace('\\', '_')
        filename = f"papers_analysis_{safe_topic}.csv"
        csv_file = output_path / filename
        
        # Write to CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["title", "url", "research_gap", "gap_direction", "gap_solution", "analysis_complete", "gap_keywords"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(papers_data)
        
        logger.info(f"Simple CSV exported: {csv_file}")
        return str(csv_file)
        
    except Exception as e:
        logger.error(f"Error exporting simple CSV: {e}")
        return None


def export_from_json_file(json_file_path: str, output_dir: str = "output") -> Optional[str]:
    """Export CSV from an existing JSON file"""
    try:
        # Load JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Extract topic from filename
        json_path = Path(json_file_path)
        topic = json_path.stem.replace("pipeline_results_", "").replace("_", " ")
        
        # Export CSV
        return export_papers_simple_csv(results, topic, output_dir)
        
    except Exception as e:
        logger.error(f"Error exporting from JSON file: {e}")
        return None


if __name__ == "__main__":
    # Test with the provided JSON file
    json_file = r"c:\Final\react_agents\output\pipeline_results_piezoelectric_energy_harvesting_optimization_--max-papers_2.json"
    
    if Path(json_file).exists():
        print("🧪 Testing Simple CSV Export")
        print("=" * 50)
        
        csv_file = export_from_json_file(json_file, "output")
        
        if csv_file:
            print(f"✅ CSV file created: {Path(csv_file).name}")
            print(f"📁 Location: {csv_file}")
            
            # Show first few rows
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:3]  # Header + first 2 rows
                    print(f"\n📊 Preview (first {len(lines)} lines):")
                    for line in lines:
                        print(line.strip())
            except Exception as e:
                print(f"Error reading preview: {e}")
        else:
            print("❌ CSV export failed")
    else:
        print(f"❌ JSON file not found: {json_file}")
