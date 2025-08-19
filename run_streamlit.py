#!/usr/bin/env python3
"""
Launch script for the ReACT AI Research Pipeline Streamlit Frontend
"""
import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    
    print("🚀 Starting ReACT AI Research Pipeline Frontend...")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit>=1.28.0", "plotly>=5.15.0"])
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_app_path = os.path.join(script_dir, "streamlit_app.py")
    
    # Launch Streamlit
    try:
        print("🌐 Launching Streamlit app...")
        print("📱 Open your browser to: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            streamlit_app_path,
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.serverAddress=localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
