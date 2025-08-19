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
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for virtual environment
    venv_python = os.path.join(script_dir, ".venv", "Scripts", "python.exe")
    venv_streamlit = os.path.join(script_dir, ".venv", "Scripts", "streamlit.exe")
    
    if os.path.exists(venv_python):
        print("✅ Using virtual environment")
        python_executable = venv_python
        streamlit_executable = venv_streamlit if os.path.exists(venv_streamlit) else None
    else:
        print("⚠️  Virtual environment not found, using system Python")
        python_executable = sys.executable
        streamlit_executable = None
    
    # Check if streamlit is installed
    try:
        if streamlit_executable and os.path.exists(streamlit_executable):
            print("✅ Streamlit found in virtual environment")
        else:
            import streamlit
            print("✅ Streamlit found in system Python")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([python_executable, "-m", "pip", "install", "streamlit>=1.28.0", "plotly>=5.15.0"])
    
    streamlit_app_path = os.path.join(script_dir, "streamlit_app.py")
    
    # Launch Streamlit
    try:
        print("🌐 Launching Streamlit app...")
        print("📱 Open your browser to: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        if streamlit_executable and os.path.exists(streamlit_executable):
            # Use virtual environment streamlit
            subprocess.run([
                streamlit_executable, "run", 
                streamlit_app_path,
                "--server.port=8501",
                "--server.address=localhost",
                "--browser.serverAddress=localhost"
            ])
        else:
            # Use python -m streamlit
            subprocess.run([
                python_executable, "-m", "streamlit", "run", 
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
