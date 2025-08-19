# PowerShell script to launch the ReACT AI Research Pipeline Streamlit Frontend

Write-Host "🚀 Starting ReACT AI Research Pipeline Frontend..." -ForegroundColor Green
Write-Host "=" * 60

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Install dependencies if needed
Write-Host "📦 Installing/checking dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Launch Streamlit
Write-Host "🌐 Launching Streamlit app..." -ForegroundColor Green
Write-Host "📱 Open your browser to: http://localhost:8501" -ForegroundColor Cyan
Write-Host "🛑 Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "=" * 60

try {
    streamlit run streamlit_app.py --server.port=8501 --server.address=localhost --browser.serverAddress=localhost
} catch {
    Write-Host "❌ Error launching Streamlit. Make sure all dependencies are installed." -ForegroundColor Red
    Write-Host "💡 Try running: pip install streamlit plotly" -ForegroundColor Yellow
}
