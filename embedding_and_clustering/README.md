
# 🤖 Embedding and Clustering Experiments


## 📋 Prerequisites
- Python 3.8 or higher
- Ollama installed and running locally
- At least one Ollama model downloaded (recommended: gemma or llama3)

## 🚀 Development Setup

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
Activate the virtual environment
```

On macOS/Linux:
```bash
source venv/bin/activate
```
On Windows:
```bash
.\venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 3️⃣ Start Ollama Server
Ensure Ollama is running on your machine (default port: 11434)

## 4️⃣ Launch the app
```bash
streamlit run app.py
```

## 🛠️ Troubleshooting
- If you see "Ollama server is not running" error, ensure Ollama is started
- If no models appear in the dropdown, download at least one model using `ollama pull mistral`