
# 🤖 Tweet Analyzer Setup Guide

## 📸 Quick Glimpse

Below is a snapshot of the NL2SQL agentic application in action:
Prompted to find top users by tweet count.

![App Screenshot](./images/nl2sql.png)

## 📋 Prerequisites
- Python 3.8 or higher
- Ollama installed and running locally
- At least one Ollama model downloaded (recommended: mistral or llama2)

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
Install Jupyter kernel dependencies
```bash
pip install ipykernel jupyter
```

## 3️⃣ Adding the virtual environment as a jupyter kernel.
```bash
python -m ipykernel install --user --name=venv
```

## 4️⃣ Start Ollama Server
Ensure Ollama is running on your machine (default port: 11434)

## 5️⃣ Launch the app
```bash
streamlit run app.py
```

## 💡 Using the App
1. Select an Ollama model in the sidebar settings
2. Adjust the temperature if desired (higher = more creative responses)
3. Enter your query about the Twitter data
4. Click "Submit Query" to analyze

## 📊 Available Data
The app connects to a local DuckDB database containing tweet data. You can preview the dataset structure by expanding the "Preview Dataset" section in the app.

## 🔍 Sample Queries
- "Show me the most retweeted tweets"
- "What are the trending hashtags?"
- "Find tweets with the highest engagement"

## 🛠️ Troubleshooting
- If you see "Ollama server is not running" error, ensure Ollama is started
- If no models appear in the dropdown, download at least one model using `ollama pull mistral`
- Please make sure you have placed the .db file inside nl2sql_agent/database. 