# 🐦 Twitter/X Political Posts Analysis  
### 🚀 USC HUMANS Lab Hackathon 2024


Welcome to my submission for the **USC HUMANS Lab Hackathon 2024**!  

This repository focuses on analyzing political tweets collected from Twitter/X, and demonstrates a pipeline to **extract**, **store**, **query**, and **analyze** the data. 

---

## 📸 Quick Glimpse

Below is a demo of the embedding and clustering experiment in action:


[![Demo Video](https://img.youtube.com/vi/sD0ibVYFg4c/0.jpg)](https://www.youtube.com/watch?v=sD0ibVYFg4c)


---

## 🗂 Repository Contents

1.**`create_db.ipynb`**  
   - Responsible for extracting compressed tweet data and **saving** it into a SQL database. Note that I have placed a limit to only parsing the first 5 parts of the db.

2. **`Analysis.ipynb`**  
   - A Jupyter Notebook demonstrating how to **explore** and **analyze** tweets stored in the SQL database. 

3. **`nl2sql_agent/`**  
   - A folder containing an AI-powered agent that converts **natural language** queries into **SQL**.  
   - For setup and usage instructions, please read the dedicated `README.md` inside this folder.

4. **`embedding_and_clustering/`**
   - Interactive visualization tool for exploring tweet clusters using embeddings
   - Features:
     - Real-time embedding generation using Ollama models
     - Dynamic clustering with adjustable parameters
     - Interactive 3D visualization of tweet clusters
     - Automatic cluster summarization and labeling
     - Caching system for faster subsequent runs

5. **`scope.pdf`**  
   - A brief **project scope** outline describing the goals, approach, and potential future directions.


---

## 🤖 About the NL2SQL Agent

The **NL2SQL Agent** is a tool that:
- Translates natural language into SQL queries on the fly.
- Executes those queries against the tweet database to quickly retrieve relevant information.
- Allows for rapid **exploratory data analysis** without writing raw SQL.

For a detailed guide on installation and usage, head to the [`nl2sql_agent` folder](./nl2sql_agent/README.md).

---

## 📊 About the Embedding & Clustering Tool

The **Embedding & Clustering Tool** enables:
- Generation of tweet embeddings using any Ollama-compatible model
- Automatic clustering of similar tweets
- Interactive 3D visualization with detailed hover information
- AI-powered cluster summarization and labeling
- Performance optimizations through caching

For detailed setup and usage, check the [`embedding_and_clustering` folder](./embedding_and_clustering/README.md).


---

## 💡 How It Works

1. **Data Extraction**  
   Use `create_db.ipynb` to unpack the compressed tweets and insert them into a SQL database.
   
2. **Data Exploration**  
   Fire up `Analysis.ipynb` to explore trends, anomalies, or other interesting facets in the data.
   
3. **Natural Language Queries**  
   Interact with the `nl2sql_agent/` to seamlessly query the database using English prompts.

4. **Embedding & Clustering Analysis**
   - **Generate Embeddings**: Convert tweets into vector representations using Ollama models
   - **Cluster Formation**: Group similar tweets using k-means clustering
   - **Visualization**: Explore clusters in an interactive 3D space
   - **Insights**: 
     - Hover over points to read tweet content
     - View AI-generated cluster summaries
     - Adjust clustering parameters in real-time
     - Export findings for further analysis


---

## 🎯 Why This Matters

By cleaning and structuring large volumes of Twitter/X data, we can uncover:
- **Topic clusters** (political or otherwise).
- **Anomalies** or out-of-place chatter.
- **Sentiment trends** and **network relationships** among users.  

This approach offers a glimpse into how modern NLP and database management can help us **understand** and **visualize** online political discourse.

---

## 🫶 Notable Resources Utilizes:

- **Streamlit:** Ease creation of user interface.
- **Ollama:** For loading models and utilizing tool calling.
- **Plotly:** Interactive 3D visualization of tweet clusters.
- **scikit-learn:** Clustering and dimensionality reduction.
- **Generative AI:** Portions of the code in this repo was generated using AI.

---
