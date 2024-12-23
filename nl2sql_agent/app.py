import streamlit as st
import pandas as pd
import requests
import ollama
import duckdb
import json
import asyncio
from typing import Dict, Any
import os
import time

# ---------------------------------------------------------------------
#                    Streamlit / App Configuration
# ---------------------------------------------------------------------
st.set_page_config(page_title='Tweet Analyzer', layout='wide')

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

SYSTEM_PROMPT = """
You are a data analysis assistant with access to a DuckDB table named 'tweets'. 
Below is the schema of 'tweets' (from DESCRIBE tweets):

{columns_info}

Your tasks:

1. **Analyze the user's question** and decide if you need to execute one or more SQL queries against the 'tweets' table. 
   - You must only reference this 'tweets' table (no other tables exist).
   - Use the column names exactly as they appear above.
   - If the question is ambiguous or impossible to answer, explain that in your final answer.

2. If you need to run an SQL query, you MUST provide a function call in valid JSON with:
   - `"name": "execute_sql_query"`
   - `"arguments": {{ "query": "<YOUR_SQL_QUERY>" }}`
   
   Example function call:
   ```json
   {{
     "name": "execute_sql_query",
     "arguments": {{
       "query": "SELECT * FROM tweets LIMIT 5"
     }}
   }}
   ```

Do not include any extra text or formatting in that JSON.
After you receive the function-call result (the query results), you can provide a final answer as role=assistant, with your explanation/analysis. This final answer should refer to the query results if needed to answer the user's question.

Do not output any Python or other code besides the JSON function call. If no SQL query is needed, just provide a direct final textual answer as role=assistant.

Do not add or remove columns arbitrarily; you only have the columns listed in the schema above.

The user says: {user_query}
"""

# ---------------------------------------------------------------------
#                    DuckDB Connection & SQL Execution
# ---------------------------------------------------------------------
@st.cache_resource
def get_db_connection():
    """
    Creates (and caches) a DuckDB connection in read-only mode
    to the actual 'tweets.duckdb' that contains your real data.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "database", "tweets1.duckdb")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Cannot find DuckDB at: {db_path}")

    try:
        # Initialize connection with 'tweets' table, adding access_mode parameter
        con = duckdb.connect(db_path, read_only=True, config={'access_mode': 'READ_ONLY'})
        
        # Add retry logic for lock conflicts
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                test_query = con.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
                #st.write(f"DEBUG: Successfully connected to database. Found {test_query} tweets.")
                return con
            except duckdb.IOException as lock_error:
                if "lock" in str(lock_error).lower() and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise
    except duckdb.CatalogException:
        st.error("The 'tweets' table was not found in the database. Please check your database setup.")
        raise
    except Exception as e:
        st.error(f"Error connecting to DuckDB: {str(e)}")
        raise

def execute_sql_query(query: str) -> str:
    """
    Execute the given SQL query against the 'tweets' table in DuckDB.
    Return JSON-serialized rows.
    """
    #st.write("DEBUG: About to execute SQL ->", query)
    try:
        con = get_db_connection()
        df = con.execute(query).df()
        #st.write("DEBUG: Query executed successfully. Rows/Cols:", df.shape)
        return json.dumps(df.to_dict(orient='records'))
    except Exception as e:
        #st.write(f"DEBUG: Error executing query -> {e}")
        return json.dumps({"error": str(e)})

# ---------------------------------------------------------------------
#                   Ollama LLM Interaction
# ---------------------------------------------------------------------
async def ai_agent_interaction(query: str, model_name: str, temperature: float) -> Dict[str, Any]:
    """
    1. Provide the LLM with the actual table schema, specifying that the table is called "tweets".
    2. Let it generate a function call with a query (execute_sql_query).
    3. Execute that query, return the final response plus query results.
    """
    client = ollama.AsyncClient()

    # Grab current schema from DuckDB
    con = get_db_connection()
    schema_df = con.execute("DESCRIBE tweets").df()
    # Turn the DESCRIBE results into a string
    columns_info = schema_df.to_string(index=False)

    # Build our final system prompt by substituting the {columns_info} and {query} placeholders
    system_prompt = SYSTEM_PROMPT.format(
        columns_info=columns_info,
        user_query=query
    )

    # DEBUG: Show what we’re sending to the model
    #st.write("DEBUG: System Prompt ->", system_prompt)

    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    # First request to Ollama: may produce a function_call
    response = await client.chat(
        model=model_name,
        messages=messages,
        options={"temperature": temperature},
        tools=[{
            "type": "function",
            "function": {
                "name": "execute_sql_query",
                "description": "Execute SQL query against the 'tweets' table",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute"
                        }
                    },
                    "required": ["query"]
                }
            }
        }]
    )

    #st.write("DEBUG: First Ollama response ->", response)

    # Keep track of conversation
    messages.append(response["message"])

    sql_query = None
    sql_results = None

    function_call = response["message"].get("function_call")
    tool_calls = response["message"].get("tool_calls")

    # If there's a new style function call
    if function_call and function_call.get("name") == "execute_sql_query":
        try:
            # Fix: Handle the arguments parsing more carefully
            args = json.loads(function_call["arguments"]) if isinstance(function_call["arguments"], str) else function_call["arguments"]
            sql_query = args.get("query")
            if not sql_query:
                #st.write("DEBUG: No query found in function arguments")
                return {"error": "No SQL query provided in function arguments"}
            
            #st.write("DEBUG: Model-proposed SQL query (function_call) ->", sql_query)
            function_resp = execute_sql_query(sql_query)
            sql_results = json.loads(function_resp)
            messages.append({
                "role": "tool",
                "name": function_call["name"],
                "content": function_resp
            })
        except Exception as e:
            st.write(f"DEBUG: Error processing function call -> {e}")
            sql_results = {"error": str(e)}

    # If there's an older style "tool_calls"
    elif tool_calls:
        # In older Ollama versions, tool_calls is a list of function calls
        for tc in tool_calls:
            if tc["function"]["name"] == "execute_sql_query":
                try:
                    query_args = tc["function"]["arguments"]
                    sql_query = query_args.get("query", "")
                    #st.write("DEBUG: Model-proposed SQL query (tool_calls) ->", sql_query)
                    function_resp = execute_sql_query(sql_query)
                    sql_results = json.loads(function_resp)
                    # Add the result as a new tool message
                    messages.append({
                        "role": "tool",
                        "name": "execute_sql_query",
                        "content": function_resp
                    })
                except Exception as e:
                    sql_results = {"error": str(e)}

    # Make a second call to get final analysis after we have the query results
    final_response = await client.chat(
        model=model_name,
        messages=messages
    )
    #st.write("DEBUG: Final Ollama response ->", final_response)

    final_analysis = final_response["message"].get("content", "")

    return {
        "sql_query": sql_query,
        "analysis": final_analysis,
        "sql_results": sql_results
    }

# ---------------------------------------------------------------------
#      Utility: Ollama server checks & model retrieval
# ---------------------------------------------------------------------
def get_available_ollama_models():
    """
    Retrieve a list of available Ollama models from the local Ollama server.
    """
    try:
        resp = requests.get('http://localhost:11434/api/tags', timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get('models', [])
            return [m['name'] for m in models if 'name' in m]
        return []
    except Exception as e:
        st.write("DEBUG: Error retrieving Ollama models ->", e)
        return []

def check_ollama_server():
    """
    Check if the Ollama server is running locally.
    """
    try:
        requests.get('http://localhost:11434/api/tags', timeout=5)
        return True
    except requests.exceptions.RequestException as e:
        st.write("DEBUG: Ollama server check failed ->", e)
        return False

# ---------------------------------------------------------------------
#                           Streamlit UI
# ---------------------------------------------------------------------
with st.sidebar:
    st.title('Tweet Analyzer 🐦')

    # Model Settings
    with st.expander("🛠️ Model Settings", expanded=False):
        if check_ollama_server():
            available_models = get_available_ollama_models()
            if available_models:
                model_name = st.selectbox('Select Ollama Model', available_models)
            else:
                st.warning('No models found. Please ensure Ollama models are loaded.')
                model_name = None
        else:
            st.error('Ollama server is not running. Please start Ollama to use the AI agent.')
            model_name = None
        
        model_temperature = st.slider('Model Temperature', min_value=0.0, max_value=1.0, value=0.7)

    st.header('AI Agent Query')
    user_query = st.text_area('Enter your query about the Twitter data:', height=100)
    submit_query = st.button('Submit Query')

# Main content area
st.header('Tweet Analysis')

# Preview the dataset
with st.expander("📊 Preview Dataset", expanded=False):
    try:
        con = get_db_connection()
        total_rows = con.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
        st.write(f"Total tweets in database: {total_rows:,}")

        sample_df = con.execute("SELECT * FROM tweets LIMIT 5").df()
        st.write(f"Sample of {sample_df.shape[0]} tweets ({sample_df.shape[1]} columns):")
        st.dataframe(sample_df)
    except Exception as e:
        st.error(f"Error loading preview: {str(e)}")

# Run the AI agent query when the user clicks "Submit Query"
if submit_query:
    if not model_name:
        st.warning('Please select an Ollama model before submitting a query.')
    elif not check_ollama_server():
        st.error('Ollama server is not running. Please start Ollama and try again.')
    else:
        with st.spinner('Analyzing tweets...'):
            #st.write("DEBUG: Sending user query to ai_agent_interaction:", user_query)
            result = asyncio.run(ai_agent_interaction(user_query, model_name, model_temperature))

            #st.write("DEBUG: Final result from AI interaction:", result)

            # Update chat history
            st.session_state['chat_history'].append({
                'role': 'user',
                'content': user_query
            })
            st.session_state['chat_history'].append({
                'role': 'assistant',
                'content': result  
            })

# Display the chat messages
for i, message_pair in enumerate(zip(st.session_state['chat_history'][::2], st.session_state['chat_history'][1::2])):
    user_message, assistant_message = message_pair
    
    # Create a single expander for the entire exchange
    with st.expander(f"🗣️ {user_message['content']}", expanded=True):
        # User Query
        st.write("Question:", user_message['content'])
        
        # Assistant Response
        content_dict = assistant_message['content']
        if isinstance(content_dict, dict):
            # SQL Query Section
            if content_dict.get('sql_query'):
                st.markdown("#### 🔍 SQL Query")
                st.code(content_dict['sql_query'], language='sql')

            # Query Results Section
            if content_dict.get('sql_results') is not None:
                st.markdown("#### 📊 Query Results")
                if isinstance(content_dict['sql_results'], list):
                    df = pd.DataFrame(content_dict['sql_results'])
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.write(content_dict['sql_results'])

            # # Analysis Section
            # if content_dict.get('analysis'):
            #     st.markdown("#### 📝 Analysis")
            #     st.write(content_dict['analysis'])
        else:
            st.write(content_dict)

# Move debug messages to use st.debug instead of st.write
def debug_message(msg, data):
    st.debug(f"{msg}: {data}")