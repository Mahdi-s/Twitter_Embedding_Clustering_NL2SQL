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
import re

st.set_page_config(page_title='Tweet Analyzer', layout='wide')

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

##############################################################################
# HELPER: EXTRACT JSON FROM OLLAMA'S "Message(role='assistant', content='...')"
##############################################################################
def extract_json_from_message_string(message_str: str) -> dict:
    """
    Attempts to extract and parse JSON from the 'content' of Ollama's
    response string or dict. If no valid JSON is found, returns {}.
    """
    # If it's already a dict from Ollama's python client, just get the content
    if isinstance(message_str, dict):
        content = message_str.get('content', '')
    else:
        # Try to extract content from string representation
        match = re.search(r"content='([^']*)'", str(message_str), re.DOTALL)
        content = match.group(1) if match else str(message_str)

    # Remove markdown code blocks and JSON markers
    content = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', content, flags=re.DOTALL)
    content = content.strip()

    # Try multiple JSON parsing approaches
    try:
        # First attempt: direct JSON parse
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            # Second attempt: find JSON-like structure within the text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # If all parsing attempts fail, return empty dict
        return {}


##############################################################################
# SYSTEM PROMPT
##############################################################################
SYSTEM_PROMPT = """
You have access to a single DuckDB table called 'tweets' with the following schema:

{columns_info}

You can respond to the user in one of two ways:

1. If the user question requires a query to the 'tweets' table, respond with a JSON function call:
   {{
     "name": "execute_sql_query",
     "arguments": {{
       "query": "SELECT ... "
     }}
   }}

2. If no SQL query is needed, provide a short direct textual answer.

Do not provide any other text when you produce the JSON function call.
Use only the columns exactly as they appear in the schema above.
"""

##############################################################################
# DUCKDB CONNECTION & EXECUTION
##############################################################################
@st.cache_resource
def get_db_connection():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "../tweets2.duckdb")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Cannot find DuckDB at: {db_path}")

    try:
        con = duckdb.connect(db_path, read_only=True, config={'access_mode': 'READ_ONLY'})
        # Quick check
        test_query = con.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
        return con
    except duckdb.CatalogException:
        st.error("The 'tweets' table was not found in the database. Please check your database setup.")
        raise
    except Exception as e:
        st.error(f"Error connecting to DuckDB: {str(e)}")
        raise

def execute_sql_query(query: str) -> str:
    try:
        con = get_db_connection()
        df = con.execute(query).df()
        return json.dumps(df.to_dict(orient='records'))
    except Exception as e:
        return json.dumps({"error": str(e)})

##############################################################################
# OLLAMA LLM INTERACTION
##############################################################################
async def ai_agent_interaction(query: str, model_name: str, temperature: float) -> Dict[str, Any]:
    """
    1. Provide the LLM with the table schema as system prompt.
    2. If the model needs to query, it will return a JSON function call in 'content' or in 'tool_calls'.
    3. Parse that JSON, execute the SQL, then call Ollama a second time for final text if needed.
    """

    client = ollama.AsyncClient()

    # Build system prompt with the table schema
    con = get_db_connection()
    schema_df = con.execute("DESCRIBE tweets").df()
    columns_info = schema_df.to_string(index=False)

    system_prompt = SYSTEM_PROMPT.format(columns_info=columns_info)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    # ----------------------------------------------------------------------
    # FIRST CALL: Get potential tool call or direct text
    # ----------------------------------------------------------------------
    response = await client.chat(
        model=model_name,
        messages=messages,
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
        }],
        options={"temperature": temperature}
    )

    # --- DEBUG STATEMENTS (FIRST CALL) ---
    # st.write("DEBUG: Raw response from Ollama (First Call) ->", response)
    # st.write("DEBUG: type(response) ->", type(response))

    # if isinstance(response["message"], dict):
    #     st.write("DEBUG: response['message'] is a dict -> keys:", list(response["message"].keys()))
    #     st.write("DEBUG: response['message'] ->", response["message"])
    # else:
    #     st.write("DEBUG: response['message'] is NOT a dict ->", response["message"])

    tool_calls = response["message"].get("tool_calls", [])
    #st.write("DEBUG: tool_calls (First Call) ->", tool_calls)

    content = response["message"].get("content", "")
    #st.write("DEBUG: content (First Call) ->", content)

    sql_query = None
    sql_results = None

    # ------------------------------------------------------------------
    # 1) If tool_calls is non-empty, parse it directly
    # ------------------------------------------------------------------
    if not tool_calls:
        # Try to parse content as a function call
        func_call = extract_json_from_message_string(content)
        if func_call.get("name") == "execute_sql_query":
            sql_query = func_call["arguments"].get("query")
            if sql_query:
                query_resp = execute_sql_query(sql_query)
                sql_results = json.loads(query_resp)

                # Instead of using tool messages, we'll add the results as part of the user's next message
                messages.append({
                    "role": "user",
                    "content": f"Here are the results of your SQL query:\n{query_resp}\n\nPlease analyze these results."
                })

                # Make the second call for analysis
                final_response = await client.chat(
                    model=model_name,
                    messages=messages,
                    options={"temperature": temperature}
                )

                return {
                    "sql_query": sql_query,
                    "analysis": final_response["message"]["content"],
                    "sql_results": sql_results
                }

    # ------------------------------------------------------------------
    # 2) If tool_calls is empty, see if the content is actually a JSON tool call
    # ------------------------------------------------------------------
    else:
        # Attempt to parse the content as a tool call
        func_call = extract_json_from_message_string(content)
        if func_call.get("name") == "execute_sql_query":
            sql_query = func_call["arguments"].get("query")
            if sql_query:
                query_resp = execute_sql_query(sql_query)
                sql_results = json.loads(query_resp)

                # Add the tool call + response to messages with correct structure
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "execute_sql_query",
                                # We must provide arguments as a *string*
                                "arguments": json.dumps({"query": sql_query})
                            }
                        }
                    ]
                })
                messages.append({
                    "role": "tool",
                    "content": query_resp,
                    "tool_call_id": "manually_parsed"
                })
        else:
            # No tool call found, just return direct text response
            return {
                "sql_query": None,
                "analysis": content,
                "sql_results": None
            }

    # ----------------------------------------------------------------------
    # SECOND CALL: If we got SQL results, pass them back for final analysis
    # ----------------------------------------------------------------------
    if sql_results:
        final_response = await client.chat(
            model=model_name,
            messages=messages,
            options={"temperature": temperature}
        )

        # --- DEBUG STATEMENTS (SECOND CALL) ---
        #st.write("DEBUG: Raw response from Ollama (Second Call) ->", final_response)
        #st.write("DEBUG: type(final_response) ->", type(final_response))

        #if isinstance(final_response["message"], dict):
            #st.write("DEBUG: final_response['message'] is a dict -> keys:", list(final_response["message"].keys()))
            #st.write("DEBUG: final_response['message'] ->", final_response["message"])
        #else:
            #st.write("DEBUG: final_response['message'] is NOT a dict ->", final_response["message"])

        final_tool_calls = final_response["message"].get("tool_calls", [])
        #st.write("DEBUG: tool_calls (Second Call) ->", final_tool_calls)

        final_content = final_response["message"].get("content", "")
        #st.write("DEBUG: content (Second Call) ->", final_content)

        # Check if there's another tool call
        if final_tool_calls:
            for tool_call in final_tool_calls:
                if tool_call["function"]["name"] == "execute_sql_query":
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                        second_query = arguments.get("query")
                        if second_query:
                            query_resp_2 = execute_sql_query(second_query)
                            sql_results_2 = json.loads(query_resp_2)
                            
                            # Combine results if both are lists
                            if isinstance(sql_results, list) and isinstance(sql_results_2, list):
                                sql_results.extend(sql_results_2)
                            else:
                                sql_results = sql_results_2
                            sql_query = f"{sql_query}\n-- Second query:\n{second_query}"
                    except json.JSONDecodeError as e:
                        st.error(f"Error parsing second tool call arguments: {e}")
            
            final_content = "Here are the combined results from multiple queries."
        
        return {
            "sql_query": sql_query,
            "analysis": final_content,
            "sql_results": sql_results
        }
    else:
        # If we got here without SQL results but had a query, something went wrong
        return {
            "sql_query": sql_query,
            "analysis": "Failed to execute SQL query or no results returned.",
            "sql_results": None
        }

##############################################################################
# OLLAMA SERVER CHECKS & MODEL RETRIEVAL
##############################################################################
def get_available_ollama_models():
    try:
        resp = requests.get('http://localhost:11434/api/tags', timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get('models', [])
            return [m['name'] for m in models if 'name' in m]
        return []
    except Exception as e:
        #st.write("DEBUG: Error retrieving Ollama models ->", e)
        return []

def check_ollama_server():
    try:
        requests.get('http://localhost:11434/api/tags', timeout=5)
        return True
    except requests.exceptions.RequestException as e:
        #st.write("DEBUG: Ollama server check failed ->", e)
        return False

##############################################################################
# STREAMLIT UI
##############################################################################
with st.sidebar:
    st.title('NL2SQL Tweet Analyzer 🐦')

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

# Handle the AI agent query
if submit_query:
    if not model_name:
        st.warning('Please select an Ollama model before submitting a query.')
    elif not check_ollama_server():
        st.error('Ollama server is not running. Please start Ollama and try again.')
    else:
        with st.spinner('Analyzing tweets...'):
            result = asyncio.run(ai_agent_interaction(user_query, model_name, model_temperature))
            # st.write("DEBUG: SQL Query:", result.get('sql_query'))
            # st.write("DEBUG: SQL Results:", result.get('sql_results'))
            # st.write("DEBUG: Analysis:", result.get('analysis'))

            # Store conversation
            st.session_state['chat_history'].append({'role': 'user', 'content': user_query})
            st.session_state['chat_history'].append({'role': 'assistant', 'content': result})

# Display conversation
for i, message_pair in enumerate(zip(st.session_state['chat_history'][::2], st.session_state['chat_history'][1::2])):
    user_message, assistant_message = message_pair
    with st.expander(f"🗣️ {user_message['content']}", expanded=False):
        st.write("**Question:**", user_message['content'])
        
        content_dict = assistant_message['content']
        if isinstance(content_dict, dict):
            # SQL Query
            if content_dict.get('sql_query'):
                st.markdown("#### 🔍 SQL Query")
                st.code(content_dict['sql_query'], language='sql')

            # Query Results
            if content_dict.get('sql_results') is not None:
                st.markdown("#### 📊 Query Results")
                if isinstance(content_dict['sql_results'], list):
                    df = pd.DataFrame(content_dict['sql_results'])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.write(content_dict['sql_results'])

            # Analysis
            if content_dict.get('analysis'):
                st.markdown("#### 📝 Analysis")
                st.write(content_dict['analysis'])
        else:
            # In case it's just plain text
            st.write(content_dict)
