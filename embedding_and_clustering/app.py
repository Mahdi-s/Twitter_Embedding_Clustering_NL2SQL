import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import re
import time
import duckdb

# Constants
DEFAULT_OLLAMA_URL = 'http://localhost:11434'
CACHE_FOLDER = "embeddings_cache"

os.makedirs(CACHE_FOLDER, exist_ok=True)

############################
#   Time Formatting Helper
############################
def format_time(seconds):
    """
    Convert a float of seconds into a string 'H hours, M minutes, S seconds'.
    Only shows non-zero components. If total time < 1 second, show '0 seconds'.
    """
    seconds = int(seconds)
    if seconds <= 0:
        return "0 seconds"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    time_str_parts = []
    if hours > 0:
        time_str_parts.append(f"{hours} hours")
    if minutes > 0:
        time_str_parts.append(f"{minutes} minutes")
    if secs > 0:
        time_str_parts.append(f"{secs} seconds")

    return ", ".join(time_str_parts)

############################
#   Ollama Connection
############################
def check_ollama_connection(ollama_url):
    try:
        requests.get(f'{ollama_url}/api/tags', timeout=5).raise_for_status()
        return True
    except requests.RequestException:
        return False

def get_ollama_models(ollama_url):
    try:
        response = requests.get(f'{ollama_url}/api/tags', timeout=5)
        response.raise_for_status()
        return [model['name'] for model in response.json()['models']]
    except requests.RequestException as e:
        st.sidebar.error(f"Error connecting to Ollama: {str(e)}")
        return []

############################
#   Embeddings & Caching
############################
def get_embeddings(ollama_url, model, texts, cache_prefix):
    """
    Generate embeddings for the given texts using the specified Ollama model.
    Includes a per-text progress bar and time estimate.
    """
    # Ensure texts are strings
    texts = [str(text) for text in texts]
    
    cache_file = os.path.join(CACHE_FOLDER, f"{cache_prefix}_{model}.npz")

    # Attempt to load from cache
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        cached_embeddings = data['embeddings']
        cached_row_count = int(data['row_count'])
        cached_model_name = str(data['model_name'])

        # If same row count & model name, use cache
        if cached_row_count == len(texts) and cached_model_name == model:
            st.info("Loading embeddings from cache...")
            return cached_embeddings

    st.write("Computing embeddings (this might take a while).")
    embedding_progress_bar = st.progress(0)
    embedding_time_text = st.empty()

    embeddings = []
    start_time = time.time()
    for i, text in enumerate(texts):
        text_str = str(text)
        try:
            response = requests.post(
                f'{ollama_url}/api/embeddings',
                json={'model': model, 'prompt': text_str},
                timeout=30
            )
            response.raise_for_status()
            emb = response.json()['embedding']
            embeddings.append(emb)
        except requests.RequestException as e:
            st.error(f"Error getting embedding for row {i}: {str(e)}")
            st.error(f"Problematic text: {text_str[:100]}...")
            return None
        except KeyError as e:
            st.error(f"Unexpected response format for row {i}: {str(e)}")
            return None

        # Update progress bar
        current_count = i + 1
        fraction_done = current_count / len(texts)
        embedding_progress_bar.progress(fraction_done)

        # Time estimate
        elapsed = time.time() - start_time
        avg_time_per_text = elapsed / current_count
        est_remaining = avg_time_per_text * (len(texts) - current_count)
        est_remaining_str = format_time(est_remaining)
        embedding_time_text.text(
            f"Embedding {current_count}/{len(texts)} completed. "
            f"Estimated time remaining: {est_remaining_str}"
        )

    embeddings_array = np.array(embeddings)

    # Save to cache
    np.savez(
        cache_file,
        embeddings=embeddings_array,
        row_count=len(texts),
        model_name=model
    )
    st.info("Embeddings computed and cached.")
    return embeddings_array

############################
#  Summaries
############################
def generate_leaf_summary(
    ollama_url, 
    model, 
    leaf_texts, 
    sample_size, 
    temperature, 
    prompt_template,
    parent_name=None  # <-- NEW
):
    """
    LLM call for leaf clusters:
    - sample up to 'sample_size' tweets from 'leaf_texts'.
    - pass them to the LLM with 'prompt_template'.
    - the prompt is aware of parent_name.
    """
    sampled_texts = leaf_texts[:sample_size]  # simple front-truncation
    prompt = prompt_template.format(
        parent_name=parent_name if parent_name else "No assigned parent name",
        texts="\n".join(sampled_texts)
    )

    try:
        response = requests.post(
            f'{ollama_url}/api/generate',
            json={'model': model, 'prompt': prompt, 'temperature': temperature},
            timeout=30
        )
        response.raise_for_status()
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_resp = json.loads(line)
                    if 'response' in json_resp:
                        full_response += json_resp['response']
                except json.JSONDecodeError:
                    continue
        final = full_response.strip()
        return final if final else "Generic Leaf Topic"
    except requests.RequestException as e:
        st.error(f"Error in generate_leaf_summary: {str(e)}")
        return "Generic Leaf Topic"


def generate_parent_summary(
    ollama_url,
    model,
    child_nodes,  # list of child ClusterNodes
    parent_sample_size,
    temperature,
    prompt_template,
    parent_name=None  # <-- NEW
):
    """
    LLM call for parent clusters:
    - gather up to 'parent_sample_size' tweets from EACH child's texts.
    - also gather child cluster names.
    - pass them all in one prompt to the LLM, being aware of parent_name.
    """
    child_names = []
    sample_texts = []
    for child in child_nodes:
        if child.name:
            child_names.append(child.name)
        # sample from the child's tweets
        sampled = child.texts[:parent_sample_size]
        # label them with the child's name for clarity
        sample_texts.append(f"Child '{child.name}':\n" + "\n".join(sampled))

    # Combine child cluster names
    joined_child_names = " | ".join(child_names) if child_names else "No child names"
    # Combine sample tweets
    joined_samples = "\n\n".join(sample_texts)

    # Format the prompt
    prompt = prompt_template.format(
        parent_name=parent_name if parent_name else "No assigned parent name",
        child_cluster_names=joined_child_names,
        child_samples=joined_samples
    )

    try:
        response = requests.post(
            f'{ollama_url}/api/generate',
            json={'model': model, 'prompt': prompt, 'temperature': temperature},
            timeout=30
        )
        response.raise_for_status()
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_resp = json.loads(line)
                    if 'response' in json_resp:
                        full_response += json_resp['response']
                except json.JSONDecodeError:
                    continue
        final = full_response.strip()
        return final if final else "Generic Parent Topic"
    except requests.RequestException as e:
        st.error(f"Error in generate_parent_summary: {str(e)}")
        return "Generic Parent Topic"

############################
#  Data Loading
############################
def load_data_from_duckdb(db_path, table_name, limit):
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"SELECT * FROM {table_name} LIMIT {limit}").df()
    con.close()
    return df

############################
#  Tree Node Class
############################
class ClusterNode:
    """
    Represents a node in the top-down hierarchy.
    """
    def __init__(self, level, indexes, parent=None):
        self.level = level            # which level in [1..num_levels]
        self.indexes = indexes        # indices of data points at this node
        self.parent = parent
        self.children = []            # list[ClusterNode]
        self.name = None              # LLM-generated name
        self.embeddings = None        # slice of embeddings for these indexes
        self.texts = None             # slice of the original texts for these indexes

def build_top_down_tree(embeddings, texts, indexes, current_level, max_levels, k, parent=None):
    """
    Recursively split the data into K sub-clusters if current_level < max_levels,
    otherwise this node is a leaf.
    """
    node = ClusterNode(level=current_level, indexes=indexes, parent=parent)
    node.embeddings = embeddings[indexes]
    node.texts = [texts[i] for i in indexes]

    if current_level < max_levels:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(node.embeddings)
        
        for cluster_label in range(k):
            child_idxes = [idx for idx, lbl in zip(indexes, labels) if lbl == cluster_label]
            if len(child_idxes) == 0:
                st.write(f"Warning: Empty cluster {cluster_label}")
                continue

            child_node = build_top_down_tree(
                embeddings=embeddings,
                texts=texts,
                indexes=child_idxes,
                current_level=current_level + 1,
                max_levels=max_levels,
                k=k,
                parent=node
            )
            node.children.append(child_node)

    return node

def count_nodes(node):
    """
    Count how many nodes in the entire subtree. We use this for a naming progress bar.
    """
    total = 1
    for c in node.children:
        total += count_nodes(c)
    return total

def postorder_naming(
    node,
    naming_progress_bar,
    naming_time_text,
    start_time,
    named_count,
    total_nodes,
    leaf_prompt_template,
    parent_prompt_template,
    leaf_sample_size,
    parent_sample_size,
    temperature,
    summary_model,
    ollama_url
):
    """
    Post-order DFS for naming:
      - name children first
      - then name this node (differently if leaf or parent)
      - pass parent's name to the prompt so LLM can be aware of previously assigned names.
    """
    for child in node.children:
        postorder_naming(
            child,
            naming_progress_bar,
            naming_time_text,
            start_time,
            named_count,
            total_nodes,
            leaf_prompt_template,
            parent_prompt_template,
            leaf_sample_size,
            parent_sample_size,
            temperature,
            summary_model,
            ollama_url
        )

    # Now name `node`
    named_count[0] += 1
    fraction_done = named_count[0] / total_nodes
    naming_progress_bar.progress(fraction_done)

    elapsed = time.time() - start_time
    avg_time_per_node = elapsed / named_count[0]
    est_remaining = avg_time_per_node * (total_nodes - named_count[0])
    est_remaining_str = format_time(est_remaining)
    naming_time_text.text(
        f"Naming cluster {named_count[0]}/{total_nodes}. "
        f"Estimated time remaining: {est_remaining_str}"
    )

    # Get parent name (if it exists), to pass to the prompt
    parent_name = node.parent.name if (node.parent and node.parent.name) else "No assigned parent name"

    if len(node.children) == 0:
        # Leaf node => use leaf_prompt_template
        node.name = generate_leaf_summary(
            ollama_url=ollama_url,
            model=summary_model,
            leaf_texts=node.texts,
            sample_size=leaf_sample_size,
            temperature=temperature,
            prompt_template=leaf_prompt_template,
            parent_name=parent_name
        )
    else:
        # Parent node => use parent_prompt_template, gather child names + samples
        node.name = generate_parent_summary(
            ollama_url=ollama_url,
            model=summary_model,
            child_nodes=node.children,
            parent_sample_size=parent_sample_size,
            temperature=temperature,
            prompt_template=parent_prompt_template,
            parent_name=parent_name
        )

def extract_leaves(node, leaf_list):
    """
    Collect leaf nodes (no children) in leaf_list.
    """
    if len(node.children) == 0:
        leaf_list.append(node)
    else:
        for c in node.children:
            extract_leaves(c, leaf_list)

def build_taxonomy_text(node, indent_level=0):
    """
    Create an indented textual representation of the tree.
    """
    lines = []
    indent = "  " * indent_level
    node_label = node.name if node.name else "Unnamed"
    lines.append(f"{indent}- {node_label}")
    for c in node.children:
        lines.extend(build_taxonomy_text(c, indent_level+1))
    return lines

############################
#  NEW: Gathering data for Plotly Treemap
############################
def gather_paths(node, path_so_far=None):
    """
    Return a list of (path, size) for every node in the hierarchy.
    Only leaf nodes will have sizes, parent nodes will have size=0.
    """
    if path_so_far is None:
        path_so_far = []
    current_path = path_so_far + [node.name or "Unnamed"]
    
    # If this is a leaf node, return its data
    if len(node.children) == 0:
        return [(current_path, len(node.indexes))]
    
    # If it's a parent node, only gather data from children
    data = []
    for child in node.children:
        data.extend(gather_paths(child, current_path))
    return data

def create_taxonomy_treemap(root_node):
    """
    Builds a Treemap figure from the hierarchical data using Plotly Express.
    Only leaf nodes will have sizes.
    """
    paths_and_sizes = gather_paths(root_node, [])
    
    # Create separate columns for each level of the path
    records = []
    max_depth = max(len(path) for path, _ in paths_and_sizes)
    
    for path, size in paths_and_sizes:
        # Pad path with None values if it's shorter than max_depth
        padded_path = path + [None] * (max_depth - len(path))
        record = {f"level_{i}": name for i, name in enumerate(padded_path)}
        record["size"] = size
        records.append(record)

    df_treemap = pd.DataFrame(records)
    
    # Create path list for treemap
    path_cols = [col for col in df_treemap.columns if col.startswith("level_")]
    
    # Create a Treemap
    fig = px.treemap(
        df_treemap,
        path=path_cols,   # list of columns for path levels
        values='size',    # the numerical column
        color='size',
        color_continuous_scale='Blues',
        branchvalues='total'  # This ensures proper size calculations
    )
    
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        title="Taxonomy Treemap (Hierarchical Naming)"
    )
    return fig

############################
#   Dimensionality Reduction
############################
def perform_dimensionality_reduction(embeddings, algorithm, n_components=3, **params):
    if algorithm == "t-SNE":
        reducer = TSNE(n_components=n_components, random_state=42, n_iter=params.get('n_iter', 1000))
    else:
        reducer = PCA(n_components=n_components, random_state=42)
    return reducer.fit_transform(embeddings)

############################
#  Visualization & Summaries
############################
def create_leaf_summary_table(leaf_nodes):
    """
    Build a summary table from the final leaf nodes.
    """
    cluster_summary = []
    total_points = 0
    for leaf in leaf_nodes:
        total_points += len(leaf.indexes)

    for i, leaf in enumerate(leaf_nodes):
        size = len(leaf.indexes)
        pct = (size / total_points * 100) if total_points else 0
        examples = leaf.texts[:3]
        cluster_summary.append({
            "Leaf ID": i,
            "Cluster Title": leaf.name or "Unnamed Leaf",
            "# of Items": size,
            "Percentage": pct,
            "Examples": "\n".join(str(x) for x in examples)
        })

    df_summary = pd.DataFrame(cluster_summary)
    df_summary = df_summary.sort_values("# of Items", ascending=False).reset_index(drop=True)
    return df_summary

def clean_html_tags(text):
    """
    Removes any <...> HTML tags from the string so they do not appear as literal HTML.
    """
    return re.sub(r'<[^>]*>', '', text)

def create_3d_scatter_plot(
    reduced_embeddings,
    leaf_nodes,
    df,
    dot_size=5,
    show_legend=True
):
    """
    Color each leaf cluster differently. We'll add a 'leaf_id' to each row 
    to indicate which leaf node it belongs to.
    """
    # Build a map from data index -> leaf_id
    index_to_leaf = {}
    for i, leaf in enumerate(leaf_nodes):
        for idx in leaf.indexes:
            index_to_leaf[idx] = i

    # Create a working copy of df
    df_plot = df.copy()
    df_plot['x'] = reduced_embeddings[:, 0]
    df_plot['y'] = reduced_embeddings[:, 1]
    df_plot['z'] = reduced_embeddings[:, 2]
    df_plot['leaf_id'] = df_plot.index.map(lambda i: index_to_leaf.get(i, -1))
    df_plot['leaf_name'] = df_plot['leaf_id'].apply(
        lambda lid: leaf_nodes[lid].name if lid >= 0 and lid < len(leaf_nodes) else "Unknown"
    )

    unique_leaf_ids = sorted(set(df_plot['leaf_id']))
    color_map = {
        lid: f'rgb({np.random.randint(0,256)}, {np.random.randint(0,256)}, {np.random.randint(0,256)})'
        for lid in unique_leaf_ids
        if lid != -1
    }

    fig = go.Figure()

    def wrap_text(text, width=40):
        text = clean_html_tags(str(text))
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            word_length = len(word)
            if current_length + word_length + 1 <= width:
                current_line.append(word)
                current_length += word_length + 1
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length + 1
        if current_line:
            lines.append(' '.join(current_line))
        return '<br>'.join(lines)

    def make_customdata(row):
        cluster_name = wrap_text(row.get('leaf_name', ''))
        tweet_text = wrap_text(row.get('text', ''))
        username = row.get('username', '')
        bio = wrap_text(str(row.get('bio', '')), width=50)
        followers_count = row.get('followersCount', '')
        location = row.get('location', '')
        favourites_count = row.get('likeCount', '')
        metadata_tuple = (
            username,
            bio,
            followers_count,
            location,
            favourites_count
        )
        return (cluster_name, tweet_text, metadata_tuple)

    df_plot['customdata'] = df_plot.apply(make_customdata, axis=1)

    for lid in unique_leaf_ids:
        if lid == -1:
            continue
        cluster_data = df_plot[df_plot['leaf_id'] == lid]
        leaf_name = leaf_nodes[lid].name or "Unknown"
        cluster_size = len(leaf_nodes[lid].indexes)

        hover_template = """
        <b>Cluster Name:</b><br>
        %{customdata[0]}
        <br><span style='color: white'>━━━━━━━━━━━━━━━━━━━━━━━━</span><br>
        <b>Tweet Text:</b><br>
        %{customdata[1]}
        <br><span style='color: white'>━━━━━━━━━━━━━━━━━━━━━━━━</span><br>
        <b>Tweet Metadata:</b><br>
        Username: %{customdata[2][0]}<br>
        Bio: %{customdata[2][1]}<br>
        Followers: %{customdata[2][2]}<br>
        Location: %{customdata[2][3]}<br>
        Favorites: %{customdata[2][4]}
        <extra></extra>
        """

        fig.add_trace(go.Scatter3d(
            x=cluster_data['x'],
            y=cluster_data['y'],
            z=cluster_data['z'],
            mode='markers',
            marker=dict(
                size=dot_size,
                color=color_map[lid],
                opacity=0.8
            ),
            customdata=cluster_data['customdata'],
            hovertemplate=hover_template,
            hoverlabel=dict(
                bgcolor=color_map[lid],
                font=dict(color='white', size=12),
                align='left',
                namelength=-1
            ),
            name=f"Leaf {lid}: {leaf_name} ({cluster_size} items)"
        ))

    fig.update_layout(
        showlegend=show_legend,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1000,
        height=800,
        title="3D Text Clustering Visualization (Top-Down Hierarchy)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            bordercolor="black",
            namelength=-1
        )
    )

    return fig

############################
#              MAIN
############################
def main():
    st.set_page_config(page_title="TextCluster", page_icon="📊", layout="wide")
    st.title("📊 Embedding & Top-Down Hierarchical Clustering")

    # Session state
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.embeddings = None

    if 'ollama_url' not in st.session_state:
        st.session_state.ollama_url = DEFAULT_OLLAMA_URL

    with st.sidebar:
        st.header("Settings")
        with st.expander("🛠️ Ollama Settings", expanded=False):
            ollama_url = st.text_input("Ollama Server URL:", value=st.session_state.ollama_url)
            if ollama_url != st.session_state.ollama_url:
                st.session_state.ollama_url = ollama_url

            if not check_ollama_connection(st.session_state.ollama_url):
                st.error(f"Cannot connect to Ollama server at {st.session_state.ollama_url}. Make sure it's running.")
            else:
                models = get_ollama_models(st.session_state.ollama_url)
                if not models:
                    st.error("No Ollama models available. Check your Ollama installation.")

        # Collapsable to load .npz embedding cache
        with st.expander("🔃 Load Embedding Cache", expanded=False):
            uploaded_file = st.file_uploader("Upload .npz embedding cache")
            if uploaded_file is not None:
                try:
                    data = np.load(uploaded_file, allow_pickle=True)
                    st.session_state.embeddings = data['embeddings']
                    st.write("Embeddings loaded from uploaded cache!")
                except Exception as e:
                    st.error(f"Error loading embeddings cache: {e}")

        st.subheader("DuckDB Configuration")
        db_path = "../tweets2.duckdb"  # Adjust path as needed
        table_name = st.text_input("DuckDB Table Name", "tweets_table")
        row_limit = st.number_input("Number of rows to load", min_value=1, max_value=1_000_000, value=1000, step=100)

        load_data_button = st.button("Load Data from DuckDB")

        if st.session_state.df is not None:
            col_options = list(st.session_state.df.columns)
            selected_column_name = st.selectbox("Select text column", col_options)
        else:
            selected_column_name = None

        # Model selection
        if check_ollama_connection(st.session_state.ollama_url):
            embedding_models = get_ollama_models(st.session_state.ollama_url)
            selected_embedding_model = st.selectbox("Select embedding model", embedding_models)
            summary_model = st.selectbox("Select summary model", embedding_models)
        else:
            embedding_models = []
            selected_embedding_model = None
            summary_model = None

        # ------------------------------
        # UPDATED PROMPTS
        # ------------------------------
        st.subheader("LLM Prompts & Sampling")
        
        # Leaf prompt: uses {parent_name} and {texts}
        leaf_prompt_template = st.text_area(
            "Leaf Prompt Template",
            """
You are a helpful AI assistant specialized in text analysis.
We have a leaf cluster, meaning it has no children. 
The parent of this leaf cluster has been named '{parent_name}'.
Below are sample texts from this leaf cluster:

{texts}

We are naming clusters from the bottom up. 
Given these sample texts and the parent's name, 
propose a short, plain-text name that captures the main theme or topic of this leaf cluster. 
Return only the short descriptive name without extra commentary.
""".strip()
        )
        
        # Parent prompt: uses {parent_name}, {child_cluster_names}, {child_samples}
        parent_prompt_template = st.text_area(
            "Parent Prompt Template",
"""
You are a helpful AI assistant specialized in text analysis.
We are building a hierarchy from the bottom up. 
This parent cluster is formed by grouping the following child clusters from the level below, each already assigned a name:

Child cluster names: {child_cluster_names}

Below are sample texts from each child cluster:
{child_samples}

The parent cluster itself belongs to a higher-level cluster named '{parent_name}' (if applicable).

Based on the child cluster names and their sample texts, 
propose a single short, plain-text name that best captures the overarching theme of these child clusters. 
Return only the short descriptive name without extra commentary.
""".strip()
        )

        leaf_sample_size = st.slider("Leaf Sampling Rate", 1, 100, 5, 1)
        parent_sample_size = st.slider("Parent Sampling Rate", 1, 100, 5, 1)
        summary_temperature = st.slider("Summary Temperature", 0.0, 1.0, 0.1, 0.1)

        st.subheader("Hierarchy Configuration")
        n_clusters = st.slider("K (clusters at each level)", 2, 20, 3)
        num_levels = st.slider("Number of Tree Levels", 1, 10, 2)

        st.subheader("Dimensionality Reduction")
        dim_reduction_algorithms = ["t-SNE", "PCA"]
        selected_dim_reduction_algorithm = st.selectbox("Algorithm", dim_reduction_algorithms)
        tsne_iterations = 1000
        if selected_dim_reduction_algorithm == "t-SNE":
            tsne_iterations = st.slider("t-SNE iterations", 250, 2000, 1000, 50)

        show_legend = st.checkbox("Show Legend", value=True)
        dot_size = st.slider("Dot Size", 1, 20, 5, 1)

        process_button = st.button("Process")

    # 1) Load data if requested
    if load_data_button:
        with st.spinner("Loading data from DuckDB..."):
            try:
                st.session_state.df = load_data_from_duckdb(db_path, table_name, row_limit)
                st.success(f"Loaded {len(st.session_state.df)} rows from {table_name}")
            except Exception as e:
                st.error(f"Error loading from DuckDB: {e}")
                st.session_state.df = None

    # Show data preview (collapsible)
    if st.session_state.df is not None:
        with st.expander("Preview of the Loaded Data", expanded=True):
            st.subheader("Preview of the Loaded Data:")
            st.dataframe(st.session_state.df.head(5))
            st.write(f"Total rows: {len(st.session_state.df)}")

    # 2) Run the pipeline if user clicks
    if process_button and st.session_state.df is not None and selected_column_name:
        df = st.session_state.df
        texts = df[selected_column_name].astype(str).tolist()  # Convert column to string type

        # Embeddings
        if st.session_state.embeddings is None:
            embeddings = get_embeddings(
                ollama_url=st.session_state.ollama_url,
                model=selected_embedding_model,
                texts=texts,
                cache_prefix=f"{table_name}_{selected_column_name}_{row_limit}"
            )
            if embeddings is None or len(embeddings) == 0:
                st.error("Failed to get embeddings. Stopping.")
                return
            st.session_state.embeddings = embeddings
        else:
            embeddings = st.session_state.embeddings
            if embeddings.shape[0] != len(texts):
                st.warning("Loaded embeddings do not match row count! Proceed with caution.")

        # Build top-down tree
        st.write("Building the top-down hierarchy...")
        root_node = build_top_down_tree(
            embeddings=embeddings,
            texts=texts,
            indexes=list(range(len(texts))),
            current_level=1,
            max_levels=num_levels,
            k=n_clusters
        )

        # Post-order naming
        st.write("Naming clusters in post-order...")
        total_nodes = count_nodes(root_node)
        naming_progress_bar = st.progress(0)
        naming_time_text = st.empty()
        start_time = time.time()
        named_count = [0]

        postorder_naming(
            node=root_node,
            naming_progress_bar=naming_progress_bar,
            naming_time_text=naming_time_text,
            start_time=start_time,
            named_count=named_count,
            total_nodes=total_nodes,
            leaf_prompt_template=leaf_prompt_template,
            parent_prompt_template=parent_prompt_template,
            leaf_sample_size=leaf_sample_size,
            parent_sample_size=parent_sample_size,
            temperature=summary_temperature,
            summary_model=summary_model,
            ollama_url=st.session_state.ollama_url
        )

        # Extract leaves
        leaf_nodes = []
        extract_leaves(root_node, leaf_nodes)

        # Build summary table
        leaf_summary_table = create_leaf_summary_table(leaf_nodes)

        # Dimensionality reduction
        st.write("Performing dimensionality reduction...")
        dim_params = {}
        if selected_dim_reduction_algorithm == "t-SNE":
            dim_params['n_iter'] = tsne_iterations
        reduced_embeddings = perform_dimensionality_reduction(
            embeddings,
            selected_dim_reduction_algorithm,
            **dim_params
        )

        # Build taxonomy text (indented)
        taxonomy_lines = build_taxonomy_text(root_node)
        taxonomy_text = "\n".join(taxonomy_lines)

        # Display Treemap hierarchy (collapsible)
        with st.expander("Taxonomy (Hierarchical Naming) - Treemap Visualization", expanded=False):
            st.subheader("Taxonomy (Hierarchical Naming) - Treemap Visualization")
            fig_tree = create_taxonomy_treemap(root_node)
            st.plotly_chart(fig_tree, use_container_width=True)

        # Display leaf cluster summary (collapsible)
        with st.expander("Leaf Cluster Summary", expanded=False):
            st.subheader("Leaf Cluster Summary")
            st.table(leaf_summary_table[["Cluster Title", "# of Items", "Percentage", "Examples"]])

        # 3D scatter plot
        st.write("### 3D Visualization (Leaf Assignments)")
        fig_3d = create_3d_scatter_plot(
            reduced_embeddings=reduced_embeddings,
            leaf_nodes=leaf_nodes,
            df=df,
            dot_size=dot_size,
            show_legend=show_legend
        )
        st.plotly_chart(fig_3d, use_container_width=True)

if __name__ == "__main__":
    main()
