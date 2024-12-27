import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import json
import os
import ast
import re
import time

# Constants
DEFAULT_OLLAMA_URL = 'http://localhost:11434'
CACHE_FOLDER = "embeddings_cache"

os.makedirs(CACHE_FOLDER, exist_ok=True)

############################
#        Ollama API
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
def get_embeddings(ollama_url, model, texts, csv_filename):
    """
    Generate embeddings for the given texts using the specified Ollama model.
    Includes a per-text progress bar and time estimate.
    """
    cache_file = os.path.join(CACHE_FOLDER, f"{csv_filename}_{model}.npz")

    # Attempt to load from cache
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        cached_embeddings = data['embeddings']
        cached_row_count = int(data['row_count'])
        cached_model_name = str(data['model_name'])

        if cached_row_count == len(texts) and cached_model_name == model:
            st.info("Loading embeddings from cache...")
            return cached_embeddings

    st.write("Computing embeddings (this might take a while).")

    # Progress bar & time estimate placeholders
    embedding_progress_bar = st.progress(0)
    embedding_time_text = st.empty()

    embeddings = []
    start_time = time.time()

    # Generate embeddings item by item, updating progress & time estimate
    for i, text in enumerate(texts):
        text_str = str(text)
        try:
            response = requests.post(
                f'{ollama_url}/api/embeddings',
                json={'model': model, 'prompt': text_str},
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json()['embedding']
            embeddings.append(embedding)
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

        # Compute average time & estimate how much is left
        elapsed = time.time() - start_time
        avg_time_per_text = elapsed / current_count
        est_remaining = avg_time_per_text * (len(texts) - current_count)

        # Update time-estimate text
        embedding_time_text.text(
            f"Embedding {current_count}/{len(texts)} completed. "
            f"Estimated time remaining: {est_remaining:.2f} seconds."
        )

    # Convert to NumPy array
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
#      Summarization
############################
def generate_cluster_summary(
    ollama_url,
    model,
    texts,
    temperature,
    prompt_template,
    previous_cluster_name=None
):
    """
    Generate a cluster summary, optionally including a previously assigned cluster name as context.
    """
    first_five_texts_as_strings = [str(t) for t in texts[:5]]

    # Incorporate previous cluster name (if any) into the prompt
    if previous_cluster_name and previous_cluster_name.strip():
        prompt = (
            f"{prompt_template}\n\n"
            f"Previously, a similar cluster was named: {previous_cluster_name}\n\n"
            f"Here is the cluster content:\n"
            f"{''.join(first_five_texts_as_strings)}"
        )
    else:
        prompt = prompt_template.format(texts='\n'.join(first_five_texts_as_strings))

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
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        full_response += json_response['response']
                except json.JSONDecodeError:
                    continue
        return full_response.strip()
    except requests.RequestException as e:
        st.error(f"Error generating summary: {str(e)}")
        return "Failed to generate summary"

############################
#   Data Loading & Caching
############################
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)

    if 'user' in df.columns:
        def parse_user(x):
            if pd.isna(x) or not isinstance(x, str):
                return {}
            # Remove any occurrences of "datetime.datetime(...)" text
            pattern = r"'created': datetime\.datetime\([^)]*\)"
            cleaned = re.sub(pattern, "", x)
            # Also remove potential trailing commas if they remain
            cleaned = re.sub(r",(\s*,)+", ",", cleaned)
            cleaned = re.sub(r",\s*\}", "}", cleaned)
            cleaned = re.sub(r"\{\s*,", "{", cleaned)
            try:
                data = ast.literal_eval(cleaned)
                if 'created' in data:
                    del data['created']
                return data
            except:
                return {}

        df['user_dict'] = df['user'].apply(parse_user)

        def make_user_metadata_str(d):
            username = d.get('username', '')
            rawDescription = d.get('rawDescription', '')
            followersCount = d.get('followersCount', 0)
            location = d.get('location', '')
            favouritesCount = d.get('favouritesCount', 0)
            meta = {
                'username': username,
                'rawDescription': rawDescription,
                'followersCount': followersCount,
                'location': location,
                'favouritesCount': favouritesCount
            }
            return json.dumps(meta, indent=2)

        df['user_metadata_str'] = df['user_dict'].apply(make_user_metadata_str)
    else:
        df['user_metadata_str'] = ""

    return df

############################
#       Clustering
############################
def perform_clustering(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(embeddings)

def perform_dimensionality_reduction(embeddings, algorithm, n_components=3, **params):
    if algorithm == "t-SNE":
        reducer = TSNE(n_components=n_components, random_state=42, n_iter=params.get('n_iter', 1000))
    else:
        reducer = PCA(n_components=n_components, random_state=42)
    return reducer.fit_transform(embeddings)

############################
#    Visualization
############################
def create_cluster_summary_table(cluster_labels, texts, cluster_summaries):
    cluster_summary = []
    unique_labels = sorted(set(cluster_labels))
    total_rows = len(texts)

    for label in unique_labels:
        cluster_texts = [texts[i] for i, l in enumerate(cluster_labels) if l == label]
        cluster_size = len(cluster_texts)
        percentage = (cluster_size / total_rows * 100) if total_rows else 0
        examples = [str(x) for x in cluster_texts[:3]]

        cluster_summary.append({
            "Cluster": label,
            "Cluster Title": cluster_summaries.get(label, "Noise"),
            "# of Items": cluster_size,
            "Percentage": percentage,
            "Examples": "\n\n".join(examples)
        })

    df_summary = pd.DataFrame(cluster_summary)
    return df_summary.sort_values("# of Items", ascending=False).reset_index(drop=True)

def clean_html_tags(text):
    """
    Removes any <...> HTML tags from the string so they do not appear as literal HTML.
    """
    return re.sub(r'<[^>]*>', '', text)

def create_3d_scatter_plot(
    reduced_embeddings,
    cluster_labels,
    cluster_summaries,
    texts,
    summary_table,
    df,
    show_legend=True
):
    df_plot = df.copy()
    df_plot['x'] = reduced_embeddings[:, 0]
    df_plot['y'] = reduced_embeddings[:, 1]
    df_plot['z'] = reduced_embeddings[:, 2]

    unique_labels = sorted(set(cluster_labels))
    color_discrete_map = {
        label: f'rgb({np.random.randint(0,256)}, {np.random.randint(0,256)}, {np.random.randint(0,256)})'
        for label in unique_labels
    }

    fig = go.Figure()

    def wrap_text(text, width=40):
        """Wrap text to specified width"""
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

    for label in unique_labels:
        cluster_data = df_plot[df_plot['cluster_label'] == label].copy()
        if cluster_data.empty:
            continue

        try:
            cluster_info = summary_table[summary_table['Cluster'] == label].iloc[0]
            cluster_title = cluster_info['Cluster Title']
            cluster_items = cluster_info['# of Items']
            cluster_percentage = cluster_info['Percentage']
        except:
            cluster_title = "Unknown"
            cluster_items = 0
            cluster_percentage = 0

        def make_customdata(row):
            cluster_name = wrap_text(str(row.get('cluster_name', '')).strip())
            tweet_text = wrap_text(str(row.get('text', '')).strip())

            try:
                metadata = json.loads(str(row.get('user_metadata_str', '{}')).strip())
                metadata_text = '\n'.join([f"{k}: {v}" for k, v in metadata.items() if v])
                metadata_text = wrap_text(metadata_text)
            except:
                metadata_text = ''

            return (cluster_name, tweet_text, metadata_text)

        cluster_data['customdata'] = cluster_data.apply(make_customdata, axis=1)
        cluster_color = color_discrete_map[label]

        hover_template = (
            "<b>Cluster:</b> %{customdata[0]}<br><br>" +
            "<b>Tweet text:</b> %{customdata[1]}<br><br>" +
            "<b>Tweet Metadata:</b> %{customdata[2]}" +
            "<extra></extra>"
        )

        fig.add_trace(go.Scatter3d(
            x=cluster_data['x'],
            y=cluster_data['y'],
            z=cluster_data['z'],
            mode='markers',
            marker=dict(
                size=5,
                color=cluster_color,
                opacity=0.8
            ),
            customdata=cluster_data['customdata'],
            hovertemplate=hover_template,
            hoverlabel=dict(
                bgcolor=cluster_color,
                font=dict(
                    color='white',
                    size=12
                ),
                align='left',
                namelength=-1
            ),
            name=(
                f"Cluster {label}: {cluster_title} "
                f"({cluster_items} items, {cluster_percentage:.2f}%)"
            )
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
        title="3D Text Clustering Visualization",
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
#          MAIN
############################
def main():
    st.set_page_config(page_title="TextCluster", page_icon="📊", layout="wide")
    st.title("📊 Embedding & Clustering Experiments")

    # CHANGED: Initialize storage in session_state for final results
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.embeddings = None
        st.session_state.cluster_labels = None
        st.session_state.cluster_summaries = None
        st.session_state.summary_table = None
        st.session_state.reduced_embeddings = None
        st.session_state.selected_texts = None
        st.session_state.column_index = None

    # Make sure session_state has a place to store old cluster names
    if 'previous_cluster_names' not in st.session_state:
        st.session_state.previous_cluster_names = {}

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

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            df = load_csv(uploaded_file)
            st.write(f"Total rows in CSV: {len(df)}")
            column_options = [f"{i}: {col}" for i, col in enumerate(df.columns)]
            selected_column = st.selectbox("Select the text column to embed", column_options)
            column_index = int(selected_column.split(":")[0])

        embedding_models = get_ollama_models(st.session_state.ollama_url)
        selected_embedding_model = st.selectbox("Select the embedding model", embedding_models)

        summary_model = st.selectbox("Select the summary model", embedding_models)
        summary_temperature = st.slider("Summary Temperature", 0.0, 1.0, 0.1, 0.1)
        summary_prompt_template = st.text_area(
            "Summary Prompt Template",
            "You are a highly capable text-summarization assistant. You will be provided with a cluster of tweets "
            "about the 2024 U.S. presidential election.\n\n"
            "Your goal is to produce a concise, descriptive title that captures the main theme or topic of those tweets.\n\n"
            "Important rules and constraints:\n"
            "1. Output Format: Only produce a short, single-line title.\n"
            "2. No Extra Text: Do not add disclaimers, explanations, or any other content besides the final title.\n"
            "3. Relevance: Ensure the title accurately reflects the common theme.\n"
            "4. Neutrality: Do not inject personal opinions, biases, or speculation.\n"
            "5. Privacy: Do not include names, usernames, or personal details.\n"
            "6. Brevity: Keep the title concise (under 15 words, if possible).\n\n"
            "cluster content:\n{texts}"
        )

        

        n_clusters = st.slider("Number of clusters", 2, 100, 5)

        dim_reduction_algorithms = ["t-SNE", "PCA"]
        selected_dim_reduction_algorithm = st.selectbox(
            "Select the dimensionality reduction algorithm",
            dim_reduction_algorithms
        )
        tsne_iterations = 1000
        if selected_dim_reduction_algorithm == "t-SNE":
            tsne_iterations = st.slider("t-SNE iterations", 250, 2000, 1000, 50)

        show_legend = st.checkbox("Show Legend", value=True)

        process_button = st.button("Process")

    # If a CSV is uploaded, preview
    if uploaded_file is not None:
        st.subheader("Preview of the uploaded data:")
        st.dataframe(df.head(5))
        st.write(f"Total rows: {len(df)}")

    if process_button and uploaded_file is not None:
        # 1) Extract texts
        texts = df.iloc[:, column_index].tolist()

        # 2) Embeddings
        embeddings = get_embeddings(
            st.session_state.ollama_url,
            selected_embedding_model,
            texts,
            csv_filename=uploaded_file.name
        )
        if embeddings is None or len(embeddings) == 0:
            st.error("Failed to get embeddings. Stopping.")
            return

        #st.success("Embeddings step complete!")

        # 3) Clustering
        st.write("Performing clustering...")
        cluster_labels = perform_clustering(embeddings, n_clusters)
        #st.success("Clustering step complete!")

        # 4) Summaries
        st.write("Generating cluster summaries...")
        unique_labels = sorted(set(cluster_labels))
        cluster_summaries = {}

        summary_progress_bar = st.progress(0)
        summary_time_text = st.empty()
        start_time = time.time()

        for i, label in enumerate(unique_labels):
            cluster_texts = [texts[idx] for idx, lbl in enumerate(cluster_labels) if lbl == label]
            previous_name = st.session_state.previous_cluster_names.get(label, "")

            # Generate the summary
            summary = generate_cluster_summary(
                st.session_state.ollama_url,
                summary_model,
                cluster_texts,
                summary_temperature,
                summary_prompt_template,
                previous_cluster_name=previous_name
            )

            # Save & store
            cluster_summaries[label] = summary
            st.session_state.previous_cluster_names[label] = summary

            # Update progress
            current_count = i + 1
            fraction_done = current_count / len(unique_labels)
            summary_progress_bar.progress(fraction_done)

            elapsed = time.time() - start_time
            avg_time_per_cluster = elapsed / current_count
            est_remaining = avg_time_per_cluster * (len(unique_labels) - current_count)
            summary_time_text.text(
                f"Summaries {current_count}/{len(unique_labels)} complete. "
                f"Estimated time remaining: {est_remaining:.2f} seconds."
            )

        #st.success("Cluster summaries step complete!")

        # 5) Create summary table
        summary_table = create_cluster_summary_table(cluster_labels, texts, cluster_summaries)
        st.subheader("Cluster Summary")
        st.table(summary_table[['Cluster Title', '# of Items', 'Percentage', 'Examples']])

        df['cluster_label'] = cluster_labels
        df['cluster_name'] = df['cluster_label'].apply(lambda c: cluster_summaries.get(c, "Unknown Cluster"))

        # 6) Dimensionality reduction
        st.write("Performing dimensionality reduction...")
        dim_reduction_params = {}
        if selected_dim_reduction_algorithm == "t-SNE":
            dim_reduction_params['n_iter'] = tsne_iterations

        reduced_embeddings = perform_dimensionality_reduction(
            embeddings,
            selected_dim_reduction_algorithm,
            **dim_reduction_params
        )
        #st.success(f"{selected_dim_reduction_algorithm} reduction complete!")

        # CHANGED: Store results in session_state so we can re-display later
        st.session_state.df = df
        st.session_state.embeddings = embeddings
        st.session_state.cluster_labels = cluster_labels
        st.session_state.cluster_summaries = cluster_summaries
        st.session_state.summary_table = summary_table
        st.session_state.reduced_embeddings = reduced_embeddings
        st.session_state.selected_texts = texts
        st.session_state.column_index = column_index

    # NEW: If we have stored results, display them again (including summary table!)
    if st.session_state.df is not None and st.session_state.summary_table is not None:
        st.subheader("Cluster Summary")  # Always show table
        st.table(st.session_state.summary_table[['Cluster Title', '# of Items', 'Percentage', 'Examples']])

        st.write("### 3D Visualization")
        fig = create_3d_scatter_plot(
            st.session_state.reduced_embeddings,
            st.session_state.cluster_labels,
            st.session_state.cluster_summaries,
            st.session_state.selected_texts,
            st.session_state.summary_table,
            st.session_state.df,
            show_legend=show_legend  # This will update on toggle without re-processing
        )
        st.plotly_chart(fig, use_container_width=True)

        #st.success("All steps complete!")


if __name__ == "__main__":
    main()
